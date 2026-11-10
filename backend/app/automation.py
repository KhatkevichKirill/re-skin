"""
Run automation — scheduled batch generation from the persona spreadsheet.

An operator enables automation on a proven Run (the *template*). The periodic
automation tick (app/tasks.py → run_automation_tick) then, for every enabled
automation:

1. **Reconcile in-flight batches** — when every run of a launched batch has
   reached a terminal status, POST the batch-completion webhook (result links
   for QA) and stamp the batch as notified. Webhook failures are retried on
   the next tick (the batch simply stays un-notified).
2. **Launch the next batch** — if the previous batch is fully notified and the
   UTC-day quota (``daily_count``) is not yet exhausted, pull the next unused
   sheet rows matching the automation's filter rule and clone the template Run
   once per row via ``_build_copied_run`` (same path as the Copy-run UI). With
   ``preprocess_refs`` set, each row's photo is first re-run through the
   template's GPT Image 2 recipe (same frames, same image prompt) and
   auto-approved, so those launches enter the pipeline at ``preprocessing``
   rather than ``queued`` — the CALLER picks the matching enqueue.

Sheet contract ("Daily Queue" tab, fixed columns):
    A  Queue ID   — used as the new run's name and as the consumed-row key
    C  Persona    D  AGE    E  GENDER    F  ETHNICITY   — filterable
    R  image_on_gdrive — reference photo for plan="close-up"
    Y  link            — reference photo for plan="midplan"

The tick is a pure reconciler: idempotent, safe to re-run, and everything it
launches goes through the normal run pipeline (locks, recovery, delivery).
"""

from __future__ import annotations

import logging
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, time, timezone
from typing import Callable, Optional

import httpx
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from . import spend as spend_mod
from .config import settings
from .models import AutomationLaunch, Run, RunAutomation, RunSegmentRangeAttempt, VideoProject
from .project_types import run_copies_are_useful
from .sheets_client import SheetsClient, SheetsError
from . import webhook_safety
from .state_machine import RANGE_ATTEMPT_ACTIVE_STATUSES, ProjectStatus, RunStatus

log = logging.getLogger(__name__)

# Sheet column positions (0-based) — see module docstring.
_COL_QUEUE_ID = 0    # A
_COL_PERSONA = 2     # C
_COL_AGE = 3         # D
_COL_GENDER = 4      # E
_COL_ETHNICITY = 5   # F
_COL_CLOSEUP = 17    # R (image_on_gdrive)
_COL_MIDPLAN = 24    # Y (link)
_SHEET_LAST_COLUMN = "Y"

PLAN_CLOSE_UP = "close-up"
PLAN_MIDPLAN = "midplan"
ALLOWED_PLANS = (PLAN_CLOSE_UP, PLAN_MIDPLAN)

# Run statuses that mean "this run will not change anymore on its own".
# `incomplete` is terminal for webhook purposes: it waits for an operator
# re-run and would otherwise block the batch (and all future batches) forever.
_TERMINAL_RUN_STATUSES = {RunStatus.done, RunStatus.failed, RunStatus.incomplete}

_WEBHOOK_TIMEOUT_SEC = 30


@dataclass
class QueueRow:
    """One parsed row of the Daily Queue sheet."""

    queue_id: str
    persona: str
    age: str
    gender: str
    ethnicity: str
    closeup_url: str
    midplan_url: str

    def ref_url_for_plan(self, plan: str) -> str:
        return self.closeup_url if plan == PLAN_CLOSE_UP else self.midplan_url


def _cell(row: list[str], idx: int) -> str:
    return row[idx].strip() if idx < len(row) else ""


def parse_queue_rows(raw_rows: list[list[str]]) -> list[QueueRow]:
    """Parse raw sheet rows, dropping rows without a Queue ID."""
    out: list[QueueRow] = []
    for row in raw_rows:
        queue_id = _cell(row, _COL_QUEUE_ID)
        if not queue_id:
            continue
        out.append(
            QueueRow(
                queue_id=queue_id,
                persona=_cell(row, _COL_PERSONA),
                age=_cell(row, _COL_AGE),
                gender=_cell(row, _COL_GENDER),
                ethnicity=_cell(row, _COL_ETHNICITY),
                closeup_url=_cell(row, _COL_CLOSEUP),
                midplan_url=_cell(row, _COL_MIDPLAN),
            )
        )
    return out


def _matches(
    filter_value: Optional[str], cell_value: str, *, exclude: bool = False
) -> bool:
    """List match: *filter_value* is one or more comma-separated values, each
    compared exactly (case-insensitive, trimmed) to the cell. A row matches
    when the cell equals ANY listed value; with *exclude* the result inverts —
    rows matching any listed value are skipped. Empty filter = any (the
    exclude flag has nothing to act on)."""
    if not filter_value or not filter_value.strip():
        return True
    wanted = {v.strip().lower() for v in filter_value.split(",") if v.strip()}
    if not wanted:
        return True
    hit = cell_value.strip().lower() in wanted
    return not hit if exclude else hit


# AGE cells hold ranges ("18–22", sometimes with a suffix like
# "26–30 (mature student)") with assorted dash characters.
_AGE_RANGE_RE = re.compile(r"(\d+)\s*[–—-]\s*(\d+)")
_AGE_NUM_RE = re.compile(r"\d+")


def _parse_age_range(text: str) -> Optional[tuple[int, int]]:
    """Extract an inclusive (lo, hi) age range from *text*; a lone number
    yields (n, n); None when no number is present."""
    m = _AGE_RANGE_RE.search(text)
    if m:
        lo, hi = sorted((int(m.group(1)), int(m.group(2))))
        return lo, hi
    m = _AGE_NUM_RE.search(text)
    if m:
        n = int(m.group())
        return n, n
    return None


def _age_matches(filter_value: Optional[str], cell_value: str) -> bool:
    """Age-aware matching: a single number matches any row whose AGE range
    contains it ("18" → "18–22"); a range matches the same numeric range no
    matter the dash style ("18-22" → "18–22"); anything non-numeric falls back
    to the exact match used by the other filters. Empty filter = any."""
    if not filter_value or not filter_value.strip():
        return True
    wanted = filter_value.strip()
    cell_range = _parse_age_range(cell_value)
    if wanted.isdigit():
        n = int(wanted)
        return cell_range is not None and cell_range[0] <= n <= cell_range[1]
    wanted_range = _AGE_RANGE_RE.search(wanted) and _parse_age_range(wanted)
    if wanted_range and cell_range:
        return wanted_range == cell_range
    return wanted.lower() == cell_value.strip().lower()


def match_rows(
    rows: list[QueueRow], automation: RunAutomation
) -> list[QueueRow]:
    """Rows matching the automation's filter rule that have a usable photo URL
    for its plan."""
    out: list[QueueRow] = []
    for r in rows:
        if not _matches(
            automation.filter_persona, r.persona, exclude=automation.exclude_persona
        ):
            continue
        if not _age_matches(automation.filter_age, r.age):
            continue
        if not _matches(
            automation.filter_gender, r.gender, exclude=automation.exclude_gender
        ):
            continue
        if not _matches(
            automation.filter_ethnicity, r.ethnicity,
            exclude=automation.exclude_ethnicity,
        ):
            continue
        url = r.ref_url_for_plan(automation.plan)
        if not (url.startswith("http://") or url.startswith("https://")):
            continue
        out.append(r)
    return out


def _utc_day_start(now: datetime) -> datetime:
    return datetime.combine(now.date(), time.min, tzinfo=timezone.utc)


def launched_today(db: Session, automation_id: str, now: datetime) -> int:
    """Number of launches this automation fired in the current UTC day."""
    day_start = _utc_day_start(now).replace(tzinfo=None)
    return db.execute(
        select(func.count()).select_from(AutomationLaunch).where(
            AutomationLaunch.automation_id == automation_id,
            AutomationLaunch.launched_at >= day_start,
        )
    ).scalar_one()


# ---------------------------------------------------------------------------
# Webhook
# ---------------------------------------------------------------------------


def _run_links(run: Optional[Run]) -> dict:
    """Absolute result/QA links for a run (best-effort; missing → None)."""
    if run is None:
        return {"run_url": None, "result_url": None, "drive_link": None}
    base = settings.PUBLIC_BASE_URL
    drive_link = (
        f"https://drive.google.com/file/d/{run.result_gdrive_file_id}/view"
        if run.result_gdrive_file_id
        else None
    )
    run_url = f"{base}/v2/runs/{run.id}" if base else None
    result_url = None
    if base and run.result_local_path:
        from .public import make_result_token  # late import: avoids app import cycle

        # AICP-151: "" = public links disabled (no secret) -> omit, never ship
        # a link that cannot work.
        token = make_result_token(run.id)
        if token:
            result_url = f"{base}/public/runs/{run.id}/result?token={token}"
    return {"run_url": run_url, "result_url": result_url, "drive_link": drive_link}


def build_batch_payload(
    automation: RunAutomation,
    batch_id: str,
    launches: list[AutomationLaunch],
    runs_by_id: dict[str, Run],
) -> dict:
    """JSON body for the batch-completion webhook."""
    items = []
    for launch in launches:
        run = runs_by_id.get(launch.new_run_id or "")
        status_val = None
        if run is not None:
            status_val = (
                run.status.value if hasattr(run.status, "value") else str(run.status)
            )
        items.append(
            {
                "queue_id": launch.queue_id,
                "ref_url": launch.ref_url,
                "run_id": launch.new_run_id,
                "run_name": run.name if run else None,
                "status": status_val or "deleted",
                "error": run.error_message if run else None,
                # Additive field. A run can be `status: "done"` and still have
                # delivered the ORIGINAL footage for swap segments its model was
                # unable to generate (too long for the model's per-clip limit) —
                # see RunSegment.source_fallback. Without this, a batch of
                # partly-swapped videos reports as clean success downstream.
                # 0 = fully swapped; null = not recorded (run predates the
                # column, was deleted, or never finalized).
                "unswapped_segments": (
                    run.source_fallback_segments if run else None
                ),
                **_run_links(run),
            }
        )
    return {
        "event": "automation.batch.completed",
        "batch_id": batch_id,
        "automation_id": automation.id,
        "template_run_id": automation.run_id,
        "plan": automation.plan,
        "filters": {
            "persona": automation.filter_persona,
            "age": automation.filter_age,
            "gender": automation.filter_gender,
            "ethnicity": automation.filter_ethnicity,
            "exclude": {
                "persona": automation.exclude_persona,
                "gender": automation.exclude_gender,
                "ethnicity": automation.exclude_ethnicity,
            },
        },
        "runs": items,
    }


# httpx logs every request URL at INFO; for a webhook that is the secret path.
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

_warned_disabled = False


def _warn_disabled_once() -> None:
    """Log (once per process) that no webhook is configured."""
    global _warned_disabled
    if not _warned_disabled:
        _warned_disabled = True
        log.warning(
            "automation webhook disabled: AUTOMATION_WEBHOOK_URL is unset and "
            "the automation has no webhook_url"
        )


def _post_webhook(url: str, payload: dict) -> bool:
    """POST *payload* to *url*; True on 2xx.

    Raises WebhookRefused when the URL breaks the policy (permanent); any
    other failure returns False (retried next tick). Logs host only.
    """
    try:
        targets = webhook_safety.pinned_targets(url)
    except webhook_safety.WebhookRefused:
        raise
    except Exception as exc:  # incl. WebhookUnresolved: retry next tick
        log.warning(
            "automation webhook: %s could not be prepared (%s)",
            webhook_safety.redact(url), type(exc).__name__,
        )
        return False
    # Connect to a vetted IP (no second DNS lookup); Host + SNI keep the name
    # so TLS still verifies against it. Redirects are never followed. Only a
    # connection-level failure moves on to the next vetted address.
    for ip_url, host, port, _ip in targets:
        try:
            # httpx.post() has no `extensions` kwarg (0.27); Client.post does.
            with httpx.Client(
                timeout=_WEBHOOK_TIMEOUT_SEC, follow_redirects=False,
                trust_env=False,  # no env proxy: it would bypass the IP pin / SNI
            ) as client:
                resp = client.post(
                    ip_url,
                    json=payload,
                    headers={"Host": host if port == 443 else f"{host}:{port}"},
                    extensions={"sni_hostname": host},
                )
        except Exception as exc:
            log.warning(
                "automation webhook: POST to %s failed (%s)",
                webhook_safety.redact(url), type(exc).__name__,
            )
            continue
        if 200 <= resp.status_code < 300:
            return True
        log.warning(
            "automation webhook: %s returned HTTP %d",
            webhook_safety.redact(url), resp.status_code,
        )
        return False
    return False


def _sheet_rows_provider(sheets: SheetsClient) -> Callable[[], list[list[str]]]:
    """Build a zero-arg callable that reads the Daily Queue tab AT MOST ONCE,
    memoizing across every call in the tick — success or failure alike.

    The spreadsheet id / tab / last-column are the same settings values for
    every automation in a tick, so re-reading per automation (the old
    behavior) was pure waste. The read is still lazy: nothing hits the Sheets
    API until the first automation that actually needs it (quota-exhausted /
    template-gone / project-not-ready automations never call this).

    Failure handling: a failed read is cached too, so a Sheets outage costs
    exactly one network round-trip per tick, not one per enabled automation.
    Each automation that subsequently calls this still gets its own
    ``SheetsError`` raised (from the cached exception) and logs its own
    "sheet read failed" warning in ``_launch_batch`` — the log output is
    byte-for-byte what it was when every automation read independently, only
    the underlying network call is deduplicated.
    """
    state: dict = {}

    def _read() -> list[list[str]]:
        if "rows" not in state and "error" not in state:
            try:
                state["rows"] = sheets.read_rows(
                    settings.AUTOMATION_SPREADSHEET_ID,
                    settings.AUTOMATION_SHEET_TAB,
                    last_column=_SHEET_LAST_COLUMN,
                )
            except SheetsError as exc:
                state["error"] = exc
        if "error" in state:
            raise state["error"]
        return state["rows"]

    return _read


# ---------------------------------------------------------------------------
# Tick phases
# ---------------------------------------------------------------------------


def _reconcile_batches(db: Session, automation: RunAutomation, now: datetime) -> bool:
    """Send webhooks for finished batches. Returns True when NO batch of this
    automation is still blocking (in-flight or awaiting webhook delivery)."""
    pending = (
        db.execute(
            select(AutomationLaunch).where(
                AutomationLaunch.automation_id == automation.id,
                AutomationLaunch.notified_at.is_(None),
            )
        )
        .scalars()
        .all()
    )
    if not pending:
        return True

    by_batch: dict[str, list[AutomationLaunch]] = {}
    for launch in pending:
        by_batch.setdefault(launch.batch_id, []).append(launch)

    all_clear = True
    for batch_id, launches in by_batch.items():
        run_ids = [l.new_run_id for l in launches if l.new_run_id]
        runs = (
            db.execute(select(Run).where(Run.id.in_(run_ids))).scalars().all()
            if run_ids
            else []
        )
        runs_by_id = {r.id: r for r in runs}
        # A deleted run (id not found) counts as terminal — never block on it.
        unfinished = [
            r for r in runs if r.status not in _TERMINAL_RUN_STATUSES
        ]
        if unfinished:
            all_clear = False
            continue

        url = (automation.webhook_url or "").strip() or settings.AUTOMATION_WEBHOOK_URL
        payload = build_batch_payload(automation, batch_id, launches, runs_by_id)
        delivered = False
        if url:
            try:
                if not _post_webhook(url, payload):
                    all_clear = False  # retry on the next tick
                    continue
                delivered = True
            except webhook_safety.WebhookRefused as exc:
                # Policy refusal is permanent: retrying would block this
                # automation forever. Same outcome as "no webhook".
                log.warning(
                    "automation %s: webhook refused (%s) — marking batch %s "
                    "notified without delivery", automation.id, exc, batch_id,
                )
        else:
            _warn_disabled_once()
        stamp = now.replace(tzinfo=None)
        for launch in launches:
            launch.notified_at = stamp
        db.commit()
        log.info(
            "automation %s: batch %s completed, webhook %s (%d runs)",
            automation.id, batch_id,
            "delivered" if delivered else "not delivered", len(launches),
        )
    return all_clear


def _launch_batch(
    db: Session,
    automation: RunAutomation,
    now: datetime,
    get_rows: Callable[[], list[list[str]]],
) -> list[str]:
    """Launch the next batch for *automation*; returns new run ids (may be []).

    *get_rows* is a zero-arg callable shared by every automation in the tick
    (see ``_sheet_rows_provider``) — it reads the Daily Queue tab at most once
    per tick regardless of how many automations call it.
    """
    # AICP-350 deploy admission gate: while paused, skip the launch — consume
    # no quota, no sheet row, launch nothing (design §2: "skip the tick").
    # A cheap unlocked read here; the authoritative share lock is taken right
    # before the batch is built, below.
    from . import ops_gate

    if ops_gate.read_gate(db).paused:
        log.info("automation %s: deploy admission gate is paused — skipping", automation.id)
        return []

    launched_before = launched_today(db, automation.id, now)
    remaining = automation.daily_count - launched_before
    if remaining <= 0:
        return []

    template = db.get(Run, automation.run_id)
    if template is None:
        log.warning("automation %s: template run is gone — skipping", automation.id)
        return []
    project = db.get(VideoProject, template.project_id)
    if project is None or project.status != ProjectStatus.ready:
        log.info(
            "automation %s: project not ready (%s) — skipping",
            automation.id, project.status if project else "missing",
        )
        return []

    # A speech-only localisation template can only produce copies of itself:
    # its prompt keeps the person on screen, so the sheet's reference photo is
    # inert (docs/localisation.md §7). The API refuses to ENABLE such an
    # automation; this is the second half of that rule, for one enabled before
    # it existed or whose template was edited afterwards — without it the
    # refusal would only bind on new automations, and the old ones would keep
    # spending a daily quota on the same clip. Deliberately here and not in
    # _reconcile_batches: a batch already in flight still needs its webhook.
    if not run_copies_are_useful(project.project_type, template.prompt):
        log.warning(
            "automation %s: template run %s is a speech-only localisation run "
            "— every copy would repeat it, so nothing is launched",
            automation.id, template.id,
        )
        return []

    # A run-local segment range regeneration in flight on the template Run
    # means Run.segment_snapshot (and possibly result_local_path) may be
    # replaced by a promotion at any moment — cloning right now risks either
    # a 409 race against a concurrent Copy, or (worse) silently cloning the
    # PRE-correction range in the window before it promotes. Skip this tick
    # for this automation without touching quota, sheet rows, or launching
    # anything; the next tick tries again, and a successful promotion is
    # picked up automatically (the template is re-read fresh every tick).
    active_attempt = db.execute(
        select(RunSegmentRangeAttempt.id).where(
            RunSegmentRangeAttempt.run_id == template.id,
            RunSegmentRangeAttempt.status.in_(
                [s.value for s in RANGE_ATTEMPT_ACTIVE_STATUSES]
            ),
        ).limit(1)
    ).scalar_one_or_none()
    if active_attempt is not None:
        log.info(
            "automation %s: template run %s has a range regeneration in "
            "flight (attempt %s) — skipping this tick, no quota consumed",
            automation.id, template.id, active_attempt,
        )
        return []

    try:
        raw = get_rows()
    except SheetsError as exc:
        log.warning("automation %s: sheet read failed: %s", automation.id, exc)
        return []

    candidates = match_rows(parse_queue_rows(raw), automation)
    if not candidates:
        log.info(
            "automation %s: no unused sheet rows match the filter — nothing to launch",
            automation.id,
        )
        return []

    # Bound the dedupe queries by the candidate Queue IDs from this tick's
    # sheet read instead of scanning the automation's entire launch history /
    # the project's entire run set (both grow forever).
    candidate_ids = [r.queue_id for r in candidates]
    lowered_candidate_ids = {qid.strip().lower() for qid in candidate_ids}

    used = {
        row[0]
        for row in db.execute(
            select(AutomationLaunch.queue_id).where(
                AutomationLaunch.automation_id == automation.id,
                AutomationLaunch.queue_id.in_(candidate_ids),
            )
        ).all()
    }
    # Project-level dedup: skip a Queue ID when ANY run in this project already
    # carries it as its name — no matter whether it was launched by this
    # automation, another automation on a sibling run, or manually via Copy run
    # with the Queue ID typed as the name. `func.trim` + `func.lower` narrows
    # the row set to plausible matches (handles the common case of surrounding
    # spaces, exactly what the Copy-run UI can produce); the exact
    # `.strip().lower()` comparison below still runs in Python on that small
    # result so the final membership test is byte-for-byte what it was before.
    existing_names = {
        (name or "").strip().lower()
        for (name,) in db.execute(
            select(Run.name).where(
                Run.project_id == template.project_id,
                func.lower(func.trim(Run.name)).in_(lowered_candidate_ids),
            )
        ).all()
    }
    fresh = [
        r
        for r in candidates
        if r.queue_id not in used
        and r.queue_id.strip().lower() not in existing_names
    ]
    take = fresh[: min(remaining, settings.AUTOMATION_MAX_PER_BATCH)]
    if not take:
        log.info(
            "automation %s: no unused sheet rows match the filter — nothing to launch",
            automation.id,
        )
        return []

    # Late import — api_v2 imports app.tasks at module load; keep the module
    # graph acyclic from tasks.py's point of view.
    from .api_v2 import _build_copied_run

    # Re-run the template's GPT Image 2 phase on each sheet photo. Only
    # meaningful on a template that used preprocessing; the API refuses to set
    # the flag otherwise, and this re-check keeps a template that was replaced
    # (or a row written before that guard existed) from failing every launch.
    preprocess = bool(
        getattr(automation, "preprocess_refs", False) and template.preprocess_enabled
    )
    if getattr(automation, "preprocess_refs", False) and not preprocess:
        log.warning(
            "automation %s: preprocess_refs is set but template run %s does not "
            "use preprocessing — launching with the raw sheet photo",
            automation.id, template.id,
        )

    # Daily kie.ai budget (AICP-155): launch only as many copies as today's
    # remaining budget covers. Rows that do not fit are left untouched — no
    # AutomationLaunch, so neither the sheet row nor the daily quota is
    # consumed — and the next tick (or tomorrow, UTC) picks them up.
    headroom = spend_mod.headroom_credits(db, now)
    if headroom is not None:
        per_copy = spend_mod.estimate_run_credits(
            db, template, include_preprocess=preprocess, fresh=True,
        )
        fits = len(take) if per_copy <= 0 else int(headroom // per_copy)
        if fits < len(take):
            log.warning(
                "automation %s: daily kie.ai budget covers %d of %d copies "
                "(~%.0f credits each, %.0f left today) — launching %d, the "
                "rest wait",
                automation.id, max(fits, 0), len(take), per_copy, headroom,
                max(fits, 0),
            )
            take = take[: max(fits, 0)]
        if not take:
            return []

    # The admission barrier (app/ops_gate.py): the gate row FOR SHARE, held
    # until this batch commits, so a deploy's `begin` waits for it and no batch
    # starts after `begin`. Taken before any row of the batch is written and
    # before the day lock (spend.reserve_new_runs takes that last).
    try:
        ops_gate.lock_admitting(db)
    except ops_gate.AdmissionPaused:
        db.rollback()
        log.info("automation %s: deploy admission gate paused — batch not launched", automation.id)
        return []

    batch_id = str(uuid.uuid4())
    stamp = now.replace(tzinfo=None)
    new_run_ids: list[str] = []
    new_runs: list[Run] = []
    for row in take:
        ref_url = row.ref_url_for_plan(automation.plan)
        new_run = _build_copied_run(
            template, db,
            resolution=None,           # keep the template's resolution
            name=row.queue_id,         # run name = Queue ID (column A)
            ref_files=[],
            ref_urls=[ref_url],
            preprocess=preprocess,
        )
        db.add(
            AutomationLaunch(
                automation_id=automation.id,
                batch_id=batch_id,
                queue_id=row.queue_id,
                ref_url=ref_url,
                new_run_id=new_run.id,
                launched_at=stamp,
            )
        )
        new_run_ids.append(new_run.id)
        new_runs.append(new_run)

    # The authoritative check, for the whole batch at once and after every
    # build (the day lock is not held across them). Lost a race to another
    # spender since the pre-trim above: roll the WHOLE batch back — nothing is
    # launched, no quota or sheet row is consumed — and retry next tick.
    try:
        spend_mod.reserve_new_runs(db, new_runs, kind="automation launch", now=now)
    except spend_mod.BudgetExceeded as exc:
        db.rollback()
        log.warning(
            "automation %s: batch of %d refused by the daily kie.ai budget — "
            "nothing launched, retrying next tick: %s",
            automation.id, len(new_runs), exc,
        )
        return []
    db.commit()
    log.info(
        "automation %s: launched batch %s — %d run(s) %s (quota %d/%d today)",
        automation.id, batch_id, len(new_run_ids), new_run_ids,
        launched_before + len(new_run_ids), automation.daily_count,
    )
    return new_run_ids


def run_tick(
    db: Session,
    *,
    sheets: Optional[SheetsClient] = None,
    now: Optional[datetime] = None,
) -> list[str]:
    """One automation tick over all enabled automations.

    Returns the ids of newly created runs; the CALLER is responsible for
    enqueueing them (keeps this module free of RQ imports and lets tests run
    the tick without Redis).
    """
    now = now or datetime.now(timezone.utc)
    automations = (
        db.execute(select(RunAutomation).where(RunAutomation.enabled.is_(True)))
        .scalars()
        .all()
    )
    if not automations:
        return []
    sheets = sheets or SheetsClient()
    get_rows = _sheet_rows_provider(sheets)

    created: list[str] = []
    for automation in automations:
        try:
            clear = _reconcile_batches(db, automation, now)
            if not clear:
                continue  # previous batch still in flight / webhook pending
            created.extend(_launch_batch(db, automation, now, get_rows))
        except Exception:
            db.rollback()
            log.exception(
                "automation %s: tick failed — continuing with others", automation.id
            )
    return created
