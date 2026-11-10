"""Import context: one batch, its report, and idempotent wrappers over the store.

Every row an import creates carries ``origin=import`` / ``import_batch_id``
where the schema has the column, and is listed in ``kb_import_members`` under
its kind (docs/knowledge-storage.md), so invalidating the batch withdraws it.
Rows found already present (an earlier batch, or live capture) are reused and
never re-registered: a rerun creates nothing twice.

Idempotency keys are derived from the source table, source id, kind and the
current case id. Keys are unique forever (storage contract), so a key whose
row was withdrawn is skipped for the next ``~n`` suffix, never replayed.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Iterable, Optional

from sqlalchemy import false, or_, select, text
from sqlalchemy.orm import Session

from .. import store
from .. import taxonomy as t
from ..access import KnowledgeScope
from ..models import (
    KbArtifact, KbAttempt, KbAttemptObservation, KbAttemptSubmission, KbCase, KbEvalManifest, KbFeedbackEvent,
    KbRequest, KbRuleEvidence, KbRuleVersion,
)

LIST_CAP = 200
MANIFEST_SHA256 = "ea5f5d52ce1d5b5d8485fb5d29116090d9cf337113ca35405f843bf194d2766f"
_MANIFEST_REL = os.path.join("docs", "knowledge", "holdout-manifest-v1.json")
_HERE = os.path.dirname(__file__)
# Repo checkout: <repo>/backend/app/knowledge/importer -> <repo>/docs. Image:
# /app/app/knowledge/importer -> /app/docs (the Dockerfile copies docs/knowledge).
# The sha256 is verified on load either way.
_MANIFEST_CANDIDATES = [os.path.normpath(os.path.join(_HERE, "..", "..", "..", "..", _MANIFEST_REL)),
                        os.path.normpath(os.path.join(_HERE, "..", "..", "..", _MANIFEST_REL))]
MANIFEST_PATH = next((p for p in _MANIFEST_CANDIDATES if os.path.isfile(p)), _MANIFEST_CANDIDATES[0])


def all_scope() -> KnowledgeScope:
    """The importer is a trusted internal writer over every project."""
    from ...assistant_access import ProjectAllowlist

    return KnowledgeScope.build(ProjectAllowlist(all_projects=True), "*")


def mcredits(credits) -> Optional[int]:
    if credits is None:
        return None
    try:
        value = float(credits)
    except (TypeError, ValueError):
        return None
    return int(round(value * 1000)) if value >= 0 else None


def status_value(value) -> Optional[str]:
    return getattr(value, "value", value)


@dataclass
class Report:
    """Counts plus capped example lists; ``*_total`` keeps the true size."""
    created: dict = field(default_factory=dict)
    reused: dict = field(default_factory=dict)
    sources: dict = field(default_factory=dict)
    lists: dict = field(default_factory=dict)
    totals: dict = field(default_factory=dict)
    extra: dict = field(default_factory=dict)

    def bump(self, bucket: dict, key: str, n: int = 1) -> None:
        bucket[key] = bucket.get(key, 0) + n

    def note(self, kind: str, **item) -> None:
        self.bump(self.totals, kind)
        items = self.lists.setdefault(kind, [])
        if len(items) < LIST_CAP:
            items.append(item)

    def source(self, name: str) -> dict:
        return self.sources.setdefault(name, {"rows": 0, "failed": 0})

    def as_dict(self) -> dict:
        out = {"created": dict(sorted(self.created.items())), "reused": dict(sorted(self.reused.items())),
               "sources": self.sources}
        for kind in ("duplicates", "missing_media", "missing_metadata", "conflicts", "failures", "excluded_cases",
                     "refused", "skipped", "historical"):
            out[kind] = {"total": self.totals.get(kind, 0), "items": self.lists.get(kind, [])}
        out.update(self.extra)
        return out


class MediaChanged(Exception):
    """A prefetched file changed before the row was written: retry next pass."""


class LeaseLost(Exception):
    """Another importer may own this source now: stop before writing anything more."""


@dataclass
class ImportContext:
    session: Session
    dry_run: bool
    code_revision: str
    source_label: str
    hasher: Optional[Callable[[str], str]] = None
    exists: Callable[[str], bool] = os.path.isfile
    stat: Optional[Callable[[str], object]] = None  # file identity for "unchanged while hashing"
    now: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    report: Report = field(default_factory=Report)
    batch_id: Optional[str] = None
    manifest_version: Optional[str] = None
    manifest_groups: dict = field(default_factory=dict)  # project_id -> group
    scope: KnowledgeScope = field(default_factory=all_scope)
    owner: Optional[str] = None
    lease_ttl_sec: int = 900
    keepalive: Optional[Callable[[], bool]] = None  # external (Redis) lease refresh; False = lost
    deadline: Optional[datetime] = None
    media: dict = field(default_factory=dict)  # path -> (present, identity, sha256) for the current row
    run_locked: bool = False
    defer_exports: bool = False  # set while a source row holds its Run lock
    deferred_exports: list = field(default_factory=list)
    _invalid: Optional[store.InvalidSet] = field(default=None, repr=False)  # per pass, see invalid()
    redeclare: Optional[Callable[[], list]] = None  # the current row's media, recomputed after the lock
    row_cache: dict = field(default_factory=dict)  # per-row values computed before the Run lock
    # Per-source durable tally ({key: value}), saved in the source's cursor in the
    # same commit as the row; a rolled-back row's entries are discarded.
    cursor_tally: dict = field(default_factory=dict)

    # ------------------------------------------------------------ leases

    def heartbeat(self, source: str) -> None:
        """Before every apply commit: refresh and re-verify every lease this write relies on.

        Runs inside the commit's transaction, so the cursor row stays locked by
        us until the commit: a lease cannot be taken over between the check and
        the write it guards.
        """
        if self.dry_run:
            return
        if self.keepalive is not None and not self.keepalive():
            raise LeaseLost("the import tick lease was lost")
        held = self.session.execute(text(
            "UPDATE kb_import_cursors SET lease_expires_at = now() + make_interval(secs => :t) "
            "WHERE source = :s AND lease_owner = :o RETURNING source"),
            {"t": self.lease_ttl_sec, "s": source, "o": self.owner}).first()
        if held is None:
            raise LeaseLost(f"the cursor lease on {source} was lost")

    def identity(self, path: str):
        """(size, mtime_ns) of a file, or None when it cannot be read."""
        if self.stat is not None:
            return self.stat(path)
        try:
            st = os.stat(path)
        except OSError:
            return None
        return st.st_size, st.st_mtime_ns

    def out_of_time(self) -> bool:
        return self.deadline is not None and datetime.now(timezone.utc) >= self.deadline

    # ------------------------------------------------------------ batch / members

    def ensure_batch(self) -> str:
        if self.batch_id is None:
            batch = store.start_import_batch(self.session, source=self.source_label[:64],
                                             code_revision=self.code_revision[:64], dry_run=self.dry_run)
            self.batch_id = batch.id
        return self.batch_id

    def member(self, kind: str, row_id: str) -> None:
        store.add_import_member(self.session, self.ensure_batch(), kind, row_id)
        self.report.bump(self.report.created, kind)

    def invalid(self) -> store.InvalidSet:
        """Invalidations as of this pass's first use: loaded once, never per row
        (a big withdrawn batch must not be re-read while a Run is locked). The
        importer writes no invalidations itself; ones made meanwhile by an
        operator are seen on the next pass."""
        if self._invalid is None:
            self._invalid = store.InvalidSet.load(self.session)
        return self._invalid

    @property
    def lock_runs(self) -> bool:
        # A dry run rolls everything back, so it needs no serialization and must
        # not hold production Run rows FOR UPDATE for its whole transaction.
        return not self.dry_run

    # ------------------------------------------------------------ manifest

    def load_manifest(self, path: str = MANIFEST_PATH, sha256: str = MANIFEST_SHA256) -> None:
        with open(path, "rb") as fh:
            manifest = store.load_eval_manifest(self.session, fh.read(), expected_sha256=sha256)
        latest = store._latest_manifest(self.session)
        manifest = latest if latest is not None else manifest
        self.manifest_version = manifest.version
        self.manifest_groups = {p: g for g in manifest.body.get("groups", ()) for p in g.get("project_ids", ())}
        self.report.extra["eval"] = {"manifest_version": manifest.version, "manifest_sha256": manifest.sha256,
                                     "assigned": 0}

    def assign_membership(self, case: KbCase) -> None:
        if self.manifest_version is None:
            return
        if case.project_id not in self.manifest_groups:
            if case.import_batch_id == self.batch_id:
                self.report.note("excluded_cases", case_key=case.case_key, project_id=case.project_id,
                                 reason="project not in the evaluation manifest")
            return
        m = store.assign_eval_membership(self.session, case.id, self.manifest_version,
                                         import_batch_id=self.ensure_batch())
        if m.import_batch_id == self.batch_id and not self.invalid().has("eval_membership", m.id):
            if self.session.info.setdefault("kb_registered", set()).isdisjoint({m.id}):
                self.member("eval_membership", m.id)
                self.session.info["kb_registered"].add(m.id)
                self.report.extra["eval"]["assigned"] += 1

    # ------------------------------------------------------------ cases

    def lock_run(self, run_id: Optional[str]) -> None:
        if self.lock_runs and run_id is not None:
            store.lock_run_and_cases(self.session, run_id, [])
            if not self.run_locked:
                # Everything read before the lock (the Run and the source row, e.g. a
                # RunSegment reset by an in-place rerun) is re-read from now on.
                self.session.flush()
                self.session.expire_all()
            self.run_locked = True

    def begin_row(self, media: Iterable = ()) -> None:
        """Start one row's transaction: probe (stat + hash) its files BEFORE any
        Run lock is taken; under the lock only a stat re-verifies them."""
        self.run_locked = False
        self.media = {}
        for path, want_hash in media:
            if path and path not in self.media:
                self.media[path] = self._probe(path, want_hash)

    def _probe(self, path: str, want_hash: bool) -> tuple:
        present = bool(self.exists(path))
        identity = self.identity(path) if present else None
        sha = None
        if present and want_hash and self.hasher is not None:
            sha = self.hasher(path)
            if self.identity(path) != identity:
                identity, sha = ("changed while hashing",), None
        return present, identity, sha

    def _media(self, path: str) -> tuple:
        entry = self.media.get(path)
        if entry is None:
            if self.run_locked:
                declared = {p for p, _ in (self.redeclare() if self.redeclare else [])}
                if path in declared:
                    # The row changed between the prefetch and the lock (e.g. a final
                    # was published meanwhile): retry next pass with fresh media.
                    raise MediaChanged(path)
                raise RuntimeError(f"file I/O under a Run lock: {path} was not probed before the lock")
            entry = self.media[path] = self._probe(path, True)
        elif self.run_locked and entry[0] and self.identity(path) != entry[1]:
            raise MediaChanged(path)
        return entry

    def present(self, path: Optional[str]) -> bool:
        return bool(path) and self._media(path)[0]

    def case(self, **kw) -> KbCase:
        case = store.ensure_case(self.session, origin="import", import_batch_id=self.ensure_batch(), **kw)
        if self.lock_runs:
            store.lock_run_and_cases(self.session, None, [case.id])
        if case.import_batch_id == self.batch_id and case.id not in self.session.info.setdefault("kb_cases", set()):
            self.session.info["kb_cases"].add(case.id)
            self.member("case", case.id)
        elif case.import_batch_id != self.batch_id:
            self.report.bump(self.report.reused, "case")
        self.assign_membership(case)
        return case

    # ------------------------------------------------------------ artifacts

    def media_sha(self, path: Optional[str], *, what: str) -> tuple[Optional[str], bool]:
        """(sha256 or None, present) from the row's probe. Hashing only when a hasher was injected."""
        if not path:
            return None, False
        present, _, sha = self._media(path)
        if not present:
            self.report.note("missing_media", what=what, path=path)
            return None, False
        return sha, True

    def artifact(self, *, kind: str, locator_kind: str, locator: str, project_id: str,
                 sha256: Optional[str] = None, present: Optional[bool] = None, verified: bool = False,
                 immutable_path: bool = False, **kw) -> KbArtifact:
        """Reuse a valid hashed artifact, else create one; a newly created artifact
        whose file was probed gets its availability check (``kb_artifact_checks``)."""
        if sha256 is not None:
            a = self.session.execute(select(KbArtifact).where(
                KbArtifact.locator_kind == locator_kind, KbArtifact.locator == locator, KbArtifact.sha256 == sha256,
                ~store.withdrawn_clause("artifact", KbArtifact.id)).order_by(KbArtifact.revision.desc()).limit(1)
            ).scalar_one_or_none()
            if a is not None:
                if a.project_id != project_id:
                    raise store.KnowledgeConflict("artifact already recorded for a different project")
                self.report.bump(self.report.reused, "artifact")
                return a
        elif immutable_path:
            # A path the pipeline never rewrites (promotion outputs) names one
            # content, so an unhashed record of it is reused rather than repeated.
            a = self.session.execute(select(KbArtifact).where(
                KbArtifact.locator_kind == locator_kind, KbArtifact.locator == locator,
                KbArtifact.sha256.is_(None), KbArtifact.project_id == project_id,
                ~store.withdrawn_clause("artifact", KbArtifact.id)).order_by(KbArtifact.seq.desc()).limit(1)
            ).scalar_one_or_none()
            if a is not None:
                self.report.bump(self.report.reused, "artifact")
                return a
        art = store.get_or_create_artifact(self.session, kind=kind, locator_kind=locator_kind, locator=locator,
                                           sha256=sha256, project_id=project_id, **kw)
        self.member("artifact", art.id)
        if present is not None:
            # ``verified``: the sha256 was computed from this file just now; a hash
            # copied from a source record is not a verification.
            store.record_artifact_check(self.session, art.id, present=present,
                                        sha256_matches=True if present and verified else None)
            self.report.bump(self.report.created, "artifact_check")
        return art

    def local_artifact(self, *, kind: str, path: str, project_id: str, what: str, **kw) -> KbArtifact:
        sha, present = self.media_sha(path, what=what)
        return self.artifact(kind=kind, locator_kind="local_path", locator=path, project_id=project_id,
                             sha256=sha, present=present, verified=sha is not None, **kw)

    # ------------------------------------------------------------ versions

    def versions(self, case: KbCase) -> list:
        return store.versions(self.session, case.id, self.invalid())

    def append_version(self, case: KbCase, artifact: KbArtifact, *, reason: str, source_record: str,
                       historical: bool = False, published_at=None):
        version = store.append_case_version(self.session, case, artifact_id=artifact.id, reason=reason,
                                            source_record=source_record, historical=historical,
                                            published_at=published_at, invalid=self.invalid())
        self.member("case_version", version.id)
        return version

    # ------------------------------------------------------------ attempts

    def _free_key(self, base: str, lookup: Callable[[str], Optional[object]],
                  is_invalid: Callable[[object], bool]) -> tuple[str, Optional[object]]:
        """First key in base, base~2, … that is unused or names a still-valid row."""
        n = 1
        while True:
            key = base if n == 1 else f"{base}~{n}"
            row = lookup(key)
            if row is None or not is_invalid(row):
                return key, row
            n += 1

    def attempt(self, case: KbCase, *, key: str, task_id: Optional[str] = None,
                **kw) -> tuple[Optional[KbAttempt], bool]:
        """(attempt, created). An existing valid attempt under the key is reused as is.

        With a provider ``task_id``, an attempt that already holds that task
        (live capture's, an orphan's, or an earlier import's: submission row or
        ``submitted`` payload) is reused for the same case — one attempt per
        provider task (capture design S9c, R25). Held by another case: reported
        and ``(None, False)``; the caller records nothing for it.
        """
        base = "import:" + store.canonical_sha256([key, case.id])[:48]
        invalid = self.invalid()
        key, existing = self._free_key(
            base, lambda k: self.session.execute(select(KbAttempt).where(KbAttempt.idempotency_key == k)
                                                 ).scalar_one_or_none(),
            invalid.attempt_invalid)
        if existing is not None:
            self.report.bump(self.report.reused, "attempt")
            return existing, False
        if task_id:
            holder = self.attempt_for_call_key(task_id)
            if holder is not None and holder.case_id != case.id and holder.target_kind == "spend_draw" \
                    and self._same_run(holder, case):
                # A draw-only placeholder (request unknown) for a task whose real row
                # is now being imported: the real attempt supersedes it. The
                # placeholder is withdrawn (append-only), its costs are recorded again
                # on the real attempt by the caller (draw_costs).
                store.invalidate(self.session, target_kind="attempt", target_id=holder.id,
                                 reason=f"superseded by the attempt of the row that names task {task_id}")
                self._invalid = None  # this pass's cached set is now stale
                self.report.bump(self.report.extra.setdefault("reconcile", {}), "ambiguous_attempts_superseded")
                holder = None
            if holder is not None:
                if holder.case_id != case.id:
                    self.report.note("conflicts", what="provider task already attributed to another case",
                                     task_id=task_id, attempt_id=holder.id)
                    return None, False
                self.report.bump(self.report.reused, "attempt")
                return holder, False
        attempt, reused = store.record_attempt_intent(self.session, case_id=case.id, idempotency_key=key,
                                                      import_batch_id=self.ensure_batch(), **kw)
        if reused or attempt.idempotency_key != key:
            raise store.KnowledgeConflict("import attempt collided with an open live intent")
        self.member("attempt", attempt.id)
        return attempt, True

    def _same_run(self, attempt: KbAttempt, case: KbCase) -> bool:
        other = self.session.get(KbCase, attempt.case_id)
        return other is not None and other.run_id is not None and other.run_id == case.run_id

    def observations(self, attempt_id: str) -> list:
        invalid = self.invalid()
        return [o for o in self.session.execute(select(KbAttemptObservation).where(
            KbAttemptObservation.attempt_id == attempt_id).order_by(KbAttemptObservation.seq)).scalars()
            if not invalid.row_invalid("observation", o)]

    def existing_output(self, attempt: KbAttempt) -> Optional[KbArtifact]:
        outs = [o for o in self.observations(attempt.id) if o.kind == "output"]
        return self.session.get(KbArtifact, outs[-1].output_artifact_id) if outs else None

    def observe(self, attempt: KbAttempt, *, kind: str, **kw) -> Optional[KbAttemptObservation]:
        """Insert once: an equal valid observation already present is not repeated."""
        same = ("status", "output_artifact_id", "mcredits", "cost_source")
        for o in self.observations(attempt.id):
            if o.kind == kind and all(getattr(o, k) == kw.get(k) for k in same):
                return None
        obs = store.record_observation(self.session, attempt.id, kind=kind, import_batch_id=self.ensure_batch(),
                                       **kw)
        self.member("observation", obs.id)
        return obs

    def submission(self, attempt: KbAttempt, *, task_id: str, call_key: Optional[str]) -> None:
        """Provider acceptance. A task id already held elsewhere is not re-recorded.

        ``kb_attempt_submissions`` keeps task ids unique across withdrawn rows too,
        so a replacement import records the acceptance as a ``submitted`` status
        whose payload carries the ids (and the draw matcher reads it).
        """
        invalid = self.invalid()
        own = self.session.execute(select(KbAttemptSubmission).where(KbAttemptSubmission.attempt_id == attempt.id)
                                   ).scalar_one_or_none()
        if own is not None:
            return
        if any(o.kind == "status" and o.status == "submitted" for o in self.observations(attempt.id)):
            return
        holders = self.session.execute(select(KbAttemptSubmission).where(or_(
            KbAttemptSubmission.provider_task_id == task_id,
            KbAttemptSubmission.spend_call_key == call_key if call_key else false(),
            KbAttemptSubmission.provider_task_id == call_key if call_key else false(),
            KbAttemptSubmission.spend_call_key == task_id))).scalars().all()
        live = [h for h in holders if not invalid.has("submission", h.id)
                and not invalid.attempt_invalid(self.session.get(KbAttempt, h.attempt_id))]
        if live:
            self.report.note("conflicts", what="provider task already attributed to another attempt",
                             task_id=task_id, attempt_id=live[0].attempt_id)
            self.observe(attempt, kind="status", status="submitted",
                         payload={"provider_task_id": task_id, "spend_call_key": call_key,
                                  "submission_withheld": "task id held by another attempt"})
            return
        if holders:
            self.observe(attempt, kind="status", status="submitted",
                         payload={"provider_task_id": task_id, "spend_call_key": call_key,
                                  "submission_withheld": "task id held by a withdrawn submission"})
            return
        sub = store.record_submission(self.session, attempt.id, provider_task_id=task_id, spend_call_key=call_key)
        self.member("submission", sub.id)
        obs = self.session.execute(select(KbAttemptObservation).where(
            KbAttemptObservation.attempt_id == attempt.id, KbAttemptObservation.kind == "status",
            KbAttemptObservation.status == "submitted").order_by(KbAttemptObservation.seq.desc()).limit(1)
        ).scalar_one()
        self.member("observation", obs.id)

    def attempt_for_call_key(self, call_key: str) -> Optional[KbAttempt]:
        invalid = self.invalid()
        for sub in self.session.execute(select(KbAttemptSubmission).where(or_(
                KbAttemptSubmission.spend_call_key == call_key,
                KbAttemptSubmission.provider_task_id == call_key))).scalars():
            attempt = self.session.get(KbAttempt, sub.attempt_id)
            if not invalid.has("submission", sub.id) and not invalid.attempt_invalid(attempt):
                return attempt
        for obs in self.session.execute(select(KbAttemptObservation).where(
                KbAttemptObservation.kind == "status", KbAttemptObservation.status == "submitted",
                or_(KbAttemptObservation.payload["provider_task_id"].as_string() == call_key,
                    KbAttemptObservation.payload["spend_call_key"].as_string() == call_key))).scalars():
            attempt = self.session.get(KbAttempt, obs.attempt_id)
            if not invalid.row_invalid("observation", obs) and not invalid.attempt_invalid(attempt):
                return attempt
        return None

    def draw_costs(self, attempt: KbAttempt, draw) -> bool:
        """Estimate and settlement are separate observations, both ``spend_draw``.

        A settlement is compared with the LATEST recorded one for this draw: a
        provider revision (new ``mcredits`` or ``settled_at``) appends a newer
        ``cost_settled``, which wins by ``seq``. Returns True when it appended one.
        """
        payload = {"call_key": draw.call_key, "draw_kind": draw.kind}
        self.observe(attempt, kind="cost_estimate", mcredits=int(draw.estimated_mcredits), cost_source="spend_draw",
                     payload=payload)
        if draw.settled_at is None:
            return False
        settled = {**payload, "settled_at": draw.settled_at.isoformat()}
        latest = next((o for o in reversed(self.observations(attempt.id))
                       if o.kind == "cost_settled" and o.cost_source == "spend_draw"
                       and (o.payload or {}).get("call_key") == draw.call_key), None)
        if latest is not None and latest.mcredits == int(draw.mcredits) \
                and (latest.payload or {}).get("settled_at") == settled["settled_at"]:
            return False
        obs = store.record_observation(self.session, attempt.id, kind="cost_settled", mcredits=int(draw.mcredits),
                                       cost_source="spend_draw", payload=settled,
                                       import_batch_id=self.ensure_batch())
        self.member("observation", obs.id)
        return True

    def snapshot_cost(self, attempt: KbAttempt, credits, *, column: str) -> None:
        """A provider-reported cost read off a mutable source column at import time.

        Only ever written for the attempt that column currently describes.
        """
        value = mcredits(credits)
        if value is None:
            return
        self.observe(attempt, kind="cost_settled", mcredits=value, cost_source="run_segment_snapshot",
                     payload={"label": "current attempt at import time", "column": column})

    # ------------------------------------------------------------ events (receipts)

    def request_key(self, base: str) -> tuple[str, bool]:
        """(key, already_recorded). A receipt whose events were all withdrawn is skipped."""
        invalid = self.invalid()

        def withdrawn(req: KbRequest) -> bool:
            if req.kind == "rule_version":
                v = self.session.get(KbRuleVersion, req.child_event_ids[0])
                return v is None or invalid.row_invalid("rule_version", v)
            events = [self.session.get(KbFeedbackEvent, i) for i in req.child_event_ids]
            return not events or all(e is None or invalid.event_invalid(e) for e in events)

        key, existing = self._free_key(base[:120], lambda k: self.session.get(KbRequest, k), withdrawn)
        return key, existing is not None

    def register_events(self, receipt) -> None:
        for event_id in receipt.ids:
            self.member("event", event_id)

    def register_rule_version(self, version: KbRuleVersion) -> None:
        self.member("rule_version", version.id)
        for ev in self.session.execute(select(KbRuleEvidence).where(KbRuleEvidence.rule_version_id == version.id)
                                       ).scalars():
            self.member("rule_evidence", ev.id)


def model_provenance(model_key: Optional[str]) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """(model_key, provider_model_id, family) from the ``ai_models`` registry at import."""
    from ... import ai_models

    if not model_key:
        return None, None, None
    spec = ai_models.AI_MODELS.get(model_key)
    return (model_key, spec.kie_model_id, spec.family) if spec is not None else (model_key, None, None)


def operation_for(project) -> str:
    op = getattr(project, "project_type", None)
    op = status_value(op)
    return op if op in t.CASE_OPERATIONS else "face_swap"


__all__ = ["ImportContext", "LeaseLost", "MediaChanged", "Report", "all_scope", "mcredits", "model_provenance", "operation_for",
           "status_value", "MANIFEST_PATH", "MANIFEST_SHA256"]
