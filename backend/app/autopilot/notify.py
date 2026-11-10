"""Chat notifications of the assistant's questions (AICP-374).

Design: ``docs/autopilot-operator-ux.md`` §2. A Telegram bot, called directly
(no n8n). The notifier never reacts to events: on every pass it computes, from
the CURRENT rows, what each message should say and brings the chat to it.

* **What gets a message** (discovery): every answerable approval
  (:func:`links.answerable` — pending, of its project's latest live attempt);
  every *system* pause of a live attempt (keyed by its journal line, so a second
  pause after a Resume is a new message); every attempt that completed, failed
  or was stopped by the system after the notifier was first switched on (the
  ``notifier:since`` marker), never history. A state that appears and goes
  between two passes (~10 s) is not seen — accepted.
* **One row per message** (``autopilot_notifications``). A row is claimed
  (owner + expiry) before any call and written back only while the claim is
  still ours. Delivery is at-least-once: a bot cannot read the chat's history,
  so a send whose outcome is unknown (timeout, crash before the write) is sent
  again and may leave a duplicate showing its last state (accepted, Operator
  2026-11-03).
* **Edits are idempotent**: the rendered text is hashed; equal → no call;
  Telegram's "message is not modified" counts as success; a deleted message is
  re-posted only while it still waits for an answer.
* Answers are not taken in the chat. Messages carry names, titles, costs and
  links only — no media, no Drive links, no operator comments, no secrets.

The token is used only to build the request URL and is scrubbed from every
error text; it is never logged.
"""

from __future__ import annotations

import hashlib
import html
import logging
import os
import random
import socket
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Callable, Optional

import httpx
from sqlalchemy import or_, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from ..config import settings
from ..models import Run, VideoProject
from ..models_autopilot import (
    LIVE_AUTOPILOT_STATUSES,
    AutopilotApproval,
    AutopilotJournal,
    AutopilotNotification as Note,
    ProjectAutopilot,
    utcnow,
)
from . import links

logger = logging.getLogger("autopilot.notify")

# httpx logs every request URL at INFO, and a Bot API URL carries the token
# (/bot<token>/sendMessage): keep those loggers at WARNING in every process
# that imports this module, whatever the root level is now or later.
for _name in ("httpx", "httpcore"):
    logging.getLogger(_name).setLevel(max(logging.getLogger(_name).level, logging.WARNING))

MARKER_KEY = "notifier:since"
CLAIM_TTL = timedelta(seconds=120)          # far beyond one 10 s request + its write
FIRST_REMINDER = timedelta(hours=2)          # after the message was posted
NEXT_REMINDER = timedelta(hours=24)          # after the previous reminder
SETTLED_FOR = timedelta(days=7)              # a closed message is re-checked this long
CLOSED_RECHECK = timedelta(minutes=5)        # … at most this often
END_LOOKBACK = timedelta(minutes=10)         # ends committed just before the marker count
END_MAX_AGE = timedelta(days=7)              # an outage shorter than this loses no end
MAX_CALLS_PER_PASS = 20
SEND_GAP_SEC = 1.1                            # Telegram: about one message per second per chat
FAILING_AFTER = 10
MAX_BACKOFF_SEC = 3600
HALT_SEC = 3600                               # bad token / bot blocked / chat moved
REASON_MAX = 300
TEXT_MAX = 3800                               # Telegram's limit is 4096 characters

_e = html.escape


# ---------------------------------------------------------------------------
# The channel (Telegram Bot API)
# ---------------------------------------------------------------------------


class ChannelError(Exception):
    def __init__(self, message: str, *, retry_after: Optional[int] = None, gone: bool = False,
                 not_modified: bool = False, halt: bool = False, ambiguous: bool = False):
        super().__init__(message)
        self.retry_after = retry_after
        self.gone = gone
        self.not_modified = not_modified
        self.halt = halt
        self.ambiguous = ambiguous


class TelegramChannel:
    """``post`` / ``edit`` — the adapter the reconciler talks to (§2.2)."""

    API = "https://api.telegram.org"

    def __init__(self, token: str, *, client: Optional[httpx.Client] = None, timeout: float = 10.0):
        self._token = token
        self._client = client or httpx.Client(timeout=timeout)

    def _scrub(self, text: str) -> str:
        return text.replace(self._token, "<token>") if self._token else text

    def _call(self, method: str, payload: dict) -> dict:
        url = f"{self.API}/bot{self._token}/{method}"
        try:
            r = self._client.post(url, json=payload)
        except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
            # The request never left: safe to retry, cannot duplicate.
            raise ChannelError(f"{method}: cannot connect ({type(exc).__name__})") from None
        except httpx.HTTPError as exc:
            raise ChannelError(f"{method}: {type(exc).__name__}, outcome unknown",
                               ambiguous=True) from None
        try:
            body = r.json()
        except ValueError:
            body = {}
        if r.status_code == 200 and body.get("ok"):
            return body.get("result") or {}
        code = int(body.get("error_code") or r.status_code)
        desc = self._scrub(str(body.get("description") or f"HTTP {r.status_code}"))[:300]
        params = body.get("parameters") or {}
        low = desc.lower()
        if code == 429:
            raise ChannelError(f"{method}: {desc}", retry_after=int(params.get("retry_after") or 30))
        if params.get("migrate_to_chat_id"):
            raise ChannelError(f"{method}: the group became a supergroup; set TELEGRAM_CHAT_ID to "
                               f"{params['migrate_to_chat_id']}", halt=True)
        if "message is not modified" in low:
            raise ChannelError(f"{method}: {desc}", not_modified=True)
        if ("message to edit not found" in low or "message can't be edited" in low
                or "message_id_invalid" in low):
            raise ChannelError(f"{method}: {desc}", gone=True)
        if code in (401, 403) or "chat not found" in low:
            raise ChannelError(f"{method}: {desc}", halt=True)
        raise ChannelError(f"{method}: {desc}", ambiguous=code >= 500)

    def _payload(self, chat_id: str, text: str) -> dict:
        return {"chat_id": chat_id, "text": text, "parse_mode": "HTML",
                "link_preview_options": {"is_disabled": True}}

    def post(self, chat_id: str, text: str, *, reply_to: Optional[int] = None) -> int:
        p = self._payload(chat_id, text)
        if reply_to:
            p["reply_parameters"] = {"message_id": int(reply_to), "allow_sending_without_reply": True}
        res = self._call("sendMessage", p)
        mid = res.get("message_id")
        if not isinstance(mid, int):
            raise ChannelError("sendMessage: no message_id in the answer", ambiguous=True)
        return mid

    def edit(self, chat_id: str, message_id: int, text: str) -> None:
        p = self._payload(chat_id, text)
        p["message_id"] = int(message_id)
        self._call("editMessageText", p)

    def me(self) -> dict:
        return self._call("getMe", {})


# ---------------------------------------------------------------------------
# What a message should say (§2.1), from the current rows
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Desired:
    state: str
    text: str
    final: bool                      # nothing more is expected to change
    waiting: bool = False            # an open question: reminders, re-post if deleted
    title: str = ""
    url: Optional[str] = None
    since: Optional[datetime] = None  # when the question was asked


def _clip(text: str, limit: int) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= limit else text[:limit - 1].rstrip() + "…"


def _link(url: Optional[str], label: str = "Open") -> str:
    return f'<a href="{_e(url)}">{label}</a>' if url else ""


def _project_name(db: Session, project_id: Optional[str]) -> str:
    p = db.get(VideoProject, project_id) if project_id else None
    if p is None:
        return "deleted project"
    return _clip(p.name or f"project {project_id[:8]}", 80)


def _decision(a: AutopilotApproval) -> str:
    if a.answered_decision == "reject":
        return "reject"
    label = None
    for o in a.options or []:
        if isinstance(o, dict) and o.get("id") == a.chosen_option:
            label = o.get("label")
    label = label or a.chosen_option
    return "approve" + (f" ({_clip(str(label), 80)})" if label else "")


def _join(*lines: str) -> str:
    text = "\n".join(x for x in lines if x)
    return text if len(text) <= TEXT_MAX else text[:TEXT_MAX - 1] + "…"


def _cost_line(a: AutopilotApproval) -> str:
    if not a.cost_mcredits:
        return ""
    from .. import spend

    credits = a.cost_mcredits / 1000.0
    return f"Cost: ~{credits:.0f} credits (${spend.usd(credits):.2f})"


def _desired_approval(db: Session, row: Note) -> Desired:
    a = db.get(AutopilotApproval, row.approval_id) if row.approval_id else None
    ap = db.get(ProjectAutopilot, a.autopilot_id) if a is not None else None
    if a is None or ap is None:
        return Desired("gone", _join("✖️ <b>No longer needed</b> — the question was deleted",
                                     _e(_project_name(db, row.project_id))), True)
    latest = links.latest_attempt(db, ap.project_id)
    pl = links.place(db, a, project_id=ap.project_id, latest=latest)
    title = links.approval_title(a.question, a.proposal)
    where = _e(f"{_project_name(db, ap.project_id)} · {pl.where}")
    struck = f"<s>{_e(title)}</s>"
    current = latest is not None and latest.id == ap.id and ap.status in LIVE_AUTOPILOT_STATUSES
    by = _e(_clip(a.answered_by or "the operator", 60))
    decision = _e(_decision(a))
    if a.status == "pending":
        if pl.open:
            return Desired("waiting", _join(
                "⏳ <b>Waiting for you</b>", f"<b>{_e(title)}</b>", where, _e(_cost_line(a)),
                "The assistant is paused — answer, then Resume." if ap.status == "paused" else "",
                _link(pl.url, "Open and answer")),
                False, waiting=True, title=title, url=pl.url, since=a.created_at)
        why = ("a newer attempt replaced this one" if ap.status in LIVE_AUTOPILOT_STATUSES
               else f"the assistant {ap.status}")
        return Desired("not_needed", _join(f"✖️ <b>No longer needed</b> — {_e(why)}", struck, where), True)
    if a.status in ("approved", "rejected"):
        if current:
            return Desired("answered", _join(
                f"☑️ <b>Answered by {by}</b>: {decision} — the assistant checks it next",
                struck, where), False)
        return Desired("answered_unused", _join(
            f"☑️ <b>Answered by {by}</b>: {decision} — the assistant stopped before using it",
            struck, where), True)
    if a.status == "consumed":
        return Desired("done", _join(f"✅ <b>Done</b>: {decision} (answered by {by})", struck, where), True)
    # superseded: by a changed run, a reopened/skipped step or "Ask again" —
    # the row does not say which, so the copy stays generic.
    if a.answered_decision:
        return Desired("not_used", _join(
            f"↩️ <b>Answer not used</b>: {decision} — the question was withdrawn or changed; "
            "if it is still needed, a new one follows", struck, where), True)
    return Desired("not_needed", _join("✖️ <b>No longer needed</b>", struck, where), True)


def current_pause_journal_id(db: Session, ap: ProjectAutopilot) -> Optional[int]:
    """The journal line of ``ap``'s current SYSTEM pause, or None (not paused,
    paused by the operator, or a pause that wrote no matching line)."""
    if ap.status != "paused" or not ap.status_reason or ap.status_reason.startswith("paused by "):
        return None
    return db.execute(select(AutopilotJournal.id).where(
        AutopilotJournal.autopilot_id == ap.id, AutopilotJournal.kind == "paused",
        AutopilotJournal.message == ap.status_reason[:4000])     # journal() cuts at 4000
        .order_by(AutopilotJournal.id.desc()).limit(1)).scalar_one_or_none()


_NOW = {"active": "running again", "paused": "paused again", "completed": "finished",
        "stopped": "stopped", "failed": "failed"}


def _desired_pause(db: Session, row: Note) -> Desired:
    ap = db.get(ProjectAutopilot, row.autopilot_id) if row.autopilot_id else None
    j = db.get(AutopilotJournal, row.journal_id) if row.journal_id else None
    reason = _e(_clip(j.message if j else "", REASON_MAX))
    name = _e(_project_name(db, row.project_id))
    url = links.absolute(f"/v2/projects/{row.project_id}")
    if ap is not None and current_pause_journal_id(db, ap) == row.journal_id:
        return Desired("paused", _join(f"⏸ <b>The assistant paused</b> — {reason}", name,
                                       _link(url, "Open the project")), False, url=url)
    now = _NOW.get(ap.status, ap.status) if ap is not None else "deleted"
    return Desired("resumed", _join(f"▶️ <b>No longer paused</b> — now {_e(now)}",
                                    f"<s>Paused — {reason}</s>", name), True)


def _public_reason(db: Session, ap: ProjectAutopilot) -> str:
    """``status_reason`` without an operator's answer comment (a rejected
    budget / regeneration appends it, core.py)."""
    reason = ap.status_reason or ""
    comments = db.execute(select(AutopilotApproval.answer_comment).where(
        AutopilotApproval.autopilot_id == ap.id, AutopilotApproval.answered_decision == "reject",
        AutopilotApproval.answer_comment.is_not(None))).scalars().all()
    for c in comments:
        if c and reason.endswith(f": {c}"):
            reason = reason[: -len(c) - 2]
    return _clip(reason, REASON_MAX)


def _deliverables(db: Session, ap: ProjectAutopilot) -> str:
    from . import run_view

    pr = run_view.project_runs(db, ap.project_id)
    out = []
    for rid, ar in pr.by_run.items():
        if ar.autopilot_id != ap.id:
            continue
        r = db.get(Run, rid)
        if r is not None:
            out.append(f"{r.short_id or rid[:8]}" + (f" ({ar.label})" if getattr(ar, "label", None) else ""))
    if not out:
        return ""
    more = f" and {len(out) - 6} more" if len(out) > 6 else ""
    return "Runs: " + ", ".join(out[:6]) + more


def _desired_end(db: Session, row: Note) -> Desired:
    ap = db.get(ProjectAutopilot, row.autopilot_id) if row.autopilot_id else None
    name = _e(_project_name(db, row.project_id))
    url = links.absolute(f"/v2/projects/{row.project_id}")
    if ap is None:
        return Desired("gone", _join("✖️ <b>Ended</b> — the project was deleted", name), True)
    if ap.status == "completed":
        return Desired("completed", _join("🏁 <b>The assistant finished</b>", name,
                                          _e(_deliverables(db, ap)), _link(url, "Open the project")),
                       True, url=url)
    return Desired("ended", _join(f"⛔ <b>The assistant ended</b> — {_e(_public_reason(db, ap))}", name,
                                  _link(url, "Open the project")), True, url=url)


def desired(db: Session, row: Note) -> Desired:
    if row.kind == "approval":
        return _desired_approval(db, row)
    if row.kind == "pause":
        return _desired_pause(db, row)
    return _desired_end(db, row)


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _ago(delta: timedelta) -> str:
    mins = int(delta.total_seconds() // 60)
    if mins < 60:
        return f"{mins} min"
    hours, mins = divmod(mins, 60)
    if hours < 48:
        return f"{hours} h {mins:02d} min"
    return f"{hours // 24} days"


def reminder_text(d: Desired, now: datetime) -> str:
    waited = _ago(now - d.since) if d.since else "a while"
    return _join(f"⏰ <b>Still waiting for you</b> ({waited}): {_e(d.title)}",
                 _link(d.url, "Open and answer"))


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def _since(db: Session, now: datetime) -> datetime:
    row = db.execute(select(Note).where(Note.key == MARKER_KEY)).scalar_one_or_none()
    if row is not None:
        return row.created_at
    db.add(Note(key=MARKER_KEY, kind="marker", chat_id="", gone=True, closed_at=now,
                created_at=now, updated_at=now))
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        return db.execute(select(Note.created_at).where(Note.key == MARKER_KEY)).scalar_one()
    logger.info("autopilot notifier: first start, notifying from %s on", now.isoformat())
    return now


def _add(db: Session, chat_id: str, now: datetime, **kw) -> bool:
    db.add(Note(chat_id=chat_id, created_at=now, updated_at=now, **kw))
    try:
        db.commit()
        return True
    except IntegrityError:          # another notifier inserted it first
        db.rollback()
        return False


def discover(db: Session, chat_id: str, now: datetime) -> int:
    """Create the rows of messages that should exist and do not yet."""
    since = _since(db, now)
    want: list[dict] = []
    for a, ap in links.answerable(db):
        want.append(dict(key=f"approval:{a.id}", kind="approval", project_id=ap.project_id,
                         autopilot_id=ap.id, approval_id=a.id))
    for ap in db.execute(select(ProjectAutopilot).where(ProjectAutopilot.status == "paused")).scalars():
        jid = current_pause_journal_id(db, ap)
        if jid is not None:
            want.append(dict(key=f"pause:{jid}", kind="pause", project_id=ap.project_id,
                             autopilot_id=ap.id, journal_id=jid))
    floor = max(since - END_LOOKBACK, now - END_MAX_AGE)
    for ap in db.execute(select(ProjectAutopilot).where(
            ProjectAutopilot.status.in_(("completed", "failed", "stopped")),
            ProjectAutopilot.finished_at >= floor)).scalars():
        if ap.status == "stopped" and (ap.status_reason or "").startswith("stopped by "):
            continue            # the operator stopped it: nothing to tell them
        want.append(dict(key=f"end:{ap.id}", kind="end", project_id=ap.project_id, autopilot_id=ap.id))
    if not want:
        db.rollback()
        return 0
    have = set(db.execute(select(Note.key).where(Note.key.in_([w["key"] for w in want]))).scalars())
    db.rollback()
    n = 0
    for w in want:
        if w["key"] not in have and _add(db, chat_id, now, **w):
            n += 1
    return n


# ---------------------------------------------------------------------------
# Delivery
# ---------------------------------------------------------------------------


class _StopPass(Exception):
    pass


@dataclass
class PassStats:
    discovered: int = 0
    calls: int = 0
    errors: int = 0
    rows: int = 0


@dataclass
class Notifier:
    channel: object
    chat_id: str
    session_factory: Callable[[], Session] = None
    owner: str = field(default_factory=lambda: f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex[:8]}")
    sleep: Callable[[float], None] = time.sleep
    clock: Callable[[], datetime] = utcnow
    pause_until: Optional[datetime] = None
    halted: Optional[str] = None
    _last_closed_check: Optional[datetime] = None

    def __post_init__(self):
        if self.session_factory is None:
            from ..db import SessionLocal

            self.session_factory = SessionLocal

    # -- one pass ------------------------------------------------------------

    def run_pass(self) -> PassStats:
        stats = PassStats()
        now = self.clock()
        if self.pause_until is not None and now < self.pause_until:
            return stats
        self.pause_until = None
        db = self.session_factory()
        try:
            stats.discovered = discover(db, self.chat_id, now)
            ids = self._due(db, now)
            db.rollback()
        finally:
            db.close()
        stats.rows = len(ids)
        for rid in ids:
            if stats.calls >= MAX_CALLS_PER_PASS:
                break
            db = self.session_factory()
            try:
                used = self._work(db, rid)
            except _StopPass:
                break
            except Exception:
                stats.errors += 1
                logger.exception("autopilot notifier: row %s failed", rid)
                used = 0
            finally:
                db.close()
            if used:
                stats.calls += used
                self.sleep(SEND_GAP_SEC)
        return stats

    def _due(self, db: Session, now: datetime) -> list[str]:
        """Every open message (no cap: open ones are few, and a cap would let old
        rows starve new ones), then — at most every CLOSED_RECHECK — the settled
        ones of the last SETTLED_FOR (a week's worth: tens of rows)."""
        ready = (Note.kind != "marker", Note.gone.is_(False),
                 or_(Note.next_try_at.is_(None), Note.next_try_at <= now))
        ids = list(db.execute(select(Note.id).where(*ready, Note.closed_at.is_(None))
                              .order_by(Note.created_at)).scalars())
        if self._last_closed_check is None or now - self._last_closed_check >= CLOSED_RECHECK:
            self._last_closed_check = now
            ids += list(db.execute(select(Note.id).where(
                *ready, Note.closed_at >= now - SETTLED_FOR)
                .order_by(Note.closed_at.desc())).scalars())
        return ids

    def _claim(self, db: Session, rid: str, now: datetime) -> bool:
        res = db.execute(update(Note).where(
            Note.id == rid, or_(Note.claim_owner.is_(None), Note.claim_expires_at < now))
            .values(claim_owner=self.owner, claim_expires_at=now + CLAIM_TTL)
            .execution_options(synchronize_session=False))
        db.commit()
        return res.rowcount == 1

    def _finish(self, db: Session, rid: str, values: dict, now: datetime) -> bool:
        """Write ``values`` and release the claim — only while it is still ours."""
        values = dict(values, claim_owner=None, claim_expires_at=None, updated_at=now)
        res = db.execute(update(Note).where(Note.id == rid, Note.claim_owner == self.owner)
                         .values(**values).execution_options(synchronize_session=False))
        db.commit()
        if res.rowcount != 1:
            # Our claim expired mid-call and another sender may have written a
            # newer state; our late call may still have landed after its edit.
            # Forget the hash so the next pass re-renders and re-edits (at worst
            # a "not modified" no-op) instead of trusting text we cannot see.
            logger.warning("autopilot notifier: lost the claim on %s before writing it back", rid)
            db.execute(update(Note).where(Note.id == rid).values(state_hash=None)
                       .execution_options(synchronize_session=False))
            db.commit()
        return res.rowcount == 1

    def _work(self, db: Session, rid: str) -> int:
        """Bring one message to its desired state. Returns the calls made."""
        now = self.clock()
        if not self._claim(db, rid, now):
            return 0
        row = db.get(Note, rid)
        d = desired(db, row)
        h = _hash(d.text)
        kind, chat, mid = row.kind, row.chat_id or self.chat_id, row.message_id
        posted_at, last_rem, old_hash = row.posted_at, row.last_reminder_at, row.state_hash
        was_closed = row.closed_at
        attempts = row.attempts or 0
        db.rollback()               # no transaction stays open across the HTTP call

        closed = (was_closed or now) if d.final else None
        settle = dict(state=d.state, state_hash=h, closed_at=closed)
        values: dict = {}
        calls = 0
        try:
            if mid is None:
                if not (d.waiting or d.state == "paused" or kind == "end"):
                    # Settled before it was ever shown (or deleted since): no message.
                    values = dict(settle, gone=posted_at is not None)
                else:
                    calls += 1
                    new_mid = self.channel.post(chat, d.text)
                    values = dict(settle, message_id=new_mid, chat_id=chat,
                                  posted_at=posted_at or now)
            elif old_hash != h:
                calls += 1
                try:
                    self.channel.edit(chat, mid, d.text)
                    values = settle
                except ChannelError as exc:
                    if exc.not_modified:
                        values = settle
                    elif exc.gone:
                        # Deleted in the chat: post it again only while it waits.
                        values = (dict(message_id=None, state_hash=None) if d.waiting
                                  else dict(settle, gone=True))
                    else:
                        raise
            elif d.waiting and posted_at is not None:
                base, gap = (last_rem, NEXT_REMINDER) if last_rem else (posted_at, FIRST_REMINDER)
                if now - base >= gap:
                    calls += 1
                    self.channel.post(chat, reminder_text(d, now), reply_to=mid)
                    values = dict(last_reminder_at=now)
            if old_hash == h and mid is not None and not values and was_closed != closed:
                values = dict(closed_at=closed)
            values.update(attempts=0, last_error=None, next_try_at=None)
            self._finish(db, rid, values, now)
            return calls
        except ChannelError as exc:
            attempts += 1
            if exc.retry_after:
                wait = exc.retry_after
                self.pause_until = now + timedelta(seconds=wait)
            elif exc.halt:
                wait = HALT_SEC
                self.pause_until = now + timedelta(seconds=wait)
                if self.halted != str(exc):
                    logger.error("autopilot notifier: stopped sending for %d min: %s", wait // 60, exc)
                self.halted = str(exc)
            else:
                wait = min(MAX_BACKOFF_SEC, 10 * 2 ** min(attempts, 9)) * random.uniform(0.8, 1.2)
            self._finish(db, rid, dict(attempts=attempts, last_error=str(exc)[:500],
                                       next_try_at=now + timedelta(seconds=wait)), now)
            if attempts == FAILING_AFTER:
                logger.error("autopilot notifier: %s keeps failing: %s", rid, exc)
            if exc.retry_after or exc.halt:
                raise _StopPass() from None
            return calls
        except Exception:
            # Our own bug or the DB: release the claim with a backoff, never spin.
            try:
                db.rollback()
                self._finish(db, rid, dict(attempts=attempts + 1, last_error="internal error (see log)",
                                           next_try_at=now + timedelta(seconds=60)), now)
            except Exception:
                logger.exception("autopilot notifier: cannot release %s (the claim expires)", rid)
            raise


# ---------------------------------------------------------------------------
# Wiring: the thread in the autopilot-worker container, status for the inbox
# ---------------------------------------------------------------------------


def config_problem() -> Optional[str]:
    """None when notifications can run, else why not (for the log / status)."""
    if not settings.TELEGRAM_BOT_TOKEN or not settings.TELEGRAM_CHAT_ID:
        return "off: TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID are not both set"
    if not settings.PUBLIC_BASE_URL:
        return "refused: PUBLIC_BASE_URL is not set, so messages could carry no links"
    return None


def make_notifier() -> Notifier:
    return Notifier(TelegramChannel(settings.TELEGRAM_BOT_TOKEN), settings.TELEGRAM_CHAT_ID)


_thread: Optional[threading.Thread] = None
_thread_lock = threading.Lock()


def _loop(n: Notifier) -> None:
    logger.info("autopilot notifier %s started (interval %ss)", n.owner, settings.AUTOPILOT_NOTIFY_INTERVAL_SEC)
    while True:
        try:
            s = n.run_pass()
            if s.calls or s.discovered or s.errors:
                logger.info("autopilot notifier: %d new, %d call(s), %d error(s)",
                            s.discovered, s.calls, s.errors)
        except Exception:
            logger.exception("autopilot notifier: pass failed (the next one retries)")
        time.sleep(max(1, settings.AUTOPILOT_NOTIFY_INTERVAL_SEC))


def ensure_thread() -> Optional[threading.Thread]:
    """Start the notifier thread, or restart it if it died. Called by the
    ticker on start and on every loop, so a dead thread never stays dead."""
    global _thread
    with _thread_lock:
        if _thread is not None and _thread.is_alive():
            return _thread
        problem = config_problem()
        if problem:
            if _thread is None:
                (logger.info if problem.startswith("off") else logger.error)(
                    "autopilot notifier %s", problem)
                _thread = threading.Thread(target=lambda: None)     # remember we checked
            return None
        if _thread is not None and _thread.ident is not None and not problem:
            logger.error("autopilot notifier thread died; restarting it")
        _thread = threading.Thread(target=_loop, args=(make_notifier(),), name="autopilot-notifier",
                                   daemon=True)
        _thread.start()
        return _thread


def failing(db: Session) -> list[Note]:
    """Messages the sender keeps failing on (the inbox's "notification failing")."""
    return list(db.execute(select(Note).where(
        Note.kind != "marker", Note.gone.is_(False), Note.attempts >= FAILING_AFTER)
        .order_by(Note.created_at)).scalars())


def status(db: Session) -> dict:
    rows = db.execute(select(Note).where(Note.kind != "marker")).scalars().all()
    since = db.execute(select(Note.created_at).where(Note.key == MARKER_KEY)).scalar_one_or_none()
    return {
        "config": config_problem() or "on",
        "since": since.isoformat() if since else None,
        "messages": len(rows),
        "open": sum(1 for r in rows if r.closed_at is None and not r.gone),
        "waiting": sum(1 for r in rows if r.state == "waiting" and not r.gone),
        "failing": [{"key": r.key, "attempts": r.attempts, "error": r.last_error}
                    for r in rows if r.attempts >= FAILING_AFTER and not r.gone],
    }
