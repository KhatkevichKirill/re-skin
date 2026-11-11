"""Project spend for the autopilot budget (docs/autopilot-design.md §6.1, §5.7).

Spend is read from the AICP-155 ledger by **scope_key**. A scope counts for a
project once it is registered in ``project_spend_scopes`` (project-owned,
append-only, so deleting a run keeps its spend). No baselines.
"""

from __future__ import annotations

import contextvars
import logging
from contextlib import contextmanager
from typing import Iterable, Optional

from sqlalchemy import case, func, select
from sqlalchemy.orm import Session

from ..models import KieSpendDraw, KieSpendReservation, Run, VideoProject
from ..models_autopilot import ProjectManualSpendEvent, ProjectSpendScope, utcnow

log = logging.getLogger(__name__)

# Set by the core around ActionHandler.execute (§5.7): reservations made inside
# it are the autopilot's own, never "manual paid actions".
_in_autopilot_admission: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "autopilot_admission", default=False
)


@contextmanager
def autopilot_admission():
    token = _in_autopilot_admission.set(True)
    try:
        yield
    finally:
        _in_autopilot_admission.reset(token)


def in_autopilot_admission() -> bool:
    return _in_autopilot_admission.get()


# ---------------------------------------------------------------------------
# Scope discovery / registration
# ---------------------------------------------------------------------------


def job_scopes(db: Session, project_id: str) -> set[str]:
    """The per-job scopes of the project's durable transcription / translate
    jobs (AICP-351: each job reserves its own §5.2 worst case)."""
    from ..localisation_jobs_models import LocalisationPromptJob, ProjectTranscription
    from ..spend import localise_job_scope, transcribe_job_scope

    out = {transcribe_job_scope(i) for (i,) in db.execute(
        select(ProjectTranscription.id).where(ProjectTranscription.project_id == project_id))}
    out |= {localise_job_scope(i) for (i,) in db.execute(
        select(LocalisationPromptJob.id).where(LocalisationPromptJob.project_id == project_id))}
    return out


def project_scopes(db: Session, project_id: str) -> set[str]:
    """Every ledger scope that belongs to the project's CURRENT rows."""
    from ..models import RunCaption, RunSegmentImageJob, RunSegmentRangeAttempt
    from ..provider_models import ProviderTask

    run_ids = [r for (r,) in db.execute(select(Run.id).where(Run.project_id == project_id))]
    scopes = {f"run:{rid}" for rid in run_ids}
    scopes |= {f"transcribe:{project_id}", f"localise:{project_id}"}
    scopes |= job_scopes(db, project_id)
    if run_ids:
        for model, prefix in ((RunCaption, "caption"), (RunSegmentRangeAttempt, "range"),
                              (RunSegmentImageJob, "segimg")):
            scopes |= {
                f"{prefix}:{i}" for (i,) in db.execute(
                    select(model.id).where(model.run_id.in_(run_ids)))
            }
    scopes |= {
        f"fal:{i}" for (i,) in db.execute(
            select(ProviderTask.id).where(ProviderTask.owner_project_id == project_id))
    }
    return scopes


def register_scopes(db: Session, project_id: str, scopes: Iterable[str], *,
                    autopilot_id: Optional[str], source: str) -> int:
    """Insert missing registrations (no commit). Returns how many were new."""
    scopes = set(scopes)
    if not scopes:
        return 0
    have = {
        s for (s,) in db.execute(
            select(ProjectSpendScope.scope_key).where(
                ProjectSpendScope.project_id == project_id,
                ProjectSpendScope.scope_key.in_(scopes)))
    }
    pending = {
        o.scope_key for o in db.new
        if isinstance(o, ProjectSpendScope) and o.project_id == project_id
    }
    new = sorted(scopes - have - pending)
    for s in new:
        db.add(ProjectSpendScope(project_id=project_id, scope_key=s,
                                 registered_by_autopilot_id=autopilot_id, source=source,
                                 registered_at=utcnow()))
    return len(new)


def registered_scopes(db: Session, project_id: str) -> list[str]:
    return [s for (s,) in db.execute(
        select(ProjectSpendScope.scope_key).where(ProjectSpendScope.project_id == project_id))]


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def figures_mcredits(db: Session, project_id: str) -> dict:
    """``{committed, consumed, settled}`` in milli-credits over registered scopes."""
    scopes = select(ProjectSpendScope.scope_key).where(
        ProjectSpendScope.project_id == project_id).scalar_subquery()
    committed = db.execute(
        select(func.coalesce(func.sum(KieSpendReservation.charged_mcredits), 0))
        .where(KieSpendReservation.scope_key.in_(scopes))
    ).scalar_one()
    consumed, settled = db.execute(
        select(
            func.coalesce(func.sum(KieSpendDraw.mcredits), 0),
            func.coalesce(func.sum(
                case((KieSpendDraw.settled_at.isnot(None), KieSpendDraw.mcredits), else_=0)
            ), 0),
        ).join(KieSpendReservation, KieSpendDraw.reservation_id == KieSpendReservation.id)
        .where(KieSpendReservation.scope_key.in_(scopes))
    ).one()
    return {"committed": int(committed or 0), "consumed": int(consumed or 0),
            "settled": int(settled or 0)}


# ---------------------------------------------------------------------------
# Manual paid actions (§5.7, operator decision Q5)
# ---------------------------------------------------------------------------


def _project_of_scope(db: Session, scope_key: str) -> Optional[str]:
    from ..models import RunCaption, RunSegmentImageJob, RunSegmentRangeAttempt
    from ..provider_models import ProviderTask

    prefix, _, ident = scope_key.partition(":")
    if not ident:
        return None
    if prefix in ("transcribe", "localise"):
        return ident
    if prefix == "run":
        return db.execute(select(Run.project_id).where(Run.id == ident)).scalar_one_or_none()
    model = {"caption": RunCaption, "range": RunSegmentRangeAttempt,
             "segimg": RunSegmentImageJob}.get(prefix)
    if model is not None:
        return db.execute(
            select(Run.project_id).join(model, model.run_id == Run.id).where(model.id == ident)
        ).scalar_one_or_none()
    if prefix == "fal":
        return db.execute(
            select(ProviderTask.owner_project_id).where(ProviderTask.id == ident)
        ).scalar_one_or_none()
    return None


def note_reservation(db: Session, scope_key: str, credits: float, kind: str) -> None:
    """Called by ``spend.reserve`` (after its flush, before the day lock).

    Records a manual paid admission on an assistant project. Never raises and
    never takes a lock on autopilot rows; runs inside a SAVEPOINT so a failure
    here cannot abort the caller's transaction.
    """
    from ..models_autopilot import LIVE_AUTOPILOT_STATUSES, ProjectAutopilot

    # Deliberately NOT gated on AUTOPILOT_ENABLED: paid work done while the
    # ticker is disabled must still pause the autopilot once it runs again.
    if credits <= 0 or in_autopilot_admission():
        return
    # No SAVEPOINT (on SQLite a savepoint opened before the caller's first
    # write commits on release) and no flush: the event row is only ADDED, so
    # it is written by the caller's own commit and vanishes with its rollback
    # (a refused reservation leaves nothing). The reads cannot conflict.
    try:
        any_live = db.execute(select(ProjectAutopilot.id).where(
            ProjectAutopilot.status.in_(LIVE_AUTOPILOT_STATUSES)).limit(1)).first()
        if any_live is None:
            return  # cheap exit: no assistant project has a live autopilot
        pid = _project_of_scope(db, scope_key)
        if not pid:
            return
        live = db.execute(select(ProjectAutopilot.id).where(
            ProjectAutopilot.project_id == pid,
            ProjectAutopilot.status.in_(LIVE_AUTOPILOT_STATUSES)).limit(1)).first()
        if live is None:
            return
        db.add(ProjectManualSpendEvent(
            project_id=pid, scope_key=scope_key, kind=str(kind)[:64],
            mcredits=int(round(float(credits) * 1000)), at=utcnow(),
        ))
    except Exception:
        log.exception("autopilot: could not record a manual paid action on %s", scope_key)
