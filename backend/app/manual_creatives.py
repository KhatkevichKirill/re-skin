"""Creatives made outside the pipelines by an operator script (AICP-239).

A one-off script can produce a deliverable the pipelines never see — e.g.
AICP-238 burned a localised card over six externally dubbed Drive videos via
``source_type="upload"`` overlay jobs. This module gives such a creative a
:class:`~app.models.ManualCreative` row (its history) and exports it to Final
Creatives under its own Job ID::

    Job ID         = languages.job_id_for(row.id, row.language)   # b73fe636-de
    Pipeline Type  = re-skin-manual-script
    Idea           = operator-given (jjuliaggr-Sport-DE)
    Input Link     = row.source_link
    Creative Link  = the Drive result

There is no UI and no HTTP endpoint: it is run inside the api container, where
the DB and the Sheets credentials are::

    docker exec -e PYTHONPATH=/app re-skin-api python -m app.manual_creatives \\
        register-overlay --overlay-job <id> --idea jjuliaggr-Sport-DE \\
        --language de --source-link <drive url> --asset-id <library asset id> \\
        --linear AICP-238
    docker exec -e PYTHONPATH=/app re-skin-api python -m app.manual_creatives export <id>
    docker exec -e PYTHONPATH=/app re-skin-api python -m app.manual_creatives show <id or Job ID>

An overlay job registered here is no longer exportable as ``re-skin-overlay``
(api_overlay.export_overlay_to_sheet 409s), and one already exported that way
cannot be registered — either would put the same Drive file in the sheet twice.

The export runs synchronously and stamps ``final_creatives_exported_at`` only
after the append (or the dedupe hit) succeeded, so a failed export leaves the
row exportable rather than looking "sent".
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import date, datetime, timezone

from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from . import languages
from .models import ManualCreative, OverlayAsset, OverlayJob

PIPELINE_TYPE = "re-skin-manual-script"
_RUN_ON_META = "need_to_rename"

# The downstream renamer's rule for the Idea cell (ai-fb-audit
# creative_rename_filename.RE_RENAME_IDEA, max 40): a row that fails it is
# skipped by Rename, so reject it here instead.
_IDEA_RE = re.compile(r"^[a-zA-Z0-9]+(?:[-_][a-zA-Z0-9]+)*$")
IDEA_MAX_LEN = 40


class ManualCreativeError(ValueError):
    """Invalid registration/export request."""


def validate_idea(idea: str | None) -> str:
    value = (idea or "").strip()
    if not value or len(value) > IDEA_MAX_LEN or not _IDEA_RE.match(value):
        raise ManualCreativeError(
            f"Idea {idea!r} must be 1-{IDEA_MAX_LEN} ASCII letters/digits "
            "with internal '-' or '_'"
        )
    return value


def _language(code: str | None) -> str | None:
    """Normalise *code*; English is stored as NULL (unsuffixed, as for runs)."""
    if code is None or not code.strip():
        return None
    key = languages.normalize(code)
    if key is None:
        raise ManualCreativeError(
            f"unknown language {code!r}; expected one of {sorted(languages.LANGUAGES)}"
        )
    return None if key == languages.DEFAULT_LANGUAGE else key


def sheet_job_id(mc: ManualCreative) -> str:
    """The Final Creatives ``Job ID``: 8-char id prefix + language suffix."""
    return languages.job_id_for(mc.id, mc.language)


def creative_link(mc: ManualCreative) -> str:
    return f"https://drive.google.com/file/d/{mc.gdrive_file_id}/view"


def register_overlay(
    db: Session,
    *,
    overlay_job_id: str,
    idea: str,
    language: str | None,
    source_link: str | None,
    asset_id: str | None = None,
    linear_issue: str | None = None,
    notes: str | None = None,
) -> ManualCreative:
    """Register a done, Drive-delivered overlay job as a manual creative.

    Snapshots what tracing needs (asset id + name, placements, source
    dimensions) into the row, because the job's files and the library asset
    can both be deleted later. Commits.
    """
    # Row lock: the regular overlay export takes the same one before its
    # "is it registered here?" check, so the two can't both pass.
    job = db.get(OverlayJob, overlay_job_id, with_for_update=True)
    if job is None:
        raise ManualCreativeError(f"overlay job {overlay_job_id} not found")
    status = job.status.value if hasattr(job.status, "value") else str(job.status)
    if status != "done" or not job.gdrive_file_id:
        raise ManualCreativeError(
            f"overlay job {overlay_job_id} is {status!r} with "
            f"gdrive_file_id={job.gdrive_file_id!r}; need a done, delivered job"
        )
    if job.final_creatives_exported_at is not None:
        raise ManualCreativeError(
            f"overlay job {overlay_job_id} was already exported as re-skin-overlay"
        )
    existing = db.execute(
        select(ManualCreative).where(
            or_(
                ManualCreative.overlay_job_id == job.id,
                ManualCreative.gdrive_file_id == job.gdrive_file_id,
            )
        )
    ).scalar_one_or_none()
    if existing is not None:
        raise ManualCreativeError(
            f"overlay job {overlay_job_id} is already registered as {existing.id}"
        )

    asset_name = None
    if asset_id:
        asset = db.get(OverlayAsset, asset_id)
        if asset is None:
            raise ManualCreativeError(f"overlay library asset {asset_id} not found")
        asset_name = asset.name

    mc = ManualCreative(
        pipeline_type=PIPELINE_TYPE,
        idea=validate_idea(idea),
        language=_language(language),
        source_link=(source_link or "").strip() or None,
        gdrive_file_id=job.gdrive_file_id,
        overlay_job_id=job.id,
        overlay_asset_id=asset_id or None,
        overlay_asset_name=asset_name,
        details={
            "overlays": job.overlays or [],
            "source_type": job.source_type,
            "src_width": job.src_width,
            "src_height": job.src_height,
            "src_duration_sec": job.src_duration_sec,
        },
        linear_issue=(linear_issue or "").strip() or None,
        notes=notes,
    )
    db.add(mc)
    db.commit()
    return mc


def find(db: Session, ref: str) -> ManualCreative:
    """Look a row up by full id, id prefix, or sheet Job ID (``b73fe636-de``)."""
    prefix = languages.strip_suffix(ref.strip())
    if not prefix:
        raise ManualCreativeError("empty id")
    rows = db.execute(
        select(ManualCreative).where(ManualCreative.id.startswith(prefix))
    ).scalars().all()
    if len(rows) != 1:
        raise ManualCreativeError(f"{ref!r} matches {len(rows)} manual creatives")
    return rows[0]


def export_to_sheet(db: Session, mc: ManualCreative, *, client=None) -> str:
    """Append *mc* to Final Creatives. Returns ``added``/``exists``/``already``.

    Dedupes on Creative Link like the overlay export. Refuses when the sheet
    already holds this Job ID for a DIFFERENT creative (an 8-char prefix
    collision), since downstream keys on it.
    """
    from .config import settings
    from .sheets_client import SheetsClient, SheetsError, build_final_creatives_row

    # Claim the row for the whole sheet round-trip, so two concurrent exports
    # of the same creative can't both see it unstamped and both append.
    db.refresh(mc, with_for_update=True)
    if mc.final_creatives_exported_at is not None:
        return "already"

    job_id = sheet_job_id(mc)
    link = creative_link(mc)
    sid = settings.FINAL_CREATIVES_SPREADSHEET_ID
    tab = settings.FINAL_CREATIVES_TAB
    client = client or SheetsClient()
    header = client.get_header(sid, tab)
    if not header:
        raise SheetsError(f"tab {tab!r} has no header row")
    norm = {h.strip().lower(): i for i, h in enumerate(header)}

    result = "added"
    link_idx, job_idx = norm.get("creative link"), norm.get("job id")
    links = client.column_values(sid, tab, link_idx) if link_idx is not None else []
    if any(v.strip() == link for v in links):
        result = "exists"
    else:
        if job_idx is not None:
            ids = client.column_values(sid, tab, job_idx)
            if any(v.strip() == job_id for v in ids):
                raise ManualCreativeError(
                    f"Job ID {job_id} is already in the sheet for another creative"
                )
        fields = {
            "Date": date.today().isoformat(),
            "Job ID": job_id,
            "Input Link": mc.source_link or "",
            "Creative Link": link,
            "Pipeline Type": mc.pipeline_type,
            "Run on Meta": _RUN_ON_META,
            "Idea": mc.idea,
        }
        client.append_row(sid, tab, build_final_creatives_row(header, fields=fields))

    mc.final_creatives_exported_at = datetime.now(timezone.utc)
    db.commit()
    return result


def describe(mc: ManualCreative) -> dict:
    return {
        "id": mc.id,
        "job_id": sheet_job_id(mc),
        "pipeline_type": mc.pipeline_type,
        "idea": mc.idea,
        "language": mc.language or languages.DEFAULT_LANGUAGE,
        "source_link": mc.source_link,
        "creative_link": creative_link(mc),
        "overlay_job_id": mc.overlay_job_id,
        "overlay_asset": f"{mc.overlay_asset_name} ({mc.overlay_asset_id})"
        if mc.overlay_asset_id else None,
        "linear_issue": mc.linear_issue,
        "notes": mc.notes,
        "created_at": mc.created_at.isoformat() if mc.created_at else None,
        "final_creatives_exported_at": mc.final_creatives_exported_at.isoformat()
        if mc.final_creatives_exported_at else None,
    }


def main(argv: list[str] | None = None) -> int:
    from .db import SessionLocal

    p = argparse.ArgumentParser(prog="python -m app.manual_creatives")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("register-overlay", help="register a done overlay job")
    r.add_argument("--overlay-job", required=True)
    r.add_argument("--idea", required=True)
    r.add_argument("--language", default=None)
    r.add_argument("--source-link", default=None)
    r.add_argument("--asset-id", default=None)
    r.add_argument("--linear", default=None)
    r.add_argument("--notes", default=None)
    e = sub.add_parser("export", help="append to Final Creatives")
    e.add_argument("ref", nargs="+")
    s = sub.add_parser("show", help="print a row (id, prefix or Job ID)")
    s.add_argument("ref", nargs="*")
    args = p.parse_args(argv)

    with SessionLocal() as db:
        try:
            if args.cmd == "register-overlay":
                mc = register_overlay(
                    db, overlay_job_id=args.overlay_job, idea=args.idea,
                    language=args.language, source_link=args.source_link,
                    asset_id=args.asset_id, linear_issue=args.linear, notes=args.notes,
                )
                print(json.dumps(describe(mc), indent=1))
            elif args.cmd == "export":
                for ref in args.ref:
                    mc = find(db, ref)
                    print(sheet_job_id(mc), export_to_sheet(db, mc))
            else:
                rows = [find(db, ref) for ref in args.ref] or db.execute(
                    select(ManualCreative).order_by(ManualCreative.created_at)
                ).scalars().all()
                print(json.dumps([describe(mc) for mc in rows], indent=1))
        except ManualCreativeError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
