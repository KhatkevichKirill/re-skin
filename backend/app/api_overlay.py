"""
api_overlay.py — REST API for the standalone overlay-burn feature.

Mounted at ``/api/v2/overlays`` (see main.py). Lets an operator pick a completed
Run's video as a source, upload transparent PNG cards, position them (normalized
coords + time window), and submit a burn-in job.

Shares NO state with the run pipeline — it only *reads* completed Run rows to
offer them as source videos.
"""

from __future__ import annotations

import logging
import os
import shutil
import struct
import tempfile
import uuid
from datetime import date, datetime, timezone

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
    UploadFile,
    status,
)
from fastapi.responses import FileResponse, Response
from sqlalchemy import desc, select
from sqlalchemy.orm import Session, selectinload

from . import overlay_runs
from .api_v2 import _admit_retained_run_source, _lock_runs_for_result_admission
from .accel import accel_or_file_response
from .config import settings
from .db import get_db, get_session
from .ops_gate import GATED
from .gdrive_client import GDriveClient, GDriveError, extract_file_id
from .models import ManualCreative, OverlayAsset, OverlayJob, Run, RunCaption, RunTailEdit
from .public import make_result_token
from .schemas_overlay import (
    OverlayAssetResponse,
    OverlayCreateRequest,
    OverlayCreateResponse,
    OverlayDriveAssetRequest,
    OverlayLibraryAssetResponse,
    OverlayResponse,
    OverlaySourceItem,
    OverlaysUpdateRequest,
)
from .state_machine import (
    CaptionStatus,
    InvalidTransition,
    OverlayStatus,
    RunStatus,
    TailEditStatus,
    transition,
)
from .storage import (
    overlay_asset_library_dir,
    overlay_assets_dir,
    overlay_job_dir,
    resolve_overlay_asset,
)
from .tasks import (
    enqueue_export_overlay_to_sheet,
    enqueue_export_or_rollback,
    enqueue_process_overlay,
)

log = logging.getLogger(__name__)

router = APIRouter(tags=["overlay"])

# PNG magic number (first 8 bytes of every PNG file).
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"

# Statuses from which a job may be (re-)submitted.
_RESUBMITTABLE = {OverlayStatus.created, OverlayStatus.failed, OverlayStatus.done}

# Statuses that block deletion (a job actively using the worker / queue).
_ACTIVE_OVERLAY_STATUSES = {
    OverlayStatus.queued,
    OverlayStatus.processing,
    OverlayStatus.delivering,
}

# Final Creatives row constants (overlay variant).
_PIPELINE_TYPE = "re-skin-overlay"
_RUN_ON_META = "need_to_rename"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_job_or_404(job_id: str, db: Session) -> OverlayJob:
    job = db.get(OverlayJob, job_id)
    # Soft-deleted jobs are hidden from the API (the row survives only for
    # DB-level source-video lookups).
    if job is None or job.deleted_at is not None:
        raise HTTPException(status_code=404, detail=f"Overlay job {job_id!r} not found")
    return job


def _save_upload(upload: UploadFile, dest: str, max_bytes: int | None = None) -> None:
    """Stream an UploadFile to *dest* in 1-MiB chunks (mirror api_v2._save_upload)."""
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    _CHUNK = 1 << 20
    written = 0
    try:
        with open(dest, "wb") as fh:
            while True:
                chunk = upload.file.read(_CHUNK)
                if not chunk:
                    break
                written += len(chunk)
                if max_bytes is not None and written > max_bytes:
                    fh.close()
                    os.unlink(dest)
                    raise HTTPException(
                        status_code=413,
                        detail=f"Upload too large: limit is {max_bytes >> 20} MiB",
                    )
                fh.write(chunk)
    except HTTPException:
        raise
    except Exception:
        if os.path.exists(dest):
            os.unlink(dest)
        raise


def _png_dimensions(path: str) -> tuple[int | None, int | None]:
    """Read (width, height) from a PNG's IHDR chunk; (None, None) on failure.

    Avoids a new dependency (Pillow) or an ffprobe subprocess for what is a
    fixed, trivial layout: an 8-byte signature followed by the IHDR chunk
    (4-byte length + b"IHDR" + 4-byte width + 4-byte height + ...).
    """
    try:
        with open(path, "rb") as fh:
            head = fh.read(24)
        if len(head) < 24 or head[:8] != _PNG_SIGNATURE or head[12:16] != b"IHDR":
            return None, None
        width, height = struct.unpack(">II", head[16:24])
        return width, height
    except OSError:
        return None, None


# ---------------------------------------------------------------------------
# Sources — selectable completed Runs
# ---------------------------------------------------------------------------


@router.get("/sources", response_model=list[OverlaySourceItem])
def list_sources(db: Session = Depends(get_db), limit: int = Query(200, ge=1, le=500)) -> list:
    """List completed Runs that can be used as an overlay source video.

    Args:
        limit: Maximum number of results (1-500, default 200).
    """
    runs = (
        db.execute(
            select(Run)
            .where(Run.status == RunStatus.done)
            .order_by(desc(Run.created_at))
            .limit(limit)
            .options(selectinload(Run.project))
        )
        .scalars()
        .all()
    )
    items: list[OverlaySourceItem] = []
    for run in runs:
        if not run.result_local_path or not os.path.exists(run.result_local_path):
            continue
        project = run.project
        tok = make_result_token(run.id)  # "" when public links are disabled
        items.append(
            OverlaySourceItem(
                run_id=run.id,
                run_name=run.name or None,
                project_name=(project.name if project else None),
                created_at=run.created_at,
                width=project.width if project else None,
                height=project.height if project else None,
                duration_sec=project.duration_sec if project else None,
                preview_url=(
                    f"/public/runs/{run.id}/result?token={tok}" if tok else None
                ),
            )
        )
    return items


# ---------------------------------------------------------------------------
# Overlay asset library — reusable PNG cards (e.g. branded logo bugs), global
# and not tied to any job. Mirrors the stitch-clip library in api_v2.py
# (create_library_clip / list_library_clips / get_library_clip_file /
# delete_library_clip), reusing this module's own PNG-signature validation
# and Drive-download flow. See models.OverlayAsset's docstring: unlike
# LibraryClip, nothing ever soft-references one of these rows — attaching one
# to a job (see the `/{job_id}/library-assets/{asset_id}` endpoint below)
# COPIES the file into the job's own assets dir — so delete here needs no
# active-job guard.
#
# Registered before the `/{job_id}` job routes further down so a literal path
# like `/library-assets` is never shadowed by the `/{job_id}` path-param route.
# ---------------------------------------------------------------------------


def _get_library_asset_or_404(asset_id: str, db: Session) -> OverlayAsset:
    asset = db.get(OverlayAsset, asset_id)
    if asset is None:
        raise HTTPException(status_code=404, detail=f"Overlay library asset not found: {asset_id}")
    return asset


@router.post(
    "/library-assets",
    status_code=status.HTTP_201_CREATED,
    response_model=OverlayLibraryAssetResponse,
)
def create_library_asset(
    file: UploadFile | None = File(None),
    gdrive_url: str | None = Form(None),
    name: str | None = Form(None),
    db: Session = Depends(get_db),
) -> OverlayLibraryAssetResponse:
    """Add a reusable PNG card to the overlay library.

    Accepts either a file upload OR a Google Drive link (not both), same
    either/or contract as POST /library-clips. A non-PNG (or failed Drive
    download) is rejected with 400 and never lands on disk.
    """
    if file and gdrive_url:
        raise HTTPException(
            status_code=400, detail="Provide either a file or a Drive link, not both"
        )
    if not file and not gdrive_url:
        raise HTTPException(
            status_code=400, detail="Provide either a file or a Drive link"
        )

    asset_id = str(uuid.uuid4())
    dest = os.path.join(overlay_asset_library_dir(asset_id), "card.png")
    clean_name: str

    try:
        if file:
            head = file.file.read(8)
            file.file.seek(0)
            if head != _PNG_SIGNATURE:
                raise HTTPException(status_code=400, detail="asset must be a PNG file")
            _save_upload(file, dest, max_bytes=settings.MAX_UPLOAD_SIZE_MB << 20)
            clean_name = (name or "").strip() or (file.filename or "card")
        else:
            # Download to a temp file first so a non-PNG never lands in the
            # library dir (mirrors add_asset_from_drive).
            fd, tmp = tempfile.mkstemp(suffix=".png", dir=overlay_asset_library_dir(asset_id))
            os.close(fd)
            try:
                GDriveClient().download_file(gdrive_url, tmp)
                with open(tmp, "rb") as fh:
                    if fh.read(8) != _PNG_SIGNATURE:
                        raise HTTPException(status_code=400, detail="linked file is not a PNG")
                os.replace(tmp, dest)
            finally:
                if os.path.exists(tmp):
                    os.unlink(tmp)
            file_id = extract_file_id(gdrive_url)
            clean_name = (name or "").strip() or file_id or "drive card"
    except GDriveError as exc:
        shutil.rmtree(overlay_asset_library_dir(asset_id), ignore_errors=True)
        raise HTTPException(status_code=400, detail=f"Drive download failed: {exc}") from exc
    except HTTPException:
        shutil.rmtree(overlay_asset_library_dir(asset_id), ignore_errors=True)
        raise
    except Exception:
        # Oversize upload (413) or unexpected failure — don't leave an orphaned
        # file/dir under <DATA>/overlay_library that nothing can ever list.
        shutil.rmtree(overlay_asset_library_dir(asset_id), ignore_errors=True)
        raise

    width, height = _png_dimensions(dest)
    asset = OverlayAsset(
        id=asset_id, name=clean_name[:255], local_path=dest, width=width, height=height,
    )
    db.add(asset)
    db.commit()
    log.info("Created overlay library asset %s (%r)", asset_id, asset.name)
    return OverlayLibraryAssetResponse.model_validate(asset)


@router.get("/library-assets", response_model=list[OverlayLibraryAssetResponse])
def list_library_assets(db: Session = Depends(get_db)) -> list:
    """Return newest 200 overlay library assets, newest first."""
    assets = (
        db.execute(select(OverlayAsset).order_by(desc(OverlayAsset.created_at)).limit(200))
        .scalars()
        .all()
    )
    return [OverlayLibraryAssetResponse.model_validate(a) for a in assets]


@router.get("/library-assets/{asset_id}/file")
def get_library_asset_file(asset_id: str):
    """Stream a library asset's PNG (for in-page preview thumbnails)."""
    # Session is closed before the file transfer (mirrors /{job_id}/asset).
    with get_session() as db:
        asset = _get_library_asset_or_404(asset_id, db)
        path = asset.local_path
        if not path or not os.path.exists(path):
            raise HTTPException(status_code=404, detail="Asset file not available on disk")

    return accel_or_file_response(path, media_type="image/png", filename=os.path.basename(path))


@router.delete("/library-assets/{asset_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_library_asset(asset_id: str, db: Session = Depends(get_db)) -> Response:
    """Delete a library asset (DB row + file).

    No reference/active-job guard is needed: attaching an asset to a job
    copies the PNG into that job's own assets dir (see the attach endpoint
    below), so an already-attached job's asset_path keeps working after this
    deletes the library row.
    """
    asset = _get_library_asset_or_404(asset_id, db)
    asset_name = asset.name
    db.delete(asset)
    db.commit()
    shutil.rmtree(overlay_asset_library_dir(asset_id), ignore_errors=True)
    log.info("Deleted overlay library asset %s (%r)", asset_id, asset_name)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


# ---------------------------------------------------------------------------
# Job CRUD
# ---------------------------------------------------------------------------


@router.post("", status_code=status.HTTP_201_CREATED, response_model=OverlayCreateResponse)
def create_overlay_job(
    body: OverlayCreateRequest, db: Session = Depends(get_db)
) -> OverlayCreateResponse:
    """Create an overlay job. For source_type='upload', upload the video next."""
    if body.source_type == "run":
        run = _lock_runs_for_result_admission(db, [body.source_run_id])[body.source_run_id]
        if run.status != RunStatus.done:
            raise HTTPException(status_code=409, detail="source run is not done")
        # A run can be done WITHOUT a final video (stitching disabled via an
        # empty stitch plan) — reject it here instead of failing at render time.
        if not run.result_local_path or not os.path.exists(run.result_local_path):
            raise HTTPException(
                status_code=409,
                detail="source run has no final video (stitching disabled?)",
            )
    elif body.source_type == "tail_edit":
        edit = db.get(RunTailEdit, body.source_tail_edit_id)
        if edit is None:
            raise HTTPException(status_code=404, detail="source tail edit not found")
        if edit.status != TailEditStatus.done:
            raise HTTPException(status_code=409, detail="source tail edit is not done")
        if not edit.output_path or not os.path.exists(edit.output_path):
            raise HTTPException(
                status_code=409,
                detail="source tail edit has no output video on disk",
            )
    elif body.source_type == "caption":
        caption = db.get(RunCaption, body.source_caption_id)
        if caption is None:
            raise HTTPException(status_code=404, detail="source caption not found")
        if caption.status != CaptionStatus.done:
            raise HTTPException(status_code=409, detail="source caption is not done")
        if not caption.output_path or not os.path.exists(caption.output_path):
            raise HTTPException(
                status_code=409,
                detail="source caption has no output video on disk",
            )

    job_id = str(uuid.uuid4())
    job = OverlayJob(
        id=job_id,
        status=OverlayStatus.created,
        source_type=body.source_type,
        source_run_id=body.source_run_id if body.source_type == "run" else None,
        source_tail_edit_id=(
            body.source_tail_edit_id if body.source_type == "tail_edit" else None
        ),
        source_caption_id=(
            body.source_caption_id if body.source_type == "caption" else None
        ),
        # Stamped now, from a source just validated as existing and done, so
        # the run link outlives that source being deleted — see
        # OverlayJob.parent_run_id.
        parent_run_id=overlay_runs.resolve_parent_run_id(
            source_type=body.source_type,
            source_run_id=body.source_run_id,
            source_tail_edit_id=body.source_tail_edit_id,
            source_caption_id=body.source_caption_id,
            db=db,
        ),
        deliver_to_drive=body.deliver_to_drive,
        overlays=[],
    )
    db.add(job)
    db.commit()
    log.info("Created overlay job %s (source_type=%s)", job_id, body.source_type)
    return OverlayCreateResponse(id=job_id, status=OverlayStatus.created.value)


@router.post("/{job_id}/source", response_model=OverlayResponse)
def upload_source_video(
    job_id: str,
    video_file: UploadFile = File(...),
    db: Session = Depends(get_db),
) -> OverlayJob:
    """Upload the source video for an ``upload``-type overlay job."""
    job = _get_job_or_404(job_id, db)
    if job.source_type != "upload":
        raise HTTPException(
            status_code=409, detail="source video upload only valid for source_type='upload'"
        )
    ext = os.path.splitext(video_file.filename or "")[1].lower() or ".mp4"
    dest = os.path.join(overlay_job_dir(job_id), f"source{ext}")
    _save_upload(video_file, dest, max_bytes=settings.MAX_UPLOAD_SIZE_MB << 20)
    job.source_local_path = dest
    db.commit()
    return job


def _next_asset_path(job_id: str) -> tuple[str, str]:
    """Return (relative, absolute) path for the next card_NN.png in the job."""
    assets = overlay_assets_dir(job_id)
    existing = len([f for f in os.listdir(assets) if f.lower().endswith(".png")])
    rel = os.path.join("assets", f"card_{existing + 1:02d}.png")
    return rel, os.path.join(overlay_job_dir(job_id), rel)


@router.post("/{job_id}/assets", response_model=OverlayAssetResponse)
def upload_asset(
    job_id: str,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
) -> OverlayAssetResponse:
    """Upload one PNG card; returns the stored relative asset_path."""
    _get_job_or_404(job_id, db)

    head = file.file.read(8)
    file.file.seek(0)
    if head != _PNG_SIGNATURE:
        raise HTTPException(status_code=400, detail="asset must be a PNG file")

    rel, dest = _next_asset_path(job_id)
    _save_upload(file, dest, max_bytes=settings.MAX_UPLOAD_SIZE_MB << 20)
    log.info("Overlay job %s: uploaded asset %s", job_id, rel)
    return OverlayAssetResponse(asset_path=rel)


@router.post("/{job_id}/assets/from-drive", response_model=OverlayAssetResponse)
def add_asset_from_drive(
    job_id: str, body: OverlayDriveAssetRequest, db: Session = Depends(get_db)
) -> OverlayAssetResponse:
    """Add a PNG card by Google Drive link/id: download, validate, store."""
    _get_job_or_404(job_id, db)
    link = body.link.strip()
    if not link:
        raise HTTPException(status_code=400, detail="Drive link is required")

    # Download to a temp file first so a non-PNG never lands in assets/.
    fd, tmp = tempfile.mkstemp(suffix=".png", dir=overlay_job_dir(job_id))
    os.close(fd)
    try:
        try:
            GDriveClient().download_file(link, tmp)
        except GDriveError as exc:
            raise HTTPException(status_code=400, detail=f"Drive download failed: {exc}") from exc
        with open(tmp, "rb") as fh:
            if fh.read(8) != _PNG_SIGNATURE:
                raise HTTPException(status_code=400, detail="linked file is not a PNG")
        rel, dest = _next_asset_path(job_id)
        os.replace(tmp, dest)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    log.info("Overlay job %s: added asset %s from Drive", job_id, rel)
    return OverlayAssetResponse(asset_path=rel)


@router.post("/{job_id}/library-assets/{asset_id}", response_model=OverlayAssetResponse)
def attach_library_asset(
    job_id: str, asset_id: str, db: Session = Depends(get_db)
) -> OverlayAssetResponse:
    """Attach a library PNG to a job by COPYING it into the job's assets dir.

    Deliberately does not store a reference to the library asset anywhere —
    see models.OverlayAsset's docstring. The response has the same shape as
    upload_asset()/add_asset_from_drive() (just {asset_path}) so the editor
    can treat "picked from the library" identically to "just uploaded".
    """
    _get_job_or_404(job_id, db)
    asset = _get_library_asset_or_404(asset_id, db)
    if not asset.local_path or not os.path.exists(asset.local_path):
        raise HTTPException(status_code=404, detail="Library asset file not available on disk")

    rel, dest = _next_asset_path(job_id)
    shutil.copy2(asset.local_path, dest)
    log.info("Overlay job %s: attached library asset %s as %s", job_id, asset_id, rel)
    return OverlayAssetResponse(asset_path=rel)


@router.put("/{job_id}/overlays", response_model=OverlayResponse)
def replace_overlays(
    job_id: str, body: OverlaysUpdateRequest, db: Session = Depends(get_db)
) -> OverlayJob:
    """Replace the full ordered list of overlay placements."""
    job = _get_job_or_404(job_id, db)
    specs: list[dict] = []
    for item in body.overlays:
        if resolve_overlay_asset(job_id, item.asset_path) is None:
            raise HTTPException(
                status_code=400, detail=f"asset not found: {item.asset_path}"
            )
        specs.append(item.model_dump(exclude_none=False))
    job.overlays = specs
    db.commit()
    log.info("Overlay job %s: saved %d placement(s)", job_id, len(specs))
    return job


@router.post("/{job_id}/submit", response_model=OverlayResponse, dependencies=[GATED])
def submit_overlay_job(job_id: str, db: Session = Depends(get_db)) -> OverlayJob:
    """created/failed/done -> queued, and enqueue the burn."""
    job = _get_job_or_404(job_id, db)
    if job.source_type == "run":
        if job.status == OverlayStatus.created:
            _lock_runs_for_result_admission(db, [job.source_run_id])
        else:
            _admit_retained_run_source(db, [job.source_run_id])
    if job.status not in _RESUBMITTABLE:
        raise HTTPException(
            status_code=409,
            detail=f"Cannot submit overlay job in status {job.status!r}",
        )
    if not job.overlays:
        raise HTTPException(status_code=409, detail="no overlays to burn")
    try:
        transition(job, OverlayStatus.queued)
    except InvalidTransition as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    db.commit()
    enqueue_process_overlay(job_id)
    log.info("Submitted overlay job %s", job_id)
    return job


@router.post("/{job_id}/retry", response_model=OverlayResponse, dependencies=[GATED])
def retry_overlay_job(job_id: str, db: Session = Depends(get_db)) -> OverlayJob:
    """failed/done -> queued, and re-enqueue."""
    return submit_overlay_job(job_id, db)


def _clone_for_rerender(job: OverlayJob, overlays: list[dict], db: Session) -> OverlayJob:
    """Fork ``job`` into a fresh OverlayJob so a re-render is a separate record.

    Copies the source reference and the uploaded PNG assets onto disk for the
    new job (self-contained), stamps it with the edited ``overlays``, and leaves
    the original job — and its prior rendered result — untouched. The result:
    unlimited renders per source run, each persisted as its own DB row.
    """
    new_id = str(uuid.uuid4())

    # Copy the PNG assets so the clone's asset_path values (assets/card_NN.png)
    # resolve inside its own dir.
    src_assets = overlay_assets_dir(job.id)
    dst_assets = overlay_assets_dir(new_id)
    if os.path.isdir(src_assets):
        for name in os.listdir(src_assets):
            if name.lower().endswith(".png"):
                shutil.copy2(os.path.join(src_assets, name), os.path.join(dst_assets, name))

    # For upload sources the original video lives in the job dir — copy it so the
    # clone is independent of the original's lifecycle.
    source_local_path = job.source_local_path
    if (
        job.source_type == "upload"
        and job.source_local_path
        and os.path.exists(job.source_local_path)
    ):
        dst = os.path.join(overlay_job_dir(new_id), os.path.basename(job.source_local_path))
        shutil.copy2(job.source_local_path, dst)
        source_local_path = dst

    clone = OverlayJob(
        id=new_id,
        status=OverlayStatus.created,
        source_type=job.source_type,
        source_run_id=job.source_run_id,
        source_tail_edit_id=job.source_tail_edit_id,
        source_caption_id=job.source_caption_id,
        parent_run_id=job.parent_run_id,
        source_local_path=source_local_path,
        src_width=job.src_width,
        src_height=job.src_height,
        src_duration_sec=job.src_duration_sec,
        overlays=overlays,
        deliver_to_drive=job.deliver_to_drive,
    )
    db.add(clone)
    return clone


@router.post("/{job_id}/render", response_model=OverlayCreateResponse, dependencies=[GATED])
def render_overlay_job(
    job_id: str, body: OverlaysUpdateRequest, db: Session = Depends(get_db)
) -> OverlayCreateResponse:
    """Save the given placements and render.

    First render of a job renders in place. Re-rendering a job that already has
    a result forks a new OverlayJob (copying the source + assets) and renders
    that instead, so each render is persisted as its own record and the prior
    result is preserved. Returns the id/status of the job that was queued (a new
    id on a re-render).
    """
    job = _get_job_or_404(job_id, db)
    if job.source_type == "run":
        if job.status == OverlayStatus.created and not job.result_local_path:
            _lock_runs_for_result_admission(db, [job.source_run_id])
        else:
            _admit_retained_run_source(db, [job.source_run_id])

    # Validate + snapshot the placements against the editing job's assets.
    specs: list[dict] = []
    for item in body.overlays:
        if resolve_overlay_asset(job_id, item.asset_path) is None:
            raise HTTPException(
                status_code=400, detail=f"asset not found: {item.asset_path}"
            )
        specs.append(item.model_dump(exclude_none=False))
    if not specs:
        raise HTTPException(status_code=409, detail="no overlays to burn")

    # A prior materialized result means this is a re-render → fork a new record.
    has_prior_render = bool(job.result_local_path) or job.status == OverlayStatus.done
    if has_prior_render:
        target = _clone_for_rerender(job, specs, db)
    else:
        if job.status not in _RESUBMITTABLE:
            raise HTTPException(
                status_code=409,
                detail=f"Cannot render overlay job in status {job.status!r}",
            )
        job.overlays = specs
        target = job

    try:
        transition(target, OverlayStatus.queued)
    except InvalidTransition as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    db.commit()
    enqueue_process_overlay(target.id)
    log.info(
        "Rendered overlay job %s (target=%s, rerender=%s)",
        job_id, target.id, has_prior_render,
    )
    return OverlayCreateResponse(id=target.id, status=OverlayStatus.queued.value)


@router.get("/{job_id}", response_model=OverlayResponse)
def get_overlay_job(job_id: str, db: Session = Depends(get_db)) -> OverlayJob:
    return _get_job_or_404(job_id, db)


@router.get("/{job_id}/result")
def get_overlay_result_info(job_id: str, db: Session = Depends(get_db)) -> dict:
    """Result metadata (local path, Drive id) without downloading."""
    job = _get_job_or_404(job_id, db)
    status_val = job.status.value if hasattr(job.status, "value") else str(job.status)
    return {
        "id": job.id,
        "status": status_val,
        "result_local_path": job.result_local_path,
        "result_available": bool(
            job.result_local_path and os.path.exists(job.result_local_path)
        ),
        "gdrive_file_id": job.gdrive_file_id,
    }


@router.get("/{job_id}/download")
def download_overlay_result(job_id: str):
    """Stream the burned-in result video.

    Fix #5: when NGINX_ACCEL_ENABLED, returns X-Accel-Redirect so nginx
    serves the bytes directly instead of streaming through Python.
    Fix #6: DB session is scoped to a plain `with` block so it's committed
    and closed before the response body is sent, freeing the pool connection
    immediately.
    Fix #1 (cache): replaced no-store with a private cacheable policy.
    """
    with get_session() as db:
        job = _get_job_or_404(job_id, db)
        path = job.result_local_path
        if not path or not os.path.exists(path):
            raise HTTPException(status_code=404, detail="Result file not found on disk")

        filename = f"overlay_{job_id[:8]}.mp4"
        gdrive_id = job.gdrive_file_id or ""

    return accel_or_file_response(
        path,
        media_type="video/mp4",
        filename=filename,
        extra_headers={
            "X-GDrive-File-Id": gdrive_id,
            "Cache-Control": "private, max-age=31536000, immutable",
        },
    )


@router.delete("/{job_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_overlay_job(job_id: str, db: Session = Depends(get_db)) -> Response:
    """Soft-delete an overlay job: hide it and free its render files on disk,
    but KEEP the DB row so the source video reference stays findable later.

    Removes only the rendered result + uploaded PNG assets from disk; the
    source upload (if any) and the row itself are preserved. Blocked (409)
    while the job is active (queued/processing/delivering).
    """
    job = _get_job_or_404(job_id, db)
    if job.status in _ACTIVE_OVERLAY_STATUSES:
        status_val = job.status.value if hasattr(job.status, "value") else str(job.status)
        raise HTTPException(
            status_code=409, detail=f"Cannot delete an overlay job while it is {status_val!r}"
        )

    job.deleted_at = datetime.now(timezone.utc)
    # The rendered result on disk is gone — reflect that in the row.
    job.result_local_path = None
    db.commit()

    # Free space: drop the burned result + uploaded PNGs, keep the source video.
    job_root = overlay_job_dir(job_id)
    shutil.rmtree(os.path.join(job_root, "output"), ignore_errors=True)
    shutil.rmtree(os.path.join(job_root, "assets"), ignore_errors=True)
    log.info("Soft-deleted overlay job %s (row kept; render files removed)", job_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post("/{job_id}/export-to-sheet", dependencies=[GATED])
def export_overlay_to_sheet(job_id: str, db: Session = Depends(get_db)) -> dict:
    """Append a row for this overlay result to the "Final Creatives" sheet.

    Mirrors the run export: idempotent via ``final_creatives_exported_at`` and a
    dedupe against the sheet. Job ID is the PARENT RUN's id (an overlay is that
    run's creative re-dressed; ``Pipeline Type`` tells the two rows apart) —
    built in the RQ task, which is where the parent is resolved, see
    ``overlay_runs.sheet_job_id``. Creative Link is the overlay's Drive result.
    """
    job = _get_job_or_404(job_id, db)
    # Row lock for the checks below: pairs with the one
    # manual_creatives.register_overlay takes, so a registration and this
    # export can't both pass their "is the other one done?" check.
    db.refresh(job, with_for_update=True)

    if job.final_creatives_exported_at is not None:
        return {"status": "exists", "message": "This overlay is already in Final Creatives."}

    status_val = job.status.value if hasattr(job.status, "value") else str(job.status)
    if status_val != "done":
        raise HTTPException(
            status_code=409,
            detail=f"Cannot export: overlay status is {status_val!r}, expected 'done'.",
        )
    if not job.gdrive_file_id:
        raise HTTPException(
            status_code=409,
            detail="Cannot export: overlay has no Google Drive result link yet.",
        )
    # A script-made creative registered in manual_creatives goes to the sheet
    # under its own Job ID (app/manual_creatives.py); exporting it here too
    # would put the same Drive file in Final Creatives twice.
    manual_id = db.execute(
        select(ManualCreative.id).where(ManualCreative.overlay_job_id == job.id)
    ).scalar_one_or_none()
    if manual_id is not None:
        raise HTTPException(
            status_code=409,
            detail=(
                f"Cannot export: this overlay is registered as manual creative "
                f"{manual_id[:8]}; export it with app.manual_creatives instead."
            ),
        )

    # Mark the job exported optimistically so the fast-path check and UI state
    # work correctly on page reload, and so a duplicate click is rejected before
    # the job even runs.  The actual Sheets I/O (3 round-trips) runs in an RQ
    # job on the overlay queue so it does not block the request.
    job.final_creatives_exported_at = datetime.now(timezone.utc)
    db.commit()

    enqueue_export_or_rollback(
        enqueue_export_overlay_to_sheet, job_id, job,
        "final_creatives_exported_at", db,
    )
    log.info("Enqueued Final Creatives export for overlay %s", job_id)
    return {"status": "added", "message": "Added to Final Creatives."}
