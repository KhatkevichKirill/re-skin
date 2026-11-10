"""Legacy admission and delivery contracts for assistant Run publication.

Isolated SQLite/files only; no queues, providers, production or model calls.
The publication service's separate PostgreSQL tests prove lock concurrency.
"""
from __future__ import annotations

import os
import tempfile
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")
os.environ.setdefault("DATA_DIR", tempfile.mkdtemp(prefix="run-result-tests-"))

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event, select, update
from sqlalchemy.orm import sessionmaker

from app import api_overlay, api_v2, public, storage, web_v2
from app.assistant_models import AssistantRunResultIntent, AssistantRunResultPromotion
from app.assistant_store import AssistantArtifactStorage, get_version_artifact, import_run_version
from app.config import settings
from app.db import Base, get_db
from app.models import InsertMerge, OverlayJob, Run, RunCaption, RunTailEdit, VideoProject


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    monkeypatch.setattr(storage, "_BASE", str(data))
    monkeypatch.setattr(settings, "NGINX_ACCEL_ENABLED", False)
    engine = create_engine(f"sqlite:///{tmp_path / 'isolated.db'}", connect_args={"check_same_thread": False})

    @event.listens_for(engine, "connect")
    def pragmas(conn, _):
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA journal_mode=WAL")

    Base.metadata.create_all(engine)
    sessions = sessionmaker(engine, expire_on_commit=False, autoflush=False)
    with sessions.begin() as db:
        project = VideoProject(source_type="upload", source_ref="fixture.mp4", status="ready",
                               width=640, height=360, duration_sec=20)
        db.add(project)
        db.flush()
        source = data / "original.mp4"
        source.write_bytes(b"original-run-video")
        project.source_local_path = str(source)
        run = Run(project_id=project.id, status="done", prompt="fixture",
                  result_local_path=str(source), result_duration_sec=20)
        db.add(run)
        db.flush()
        version = import_run_version(db, run, actor="operator", storage=AssistantArtifactStorage(data))
        ids = dict(run=run.id, project=project.id, version=version.id, conversation=version.conversation_id,
                   artifact=get_version_artifact(db, version.id, role="full").id)

    calls = []
    for module, names in ((api_v2, ("enqueue_process_run", "enqueue_process_tail_edit", "enqueue_process_caption", "enqueue_process_merge")),
                          (api_overlay, ("enqueue_process_overlay",))):
        for name in names:
            monkeypatch.setattr(module, name, lambda *args, _name=name: calls.append((_name, args)))
    monkeypatch.setattr(api_v2, "_spend_reserve", lambda *args, **kwargs: None)
    monkeypatch.setattr(api_v2, "_budget_precheck", lambda *args, **kwargs: None)

    def database():
        with sessions() as db:
            try:
                yield db
                db.commit()
            except BaseException:
                db.rollback()
                raise

    @contextmanager
    def download_session():
        with sessions() as db:
            yield db

    monkeypatch.setattr(api_v2, "get_session", download_session)
    monkeypatch.setattr(public, "get_session", download_session)
    app = FastAPI()
    app.include_router(api_v2.router, prefix="/api/v2")
    app.include_router(api_overlay.router, prefix="/api/v2/overlays")
    app.include_router(public.router, prefix="/public")
    app.dependency_overrides[get_db] = database
    client = TestClient(app)
    yield SimpleNamespace(sessions=sessions, ids=ids, data=data, source=source, calls=calls, client=client)
    client.close()
    engine.dispose()


def fence(fixture, *, receipt=False, run_id=None):
    ids = fixture.ids
    now = datetime.now(timezone.utc)
    with fixture.sessions.begin() as db:
        intent = AssistantRunResultIntent(id=str(uuid.uuid4()), run_id=run_id or ids["run"],
            conversation_id=ids["conversation"], version_id=ids["version"], artifact_id=ids["artifact"],
            idempotency_key=str(uuid.uuid4()), request_sha256="a" * 64, actor="operator",
            request_json={}, before_json={}, after_json={}, completed_at=now if receipt else None)
        db.add(intent)
        db.flush()
        if receipt:
            db.add(AssistantRunResultPromotion(id=intent.id, run_id=intent.run_id,
                conversation_id=intent.conversation_id, version_id=intent.version_id,
                artifact_id=intent.artifact_id, idempotency_key=intent.idempotency_key,
                request_sha256=intent.request_sha256, actor="operator", before_json={},
                after_json={}, delivery="local_only", noop=False))


def retained_jobs(fixture):
    with fixture.sessions.begin() as db:
        tail = RunTailEdit(run_id=fixture.ids["run"], source_type="run", cut_sec=10, status="done", output_path=str(fixture.source))
        caption = RunCaption(run_id=fixture.ids["run"], source_type="run", status="done", output_path=str(fixture.source))
        overlay = OverlayJob(source_type="run", source_run_id=fixture.ids["run"], status="done",
                             overlays=[{"asset_path": "assets/card.png"}], result_local_path=str(fixture.source))
        merge = InsertMerge(child_run_id=fixture.ids["run"], target_project_id=fixture.ids["project"],
                            x=0, y=0, width=20, height=20, start_sec=0, end_sec=10, status="failed")
        db.add_all([tail, caption, overlay, merge])
        db.flush()
        return dict(tail=tail.id, caption=caption.id, overlay=overlay.id, merge=merge.id)


@pytest.mark.parametrize("route", ["delete_run", "delete_project", "rerun", "range", "retry", "audio", "tail", "caption", "overlay", "merge"])
def test_unresolved_intent_blocks_legacy_admission_and_deletion(isolated, route):
    fence(isolated)
    rid, pid = isolated.ids["run"], isolated.ids["project"]
    requests = {
        "delete_run": ("delete", f"/api/v2/runs/{rid}", {}),
        "delete_project": ("delete", f"/api/v2/projects/{pid}", {}),
        "rerun": ("post", f"/api/v2/runs/{rid}/segments/{uuid.uuid4()}/rerun", {}),
        "range": ("post", f"/api/v2/runs/{rid}/segments/{uuid.uuid4()}/range-rerun", {"data": {"start_sec": 1, "end_sec": 3}}),
        "retry": ("post", f"/api/v2/runs/{rid}/retry", {}),
        "audio": ("post", f"/api/v2/runs/{rid}/audio-patches", {"data": {"donor_run_id": rid, "target_start_sec": 1, "target_end_sec": 2, "donor_start_sec": 1}}),
        "tail": ("post", f"/api/v2/runs/{rid}/tail-edits", {"data": {"cut_sec": 10}}),
        "caption": ("post", f"/api/v2/runs/{rid}/captions", {"data": {"source_tail_edit_id": ""}}),
        "overlay": ("post", "/api/v2/overlays", {"json": {"source_type": "run", "source_run_id": rid}}),
        "merge": ("post", f"/api/v2/runs/{rid}/merge", {"data": {"target_project_id": pid}}),
    }
    method, url, kwargs = requests[route]
    response = getattr(isolated.client, method)(url, **kwargs)
    assert response.status_code == 409, response.text
    assert "reconcil" in response.json()["detail"]
    assert isolated.source.read_bytes() == b"original-run-video"
    assert isolated.calls == []


@pytest.mark.parametrize("receipt", [False, True])
@pytest.mark.parametrize("kind", ["tail", "caption", "overlay", "overlay_render", "merge"])
def test_retained_direct_jobs_cannot_restart_across_publication(isolated, receipt, kind):
    jobs = retained_jobs(isolated)
    fence(isolated, receipt=receipt)
    if kind == "overlay_render":
        url = f"/api/v2/overlays/{jobs['overlay']}/render"
        response = isolated.client.post(url, json={"overlays": []})
    else:
        prefix = {"tail": "tail-edits", "caption": "captions", "overlay": "overlays", "merge": "merges"}[kind]
        response = isolated.client.post(f"/api/v2/{prefix}/{jobs[kind]}/retry")
    assert response.status_code == 409, response.text
    assert isolated.calls == []
    assert isolated.source.read_bytes() == b"original-run-video"


@pytest.mark.parametrize("route", ["rerun", "range", "retry"])
def test_receipt_permanently_blocks_segment_reconstruction_after_other_pointer_changes(isolated, route):
    fence(isolated, receipt=True)
    # A later audio repair can move the pointer; receipt provenance still wins.
    with isolated.sessions.begin() as db:
        db.get(Run, isolated.ids["run"]).result_local_path = str(isolated.data / "audio-repair.mp4")
    path = f"/api/v2/runs/{isolated.ids['run']}"
    suffix = {"rerun": f"/segments/{uuid.uuid4()}/rerun", "range": f"/segments/{uuid.uuid4()}/range-rerun", "retry": "/retry"}[route]
    response = isolated.client.post(path + suffix, data={"start_sec": 1, "end_sec": 3})
    assert response.status_code == 409
    assert "copy" in response.json()["detail"]
    assert isolated.calls == []


def test_fresh_direct_jobs_and_created_overlay_first_submit_remain_available(isolated):
    fence(isolated, receipt=True)
    rid = isolated.ids["run"]
    assert isolated.client.post(f"/api/v2/runs/{rid}/tail-edits", data={"cut_sec": 10}).status_code == 201
    assert isolated.client.post(f"/api/v2/runs/{rid}/captions", data={"source_tail_edit_id": ""}).status_code == 201
    response = isolated.client.post("/api/v2/overlays", json={"source_type": "run", "source_run_id": rid})
    assert response.status_code == 201, response.text
    oid = response.json()["id"]
    with isolated.sessions.begin() as db:
        db.get(OverlayJob, oid).overlays = [{"asset_path": "assets/card.png"}]
    assert isolated.client.post(f"/api/v2/overlays/{oid}/submit").status_code == 200


def test_completed_derived_source_retry_does_not_follow_mutable_run(isolated):
    jobs = retained_jobs(isolated)
    fence(isolated, receipt=True)
    with isolated.sessions.begin() as db:
        caption = db.get(RunCaption, jobs["caption"])
        caption.source_type = "tail_edit"
        caption.source_tail_edit_id = jobs["tail"]
    response = isolated.client.post(f"/api/v2/captions/{jobs['caption']}/retry")
    assert response.status_code == 200, response.text
    assert isolated.calls[0][0] == "enqueue_process_caption"


def test_pip_admission_locks_parent_and_child_in_sorted_order(isolated, monkeypatch):
    with isolated.sessions.begin() as db:
        parent = Run(project_id=isolated.ids["project"], prompt="parent", status="done")
        db.add(parent)
        db.flush()
        db.get(VideoProject, isolated.ids["project"]).source_run_id = parent.id
        parent_id = parent.id
    seen = []
    original = api_v2._lock_run_for_result_mutation

    def record(run, db):
        seen.append(run.id)
        return original(run, db)

    monkeypatch.setattr(api_v2, "_lock_run_for_result_mutation", record)
    response = isolated.client.post(f"/api/v2/runs/{isolated.ids['run']}/merge", data={
        "target_project_id": isolated.ids["project"], "x": 0, "y": 0, "width": 20,
        "height": 20, "start_sec": 0, "end_sec": 10})
    assert response.status_code == 201, response.text
    assert seen == sorted([parent_id, isolated.ids["run"]])


def test_pip_parent_intent_blocks_creation_and_retained_retry(isolated):
    jobs = retained_jobs(isolated)
    with isolated.sessions.begin() as db:
        parent = Run(project_id=isolated.ids["project"], prompt="parent", status="done")
        db.add(parent)
        db.flush()
        db.get(VideoProject, isolated.ids["project"]).source_run_id = parent.id
        parent_id = parent.id
    fence(isolated, run_id=parent_id)
    for url, data in ((f"/api/v2/runs/{isolated.ids['run']}/merge", {"target_project_id": isolated.ids["project"]}),
                      (f"/api/v2/merges/{jobs['merge']}/retry", {})):
        response = isolated.client.post(url, data=data)
        assert response.status_code == 409, response.text
    assert isolated.calls == []


def test_result_cache_identity_distinguishes_same_size_same_mtime_paths(isolated):
    other = isolated.data / "published.mp4"
    other.write_bytes(b"different-run-data")
    stat = isolated.source.stat()
    os.utime(other, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    assert other.stat().st_size == stat.st_size
    run = SimpleNamespace(result_local_path=str(isolated.source))
    before = web_v2._result_version_for_run(run)
    run.result_local_path = str(other)
    assert web_v2._result_version_for_run(run) != before


def test_mutable_result_routes_revalidate_and_public_accepts_revision(isolated):
    rid = isolated.ids["run"]
    paths = [f"/api/v2/runs/{rid}/result?v=new", f"/public/runs/{rid}/result?token={public.make_result_token(rid)}&v=new"]
    for path in paths:
        response = isolated.client.get(path)
        assert response.status_code == 200
        assert response.content == isolated.source.read_bytes()
        assert response.headers["cache-control"] == "private, no-cache"


@pytest.mark.parametrize("consumer", ["web", "tail_create"])
def test_duration_backfill_cannot_overwrite_a_result_published_during_probe(isolated, monkeypatch, consumer):
    rid = isolated.ids["run"]
    other = isolated.data / "new-main.mp4"
    other.write_bytes(b"new-main-video")
    with isolated.sessions.begin() as db:
        db.get(Run, rid).result_duration_sec = None

    def concurrent_publication(path):
        assert path == str(isolated.source)
        with isolated.sessions.begin() as db:
            db.execute(update(Run).where(Run.id == rid).values(result_local_path=str(other), result_duration_sec=8))
        return SimpleNamespace(duration_sec=20)

    monkeypatch.setattr(api_v2.media_mod, "probe", concurrent_publication)
    if consumer == "web":
        with isolated.sessions() as db:
            context = web_v2._get_tail_edit_context(db.get(Run, rid), db)
            assert context["tail_edit_result_duration_sec"] == 8
    else:
        response = isolated.client.post(f"/api/v2/runs/{rid}/tail-edits", data={"cut_sec": 10})
        assert response.status_code == 409, response.text
        assert isolated.calls == []
    with isolated.sessions() as db:
        run = db.get(Run, rid)
        assert (run.result_local_path, run.result_duration_sec) == (str(other), 8)
