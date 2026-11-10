"""Operator feedback API + Run-page Outcome card on PostgreSQL (AICP-299).

Design: docs/knowledge-feedback-design.md §4.5, §6.2, §6.6, §7;
docs/knowledge-capture-hooks.md §8 and §10. Opt-in like test_knowledge_pg.py:
RESKIN_TEST_PG_URL must name a DISPOSABLE server (this creates and drops a
database).
"""
from __future__ import annotations

import hashlib
import io
import os
import re
import sys
import tempfile
import uuid

import pytest

_PG_URL = os.environ.get("RESKIN_TEST_PG_URL")
pytestmark = pytest.mark.skipif(not _PG_URL, reason="RESKIN_TEST_PG_URL not set (needs a disposable PostgreSQL)")

os.environ.setdefault("DATA_DIR", tempfile.mkdtemp(prefix="reskin_kb_fb_"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

BACKEND = os.path.join(os.path.dirname(__file__), "..")
MANIFEST = os.path.join(BACKEND, "..", "docs", "knowledge", "holdout-manifest-v1.json")
MANIFEST_SHA = "ea5f5d52ce1d5b5d8485fb5d29116090d9cf337113ca35405f843bf194d2766f"


@pytest.fixture(scope="module")
def engine():
    from sqlalchemy import create_engine, text
    from sqlalchemy.engine import make_url

    import app.models  # noqa: F401
    from app.db import Base

    name = "kbfb_" + uuid.uuid4().hex[:12]
    admin = create_engine(_PG_URL, isolation_level="AUTOCOMMIT")
    with admin.connect() as c:
        c.execute(text(f"CREATE DATABASE {name}"))
    eng = create_engine(make_url(_PG_URL).set(database=name).render_as_string(hide_password=False), pool_size=20)
    Base.metadata.create_all(eng)
    yield eng
    eng.dispose()
    with admin.connect() as c:
        c.execute(text(f"DROP DATABASE {name} WITH (FORCE)"))
    admin.dispose()


@pytest.fixture(scope="module")
def Session(engine):
    from sqlalchemy.orm import sessionmaker

    from app.knowledge import store

    factory = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
    s = factory()
    store.load_eval_manifest(s, open(MANIFEST, "rb").read(), expected_sha256=MANIFEST_SHA)
    s.commit()
    s.close()
    return factory


@pytest.fixture()
def s(Session):
    session = Session()
    yield session
    session.rollback()
    session.close()


@pytest.fixture()
def flags(monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "KNOWLEDGE_ENABLED", True, raising=False)
    monkeypatch.setattr(settings, "KNOWLEDGE_FEEDBACK_UI", True, raising=False)
    monkeypatch.setattr(settings, "KNOWLEDGE_PROJECT_IDS", "*", raising=False)
    monkeypatch.setattr(settings, "ASSISTANT_ALL_PROJECTS", True, raising=False)
    return settings


@pytest.fixture()
def client(Session, flags):
    from fastapi.testclient import TestClient

    from app.db import get_db
    from app.main import app

    def override():
        session = Session()
        try:
            yield session
        finally:
            session.rollback()
            session.close()

    app.dependency_overrides[get_db] = override
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c
    app.dependency_overrides.pop(get_db, None)


# ---------------------------------------------------------------- helpers


def project(s):
    from app.models import VideoProject

    p = VideoProject(id=str(uuid.uuid4()), source_type="upload", source_ref="fixture", project_type="face_swap")
    s.add(p)
    s.commit()
    return p.id


def done_run(s, tmp_path, content=b"final-v1", status="done", pid=None):
    from app.models import Run
    from app.state_machine import RunStatus

    pid = pid or project(s)
    path = tmp_path / f"{uuid.uuid4().hex}.mp4"
    path.write_bytes(content)
    r = Run(project_id=pid, result_local_path=str(path), status=RunStatus(status))
    s.add(r)
    s.commit()
    return pid, r.id, str(path)


def set_status(s, rid, status):
    from app.models import Run
    from app.state_machine import RunStatus

    s.get(Run, rid).status = RunStatus(status)
    s.commit()


def started(client, s, tmp_path, **kw):
    pid, rid, path = done_run(s, tmp_path, **kw)
    r = client.post(f"/api/v2/knowledge/runs/{rid}/start")
    assert r.status_code == 200, r.text
    return pid, rid, path, r.json()["case_id"]


def key():
    return uuid.uuid4().hex


def fb(client, cid, **body):
    body.setdefault("idempotency_key", key())
    return client.post(f"/api/v2/knowledge/cases/{cid}/feedback", json=body)


def state(Session, cid):
    from app.knowledge import store

    with Session() as x:
        return store.case_state(x, cid)


def events(Session, cid):
    from sqlalchemy import select

    from app.knowledge.models import KbFeedbackEvent

    with Session() as x:
        return x.execute(select(KbFeedbackEvent).where(KbFeedbackEvent.case_id == cid)
                         .order_by(KbFeedbackEvent.seq)).scalars().all()


def version_count(Session, cid):
    from app.knowledge import store

    with Session() as x:
        return len(store.versions(x, cid))


# ---------------------------------------------------------------- flags / scope


def test_flag_off_404s_every_endpoint(client, s, Session, tmp_path, flags, monkeypatch):
    pid, rid, path, cid = started(client, s, tmp_path)
    monkeypatch.setattr(flags, "KNOWLEDGE_ENABLED", False)
    assert fb(client, cid, expected_version_no=1, verdict="accepted_as_is").status_code == 404
    assert client.post(f"/api/v2/knowledge/runs/{rid}/start").status_code == 404
    assert client.get(f"/v2/runs/{rid}/knowledge-fragment").status_code == 404
    monkeypatch.setattr(flags, "KNOWLEDGE_ENABLED", True)
    monkeypatch.setattr(flags, "KNOWLEDGE_FEEDBACK_UI", False)
    assert client.get(f"/v2/runs/{rid}/knowledge-fragment").status_code == 404
    assert client.post(f"/v2/runs/{rid}/knowledge/start").status_code == 404
    assert state(Session, cid)["verdict"] == "unreviewed"


def test_out_of_scope_is_404_everywhere(client, s, tmp_path, flags, monkeypatch):
    pid, rid, path, cid = started(client, s, tmp_path)
    other = project(s)
    monkeypatch.setattr(flags, "KNOWLEDGE_PROJECT_IDS", other)
    assert fb(client, cid, expected_version_no=1, verdict="accepted_as_is").status_code == 404
    assert client.post(f"/api/v2/knowledge/runs/{rid}/start").status_code == 404
    assert client.get(f"/v2/runs/{rid}/knowledge-fragment").status_code == 404
    assert client.post(f"/api/v2/knowledge/runs/{rid}/external-repair",
                       json={"idempotency_key": key(), "drive_link": "1AbCdEfGhIjKlMnOp"}).status_code == 404
    assert client.post(f"/api/v2/knowledge/cases/{cid}/corrections",
                       json={"idempotency_key": key(), "before_artifact_id": "x", "method": "manual"}
                       ).status_code == 404
    assert client.post(f"/api/v2/knowledge/events/{uuid.uuid4()}/retract",
                       json={"idempotency_key": key(), "reason": "x"}).status_code == 404
    assert client.post(f"/v2/knowledge/cases/{cid}/feedback",
                       data={"idempotency_key": key(), "expected_version_no": "1",
                             "verdict": "accepted_as_is"}).status_code == 404


# ---------------------------------------------------------------- start bootstrap


def test_start_requires_a_done_run(client, s, Session, tmp_path):
    from app.knowledge import store

    pid, rid, path = done_run(s, tmp_path, status="stitching")
    r = client.post(f"/api/v2/knowledge/runs/{rid}/start")
    assert r.status_code == 409 and r.json()["detail"]["kind"] == "not_delivered"
    with Session() as x:
        assert store.current_case(x, store.run_case_key(rid)) is None  # nothing created


def test_start_creates_the_case_and_version_once(client, s, Session, tmp_path):
    from app.knowledge import store

    pid, rid, path = done_run(s, tmp_path, content=b"delivered bytes")
    first = client.post(f"/api/v2/knowledge/runs/{rid}/start").json()
    again = client.post(f"/api/v2/knowledge/runs/{rid}/start").json()
    assert first == again and first["version_no"] == 1
    with Session() as x:
        vs = store.versions(x, first["case_id"])
        assert len(vs) == 1
        art = x.get(store.KbArtifact, vs[0].artifact_id)
        assert art.sha256 == hashlib.sha256(b"delivered bytes").hexdigest()
        assert art.locator == path and art.locator_kind == "local_path"


def test_start_without_a_file_on_disk_is_409(client, s, tmp_path):
    pid, rid, path = done_run(s, tmp_path)
    os.unlink(path)
    r = client.post(f"/api/v2/knowledge/runs/{rid}/start")
    assert r.status_code == 409 and "on disk" in r.json()["detail"]["detail"]


def test_start_uses_the_prehash_outside_the_lock(client, s, tmp_path, monkeypatch):
    """The sha is computed before the transaction; under the lock the memo hasher reuses it."""
    from app.knowledge import feedback, store

    calls = []
    real = store.sha256_file
    monkeypatch.setattr(store, "sha256_file", lambda p: calls.append(p) or real(p))
    pid, rid, path = done_run(s, tmp_path)
    in_lock = []
    real_memo = feedback.memo_hasher

    def spy(pre):
        inner = real_memo(pre)
        return lambda p: in_lock.append(len(calls)) or inner(p)

    monkeypatch.setattr(feedback, "memo_hasher", spy)
    assert client.post(f"/api/v2/knowledge/runs/{rid}/start").status_code == 200
    assert calls == [path] and in_lock == [1]  # one hash, done before the locked sync ran


def test_start_passes_the_live_link_lookup_hook(client, s, tmp_path, monkeypatch):
    from app.knowledge import feedback, store

    seen = {}
    real = store.sync_run_case_version

    def sync(session, case, **kw):
        seen.update(kw)
        kw.pop("live_link_lookup", None)
        return real(session, case, **kw)

    monkeypatch.setattr(store, "sync_run_case_version", sync)
    monkeypatch.setattr(feedback, "live_link_lookup", lambda run: None)
    pid, rid, path = done_run(s, tmp_path)
    assert client.post(f"/api/v2/knowledge/runs/{rid}/start").status_code == 200
    assert seen["live_link_lookup"] is feedback.live_link_lookup


# ---------------------------------------------------------------- feedback


def test_accept_repair_reject(client, s, Session, tmp_path):
    for verdict in ("accepted_as_is", "needs_repair", "rejected"):
        pid, rid, path, cid = started(client, s, tmp_path)
        body = {"expected_version_no": 1, "verdict": verdict}
        if verdict == "needs_repair":
            body["defects"] = [{"code": "lip_sync", "severity": "major", "start": 1.0, "end": 2.5},
                               {"code": "other", "severity": "minor", "note": "logo flicker"}]
        r = fb(client, cid, **body)
        assert r.status_code == 200, r.text
        out = r.json()
        assert out["state"]["verdict"] == verdict and out["replayed"] is False
        assert len(out["event_ids"]) == 1 + len(body.get("defects", []))
    evs = events(Session, cid)
    assert [e.event_type for e in evs] == ["operator_verdict"] and evs[0].actor_kind == "operator"


def test_defect_validation_and_contradiction(client, s, tmp_path):
    pid, rid, path, cid = started(client, s, tmp_path)
    assert fb(client, cid, expected_version_no=1, verdict="great").status_code == 422
    assert fb(client, cid, expected_version_no=1, defects=[{"code": "nope", "severity": "minor"}]).status_code == 422
    assert fb(client, cid, expected_version_no=1, defects=[{"code": "other", "severity": "minor"}]).status_code == 422
    assert fb(client, cid, expected_version_no=1).status_code == 422  # nothing to record
    assert fb(client, cid, expected_version_no="1", verdict="rejected").status_code == 422  # strict int
    assert fb(client, cid, expected_version_no=1, verdict="rejected", extra=1).status_code == 422
    assert fb(client, cid, expected_version_no=1, verdict="accepted_as_is").status_code == 200
    r = fb(client, cid, expected_version_no=1, defects=[{"code": "stitching", "severity": "blocker"}])
    assert r.status_code == 409 and r.json()["detail"]["current_version_no"] == 1


def test_stale_version_409_carries_the_current_version(client, s, Session, tmp_path):
    pid, rid, path, cid = started(client, s, tmp_path)
    with open(path, "wb") as fh:
        fh.write(b"restitched bytes")
    r = fb(client, cid, expected_version_no=1, verdict="accepted_as_is")
    assert r.status_code == 409
    assert r.json()["detail"]["current_version_no"] == 2
    # The refused request's sync rolled back with it; the next submit syncs again.
    assert version_count(Session, cid) == 1
    assert fb(client, cid, expected_version_no=2, verdict="accepted_as_is").json()["state"]["verdict"] == \
        "accepted_as_is"


def test_feedback_on_a_run_that_is_no_longer_done_is_409(client, s, Session, tmp_path):
    pid, rid, path, cid = started(client, s, tmp_path)
    set_status(s, rid, "stitching")
    r = fb(client, cid, expected_version_no=1, verdict="accepted_as_is")
    assert r.status_code == 409 and r.json()["detail"]["kind"] == "not_delivered"
    assert events(Session, cid) == []


def test_duplicate_submit_returns_the_same_receipt(client, s, Session, tmp_path):
    pid, rid, path, cid = started(client, s, tmp_path)
    k = key()
    a = fb(client, cid, idempotency_key=k, expected_version_no=1, verdict="needs_repair",
           defects=[{"code": "truncation", "severity": "major"}]).json()
    b = fb(client, cid, idempotency_key=k, expected_version_no=1, verdict="needs_repair",
           defects=[{"code": "truncation", "severity": "major"}]).json()
    assert a["request_id"] == b["request_id"] and a["event_ids"] == b["event_ids"] and b["replayed"] is True
    assert len(events(Session, cid)) == 2
    # Same key, different body: refused, never silently merged.
    r = fb(client, cid, idempotency_key=k, expected_version_no=1, verdict="rejected")
    assert r.status_code == 409


def test_reload_mints_a_new_key_and_records_a_new_judgement(client, s, Session, tmp_path):
    pid, rid, path, cid = started(client, s, tmp_path)
    keys = set()
    for verdict in ("needs_repair", "accepted_as_is"):
        html = client.get(f"/v2/runs/{rid}/knowledge-fragment").text
        k = re.search(r'name="idempotency_key" value="([0-9a-f]{32})"', html).group(1)
        assert k not in keys
        keys.add(k)
        r = client.post(f"/v2/knowledge/cases/{cid}/feedback",
                        data={"idempotency_key": k, "expected_version_no": "1", "verdict": verdict})
        assert r.status_code == 200 and "Saved." in r.text
    assert state(Session, cid)["verdict"] == "accepted_as_is"
    assert len(events(Session, cid)) == 2


def test_historical_judgement_needs_a_preserved_version(client, s, Session, tmp_path):
    from app.knowledge import store

    pid, rid, path, cid = started(client, s, tmp_path)
    with open(path, "wb") as fh:
        fh.write(b"v2 bytes")
    assert client.post(f"/api/v2/knowledge/runs/{rid}/start").json()["version_no"] == 2
    # v1 was a mutable local path: it cannot be judged after the fact (S10).
    r = fb(client, cid, expected_version_no=1, historical=True, verdict="rejected")
    assert r.status_code == 422 and "not preserved" in r.text
    # A preserved historical version can be, and it never changes current state.
    with Session() as x:
        case = x.get(store.KbCase, cid)
        art = store.get_or_create_artifact(x, kind="delivered_final", locator_kind="preserved_copy",
                                           locator=f"/data/knowledge/finals/{pid}/{rid}/x.mp4",
                                           sha256=hashlib.sha256(b"old").hexdigest(), project_id=pid)
        store.lock_run_and_cases(x, rid, [cid])
        old = store.append_case_version(x, case, artifact_id=art.id, reason="restitch", historical=True)
        x.commit()
        old_no = old.version_no
    r = fb(client, cid, expected_version_no=old_no, historical=True, verdict="rejected")
    assert r.status_code == 200, r.text
    assert r.json()["state"]["verdict"] == "unreviewed"
    assert fb(client, cid, expected_version_no=2, historical=True, verdict="rejected").status_code == 422


def test_retract(client, s, Session, tmp_path):
    pid, rid, path, cid = started(client, s, tmp_path)
    eid = fb(client, cid, expected_version_no=1, verdict="accepted_as_is").json()["event_ids"][0]
    k = key()
    r = client.post(f"/api/v2/knowledge/events/{eid}/retract", json={"idempotency_key": k, "reason": "misclick"})
    assert r.status_code == 200
    assert state(Session, cid)["verdict"] == "unreviewed"
    again = client.post(f"/api/v2/knowledge/events/{eid}/retract", json={"idempotency_key": k, "reason": "misclick"})
    assert again.json()["replayed"] is True
    other = client.post(f"/api/v2/knowledge/events/{eid}/retract", json={"idempotency_key": key(), "reason": "x"})
    assert other.status_code == 409
    rid_event = r.json()["event_ids"][0]
    nested = client.post(f"/api/v2/knowledge/events/{rid_event}/retract",
                         json={"idempotency_key": key(), "reason": "undo"})
    assert nested.status_code == 422
    assert client.post(f"/api/v2/knowledge/events/{uuid.uuid4()}/retract",
                       json={"idempotency_key": key(), "reason": "x"}).status_code == 404
    assert client.post(f"/api/v2/knowledge/events/{eid}/retract",
                       json={"idempotency_key": key(), "reason": ""}).status_code == 422


def test_corrections(client, s, Session, tmp_path):
    from app.knowledge import store

    pid, rid, path, cid = started(client, s, tmp_path)
    with Session() as x:
        art = store.latest_version(x, cid).artifact_id
    r = client.post(f"/api/v2/knowledge/cases/{cid}/corrections",
                    json={"idempotency_key": key(), "before_artifact_id": art, "method": "manual",
                          "resolved": "yes", "minutes": 12.5, "note": "re-cut ending"})
    assert r.status_code == 200, r.text
    ev = events(Session, cid)[-1]
    assert (ev.event_type, ev.method, ev.resolved, ev.operator_minutes) == ("correction", "manual", "yes", 12.5)
    assert client.post(f"/api/v2/knowledge/cases/{cid}/corrections",
                       json={"idempotency_key": key(), "before_artifact_id": str(uuid.uuid4()),
                             "method": "manual"}).status_code == 404  # not this case's artifact
    assert client.post(f"/api/v2/knowledge/cases/{cid}/corrections",
                       json={"idempotency_key": key(), "before_artifact_id": art, "method": "manual",
                             "mcredits": 5}).status_code == 422  # a cost needs its source


def test_current_judging_of_a_derived_local_path_needs_its_recorded_bytes(client, s, Session, tmp_path):
    """S10 for non-Run cases: the file must still hold the recorded sha."""
    from app.knowledge import store

    pid, rid, path, cid = started(client, s, tmp_path)
    tail = tmp_path / "tail.mp4"
    tail.write_bytes(b"tail edit")
    with Session() as x:
        case = store.ensure_case(x, case_key=store.derived_case_key("run_tail_edits", str(uuid.uuid4())),
                                 scope="derived", operation="face_swap", project_id=pid, run_id=rid,
                                 parent_case_id=cid)
        art = store.get_or_create_artifact(x, kind="tail_edit", locator_kind="local_path", locator=str(tail),
                                           sha256=hashlib.sha256(b"tail edit").hexdigest(), project_id=pid)
        store.append_case_version(x, case, artifact_id=art.id, reason="initial_output")
        x.commit()
        did = case.id
    assert fb(client, did, expected_version_no=1, verdict="accepted_as_is").status_code == 200
    tail.write_bytes(b"overwritten")
    r = fb(client, did, expected_version_no=1, verdict="rejected")
    assert r.status_code == 409 and "changed on disk" in r.text


# ---------------------------------------------------------------- external repair


def test_external_repair_drive_link(client, s, Session, tmp_path):
    from app.knowledge import store

    pid, rid, path, cid = started(client, s, tmp_path)
    k = key()
    body = {"idempotency_key": k, "drive_link": "https://drive.google.com/file/d/1AbCdEfGhIjKlMnOp/view",
            "replaced_delivered": True}
    r = client.post(f"/api/v2/knowledge/runs/{rid}/external-repair", json=body)
    assert r.status_code == 201, r.text
    out = r.json()
    again = client.post(f"/api/v2/knowledge/runs/{rid}/external-repair", json=body).json()
    assert again["replayed"] is True and again["external_case_id"] == out["external_case_id"]
    with Session() as x:
        ext = x.get(store.KbCase, out["external_case_id"])
        assert (ext.scope, ext.operation, ext.parent_case_id, ext.case_key) == \
            ("external", "repair", cid, f"external:{rid}:1")
        assert ext.snapshot["replaced_delivered"] is True
        v = store.versions(x, ext.id)
        assert len(v) == 1
        art = x.get(store.KbArtifact, v[0].artifact_id)
        assert (art.locator_kind, art.locator, art.sha256) == ("drive_file", "1AbCdEfGhIjKlMnOp", None)
    ev = events(Session, cid)[-1]
    assert ev.event_type == "link_external" and ev.note is None  # no invented reason
    # A second, different repair gets the next gap-free number.
    r2 = client.post(f"/api/v2/knowledge/runs/{rid}/external-repair",
                     json={"idempotency_key": key(), "drive_link": "1ZyXwVuTsRqPoNm", "note": "editor fixed hands"})
    with Session() as x:
        assert x.get(store.KbCase, r2.json()["external_case_id"]).case_key == f"external:{rid}:2"
    assert client.post(f"/api/v2/knowledge/runs/{rid}/external-repair",
                       json={"idempotency_key": key(), "drive_link": "not a link"}).status_code == 422
    assert client.post(f"/api/v2/knowledge/runs/{rid}/external-repair",
                       json={"idempotency_key": key()}).status_code == 422


def test_external_repair_upload(client, s, Session, tmp_path, monkeypatch):
    from app import media
    from app.knowledge import feedback, store

    monkeypatch.setattr(media, "probe", lambda p: type("I", (), {"duration_sec": 12.0})())
    pid, rid, path, cid = started(client, s, tmp_path)
    payload = b"repaired video bytes"
    r = client.post(f"/api/v2/knowledge/runs/{rid}/external-repair",
                    data={"idempotency_key": key(), "note": "re-graded"},
                    files={"upload": ("fix.mp4", io.BytesIO(payload), "video/mp4")})
    assert r.status_code == 201, r.text
    sha = hashlib.sha256(payload).hexdigest()
    with Session() as x:
        v = store.versions(x, r.json()["external_case_id"])[0]
        art = x.get(store.KbArtifact, v.artifact_id)
        assert (art.locator_kind, art.sha256, art.bytes, art.duration_sec) == ("local_path", sha, len(payload), 12.0)
        assert art.locator == os.path.join(feedback.knowledge_root(), "external", pid, f"{sha}.mp4")
        assert open(art.locator, "rb").read() == payload
    # The same bytes linked again would make the receipt ambiguous.
    dup = client.post(f"/api/v2/knowledge/runs/{rid}/external-repair", data={"idempotency_key": key()},
                      files={"upload": ("again.mp4", io.BytesIO(payload), "video/mp4")})
    assert dup.status_code == 409
    bad = client.post(f"/api/v2/knowledge/runs/{rid}/external-repair", data={"idempotency_key": key()},
                      files={"upload": ("fix.txt", io.BytesIO(b"x"), "text/plain")})
    assert bad.status_code == 422
    both = client.post(f"/api/v2/knowledge/runs/{rid}/external-repair",
                       data={"idempotency_key": key(), "drive_link": "1AbCdEfGhIjKlMnOp"},
                       files={"upload": ("fix.mp4", io.BytesIO(b"y"), "video/mp4")})
    assert both.status_code == 422
    monkeypatch.setattr(feedback, "_disk_ok", lambda root, n: False)
    full = client.post(f"/api/v2/knowledge/runs/{rid}/external-repair", data={"idempotency_key": key()},
                       files={"upload": ("other.mp4", io.BytesIO(b"other"), "video/mp4")})
    assert full.status_code == 507
    monkeypatch.setattr(feedback, "_disk_ok", lambda root, n: True)
    monkeypatch.setattr(feedback, "UPLOAD_MAX_BYTES", 4)
    big = client.post(f"/api/v2/knowledge/runs/{rid}/external-repair", data={"idempotency_key": key()},
                      files={"upload": ("big.mp4", io.BytesIO(b"0123456789"), "video/mp4")})
    assert big.status_code == 413
    # A declared body over the cap is refused before multipart parsing spools anything.
    monkeypatch.setattr(feedback, "MULTIPART_OVERHEAD_BYTES", 0)
    early = client.post(f"/api/v2/knowledge/runs/{rid}/external-repair", data={"idempotency_key": key()},
                        files={"upload": ("big.mp4", io.BytesIO(b"0123456789"), "video/mp4")})
    assert early.status_code == 413
    assert not os.listdir(os.path.join(feedback.knowledge_root(), "tmp"))  # no stray .part files


# ---------------------------------------------------------------- the card (HTMX)


def test_card_states_are_read_only(client, s, Session, tmp_path):
    from app.knowledge import store

    pid, rid, path = done_run(s, tmp_path, status="processing")
    html = client.get(f"/v2/runs/{rid}/knowledge-fragment").text
    assert 'data-kb-state="not_done"' in html and "Start recording" not in html
    set_status(s, rid, "done")
    html = client.get(f"/v2/runs/{rid}/knowledge-fragment").text
    assert 'data-kb-state="start"' in html and "Start recording outcome" in html
    with Session() as x:
        assert store.current_case(x, store.run_case_key(rid)) is None  # the GET created nothing
    r = client.post(f"/v2/runs/{rid}/knowledge/start", data={"expected_status": "done"})
    assert r.status_code == 200 and 'data-kb-state="cases"' in r.text
    html = r.text
    assert "version 1</strong>" in html and "not preserved" in html
    assert hashlib.sha256(b"final-v1").hexdigest()[:12] in html
    assert " checked" not in html  # nothing preselected
    for label in ("Accept", "Needs repair", "Reject", "Add defect", "Set start", "Set end", "Link external repair"):
        assert label in html
    # The card mounts outside the status poll target and never self-polls.
    assert "hx-trigger" not in html and "run-status-content" not in html.split("<script>")[0]


def test_card_start_on_a_run_that_is_not_done_renders_the_not_delivered_state(client, s, tmp_path):
    pid, rid, path = done_run(s, tmp_path, status="stitching")
    r = client.post(f"/v2/runs/{rid}/knowledge/start", data={"expected_status": "done"})
    assert r.status_code == 409 and 'data-kb-state="not_done"' in r.text


def test_card_409_rerenders_the_current_version_with_a_fresh_key(client, s, tmp_path):
    pid, rid, path, cid = started(client, s, tmp_path)
    with open(path, "wb") as fh:
        fh.write(b"new final")
    k = key()
    r = client.post(f"/v2/knowledge/cases/{cid}/feedback",
                    data={"idempotency_key": k, "expected_version_no": "1", "verdict": "accepted_as_is",
                          "note": "looks fine"})
    assert r.status_code == 409
    assert "version 2</strong>" in r.text and "now showing version 2" in r.text
    assert 'name="expected_version_no" value="2"' in r.text
    assert k not in r.text
    assert 'value="accepted_as_is" checked' in r.text and "looks fine" in r.text  # entered values kept


def test_card_timeline_badges_markers_and_older_versions(client, s, Session, tmp_path):
    from app.knowledge import store

    pid, rid, path, cid = started(client, s, tmp_path)
    client.post(f"/v2/knowledge/cases/{cid}/feedback",
                data={"idempotency_key": key(), "expected_version_no": "1", "verdict": "needs_repair",
                      "defect_code": ["stitching", ""], "defect_severity": ["major", ""],
                      "defect_start": ["1.5", ""], "defect_end": ["3", ""], "defect_note": ["", ""],
                      "actor_label": "Operator"})
    with Session() as x:
        store.record_auto_assessment(x, case_id=cid, idempotency_key=key(), case_version_no=1,
                                     check_kind="duration_probe", claim="full_length")
        batch = store.start_import_batch(x, source="test", code_revision="t", dry_run=False)
        imported = store.record_auto_assessment(x, case_id=cid, idempotency_key=key(), case_version_no=1,
                                                check_kind="duration_probe", claim="short_result",
                                                actor_kind="importer", import_batch_id=batch.id)
        store.invalidate(x, target_kind="event", target_id=imported.ids[0], reason="bad import")
        x.commit()
    with open(path, "wb") as fh:
        fh.write(b"second final")
    client.post(f"/api/v2/knowledge/runs/{rid}/start")
    html = client.get(f"/v2/runs/{rid}/knowledge-fragment").text
    for badge in ('data-kb-actor="operator">Operator', 'data-kb-actor="system_check">Auto',
                  'data-kb-actor="importer">Imported'):
        assert badge in html
    assert "invalidated import" in html and "Operator (unverified)" in html
    assert "Defect stitching (major) 1.50s–3.00s" in html
    assert 'data-kb-older="1"' in html and "cannot be judged after the fact" in html
    assert "Judge this older version" not in html  # S10: v1 was never preserved
    assert "Retract" in html


def test_card_holdout_hides_operator_judgements(client, s, tmp_path, monkeypatch):
    from app.knowledge import store

    pid, rid, path, cid = started(client, s, tmp_path)
    fb(client, cid, expected_version_no=1, verdict="rejected", note="secret label")
    monkeypatch.setattr(store, "is_holdout", lambda session, case, holdout=None: True)
    html = client.get(f"/v2/runs/{rid}/knowledge-fragment").text
    assert "data-kb-holdout" in html
    assert "secret label" not in html and 'data-kb-actor="operator"' not in html
    assert "Hidden (evaluation holdout)" in html


def test_card_external_repair_and_retract_forms(client, s, Session, tmp_path):
    pid, rid, path, cid = started(client, s, tmp_path)
    r = client.post(f"/v2/runs/{rid}/knowledge/external-repair",
                    data={"idempotency_key": key(), "drive_link": "1AbCdEfGhIjKlMnOp", "replaced_delivered": ""})
    assert r.status_code == 200 and "External repair 1" in r.text and "Linked an external repair" in r.text
    eid = events(Session, cid)[-1].id
    r = client.post(f"/v2/knowledge/events/{eid}/retract", data={"idempotency_key": key(), "reason": "wrong file"})
    assert r.status_code == 200 and "Retracted." in r.text
    bad = client.post(f"/v2/runs/{rid}/knowledge/external-repair", data={"idempotency_key": key()})
    assert bad.status_code == 422 and "Drive link or an upload" in bad.text

