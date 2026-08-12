"""
Tests for the v2 REST API (app/api_v2.py).

Strategy
--------
- Temp SQLite DB per test session (DATABASE_URL set before importing app).
- enqueue_analyze_project and enqueue_process_run are monkeypatched to spies.
- A TestClient wraps the FastAPI app for request-level tests.
- ffmpeg is used for the frame endpoint test (generates a tiny synthetic video).
"""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import tempfile

# Must set DATABASE_URL and DATA_DIR before any app import
_db_fd, _db_path = tempfile.mkstemp(suffix=".db")
os.close(_db_fd)
os.environ["DATABASE_URL"] = f"sqlite:///{_db_path}"
_data_tmp = tempfile.mkdtemp()
os.environ["DATA_DIR"] = _data_tmp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

from app.config import settings
from app.db import Base, get_db
from app.models import Run, RunSegment, SegmentDef, VideoProject
from app.state_machine import ProjectStatus, RunStatus, SegmentStatus
import app.api_v2 as api_v2_module


# ---------------------------------------------------------------------------
# Session-scoped engine + schema creation
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def engine():
    eng = create_engine(
        f"sqlite:///{_db_path}",
        connect_args={"check_same_thread": False},
    )

    @event.listens_for(eng, "connect")
    def set_pragmas(dbapi_conn, _):
        cur = dbapi_conn.cursor()
        cur.execute("PRAGMA foreign_keys=ON")
        cur.close()

    Base.metadata.create_all(eng)
    yield eng
    Base.metadata.drop_all(eng)


@pytest.fixture(scope="session")
def SessionFactory(engine):
    return sessionmaker(
        bind=engine, autocommit=False, autoflush=False, expire_on_commit=False
    )


# ---------------------------------------------------------------------------
# Per-test DB session + TestClient
# ---------------------------------------------------------------------------


@pytest.fixture()
def db_session(SessionFactory):
    """Yield a session; rollback after the test so state doesn't leak."""
    session = SessionFactory()
    yield session
    session.rollback()
    session.close()


@pytest.fixture()
def client(engine, SessionFactory, monkeypatch):
    """TestClient with DB dependency overridden and task queues mocked."""
    import app.api_v2 as api_v2_module

    monkeypatch.setattr(api_v2_module, "enqueue_analyze_project", lambda pid: None)
    monkeypatch.setattr(api_v2_module, "enqueue_process_run", lambda rid: None)
    monkeypatch.setattr(api_v2_module, "enqueue_transcribe_project", lambda pid: None)

    from app.main import app

    def override_get_db():
        session = SessionFactory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    app.dependency_overrides[get_db] = override_get_db

    with TestClient(app, raise_server_exceptions=True) as c:
        yield c

    app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# Spy fixture: track enqueue calls
# ---------------------------------------------------------------------------


@pytest.fixture()
def enqueue_spy(monkeypatch):
    """Returns dicts that record calls to enqueue functions."""
    import app.api_v2 as api_v2_module

    calls: dict[str, list[str]] = {
        "analyze_project": [],
        "process_run": [],
        "transcribe_project": [],
    }

    monkeypatch.setattr(
        api_v2_module,
        "enqueue_analyze_project",
        lambda pid: calls["analyze_project"].append(pid),
    )
    monkeypatch.setattr(
        api_v2_module,
        "enqueue_process_run",
        lambda rid: calls["process_run"].append(rid),
    )
    monkeypatch.setattr(
        api_v2_module,
        "enqueue_transcribe_project",
        lambda pid: calls["transcribe_project"].append(pid),
    )
    return calls


@pytest.fixture()
def spy_client(engine, SessionFactory, enqueue_spy):
    """TestClient that also tracks enqueue calls."""
    from app.main import app
    from app.db import get_db

    def override_get_db():
        session = SessionFactory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    app.dependency_overrides[get_db] = override_get_db

    with TestClient(app, raise_server_exceptions=True) as c:
        yield c, enqueue_spy

    app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tiny_video_bytes() -> bytes:
    """Return a few bytes that pass as a video upload (storage test only)."""
    return b"\x00\x01\x02\x03" * 16


class _UploadStub:
    def __init__(self, data: bytes):
        self.file = io.BytesIO(data)


def test_safe_filename_strips_path_and_leading_dots():
    assert api_v2_module._safe_filename("../../evil.mp4") == "evil.mp4"
    assert api_v2_module._safe_filename("..\\..\\evil.jpg") == "evil.jpg"
    assert api_v2_module._safe_filename(".hidden") == "hidden"
    assert api_v2_module._safe_filename("...") == "upload"


def test_save_upload_streams_and_rejects_oversize(tmp_path):
    ok_path = tmp_path / "ok.bin"
    api_v2_module._save_upload(_UploadStub(b"abcd"), str(ok_path), max_bytes=4)
    assert ok_path.read_bytes() == b"abcd"

    too_large = tmp_path / "too-large.bin"
    with pytest.raises(Exception) as excinfo:
        api_v2_module._save_upload(_UploadStub(b"abcde"), str(too_large), max_bytes=4)
    assert getattr(excinfo.value, "status_code", None) == 413
    assert not too_large.exists()


def _make_project(session, **kwargs) -> VideoProject:
    defaults = dict(
        source_type="upload",
        source_ref="test.mp4",
        status=ProjectStatus.created,
    )
    defaults.update(kwargs)
    p = VideoProject(**defaults)
    session.add(p)
    session.commit()
    return p


def _make_segment_def(session, project_id: str, index: int, **kwargs) -> SegmentDef:
    defaults = dict(
        project_id=project_id,
        index=index,
        start_sec=float(index * 5),
        end_sec=float(index * 5 + 5),
        has_face=True,
        action="swap",
    )
    defaults.update(kwargs)
    s = SegmentDef(**defaults)
    session.add(s)
    session.commit()
    return s


def _make_run_segment(session, run_id: str, segment_def_id: str, index: int = 0, **kwargs) -> RunSegment:
    defaults = dict(
        run_id=run_id,
        segment_def_id=segment_def_id,
        index=index,
        status=SegmentStatus.pending,
    )
    defaults.update(kwargs)
    rs = RunSegment(**defaults)
    session.add(rs)
    session.commit()
    return rs


def _make_run(session, project_id: str, **kwargs) -> Run:
    defaults = dict(
        project_id=project_id,
        prompt="swap character",
        resolution="720p",
        status=RunStatus.created,
        reference_image_urls=[],
    )
    defaults.update(kwargs)
    r = Run(**defaults)
    session.add(r)
    session.commit()
    return r


def _make_ffmpeg_video(path: str) -> None:
    """Generate a tiny 1-second H.264 video at *path* using ffmpeg."""
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "lavfi",
            "-i", "color=c=blue:size=64x64:rate=10:duration=1",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            path,
        ],
        capture_output=True,
        check=True,
        timeout=30,
    )


# ---------------------------------------------------------------------------
# POST /api/v2/projects
# ---------------------------------------------------------------------------


class TestCreateProject:
    def test_upload_video_returns_201(self, spy_client, SessionFactory):
        client, spy = spy_client
        response = client.post(
            "/api/v2/projects",
            files={"video_file": ("clip.mp4", io.BytesIO(_tiny_video_bytes()), "video/mp4")},
        )
        assert response.status_code == 201
        body = response.json()
        assert "project_id" in body
        assert body["status"] == "created"
        # enqueue_analyze_project called exactly once with the project id
        assert spy["analyze_project"] == [body["project_id"]]

        # project row exists in DB
        session = SessionFactory()
        project = session.get(VideoProject, body["project_id"])
        session.close()
        assert project is not None
        assert project.source_type == "upload"

    def test_gdrive_link_variant(self, spy_client):
        client, spy = spy_client
        response = client.post(
            "/api/v2/projects",
            data={"gdrive_link": "https://drive.google.com/file/d/FAKE_ID/view"},
        )
        assert response.status_code == 201
        body = response.json()
        assert "project_id" in body
        assert spy["analyze_project"] == [body["project_id"]]

    def test_neither_source_is_400(self, client):
        response = client.post("/api/v2/projects")
        assert response.status_code == 400
        assert "exactly one" in response.json()["detail"].lower()

    def test_both_sources_is_400(self, client):
        response = client.post(
            "/api/v2/projects",
            data={"gdrive_link": "https://drive.google.com/file/d/X/view"},
            files={"video_file": ("clip.mp4", io.BytesIO(_tiny_video_bytes()), "video/mp4")},
        )
        assert response.status_code == 400
        assert "exactly one" in response.json()["detail"].lower()


# ---------------------------------------------------------------------------
# GET /api/v2/projects  and  GET /api/v2/projects/{id}
# ---------------------------------------------------------------------------


class TestGetProject:
    def test_get_existing_project_200(self, client, db_session):
        project = _make_project(db_session)
        response = client.get(f"/api/v2/projects/{project.id}")
        assert response.status_code == 200
        body = response.json()
        assert body["id"] == project.id
        assert body["status"] == "created"

    def test_get_missing_project_404(self, client):
        response = client.get("/api/v2/projects/does-not-exist-id")
        assert response.status_code == 404

    def test_list_projects_returns_array(self, client, db_session):
        _make_project(db_session)
        response = client.get("/api/v2/projects")
        assert response.status_code == 200
        body = response.json()
        assert isinstance(body, list)
        assert len(body) >= 1
        assert "id" in body[0]
        assert "status" in body[0]


# ---------------------------------------------------------------------------
# GET /api/v2/projects/{pid}/segments  and  PATCH
# ---------------------------------------------------------------------------


class TestProjectSegments:
    def test_get_segments_returns_list(self, client, db_session):
        project = _make_project(db_session, status=ProjectStatus.ready)
        _make_segment_def(db_session, project.id, 0)
        _make_segment_def(db_session, project.id, 1)

        response = client.get(f"/api/v2/projects/{project.id}/segments")
        assert response.status_code == 200
        segs = response.json()
        assert len(segs) == 2
        assert segs[0]["index"] < segs[1]["index"]

    def test_get_segments_404_for_missing_project(self, client):
        response = client.get("/api/v2/projects/no-such-project/segments")
        assert response.status_code == 404

    def test_patch_segments_edits_field(self, client, db_session):
        project = _make_project(db_session, status=ProjectStatus.ready)
        seg = _make_segment_def(db_session, project.id, 0, action="keep")

        response = client.patch(
            f"/api/v2/projects/{project.id}/segments",
            json={"updates": [{"id": seg.id, "action": "swap", "end_sec": 7.5}]},
        )
        assert response.status_code == 200
        segs = response.json()
        assert len(segs) == 1
        assert segs[0]["action"] == "swap"
        assert segs[0]["end_sec"] == pytest.approx(7.5)

    def test_patch_segments_renumbers_by_start_sec(self, client, db_session):
        project = _make_project(db_session, status=ProjectStatus.ready)
        s0 = _make_segment_def(db_session, project.id, 0, start_sec=0.0, end_sec=5.0)
        s1 = _make_segment_def(db_session, project.id, 1, start_sec=5.0, end_sec=10.0)

        # Move s1 to start before s0
        response = client.patch(
            f"/api/v2/projects/{project.id}/segments",
            json={"updates": [{"id": s1.id, "start_sec": -1.0, "end_sec": 4.0}]},
        )
        assert response.status_code == 200
        segs = response.json()
        assert segs[0]["id"] == s1.id
        assert segs[0]["index"] == 0
        assert segs[1]["id"] == s0.id
        assert segs[1]["index"] == 1

    def test_patch_on_non_ready_project_is_409(self, client, db_session):
        project = _make_project(db_session, status=ProjectStatus.analyzing)
        seg = _make_segment_def(db_session, project.id, 0)

        response = client.patch(
            f"/api/v2/projects/{project.id}/segments",
            json={"updates": [{"id": seg.id, "action": "swap"}]},
        )
        assert response.status_code == 409

    def test_patch_on_created_project_is_409(self, client, db_session):
        project = _make_project(db_session, status=ProjectStatus.created)
        seg = _make_segment_def(db_session, project.id, 0)

        response = client.patch(
            f"/api/v2/projects/{project.id}/segments",
            json={"updates": [{"id": seg.id, "action": "swap"}]},
        )
        assert response.status_code == 409


# ---------------------------------------------------------------------------
# TR8: linked/contiguous segment boundaries
# ---------------------------------------------------------------------------


class TestLinkedBoundaries:
    """Tests for TR8: _normalize_partition and contiguous partition invariants."""

    def _make_contiguous_project(self, db_session, duration=30.0, n=3):
        """Create a ready project with n contiguous segments covering [0, duration]."""
        project = _make_project(
            db_session,
            status=ProjectStatus.ready,
            duration_sec=duration,
        )
        step = duration / n
        segs = []
        for i in range(n):
            seg = _make_segment_def(
                db_session,
                project.id,
                index=i,
                start_sec=round(i * step, 6),
                end_sec=round((i + 1) * step, 6),
            )
            segs.append(seg)
        return project, segs

    def test_lengthen_seg0_shifts_seg1_start(self, client, db_session):
        """Lengthening seg0's end by +1.0 s → seg1.start shifts by +1.0 s,
        seg1 is 1.0 s shorter, all later boundaries unchanged, total = [0, 30]."""
        project, segs = self._make_contiguous_project(db_session, duration=30.0, n=3)
        # segs: [0..10], [10..20], [20..30]
        s0, s1, s2 = segs

        new_s0_end = 11.0  # extended by 1.0
        response = client.patch(
            f"/api/v2/projects/{project.id}/segments",
            json={"updates": [{"id": s0.id, "end_sec": new_s0_end}]},
        )
        assert response.status_code == 200, response.json()
        result = response.json()

        # Sort by index for deterministic access
        result.sort(key=lambda s: s["index"])
        r0, r1, r2 = result

        assert r0["start_sec"] == pytest.approx(0.0)
        assert r0["end_sec"] == pytest.approx(11.0)

        # seg1 start must equal seg0 new end
        assert r1["start_sec"] == pytest.approx(11.0)
        # seg1 end is unchanged (only the one boundary moved)
        assert r1["end_sec"] == pytest.approx(20.0)

        # seg2 boundaries completely unchanged
        assert r2["start_sec"] == pytest.approx(20.0)
        assert r2["end_sec"] == pytest.approx(30.0)

        # Contiguity invariant
        assert r0["end_sec"] == pytest.approx(r1["start_sec"])
        assert r1["end_sec"] == pytest.approx(r2["start_sec"])
        assert r0["start_sec"] == pytest.approx(0.0)
        assert r2["end_sec"] == pytest.approx(30.0)

        # Indices are 0..n-1
        assert [s["index"] for s in result] == [0, 1, 2]

    def test_contiguity_invariant_after_edit(self, client, db_session):
        """For any edit, seg[i].end == seg[i+1].start for all i,
        seg[0].start == 0, seg[-1].end == duration."""
        project, segs = self._make_contiguous_project(db_session, duration=20.0, n=4)
        s0, s1, s2, s3 = segs

        # Edit middle boundary
        response = client.patch(
            f"/api/v2/projects/{project.id}/segments",
            json={"updates": [{"id": s1.id, "end_sec": 8.5}]},
        )
        assert response.status_code == 200
        result = sorted(response.json(), key=lambda s: s["index"])

        assert result[0]["start_sec"] == pytest.approx(0.0)
        assert result[-1]["end_sec"] == pytest.approx(20.0)
        for i in range(len(result) - 1):
            assert result[i]["end_sec"] == pytest.approx(result[i + 1]["start_sec"]), \
                f"Contiguity broken between index {i} and {i+1}"

    def test_collapsing_a_segment_drops_it(self, client, db_session):
        """Collapsing a segment (end <= start) DROPS it (200), leaving a
        contiguous partition — this is how the editor 'deletes' a keep."""
        project, segs = self._make_contiguous_project(db_session, duration=30.0, n=3)
        s0, s1, s2 = segs

        # Collapse s0 (end=0). It should be dropped, not rejected.
        response = client.patch(
            f"/api/v2/projects/{project.id}/segments",
            json={"updates": [{"id": s0.id, "end_sec": 0.0}]},
        )
        assert response.status_code == 200
        result = response.json()
        assert len(result) == 2  # s0 dropped
        # Contiguous coverage of [0, 30]
        assert result[0]["start_sec"] == pytest.approx(0.0)
        assert result[-1]["end_sec"] == pytest.approx(30.0)
        for a, b in zip(result, result[1:]):
            assert a["end_sec"] == pytest.approx(b["start_sec"])

    def test_end_beyond_duration_is_clamped(self, client, db_session):
        """An end_sec beyond project duration is clamped to duration (200);
        segments pushed past the end collapse and are dropped."""
        project, segs = self._make_contiguous_project(db_session, duration=30.0, n=3)
        s0, s1, s2 = segs

        response = client.patch(
            f"/api/v2/projects/{project.id}/segments",
            json={"updates": [{"id": s1.id, "end_sec": 35.0}]},
        )
        assert response.status_code == 200
        result = response.json()
        assert result[-1]["end_sec"] == pytest.approx(30.0)
        for a, b in zip(result, result[1:]):
            assert a["end_sec"] == pytest.approx(b["start_sec"])

    def test_patch_still_409_when_not_ready(self, client, db_session):
        """PATCH on a non-ready project is still 409."""
        project = _make_project(
            db_session,
            status=ProjectStatus.analyzing,
            duration_sec=30.0,
        )
        seg = _make_segment_def(db_session, project.id, 0, start_sec=0.0, end_sec=10.0)

        response = client.patch(
            f"/api/v2/projects/{project.id}/segments",
            json={"updates": [{"id": seg.id, "end_sec": 12.0}]},
        )
        assert response.status_code == 409


# ---------------------------------------------------------------------------
# TR8: _normalize_partition unit tests (pure helper)
# ---------------------------------------------------------------------------


class TestNormalizePartition:
    """Direct unit tests for the _normalize_partition helper."""

    def setup_method(self):
        """Import the helper fresh each test."""
        import app.api_v2 as m
        self.normalize = m._normalize_partition

        class FakeDB:
            def __init__(self):
                self.deleted = []
            def delete(self, x):
                self.deleted.append(x)

        self.db = FakeDB()

    def _fake_seg(self, idx, start, end, seg_id=None):
        """Minimal object with the fields _normalize_partition reads/writes."""
        class FakeSeg:
            pass
        s = FakeSeg()
        s.id = seg_id or f"seg-{idx}"
        s.index = idx
        s.start_sec = float(start)
        s.end_sec = float(end)
        return s

    def test_contiguous_input_unchanged(self):
        segs = [
            self._fake_seg(0, 0, 10),
            self._fake_seg(1, 10, 20),
            self._fake_seg(2, 20, 30),
        ]
        self.normalize(segs, 30.0, self.db)
        assert self.db.deleted == []
        assert segs[0].start_sec == pytest.approx(0.0)
        assert segs[0].end_sec == pytest.approx(10.0)
        assert segs[1].start_sec == pytest.approx(10.0)
        assert segs[1].end_sec == pytest.approx(20.0)
        assert segs[2].start_sec == pytest.approx(20.0)
        assert segs[2].end_sec == pytest.approx(30.0)

    def test_derives_starts_from_ends(self):
        """Starts are derived from the running cursor; only ends matter."""
        segs = [
            self._fake_seg(0, 0, 10),
            self._fake_seg(1, 9999, 20),  # stale start_sec; end=20 is the boundary
        ]
        self.normalize(segs, 30.0, self.db)
        assert segs[0].start_sec == pytest.approx(0.0)
        assert segs[0].end_sec == pytest.approx(10.0)
        assert segs[1].start_sec == pytest.approx(10.0)
        assert segs[1].end_sec == pytest.approx(30.0)  # last extended to duration

    def test_indices_reassigned_zero_based(self):
        segs = [
            self._fake_seg(5, 0, 10),
            self._fake_seg(7, 10, 20),
            self._fake_seg(3, 20, 30),
        ]
        self.normalize(segs, 30.0, self.db)
        indices = sorted(s.index for s in segs)
        assert indices == [0, 1, 2]

    def test_empty_list_is_noop(self):
        self.normalize([], 30.0, self.db)  # should not raise

    def test_collapsed_segment_is_dropped_not_rejected(self):
        """A zero-duration segment is DROPPED (deleted), not a 400 — and the
        remaining segments stay contiguous. This is the 'delete the keep by
        collapsing it' behaviour the editor relies on."""
        zero = self._fake_seg(1, 6.0, 6.0, seg_id="keep")   # collapsed
        s0 = self._fake_seg(0, 0, 6.0, seg_id="a")
        s2 = self._fake_seg(2, 6.0, 30.0, seg_id="b")
        segs = [s0, zero, s2]
        self.normalize(segs, 30.0, self.db)
        assert self.db.deleted == [zero]            # the collapsed one dropped
        assert s0.start_sec == pytest.approx(0.0) and s0.end_sec == pytest.approx(6.0)
        assert s2.start_sec == pytest.approx(6.0) and s2.end_sec == pytest.approx(30.0)
        assert s0.index == 0 and s2.index == 1      # reindexed over the gap

    def test_negative_duration_segment_dropped(self):
        """A neighbour extended over a segment (start>end) drops that segment."""
        s0 = self._fake_seg(0, 0, 6.5, seg_id="a")
        bad = self._fake_seg(1, 6.5, 6.0, seg_id="keep")  # end<start after edit
        s2 = self._fake_seg(2, 6.0, 30.0, seg_id="b")
        segs = [s0, bad, s2]
        self.normalize(segs, 30.0, self.db)
        assert bad in self.db.deleted
        assert s0.end_sec == pytest.approx(6.5)
        assert s2.start_sec == pytest.approx(6.5) and s2.end_sec == pytest.approx(30.0)

    def test_end_beyond_duration_is_clamped(self):
        """An end beyond duration is clamped; segments past it are dropped."""
        s0 = self._fake_seg(0, 0, 50, seg_id="a")   # end > duration 30
        s1 = self._fake_seg(1, 60, 80, seg_id="b")
        segs = [s0, s1]
        self.normalize(segs, 30.0, self.db)
        assert s0.start_sec == pytest.approx(0.0) and s0.end_sec == pytest.approx(30.0)
        assert s1 in self.db.deleted

    def test_all_collapsed_raises_400(self):
        from fastapi import HTTPException
        segs = [self._fake_seg(0, 0, 0)]
        with pytest.raises(HTTPException) as exc_info:
            self.normalize(segs, 30.0, self.db)
        assert exc_info.value.status_code == 400

    def test_single_segment_spans_full_duration(self):
        segs = [self._fake_seg(0, 5, 25)]  # start/end arbitrary; will be pinned
        self.normalize(segs, 30.0, self.db)
        assert segs[0].start_sec == pytest.approx(0.0)
        assert segs[0].end_sec == pytest.approx(30.0)


# ---------------------------------------------------------------------------
# POST /api/v2/projects/{pid}/runs
# ---------------------------------------------------------------------------


class TestCreateRun:
    def test_create_run_on_ready_project_returns_201(self, spy_client, SessionFactory):
        client, spy = spy_client
        session = SessionFactory()
        project = _make_project(session, status=ProjectStatus.ready)
        session.close()

        response = client.post(
            f"/api/v2/projects/{project.id}/runs",
            data={"prompt": "swap to redhead", "resolution": "720p"},
        )
        assert response.status_code == 201
        body = response.json()
        assert "run_id" in body
        assert body["status"] == "queued"
        assert spy["process_run"] == [body["run_id"]]

        # Verify Run in DB
        session = SessionFactory()
        run = session.get(Run, body["run_id"])
        session.close()
        assert run is not None
        assert run.project_id == project.id
        assert run.prompt == "swap to redhead"
        # status should be queued after create_run transitions it
        status_val = run.status.value if hasattr(run.status, "value") else str(run.status)
        assert status_val == "queued"

    def test_create_run_on_non_ready_project_is_409(self, client, db_session):
        project = _make_project(db_session, status=ProjectStatus.created)
        response = client.post(
            f"/api/v2/projects/{project.id}/runs",
            data={"prompt": "swap"},
        )
        assert response.status_code == 409

    def test_create_run_on_analyzing_project_is_409(self, client, db_session):
        project = _make_project(db_session, status=ProjectStatus.analyzing)
        response = client.post(
            f"/api/v2/projects/{project.id}/runs",
            data={"prompt": "swap"},
        )
        assert response.status_code == 409

    def test_too_many_reference_images_is_400(self, client, db_session):
        project = _make_project(db_session, status=ProjectStatus.ready)
        response = client.post(
            f"/api/v2/projects/{project.id}/runs",
            data={
                "prompt": "swap",
                "reference_urls": (
                    "https://example.com/ref1.jpg,"
                    "https://example.com/ref2.jpg,"
                    "https://example.com/ref3.jpg"
                ),
            },
        )
        assert response.status_code == 400
        assert "too many reference" in response.json()["detail"].lower()

    def test_bad_resolution_is_400(self, client, db_session):
        project = _make_project(db_session, status=ProjectStatus.ready)
        response = client.post(
            f"/api/v2/projects/{project.id}/runs",
            data={"prompt": "swap", "resolution": "4k"},
        )
        assert response.status_code == 400
        assert "resolution" in response.json()["detail"].lower()

    def test_create_run_missing_project_is_404(self, client):
        response = client.post(
            "/api/v2/projects/no-such-project/runs",
            data={"prompt": "swap"},
        )
        assert response.status_code == 404

    def test_create_run_with_name_stored(self, spy_client, SessionFactory):
        client, spy = spy_client
        session = SessionFactory()
        project = _make_project(session, status=ProjectStatus.ready)
        session.close()

        response = client.post(
            f"/api/v2/projects/{project.id}/runs",
            data={"prompt": "swap character", "name": "Redhead woman"},
        )
        assert response.status_code == 201
        run_id = response.json()["run_id"]

        session = SessionFactory()
        run = session.get(Run, run_id)
        session.close()
        assert run.name == "Redhead woman"

    def test_create_run_with_reference_urls(self, spy_client, SessionFactory):
        client, spy = spy_client
        session = SessionFactory()
        project = _make_project(session, status=ProjectStatus.ready)
        session.close()

        response = client.post(
            f"/api/v2/projects/{project.id}/runs",
            data={
                "prompt": "swap character",
                "reference_urls": (
                    "https://example.com/ref1.jpg,"
                    "https://example.com/ref2.jpg"
                ),
            },
        )
        assert response.status_code == 201
        run_id = response.json()["run_id"]

        session = SessionFactory()
        run = session.get(Run, run_id)
        session.close()
        assert len(run.reference_image_urls) == 2


# ---------------------------------------------------------------------------
# Model selection (Seedance vs Gemini Omni) on Run creation
# ---------------------------------------------------------------------------


class TestModelSelection:
    """Tests for the per-run model field and its model-specific resolution rules."""

    def test_create_run_with_gemini_model_persists(self, spy_client, SessionFactory):
        client, spy = spy_client
        session = SessionFactory()
        project = _make_project(session, status=ProjectStatus.ready)
        session.close()

        response = client.post(
            f"/api/v2/projects/{project.id}/runs",
            data={"prompt": "swap", "model": "gemini-omni", "resolution": "1080p"},
        )
        assert response.status_code == 201
        run_id = response.json()["run_id"]

        session = SessionFactory()
        run = session.get(Run, run_id)
        session.close()
        assert run.model == "gemini-omni"
        assert run.resolution == "1080p"

        # Visible in GET /api/v2/runs/{rid}
        get_resp = client.get(f"/api/v2/runs/{run_id}")
        assert get_resp.json()["model"] == "gemini-omni"

    def test_create_run_default_model_is_seedance(self, spy_client, SessionFactory):
        client, spy = spy_client
        session = SessionFactory()
        project = _make_project(session, status=ProjectStatus.ready)
        session.close()

        response = client.post(
            f"/api/v2/projects/{project.id}/runs",
            data={"prompt": "swap", "resolution": "720p"},
        )
        assert response.status_code == 201
        run_id = response.json()["run_id"]

        get_resp = client.get(f"/api/v2/runs/{run_id}")
        assert get_resp.json()["model"] == "seedance"

    def test_invalid_model_is_400(self, client, db_session):
        project = _make_project(db_session, status=ProjectStatus.ready)
        response = client.post(
            f"/api/v2/projects/{project.id}/runs",
            data={"prompt": "swap", "model": "midjourney"},
        )
        assert response.status_code == 400
        assert "model" in response.json()["detail"].lower()

    def test_gemini_allows_4k(self, spy_client, SessionFactory):
        client, spy = spy_client
        session = SessionFactory()
        project = _make_project(session, status=ProjectStatus.ready)
        session.close()

        response = client.post(
            f"/api/v2/projects/{project.id}/runs",
            data={"prompt": "swap", "model": "gemini-omni", "resolution": "4k"},
        )
        assert response.status_code == 201

    def test_gemini_rejects_480p(self, client, db_session):
        project = _make_project(db_session, status=ProjectStatus.ready)
        response = client.post(
            f"/api/v2/projects/{project.id}/runs",
            data={"prompt": "swap", "model": "gemini-omni", "resolution": "480p"},
        )
        assert response.status_code == 400
        assert "resolution" in response.json()["detail"].lower()

    def test_seedance_rejects_4k(self, client, db_session):
        project = _make_project(db_session, status=ProjectStatus.ready)
        response = client.post(
            f"/api/v2/projects/{project.id}/runs",
            data={"prompt": "swap", "model": "seedance", "resolution": "4k"},
        )
        assert response.status_code == 400
        assert "resolution" in response.json()["detail"].lower()

    @pytest.mark.parametrize("variant", ["seedance-fast", "seedance-mini"])
    @pytest.mark.parametrize("resolution", ["480p", "720p"])
    def test_seedance_variant_accepts_480p_and_720p(
        self, spy_client, SessionFactory, variant, resolution
    ):
        """seedance-fast/seedance-mini accept both of their supported resolutions."""
        client, spy = spy_client
        session = SessionFactory()
        project = _make_project(session, status=ProjectStatus.ready)
        session.close()

        response = client.post(
            f"/api/v2/projects/{project.id}/runs",
            data={"prompt": "swap", "model": variant, "resolution": resolution},
        )
        assert response.status_code == 201
        run_id = response.json()["run_id"]

        get_resp = client.get(f"/api/v2/runs/{run_id}")
        assert get_resp.json()["model"] == variant
        assert get_resp.json()["resolution"] == resolution

    @pytest.mark.parametrize("variant", ["seedance-fast", "seedance-mini"])
    def test_seedance_variant_rejects_1080p(self, client, db_session, variant):
        """seedance-fast/seedance-mini do not support 1080p (or 4k)."""
        project = _make_project(db_session, status=ProjectStatus.ready)
        response = client.post(
            f"/api/v2/projects/{project.id}/runs",
            data={"prompt": "swap", "model": variant, "resolution": "1080p"},
        )
        assert response.status_code == 400
        assert "resolution" in response.json()["detail"].lower()


# ---------------------------------------------------------------------------
# TR7: audio_mode field on Run creation and RunResponse
# ---------------------------------------------------------------------------


class TestAudioMode:
    """Tests for TR7: per-run audio_mode field."""

    def test_create_run_with_audio_mode_seedance_persists(
        self, spy_client, SessionFactory
    ):
        """Creating a run with audio_mode=seedance persists that value and shows in RunResponse."""
        client, spy = spy_client
        session = SessionFactory()
        project = _make_project(session, status=ProjectStatus.ready)
        session.close()

        response = client.post(
            f"/api/v2/projects/{project.id}/runs",
            data={"prompt": "swap", "audio_mode": "seedance"},
        )
        assert response.status_code == 201
        run_id = response.json()["run_id"]

        # Verify persisted in DB.
        session = SessionFactory()
        run = session.get(Run, run_id)
        session.close()
        assert run is not None
        audio_mode_val = run.audio_mode.value if hasattr(run.audio_mode, "value") else str(run.audio_mode)
        assert audio_mode_val == "seedance"

        # Verify visible in GET /api/v2/runs/{rid}
        get_resp = client.get(f"/api/v2/runs/{run_id}")
        assert get_resp.status_code == 200
        assert get_resp.json()["audio_mode"] == "seedance"

    def test_create_run_default_audio_mode_is_original(
        self, spy_client, SessionFactory
    ):
        """When audio_mode is not provided, it defaults to 'original'."""
        client, spy = spy_client
        session = SessionFactory()
        project = _make_project(session, status=ProjectStatus.ready)
        session.close()

        response = client.post(
            f"/api/v2/projects/{project.id}/runs",
            data={"prompt": "swap"},
        )
        assert response.status_code == 201
        run_id = response.json()["run_id"]

        get_resp = client.get(f"/api/v2/runs/{run_id}")
        assert get_resp.status_code == 200
        assert get_resp.json()["audio_mode"] == "original"

    def test_create_run_invalid_audio_mode_is_400(self, client, db_session):
        """Passing an unrecognised audio_mode value returns HTTP 400."""
        project = _make_project(db_session, status=ProjectStatus.ready)
        response = client.post(
            f"/api/v2/projects/{project.id}/runs",
            data={"prompt": "swap", "audio_mode": "dolby"},
        )
        assert response.status_code == 400
        assert "audio_mode" in response.json()["detail"].lower()

    def test_run_response_includes_audio_mode_field(self, client, db_session):
        """GET /api/v2/runs/{rid} returns audio_mode in the response body."""
        project = _make_project(db_session)
        run = _make_run(db_session, project.id, audio_mode="original")

        response = client.get(f"/api/v2/runs/{run.id}")
        assert response.status_code == 200
        body = response.json()
        assert "audio_mode" in body
        assert body["audio_mode"] in ("original", "seedance")


# ---------------------------------------------------------------------------
# GET /api/v2/runs/{rid}  and  GET /api/v2/projects/{pid}/runs
# ---------------------------------------------------------------------------


class TestGetRun:
    def test_get_existing_run_200(self, client, db_session):
        project = _make_project(db_session)
        run = _make_run(db_session, project.id)

        response = client.get(f"/api/v2/runs/{run.id}")
        assert response.status_code == 200
        body = response.json()
        assert body["id"] == run.id
        assert body["project_id"] == project.id

    def test_get_missing_run_404(self, client):
        response = client.get("/api/v2/runs/does-not-exist-id")
        assert response.status_code == 404

    def test_list_project_runs(self, client, db_session):
        project = _make_project(db_session)
        _make_run(db_session, project.id)
        _make_run(db_session, project.id)

        response = client.get(f"/api/v2/projects/{project.id}/runs")
        assert response.status_code == 200
        body = response.json()
        assert isinstance(body, list)
        assert len(body) >= 2

    def test_list_project_runs_404_missing_project(self, client):
        response = client.get("/api/v2/projects/no-such/runs")
        assert response.status_code == 404


# ---------------------------------------------------------------------------
# GET /api/v2/runs/{rid}/result
# ---------------------------------------------------------------------------


class TestRunResult:
    def test_result_on_non_done_run_is_409(self, client, db_session):
        project = _make_project(db_session)
        run = _make_run(db_session, project.id, status=RunStatus.processing)

        response = client.get(f"/api/v2/runs/{run.id}/result")
        assert response.status_code == 409

    def test_result_on_queued_run_is_409(self, client, db_session):
        project = _make_project(db_session)
        run = _make_run(db_session, project.id, status=RunStatus.queued)

        response = client.get(f"/api/v2/runs/{run.id}/result")
        assert response.status_code == 409

    def test_result_on_done_run_returns_file(self, client, db_session, tmp_path):
        result_file = tmp_path / "final.mp4"
        result_file.write_bytes(b"\x00VIDEO\xff")

        project = _make_project(db_session)
        run = _make_run(
            db_session,
            project.id,
            status=RunStatus.done,
            result_local_path=str(result_file),
        )

        response = client.get(f"/api/v2/runs/{run.id}/result")
        assert response.status_code == 200
        assert response.content == b"\x00VIDEO\xff"
        assert response.headers["cache-control"] == "no-store, max-age=0"

    def test_result_info_on_done_run(self, client, db_session, tmp_path):
        result_file = tmp_path / "output.mp4"
        result_file.write_bytes(b"\x00\xff")

        project = _make_project(db_session)
        run = _make_run(
            db_session,
            project.id,
            status=RunStatus.done,
            result_local_path=str(result_file),
            result_gdrive_file_id="GDRIVE_RUN_123",
        )

        response = client.get(f"/api/v2/runs/{run.id}/result/info")
        assert response.status_code == 200
        body = response.json()
        assert body["result_gdrive_file_id"] == "GDRIVE_RUN_123"
        assert "drive.google.com" in body["result_gdrive_link"]

    def test_result_info_on_non_done_run_is_409(self, client, db_session):
        project = _make_project(db_session)
        run = _make_run(db_session, project.id, status=RunStatus.processing)

        response = client.get(f"/api/v2/runs/{run.id}/result/info")
        assert response.status_code == 409


# ---------------------------------------------------------------------------
# GET /api/v2/projects/{pid}/frame
# ---------------------------------------------------------------------------


class TestProjectFrame:
    def test_frame_endpoint_returns_jpeg(self, client, db_session, tmp_path):
        """Generate a real 1-second video with ffmpeg and verify the frame endpoint."""
        video_path = str(tmp_path / "source.mp4")
        _make_ffmpeg_video(video_path)

        project = _make_project(
            db_session,
            source_local_path=video_path,
            status=ProjectStatus.ready,
        )

        response = client.get(f"/api/v2/projects/{project.id}/frame?t=0")
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/jpeg"
        assert len(response.content) > 100  # non-empty JPEG

    def test_frame_endpoint_404_missing_source(self, client, db_session):
        """Project exists but source file not on disk → 404."""
        project = _make_project(
            db_session,
            source_local_path="/tmp/nonexistent_source_xyz.mp4",
            status=ProjectStatus.ready,
        )
        response = client.get(f"/api/v2/projects/{project.id}/frame?t=0")
        assert response.status_code == 404

    def test_frame_endpoint_404_no_source(self, client, db_session):
        """Project with no source_local_path → 404."""
        project = _make_project(db_session)
        response = client.get(f"/api/v2/projects/{project.id}/frame?t=0")
        assert response.status_code == 404

    def test_source_endpoint_returns_video(self, client, db_session, tmp_path):
        """The source endpoint streams the original video file."""
        video_path = str(tmp_path / "source.mp4")
        _make_ffmpeg_video(video_path)
        project = _make_project(
            db_session, source_local_path=video_path, status=ProjectStatus.ready
        )
        response = client.get(f"/api/v2/projects/{project.id}/source")
        assert response.status_code == 200
        assert response.headers["content-type"] == "video/mp4"
        assert len(response.content) > 100

    def test_source_endpoint_404_missing(self, client, db_session):
        """No source file on disk → 404."""
        project = _make_project(
            db_session, source_local_path="/tmp/nope_xyz.mp4", status=ProjectStatus.ready
        )
        response = client.get(f"/api/v2/projects/{project.id}/source")
        assert response.status_code == 404


# ---------------------------------------------------------------------------
# GET /api/v2/runs/{rid}/segments
# ---------------------------------------------------------------------------


class TestRunSegments:
    def test_get_run_segments_empty(self, client, db_session):
        project = _make_project(db_session)
        run = _make_run(db_session, project.id)

        response = client.get(f"/api/v2/runs/{run.id}/segments")
        assert response.status_code == 200
        assert response.json() == []

    def test_get_run_segments_404_missing_run(self, client):
        response = client.get("/api/v2/runs/no-such-run/segments")
        assert response.status_code == 404

    def test_get_run_segments_with_data(self, client, db_session):
        project = _make_project(db_session, status=ProjectStatus.ready)
        seg_def = _make_segment_def(db_session, project.id, 0)
        run = _make_run(db_session, project.id, status=RunStatus.processing)

        # Insert a RunSegment directly
        rs = RunSegment(
            run_id=run.id,
            segment_def_id=seg_def.id,
            index=0,
            status=SegmentStatus.pending,
        )
        db_session.add(rs)
        db_session.commit()

        response = client.get(f"/api/v2/runs/{run.id}/segments")
        assert response.status_code == 200
        body = response.json()
        assert len(body) == 1
        assert body[0]["run_id"] == run.id
        assert body[0]["segment_def_id"] == seg_def.id


# ---------------------------------------------------------------------------
# POST /api/v2/runs/{rid}/retry
# ---------------------------------------------------------------------------


class TestRetryRun:
    def test_retry_failed_run_transitions_to_queued(self, spy_client, SessionFactory):
        client, spy = spy_client
        session = SessionFactory()
        project = _make_project(session)
        run = _make_run(session, project.id, status=RunStatus.failed)
        session.close()

        response = client.post(f"/api/v2/runs/{run.id}/retry")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "queued"
        assert spy["process_run"] == [run.id]

    def test_retry_non_retryable_run_is_409(self, client, db_session):
        """done status is not retryable via /retry (use /runs/{rid}/segments/{rsid}/rerun for that)."""
        project = _make_project(db_session)
        run = _make_run(db_session, project.id, status=RunStatus.done)

        response = client.post(f"/api/v2/runs/{run.id}/retry")
        assert response.status_code == 409

    def test_retry_processing_run_succeeds(self, spy_client, SessionFactory):
        """TR5b: /retry must accept a run stuck in processing (orphan resume)."""
        client, spy = spy_client
        session = SessionFactory()
        project = _make_project(session)
        run = _make_run(session, project.id, status=RunStatus.processing)
        session.close()

        response = client.post(f"/api/v2/runs/{run.id}/retry")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "queued"
        assert run.id in spy["process_run"]

    def test_retry_incomplete_run_succeeds(self, spy_client, SessionFactory):
        """/retry must accept an `incomplete` run (re-send the failed segments)."""
        client, spy = spy_client
        session = SessionFactory()
        project = _make_project(session)
        run = _make_run(session, project.id, status=RunStatus.incomplete)
        session.close()

        response = client.post(f"/api/v2/runs/{run.id}/retry")
        assert response.status_code == 200
        assert response.json()["status"] == "queued"
        assert run.id in spy["process_run"]

    def test_retry_queued_run_succeeds(self, spy_client, SessionFactory):
        """TR5b: /retry must accept a run stuck in queued (orphan resume)."""
        client, spy = spy_client
        session = SessionFactory()
        project = _make_project(session)
        run = _make_run(session, project.id, status=RunStatus.queued)
        session.close()

        response = client.post(f"/api/v2/runs/{run.id}/retry")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "queued"
        assert run.id in spy["process_run"]

    def test_retry_missing_run_is_404(self, client):
        response = client.post("/api/v2/runs/no-such-run/retry")
        assert response.status_code == 404


# ---------------------------------------------------------------------------
# PATCH /api/v2/runs/{rid}/segments/{rsid}
# ---------------------------------------------------------------------------


class TestPatchRunSegment:
    def test_patch_sets_prompt_override(self, client, db_session):
        project = _make_project(db_session, status=ProjectStatus.ready)
        sd = _make_segment_def(db_session, project.id, 0)
        run = _make_run(db_session, project.id, status=RunStatus.done)
        rs = _make_run_segment(db_session, run.id, sd.id, status=SegmentStatus.completed)

        response = client.patch(
            f"/api/v2/runs/{run.id}/segments/{rs.id}",
            data={"prompt": "per-segment override"},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["prompt_override"] == "per-segment override"

    def test_patch_while_processing_is_409(self, client, db_session):
        project = _make_project(db_session, status=ProjectStatus.ready)
        sd = _make_segment_def(db_session, project.id, 0)
        run = _make_run(db_session, project.id, status=RunStatus.processing)
        rs = _make_run_segment(db_session, run.id, sd.id, status=SegmentStatus.generating)

        response = client.patch(
            f"/api/v2/runs/{run.id}/segments/{rs.id}",
            data={"prompt": "override"},
        )
        assert response.status_code == 409

    def test_patch_empty_prompt_clears_override(self, client, db_session):
        project = _make_project(db_session, status=ProjectStatus.ready)
        sd = _make_segment_def(db_session, project.id, 0)
        run = _make_run(db_session, project.id, status=RunStatus.done)
        rs = _make_run_segment(
            db_session, run.id, sd.id,
            status=SegmentStatus.completed,
            prompt_override="old override",
        )

        response = client.patch(
            f"/api/v2/runs/{run.id}/segments/{rs.id}",
            data={"prompt": ""},
        )
        assert response.status_code == 200
        assert response.json()["prompt_override"] is None

    def test_patch_too_many_refs_is_400(self, client, db_session):
        project = _make_project(db_session, status=ProjectStatus.ready)
        sd = _make_segment_def(db_session, project.id, 0)
        run = _make_run(db_session, project.id, status=RunStatus.failed)
        rs = _make_run_segment(db_session, run.id, sd.id)

        response = client.patch(
            f"/api/v2/runs/{run.id}/segments/{rs.id}",
            data={"reference_urls": "https://a.com/1.jpg,https://a.com/2.jpg,https://a.com/3.jpg"},
        )
        assert response.status_code == 400
        assert "too many" in response.json()["detail"].lower()

    def test_patch_missing_segment_is_404(self, client, db_session):
        project = _make_project(db_session)
        run = _make_run(db_session, project.id, status=RunStatus.done)

        response = client.patch(
            f"/api/v2/runs/{run.id}/segments/no-such-seg",
            data={"prompt": "override"},
        )
        assert response.status_code == 404

    def test_patch_sets_reference_urls_override(self, client, db_session):
        project = _make_project(db_session, status=ProjectStatus.ready)
        sd = _make_segment_def(db_session, project.id, 0)
        run = _make_run(db_session, project.id, status=RunStatus.done)
        rs = _make_run_segment(db_session, run.id, sd.id, status=SegmentStatus.completed)

        response = client.patch(
            f"/api/v2/runs/{run.id}/segments/{rs.id}",
            data={"reference_urls": "https://example.com/ref1.jpg"},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["reference_image_urls_override"] is not None
        assert len(body["reference_image_urls_override"]) == 1


# ---------------------------------------------------------------------------
# POST /api/v2/runs/{rid}/segments/{rsid}/rerun
# ---------------------------------------------------------------------------


class TestRerunSegment:
    def test_rerun_resets_segment_and_enqueues(self, spy_client, SessionFactory):
        client, spy = spy_client
        session = SessionFactory()
        project = _make_project(session, status=ProjectStatus.ready)
        sd = _make_segment_def(session, project.id, 0)
        run = _make_run(
            session, project.id, status=RunStatus.done,
            result_local_path="/data/fake/final.mp4",
        )
        rs = _make_run_segment(
            session, run.id, sd.id,
            status=SegmentStatus.completed,
            seedance_task_id="old-task-id",
        )
        session.close()

        response = client.post(f"/api/v2/runs/{run.id}/segments/{rs.id}/rerun")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "queued"
        # enqueue_process_run should have been called
        assert spy["process_run"] == [run.id]

        # Verify the RunSegment was reset
        session2 = SessionFactory()
        rs_fetched = session2.get(RunSegment, rs.id)
        run_fetched = session2.get(Run, run.id)
        session2.close()
        assert rs_fetched.status == SegmentStatus.pending
        assert rs_fetched.seedance_task_id is None
        # The stitched final contains the OLD segment result — a rerun must
        # invalidate it so finalize re-stitches instead of re-delivering it.
        assert run_fetched.result_local_path is None

    def test_rerun_failed_segment_from_incomplete_run(self, spy_client, SessionFactory):
        """An `incomplete` run can re-run its failed segment; run → queued."""
        client, spy = spy_client
        session = SessionFactory()
        project = _make_project(session, status=ProjectStatus.ready)
        sd = _make_segment_def(session, project.id, 0)
        run = _make_run(session, project.id, status=RunStatus.incomplete)
        rs = _make_run_segment(
            session, run.id, sd.id,
            status=SegmentStatus.failed,
            seedance_task_id="failed-task-id",
            error_message="failed after 3 attempt(s): Internal Error",
        )
        session.close()

        response = client.post(f"/api/v2/runs/{run.id}/segments/{rs.id}/rerun")
        assert response.status_code == 200
        assert response.json()["status"] == "queued"
        assert spy["process_run"] == [run.id]

        session2 = SessionFactory()
        rs_fetched = session2.get(RunSegment, rs.id)
        session2.close()
        assert rs_fetched.status == SegmentStatus.pending
        assert rs_fetched.error_message is None

    def test_rerun_with_prompt_applies_override(self, spy_client, SessionFactory):
        """Re-run carrying a prompt persists it as the segment override atomically."""
        client, spy = spy_client
        session = SessionFactory()
        project = _make_project(session, status=ProjectStatus.ready)
        sd = _make_segment_def(session, project.id, 0)
        run = _make_run(session, project.id, status=RunStatus.done)
        rs = _make_run_segment(
            session, run.id, sd.id, status=SegmentStatus.completed,
        )
        rs_id = rs.id
        session.close()

        response = client.post(
            f"/api/v2/runs/{run.id}/segments/{rs_id}/rerun",
            data={"prompt": "make the character a red panda"},
        )
        assert response.status_code == 200
        assert response.json()["status"] == "queued"
        assert spy["process_run"] == [run.id]

        session2 = SessionFactory()
        rs_fetched = session2.get(RunSegment, rs_id)
        session2.close()
        assert rs_fetched.prompt_override == "make the character a red panda"
        assert rs_fetched.status == SegmentStatus.pending

    def test_rerun_on_failed_run_also_works(self, spy_client, SessionFactory):
        client, spy = spy_client
        session = SessionFactory()
        project = _make_project(session, status=ProjectStatus.ready)
        sd = _make_segment_def(session, project.id, 0)
        run = _make_run(session, project.id, status=RunStatus.failed)
        rs = _make_run_segment(session, run.id, sd.id, status=SegmentStatus.failed)
        session.close()

        response = client.post(f"/api/v2/runs/{run.id}/segments/{rs.id}/rerun")
        assert response.status_code == 200
        assert response.json()["status"] == "queued"

    def test_rerun_on_processing_run_is_409(self, client, db_session):
        project = _make_project(db_session, status=ProjectStatus.ready)
        sd = _make_segment_def(db_session, project.id, 0)
        run = _make_run(db_session, project.id, status=RunStatus.processing)
        rs = _make_run_segment(db_session, run.id, sd.id, status=SegmentStatus.generating)

        response = client.post(f"/api/v2/runs/{run.id}/segments/{rs.id}/rerun")
        assert response.status_code == 409

    def test_rerun_missing_segment_is_404(self, client, db_session):
        project = _make_project(db_session)
        run = _make_run(db_session, project.id, status=RunStatus.done)

        response = client.post(f"/api/v2/runs/{run.id}/segments/no-such-seg/rerun")
        assert response.status_code == 404


# ---------------------------------------------------------------------------
# PATCH /api/v2/projects/{pid} — editable project name
# ---------------------------------------------------------------------------


class TestProjectName:
    def test_patch_name_persists_and_shows_in_get(self, client, db_session):
        project = _make_project(db_session, status=ProjectStatus.ready)

        resp = client.patch(f"/api/v2/projects/{project.id}", json={"name": "  Erewhon promo  "})
        assert resp.status_code == 200
        assert resp.json()["name"] == "Erewhon promo"  # trimmed

        get_resp = client.get(f"/api/v2/projects/{project.id}")
        assert get_resp.json()["name"] == "Erewhon promo"

    def test_patch_empty_name_clears_to_null(self, client, db_session):
        project = _make_project(db_session, status=ProjectStatus.ready, name="old name")
        resp = client.patch(f"/api/v2/projects/{project.id}", json={"name": "   "})
        assert resp.status_code == 200
        assert resp.json()["name"] is None

    def test_patch_missing_project_is_404(self, client):
        resp = client.patch("/api/v2/projects/no-such-project", json={"name": "x"})
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# DELETE /api/v2/projects/{pid} and /api/v2/runs/{rid} — DB + disk
# ---------------------------------------------------------------------------


class TestDeleteProject:
    def test_delete_removes_db_rows_and_disk(self, client, db_session, SessionFactory):
        from app.storage import project_dir

        project = _make_project(db_session, status=ProjectStatus.ready)
        sd = _make_segment_def(db_session, project.id, 0)
        run = _make_run(db_session, project.id, status=RunStatus.done)
        pid, rid = project.id, run.id

        pdir = project_dir(pid)  # creates the dir
        assert os.path.isdir(pdir)

        resp = client.delete(f"/api/v2/projects/{pid}")
        assert resp.status_code == 204
        assert not os.path.exists(pdir)

        s = SessionFactory()
        assert s.get(VideoProject, pid) is None
        assert s.get(Run, rid) is None  # cascade
        s.close()

    def test_delete_blocked_while_analyzing(self, client, db_session):
        from app.storage import project_dir

        project = _make_project(db_session, status=ProjectStatus.analyzing)
        pdir = project_dir(project.id)
        resp = client.delete(f"/api/v2/projects/{project.id}")
        assert resp.status_code == 409
        assert os.path.isdir(pdir)  # untouched

    def test_delete_blocked_while_run_active(self, client, db_session):
        project = _make_project(db_session, status=ProjectStatus.ready)
        _make_run(db_session, project.id, status=RunStatus.processing)
        resp = client.delete(f"/api/v2/projects/{project.id}")
        assert resp.status_code == 409

    def test_delete_missing_project_is_404(self, client):
        assert client.delete("/api/v2/projects/no-such-project").status_code == 404


class TestDeleteRun:
    def test_delete_removes_db_rows_and_disk(self, client, db_session, SessionFactory):
        from app.storage import run_clips_dir, run_dir

        project = _make_project(db_session, status=ProjectStatus.ready)
        sd = _make_segment_def(db_session, project.id, 0)
        run = _make_run(db_session, project.id, status=RunStatus.done)
        rs = _make_run_segment(db_session, run.id, sd.id)
        rid, rs_id = run.id, rs.id

        run_clips_dir(rid, project.id)  # creates runs/<rid>/clips
        rdir = run_dir(rid, project.id)
        assert os.path.isdir(rdir)

        resp = client.delete(f"/api/v2/runs/{rid}")
        assert resp.status_code == 204
        assert not os.path.exists(rdir)

        s = SessionFactory()
        assert s.get(Run, rid) is None
        assert s.get(RunSegment, rs_id) is None  # cascade
        s.close()

    def test_delete_blocked_while_active(self, client, db_session):
        project = _make_project(db_session, status=ProjectStatus.ready)
        run = _make_run(db_session, project.id, status=RunStatus.stitching)
        resp = client.delete(f"/api/v2/runs/{run.id}")
        assert resp.status_code == 409

    def test_delete_missing_run_is_404(self, client):
        assert client.delete("/api/v2/runs/no-such-run").status_code == 404


# ---------------------------------------------------------------------------
# Single-file uploads (regression: Optional[List[UploadFile]] coerced a single
# file to a 422 "Input should be a valid list" on FastAPI 0.104)
# ---------------------------------------------------------------------------


class TestSingleFileUpload:
    def test_patch_segment_with_one_reference_file(self, client, db_session):
        project = _make_project(db_session, status=ProjectStatus.ready)
        sd = _make_segment_def(db_session, project.id, 0)
        run = _make_run(db_session, project.id, status=RunStatus.done)
        rs = _make_run_segment(db_session, run.id, sd.id, status=SegmentStatus.completed)

        resp = client.patch(
            f"/api/v2/runs/{run.id}/segments/{rs.id}",
            data={"prompt": "new prompt"},
            files={"reference_files": ("a.jpg", io.BytesIO(b"img-bytes"), "image/jpeg")},
        )
        assert resp.status_code == 200, resp.text
        assert len(resp.json()["reference_image_urls_override"]) == 1

    def test_rerun_segment_with_one_reference_file(self, client, db_session):
        project = _make_project(db_session, status=ProjectStatus.ready)
        sd = _make_segment_def(db_session, project.id, 0)
        run = _make_run(db_session, project.id, status=RunStatus.done)
        rs = _make_run_segment(db_session, run.id, sd.id, status=SegmentStatus.completed)

        resp = client.post(
            f"/api/v2/runs/{run.id}/segments/{rs.id}/rerun",
            data={"prompt": "redo"},
            files={"reference_files": ("a.jpg", io.BytesIO(b"img-bytes"), "image/jpeg")},
        )
        assert resp.status_code == 200, resp.text

    def test_create_run_with_one_reference_file(self, spy_client, SessionFactory):
        client, spy = spy_client
        session = SessionFactory()
        project = _make_project(session, status=ProjectStatus.ready)
        session.close()

        resp = client.post(
            f"/api/v2/projects/{project.id}/runs",
            data={"prompt": "swap"},
            files={"reference_files": ("a.jpg", io.BytesIO(b"img-bytes"), "image/jpeg")},
        )
        assert resp.status_code == 201, resp.text
        run_id = resp.json()["run_id"]
        session = SessionFactory()
        run = session.get(Run, run_id)
        session.close()
        assert len(run.reference_image_urls) == 1


# ---------------------------------------------------------------------------
# POST /api/v2/runs/{rid}/copy — duplicate a run at a new resolution
# ---------------------------------------------------------------------------


class TestCopyRun:
    def test_copy_clones_config_and_enqueues(self, spy_client, SessionFactory):
        client, spy = spy_client
        s = SessionFactory()
        project = _make_project(s, status=ProjectStatus.ready)
        run = _make_run(
            s, project.id, name="Test", prompt="hello there",
            model="seedance", resolution="480p", audio_mode="seedance",
            status=RunStatus.done,
        )
        pid, rid = project.id, run.id
        s.close()

        resp = client.post(f"/api/v2/runs/{rid}/copy", data={"resolution": "1080p"})
        assert resp.status_code == 201, resp.text
        new_id = resp.json()["run_id"]
        assert resp.json()["status"] == "queued"
        assert new_id in spy["process_run"]
        assert new_id != rid

        s = SessionFactory()
        nr = s.get(Run, new_id)
        s.close()
        assert nr.project_id == pid
        assert nr.prompt == "hello there"
        assert nr.model == "seedance"
        assert nr.audio_mode == "seedance"
        assert nr.resolution == "1080p"
        assert nr.status == RunStatus.queued

    def test_copy_uses_custom_name(self, spy_client, SessionFactory):
        client, spy = spy_client
        s = SessionFactory()
        project = _make_project(s, status=ProjectStatus.ready)
        run = _make_run(s, project.id, resolution="480p", status=RunStatus.done)
        rid = run.id
        s.close()
        resp = client.post(
            f"/api/v2/runs/{rid}/copy", data={"resolution": "720p", "name": "Prod cut"}
        )
        assert resp.status_code == 201
        s = SessionFactory()
        nr = s.get(Run, resp.json()["run_id"])
        s.close()
        assert nr.name == "Prod cut"

    def test_copy_clones_segment_overrides(self, spy_client, SessionFactory):
        client, spy = spy_client
        s = SessionFactory()
        project = _make_project(s, status=ProjectStatus.ready)
        sd = _make_segment_def(s, project.id, 0)
        run = _make_run(s, project.id, resolution="480p", status=RunStatus.done)
        _make_run_segment(
            s, run.id, sd.id, status=SegmentStatus.completed,
            prompt_override="tuned segment prompt",
        )
        rid, sd_id = run.id, sd.id
        s.close()

        resp = client.post(f"/api/v2/runs/{rid}/copy", data={"resolution": "720p"})
        assert resp.status_code == 201
        new_id = resp.json()["run_id"]

        s = SessionFactory()
        nr = s.get(Run, new_id)
        overrides = [rs for rs in nr.run_segments if rs.prompt_override]
        s.close()
        assert len(overrides) == 1
        assert overrides[0].segment_def_id == sd_id
        assert overrides[0].prompt_override == "tuned segment prompt"

    def test_copy_clones_local_reference_file(self, spy_client, SessionFactory, tmp_path):
        client, spy = spy_client
        ref = tmp_path / "face.jpg"
        ref.write_bytes(b"img-bytes")
        s = SessionFactory()
        project = _make_project(s, status=ProjectStatus.ready)
        run = _make_run(
            s, project.id, resolution="480p", status=RunStatus.done,
            reference_image_urls=[str(ref)],
        )
        rid = run.id
        s.close()

        resp = client.post(f"/api/v2/runs/{rid}/copy", data={"resolution": "720p"})
        assert resp.status_code == 201
        s = SessionFactory()
        nr = s.get(Run, resp.json()["run_id"])
        new_refs = list(nr.reference_image_urls)
        s.close()
        assert len(new_refs) == 1
        assert new_refs[0] != str(ref)       # copied into the new run's dir
        assert os.path.exists(new_refs[0])   # and the copy is on disk

    def test_copy_with_new_reference_url_replaces_refs(self, spy_client, SessionFactory):
        """A new reference URL on copy replaces the run-level references."""
        client, spy = spy_client
        s = SessionFactory()
        project = _make_project(s, status=ProjectStatus.ready)
        run = _make_run(
            s, project.id, resolution="480p", status=RunStatus.done,
            reference_image_urls=["https://old.example/face1.jpg"],
        )
        rid = run.id
        s.close()

        resp = client.post(
            f"/api/v2/runs/{rid}/copy",
            data={"reference_urls": "https://new.example/newface.jpg"},
        )
        assert resp.status_code == 201, resp.text
        new_id = resp.json()["run_id"]
        assert new_id in spy["process_run"]

        s = SessionFactory()
        nr = s.get(Run, new_id)
        refs = list(nr.reference_image_urls)
        res = nr.resolution
        s.close()
        assert refs == ["https://new.example/newface.jpg"]
        # resolution omitted → defaults to the source run's resolution
        assert res == "480p"

    def test_copy_new_ref_drops_segment_photo_override_keeps_prompt(
        self, spy_client, SessionFactory
    ):
        """With a new photo, per-segment photo overrides are dropped (so every
        segment uses the new photo) but per-segment prompt tweaks are kept."""
        client, spy = spy_client
        s = SessionFactory()
        project = _make_project(s, status=ProjectStatus.ready)
        sd = _make_segment_def(s, project.id, 0)
        run = _make_run(
            s, project.id, resolution="480p", status=RunStatus.done,
            reference_image_urls=["https://old.example/face.jpg"],
        )
        _make_run_segment(
            s, run.id, sd.id, status=SegmentStatus.completed,
            prompt_override="tuned segment prompt",
            reference_image_urls_override=["https://old.example/seg-specific.jpg"],
        )
        rid = run.id
        s.close()

        resp = client.post(
            f"/api/v2/runs/{rid}/copy",
            data={"reference_urls": "https://new.example/newface.jpg"},
        )
        assert resp.status_code == 201, resp.text
        new_id = resp.json()["run_id"]

        s = SessionFactory()
        nr = s.get(Run, new_id)
        segs = list(nr.run_segments)
        run_refs = list(nr.reference_image_urls)
        s.close()
        assert run_refs == ["https://new.example/newface.jpg"]
        assert len(segs) == 1
        # prompt tweak preserved, photo override dropped (segment now inherits the
        # new run-level photo).
        assert segs[0].prompt_override == "tuned segment prompt"
        assert not segs[0].reference_image_urls_override

    def test_copy_no_resolution_no_ref_clones_same_resolution(
        self, spy_client, SessionFactory
    ):
        """Backward-compat: copy with neither resolution nor photo clones as-is."""
        client, spy = spy_client
        s = SessionFactory()
        project = _make_project(s, status=ProjectStatus.ready)
        run = _make_run(
            s, project.id, resolution="720p", status=RunStatus.done,
            reference_image_urls=["https://old.example/face.jpg"],
        )
        rid = run.id
        s.close()

        resp = client.post(f"/api/v2/runs/{rid}/copy", data={})
        assert resp.status_code == 201, resp.text
        s = SessionFactory()
        nr = s.get(Run, resp.json()["run_id"])
        res = nr.resolution
        refs = list(nr.reference_image_urls)
        s.close()
        assert res == "720p"
        assert refs == ["https://old.example/face.jpg"]

    def test_copy_too_many_new_refs_is_400(self, client, db_session):
        project = _make_project(db_session, status=ProjectStatus.ready)
        run = _make_run(db_session, project.id, resolution="480p", status=RunStatus.done)
        many = ",".join(f"https://e/{i}.jpg" for i in range(settings.MAX_REFERENCE_IMAGES + 1))
        resp = client.post(
            f"/api/v2/runs/{run.id}/copy", data={"reference_urls": many}
        )
        assert resp.status_code == 400
        assert "reference images" in resp.json()["detail"].lower()

    def test_copy_invalid_resolution_for_model_is_400(self, client, db_session):
        project = _make_project(db_session, status=ProjectStatus.ready)
        run = _make_run(
            db_session, project.id, model="gemini-omni", resolution="720p",
            status=RunStatus.done,
        )
        resp = client.post(f"/api/v2/runs/{run.id}/copy", data={"resolution": "480p"})
        assert resp.status_code == 400
        assert "resolution" in resp.json()["detail"].lower()

    def test_copy_blocked_when_project_not_ready(self, client, db_session):
        project = _make_project(db_session, status=ProjectStatus.created)
        run = _make_run(db_session, project.id, resolution="480p", status=RunStatus.done)
        resp = client.post(f"/api/v2/runs/{run.id}/copy", data={"resolution": "720p"})
        assert resp.status_code == 409

    def test_copy_missing_run_is_404(self, client):
        resp = client.post("/api/v2/runs/no-such-run/copy", data={"resolution": "720p"})
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# POST /api/v2/runs/{rid}/copy-batch — launch up to 10 copies in one request
# ---------------------------------------------------------------------------


class TestCopyRunBatch:
    def test_batch_creates_multiple_runs_and_enqueues(self, spy_client, SessionFactory):
        client, spy = spy_client
        s = SessionFactory()
        project = _make_project(s, status=ProjectStatus.ready)
        run = _make_run(
            s, project.id, name="Base", prompt="hi", model="seedance",
            resolution="480p", status=RunStatus.done,
        )
        pid, rid = project.id, run.id
        s.close()

        resp = client.post(
            f"/api/v2/runs/{rid}/copy-batch",
            data={
                "runs[0][resolution]": "480p", "runs[0][name]": "First",
                "runs[1][resolution]": "720p", "runs[1][name]": "Second",
                "runs[2][resolution]": "1080p",
            },
        )
        assert resp.status_code == 201, resp.text
        runs = resp.json()["runs"]
        assert len(runs) == 3
        assert all(r["status"] == "queued" for r in runs)
        ids = [r["run_id"] for r in runs]
        assert all(i in spy["process_run"] for i in ids)
        assert len(set(ids)) == 3 and rid not in ids

        s = SessionFactory()
        rows = [s.get(Run, i) for i in ids]
        names = [r.name for r in rows]
        resolutions = [r.resolution for r in rows]
        s.close()
        # Order is preserved (runs[0], runs[1], runs[2]).
        assert names[0] == "First" and names[1] == "Second"
        assert resolutions == ["480p", "720p", "1080p"]
        # Unnamed copy gets the default name suffix.
        assert names[2].startswith("Base ·")

    def test_batch_per_run_reference_url(self, spy_client, SessionFactory):
        client, spy = spy_client
        s = SessionFactory()
        project = _make_project(s, status=ProjectStatus.ready)
        run = _make_run(
            s, project.id, resolution="480p", status=RunStatus.done,
            reference_image_urls=["https://old.example/face.jpg"],
        )
        rid = run.id
        s.close()

        resp = client.post(
            f"/api/v2/runs/{rid}/copy-batch",
            data={
                "runs[0][resolution]": "480p",
                "runs[0][reference_urls]": "https://new.example/a.jpg",
                "runs[1][resolution]": "720p",  # no new photo → clones source refs
            },
        )
        assert resp.status_code == 201, resp.text
        ids = [r["run_id"] for r in resp.json()["runs"]]

        s = SessionFactory()
        refs0 = list(s.get(Run, ids[0]).reference_image_urls)
        refs1 = list(s.get(Run, ids[1]).reference_image_urls)
        s.close()
        assert refs0 == ["https://new.example/a.jpg"]
        assert refs1 == ["https://old.example/face.jpg"]

    def test_batch_with_uploaded_file(self, spy_client, SessionFactory):
        client, spy = spy_client
        s = SessionFactory()
        project = _make_project(s, status=ProjectStatus.ready)
        run = _make_run(s, project.id, resolution="480p", status=RunStatus.done)
        rid = run.id
        s.close()

        resp = client.post(
            f"/api/v2/runs/{rid}/copy-batch",
            data={"runs[0][resolution]": "720p"},
            files=[("runs[0][reference_files]", ("face.jpg", io.BytesIO(b"img"), "image/jpeg"))],
        )
        assert resp.status_code == 201, resp.text
        new_id = resp.json()["runs"][0]["run_id"]

        s = SessionFactory()
        refs = list(s.get(Run, new_id).reference_image_urls)
        s.close()
        assert len(refs) == 1
        assert os.path.exists(refs[0])

    def test_batch_empty_is_400(self, client, db_session):
        project = _make_project(db_session, status=ProjectStatus.ready)
        run = _make_run(db_session, project.id, resolution="480p", status=RunStatus.done)
        resp = client.post(f"/api/v2/runs/{run.id}/copy-batch", data={})
        assert resp.status_code == 400
        assert "no runs" in resp.json()["detail"].lower()

    def test_batch_too_many_is_400(self, client, db_session):
        project = _make_project(db_session, status=ProjectStatus.ready)
        run = _make_run(db_session, project.id, resolution="480p", status=RunStatus.done)
        data = {f"runs[{i}][resolution]": "480p" for i in range(11)}
        resp = client.post(f"/api/v2/runs/{run.id}/copy-batch", data=data)
        assert resp.status_code == 400
        assert "too many" in resp.json()["detail"].lower()

    def test_batch_invalid_spec_rolls_back_whole_batch(self, spy_client, SessionFactory):
        """One bad resolution fails the entire batch — no partial runs created."""
        client, spy = spy_client
        s = SessionFactory()
        project = _make_project(s, status=ProjectStatus.ready)
        run = _make_run(
            s, project.id, model="seedance", resolution="480p", status=RunStatus.done,
        )
        pid, rid = project.id, run.id
        s.close()

        resp = client.post(
            f"/api/v2/runs/{rid}/copy-batch",
            data={
                "runs[0][resolution]": "720p",   # valid
                "runs[1][resolution]": "4k",     # invalid for seedance
            },
        )
        assert resp.status_code == 400
        assert spy["process_run"] == []  # nothing enqueued

        s = SessionFactory()
        # Only the original run exists; neither copy was committed.
        count = len([r for r in s.query(Run).filter(Run.project_id == pid).all()])
        s.close()
        assert count == 1

    def test_batch_blocked_when_project_not_ready(self, client, db_session):
        project = _make_project(db_session, status=ProjectStatus.created)
        run = _make_run(db_session, project.id, resolution="480p", status=RunStatus.done)
        resp = client.post(
            f"/api/v2/runs/{run.id}/copy-batch", data={"runs[0][resolution]": "720p"}
        )
        assert resp.status_code == 409

    def test_batch_missing_run_is_404(self, client):
        resp = client.post(
            "/api/v2/runs/no-such-run/copy-batch", data={"runs[0][resolution]": "720p"}
        )
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Per-segment prompts supplied at run creation
# ---------------------------------------------------------------------------


class TestSegmentPromptsOnCreate:
    def test_segment_prompt_appended_as_override(self, client, db_session, SessionFactory):
        import json as _json

        project = _make_project(db_session, status=ProjectStatus.ready)
        sd0 = _make_segment_def(db_session, project.id, 0)
        sd1 = _make_segment_def(db_session, project.id, 1)

        resp = client.post(
            f"/api/v2/projects/{project.id}/runs",
            data={
                "prompt": "base prompt",
                "segment_prompts": _json.dumps({sd0.id: "make the jacket red"}),
            },
        )
        assert resp.status_code == 201, resp.text

        s = SessionFactory()
        run = s.get(Run, resp.json()["run_id"])
        rss = {rs.segment_def_id: rs for rs in run.run_segments}
        s.close()
        # Only the segment with extra text gets a pre-created override RunSegment.
        assert sd0.id in rss
        assert rss[sd0.id].prompt_override == "base prompt\nmake the jacket red"
        assert sd1.id not in rss

    def test_blank_and_unknown_segment_prompts_ignored(self, client, db_session, SessionFactory):
        import json as _json

        project = _make_project(db_session, status=ProjectStatus.ready)
        sd0 = _make_segment_def(db_session, project.id, 0)

        resp = client.post(
            f"/api/v2/projects/{project.id}/runs",
            data={
                "prompt": "base",
                "segment_prompts": _json.dumps({sd0.id: "   ", "no-such-id": "x"}),
            },
        )
        assert resp.status_code == 201
        s = SessionFactory()
        run = s.get(Run, resp.json()["run_id"])
        n = len(run.run_segments)
        s.close()
        assert n == 0  # blank text + unknown id both ignored

    def test_invalid_segment_prompts_json_is_400(self, client, db_session):
        project = _make_project(db_session, status=ProjectStatus.ready)
        resp = client.post(
            f"/api/v2/projects/{project.id}/runs",
            data={"prompt": "base", "segment_prompts": "{not valid json"},
        )
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# Public token-signed source link (/public/projects/{pid}/source)
# ---------------------------------------------------------------------------


class TestPublicSourceLink:
    def test_valid_token_streams_source(self, client, db_session, tmp_path):
        from app.public import make_source_token

        src = tmp_path / "source.mp4"
        src.write_bytes(b"\x00\x01videodata")
        project = _make_project(
            db_session, status=ProjectStatus.ready, source_local_path=str(src)
        )
        tok = make_source_token(project.id)

        r = client.get(f"/public/projects/{project.id}/source", params={"token": tok})
        assert r.status_code == 200
        assert r.content == b"\x00\x01videodata"
        assert r.headers["content-type"].startswith("video/mp4")

    def test_bad_token_is_403(self, client, db_session, tmp_path):
        src = tmp_path / "s.mp4"
        src.write_bytes(b"x")
        project = _make_project(
            db_session, status=ProjectStatus.ready, source_local_path=str(src)
        )
        r = client.get(
            f"/public/projects/{project.id}/source", params={"token": "deadbeef"}
        )
        assert r.status_code == 403

    def test_missing_token_is_422(self, client, db_session):
        project = _make_project(db_session, status=ProjectStatus.ready)
        r = client.get(f"/public/projects/{project.id}/source")
        assert r.status_code == 422

    def test_valid_token_missing_file_is_404(self, client, db_session):
        from app.public import make_source_token

        project = _make_project(
            db_session, status=ProjectStatus.ready, source_local_path="/nope/x.mp4"
        )
        tok = make_source_token(project.id)
        r = client.get(f"/public/projects/{project.id}/source", params={"token": tok})
        assert r.status_code == 404


class TestPublicResultLink:
    def test_valid_token_streams_result(self, client, db_session, tmp_path):
        from app.public import make_result_token

        out = tmp_path / "final.mp4"
        out.write_bytes(b"resultbytes")
        project = _make_project(db_session, status=ProjectStatus.ready)
        run = _make_run(
            db_session, project.id, status=RunStatus.done, result_local_path=str(out)
        )
        tok = make_result_token(run.id)

        r = client.get(f"/public/runs/{run.id}/result", params={"token": tok})
        assert r.status_code == 200
        assert r.content == b"resultbytes"
        assert r.headers["content-type"].startswith("video/mp4")
        assert r.headers["cache-control"] == "no-store, max-age=0"

    def test_bad_token_is_403(self, client, db_session, tmp_path):
        out = tmp_path / "final.mp4"
        out.write_bytes(b"x")
        project = _make_project(db_session, status=ProjectStatus.ready)
        run = _make_run(
            db_session, project.id, status=RunStatus.done, result_local_path=str(out)
        )
        r = client.get(f"/public/runs/{run.id}/result", params={"token": "nope"})
        assert r.status_code == 403

    def test_missing_result_file_is_404(self, client, db_session):
        from app.public import make_result_token

        project = _make_project(db_session, status=ProjectStatus.ready)
        run = _make_run(db_session, project.id, status=RunStatus.failed)
        tok = make_result_token(run.id)
        r = client.get(f"/public/runs/{run.id}/result", params={"token": tok})
        assert r.status_code == 404


# ---------------------------------------------------------------------------
# Per-project segmentation cap (max_segment_sec)
# ---------------------------------------------------------------------------


class TestMaxSegmentSec:
    """The analyze-time segmentation cap: settable at create, patchable after.

    NULL is a real value meaning "use the universal default", so it must survive
    a round-trip as NULL rather than being coerced to a number — freezing today's
    default into every row would silently diverge if the registry changes.
    """

    _LINK = "https://drive.google.com/file/d/FAKE_ID/view"

    def _create(self, client, **data):
        return client.post(
            "/api/v2/projects", data={"gdrive_link": self._LINK, **data}
        )

    def test_omitted_stays_null(self, client):
        resp = self._create(client)
        assert resp.status_code == 201
        pid = resp.json()["project_id"]
        assert client.get(f"/api/v2/projects/{pid}").json()["max_segment_sec"] is None

    def test_accepted_at_create(self, client):
        resp = self._create(client, max_segment_sec=30)
        assert resp.status_code == 201
        pid = resp.json()["project_id"]
        assert client.get(f"/api/v2/projects/{pid}").json()["max_segment_sec"] == 30.0

    def test_above_the_longest_model_is_400(self, client):
        """No backend could generate a 45s clip, so this is rejected rather than
        clamped — silently halving what the operator asked for is worse."""
        resp = self._create(client, max_segment_sec=45)
        assert resp.status_code == 400
        assert "max_segment_sec" in resp.json()["detail"]

    def test_below_the_shortest_generatable_clip_is_400(self, client):
        """A cap under every model's min_duration_sec is still billed at that
        floor per segment, so a 2s cap quietly multiplies spend."""
        resp = self._create(client, max_segment_sec=2)
        assert resp.status_code == 400
        assert "max_segment_sec" in resp.json()["detail"]

    def test_nan_is_400(self, client):
        """NaN parses as a float and Postgres stores it happily; analysis then
        dies in math.ceil(dur / nan) and every re-analysis repeats it."""
        resp = self._create(client, max_segment_sec="nan")
        assert resp.status_code == 400

    def test_infinity_is_400(self, client):
        resp = self._create(client, max_segment_sec="inf")
        assert resp.status_code == 400

    def test_patch_sets_it(self, client):
        pid = self._create(client).json()["project_id"]
        resp = client.patch(
            f"/api/v2/projects/{pid}", json={"max_segment_sec": 15}
        )
        assert resp.status_code == 200
        assert resp.json()["max_segment_sec"] == 15.0

    def test_patch_null_clears_it(self, client):
        """Explicit null means "back to the universal default", which is
        different from omitting the field."""
        pid = self._create(client, max_segment_sec=30).json()["project_id"]
        resp = client.patch(
            f"/api/v2/projects/{pid}", json={"max_segment_sec": None}
        )
        assert resp.status_code == 200
        assert resp.json()["max_segment_sec"] is None

    def test_patch_omitting_the_field_leaves_it_alone(self, client):
        """A name-only PATCH must not wipe the cap — the whole reason the
        endpoint checks model_fields_set instead of `is not None`."""
        pid = self._create(client, max_segment_sec=30).json()["project_id"]
        resp = client.patch(f"/api/v2/projects/{pid}", json={"name": "renamed"})
        assert resp.status_code == 200
        assert resp.json()["name"] == "renamed"
        assert resp.json()["max_segment_sec"] == 30.0

    def test_patch_out_of_range_is_400_and_does_not_persist(self, client):
        pid = self._create(client, max_segment_sec=15).json()["project_id"]
        assert client.patch(
            f"/api/v2/projects/{pid}", json={"max_segment_sec": 999}
        ).status_code == 400
        assert client.get(
            f"/api/v2/projects/{pid}"
        ).json()["max_segment_sec"] == 15.0


# ---------------------------------------------------------------------------
# create_run — Seedance 2.5 as a selectable model
# ---------------------------------------------------------------------------


class TestCreateRunSeedance25:
    @pytest.fixture()
    def ready_project(self, SessionFactory, tmp_path):
        src = tmp_path / "src.mp4"
        src.write_bytes(_tiny_video_bytes())
        with SessionFactory() as s:
            project = VideoProject(
                source_type="upload",
                source_ref="src.mp4",
                source_local_path=str(src),
                status=ProjectStatus.ready,
                duration_sec=30.0,
                width=1080,
                height=1920,
                fps=25.0,
                aspect_ratio="9:16",
            )
            s.add(project)
            s.flush()
            s.add(
                SegmentDef(project_id=project.id, index=0, start_sec=0.0,
                           end_sec=10.0, has_face=True, action="swap")
            )
            s.commit()
            return project.id

    def _create_run(self, client, pid, **data):
        return client.post(
            f"/api/v2/projects/{pid}/runs",
            data={"name": "c1", "prompt": "swap", **data},
        )

    def test_seedance_2_5_is_accepted(self, client, ready_project, enqueue_spy):
        resp = self._create_run(
            client, ready_project, model="seedance-2-5", resolution="720p"
        )
        assert resp.status_code == 201, resp.text
        rid = resp.json()["run_id"]
        run = client.get(f"/api/v2/runs/{rid}").json()
        assert run["model"] == "seedance-2-5"
        assert run["resolution"] == "720p"

    def test_1080p_is_rejected_for_seedance_2_5(
        self, client, ready_project, enqueue_spy
    ):
        """2.5 tops out at 720p. Rejecting at the API is better than silently
        coercing: the operator asked for production quality and would not know
        they did not get it."""
        resp = self._create_run(
            client, ready_project, model="seedance-2-5", resolution="1080p"
        )
        assert resp.status_code == 400
        assert "1080p" in resp.json()["detail"]

    def test_seedance_2_5_keeps_a_requested_generated_audio_mode(
        self, client, ready_project, enqueue_spy
    ):
        """Unlike Gemini Omni, 2.5 does emit audio, so audio_mode is not forced."""
        resp = self._create_run(
            client, ready_project, model="seedance-2-5", resolution="720p",
            audio_mode="seedance",
        )
        assert resp.status_code == 201
        rid = resp.json()["run_id"]
        assert client.get(f"/api/v2/runs/{rid}").json()["audio_mode"] == "seedance"

    def test_unknown_model_is_400(self, client, ready_project, enqueue_spy):
        resp = self._create_run(
            client, ready_project, model="seedance-9", resolution="720p"
        )
        assert resp.status_code == 400
        assert "model must be one of" in resp.json()["detail"]


def _line(line_id, start, end, text, speaker="the woman in the red jacket", **extra):
    """One docs/localisation.md §4.1 transcript line."""
    line = {
        "id": line_id,
        "start": start,
        "end": end,
        "speaker": speaker,
        "on_screen": True,
        "text": text,
    }
    line.update(extra)
    return line


def _transcript(lines, *, source_language="en", **extra):
    """A §4.1 transcript envelope around *lines*."""
    doc = {
        "schema_version": 1,
        "model": "gemini-2.5-pro",
        "prompt_version": "transcribe/v1",
        "created_at": "2026-08-09T12:00:00+00:00",
        "source_language": source_language,
        "lines": lines,
        "on_screen_text": [],
    }
    doc.update(extra)
    return doc


# Two short lines inside a 10s hook. Lengths are chosen so the crude
# chars-per-second estimate in api_v2._localisation_warnings finds them
# plausible in a Latin-script target — a warning here would be noise, and the
# warning tests below make their own over/under-long lines on purpose.
_HOOK_LINES = [
    _line(1, 0.0, 2.5, "Hey, what are you doing?"),
    _line(2, 2.5, 6.0, "I'm just listening to my emails.", speaker="the man"),
]


def _localisation_project(session, **kwargs):
    """A ready localisation project whose transcript is ready to translate."""
    defaults = dict(
        status=ProjectStatus.ready,
        project_type="localisation",
        source_local_path="/tmp/source.mp4",
        transcript=_transcript(_HOOK_LINES),
        transcript_status="ready",
    )
    defaults.update(kwargs)
    return _make_project(session, **defaults)


@pytest.fixture()
def fake_translate(monkeypatch):
    """Replace the kie.ai translation call — no live network in tests.

    Honours the real contract of localisation.translate_lines (same ids, same
    order, ``text`` replaced, ``source_text`` carrying the original) so the
    endpoint is exercised exactly as it would be in production. Returns the
    list of calls it received. Accepts the optional video_summary /
    scene_context kwargs so a real caller that forwards cached context does
    not TypeError the stub.
    """
    calls: list[dict] = []

    def _translate(
        lines,
        *,
        source_language,
        target_language,
        model=None,
        video_summary="",
        scene_context="",
    ):
        calls.append(
            {
                "lines": lines,
                "source_language": source_language,
                "target_language": target_language,
                "video_summary": video_summary,
                "scene_context": scene_context,
            }
        )
        out = []
        for line in lines:
            translated = dict(line)
            translated["source_text"] = line["text"]
            translated["text"] = f"<{target_language}> {line['text']}"
            out.append(translated)
        return out

    monkeypatch.setattr(api_v2_module.localisation, "translate_lines", _translate)
    return calls


class TestProjectHookSec:
    """The analyze-time hook length (docs/localisation.md §3.1, §5).

    Same contract as max_segment_sec: NULL means "no explicit choice" (analysis
    falls back to settings.LOCALISATION_DEFAULT_HOOK_SEC) and the value is
    consumed at ANALYZE time, so changing it re-cuts nothing. Unlike
    max_segment_sec it has no upper bound — a hook longer than the video is
    clamped at analyze time, not rejected — but 0 and NaN are refused, because
    neither can ever produce a segment.
    """

    DEFAULT = settings.LOCALISATION_DEFAULT_HOOK_SEC

    # -- POST /api/v2/projects -------------------------------------------

    def test_omitted_is_stored_as_null(self, spy_client, SessionFactory):
        """NULL, not the resolved default — the default stays in one place."""
        client, _spy = spy_client
        response = client.post(
            "/api/v2/projects",
            data={"project_type": "localisation"},
            files={"video_file": ("clip.mp4", io.BytesIO(_tiny_video_bytes()), "video/mp4")},
        )
        assert response.status_code == 201, response.text

        session = SessionFactory()
        project = session.get(VideoProject, response.json()["project_id"])
        session.close()
        assert project.hook_sec is None

    def test_localisation_is_marked_pending_from_creation(
        self, spy_client, SessionFactory
    ):
        """NULL must mean exactly "nobody asked for a transcript".

        A localisation project is transcribed automatically once analysis
        returns, so leaving transcript_status NULL for the whole analyze window
        made NULL mean two things at once — and the panel, reading it, offered
        a Transcribe button next to a job that was already promised. One click
        there paid for a second full model call on the same video.
        """
        client, _spy = spy_client
        response = client.post(
            "/api/v2/projects",
            data={"project_type": "localisation"},
            files={"video_file": ("clip.mp4", io.BytesIO(_tiny_video_bytes()), "video/mp4")},
        )
        assert response.status_code == 201, response.text

        session = SessionFactory()
        project = session.get(VideoProject, response.json()["project_id"])
        session.close()
        assert project.transcript_status == "pending"
        assert project.transcript is None

    def test_other_project_types_are_not_marked_pending(
        self, spy_client, SessionFactory
    ):
        """Nothing transcribes a face-swap project, so a "pending" there would
        poll a panel that is never rendered and never resolves."""
        client, _spy = spy_client
        response = client.post(
            "/api/v2/projects",
            data={"project_type": "face_swap"},
            files={"video_file": ("clip.mp4", io.BytesIO(_tiny_video_bytes()), "video/mp4")},
        )
        assert response.status_code == 201, response.text

        session = SessionFactory()
        project = session.get(VideoProject, response.json()["project_id"])
        session.close()
        assert project.transcript_status is None

    def test_create_with_explicit_hook(self, spy_client, SessionFactory):
        client, _spy = spy_client
        response = client.post(
            "/api/v2/projects",
            data={"project_type": "localisation", "hook_sec": 12.5},
            files={"video_file": ("clip.mp4", io.BytesIO(_tiny_video_bytes()), "video/mp4")},
        )
        assert response.status_code == 201, response.text

        session = SessionFactory()
        project = session.get(VideoProject, response.json()["project_id"])
        session.close()
        assert project.hook_sec == 12.5

    def test_blank_form_value_is_stored_as_null(self, spy_client, SessionFactory):
        """The create form posts its input verbatim: "" must mean unset, not 0."""
        client, _spy = spy_client
        response = client.post(
            "/api/v2/projects",
            data={"project_type": "localisation", "hook_sec": ""},
            files={"video_file": ("clip.mp4", io.BytesIO(_tiny_video_bytes()), "video/mp4")},
        )
        assert response.status_code == 201, response.text

        session = SessionFactory()
        project = session.get(VideoProject, response.json()["project_id"])
        session.close()
        assert project.hook_sec is None

    def test_create_gdrive_with_hook(self, spy_client, SessionFactory):
        client, _spy = spy_client
        response = client.post(
            "/api/v2/projects",
            data={
                "gdrive_link": "https://drive.google.com/file/d/FAKE_ID/view",
                "project_type": "localisation",
                "hook_sec": 8,
            },
        )
        assert response.status_code == 201, response.text

        session = SessionFactory()
        project = session.get(VideoProject, response.json()["project_id"])
        session.close()
        assert project.hook_sec == 8.0

    def test_zero_is_a_value_and_is_rejected(self, client, SessionFactory):
        """0.0 is NOT "unset": a zero-length hook has no speech and no segment.

        This is the whole reason blank/absent maps to None rather than 0.0 —
        the two have to stay distinguishable, and only one of them is legal.
        """
        response = client.post(
            "/api/v2/projects",
            data={"project_type": "localisation", "hook_sec": 0},
            files={"video_file": ("clip.mp4", io.BytesIO(_tiny_video_bytes()), "video/mp4")},
        )
        assert response.status_code == 400
        assert "hook_sec" in response.json()["detail"]

    @pytest.mark.parametrize("value", [-1, -0.5, "nan", "inf", "-inf"])
    def test_create_out_of_range_is_400(self, client, value):
        response = client.post(
            "/api/v2/projects",
            data={"project_type": "localisation", "hook_sec": value},
            files={"video_file": ("clip.mp4", io.BytesIO(_tiny_video_bytes()), "video/mp4")},
        )
        assert response.status_code == 400, response.text
        assert "hook_sec" in response.json()["detail"]

    def test_a_hook_longer_than_any_segment_cap_is_allowed(
        self, spy_client, SessionFactory
    ):
        """No ceiling: analysis clamps a too-long hook, it is not a bad request."""
        client, _spy = spy_client
        response = client.post(
            "/api/v2/projects",
            data={"project_type": "localisation", "hook_sec": 600},
            files={"video_file": ("clip.mp4", io.BytesIO(_tiny_video_bytes()), "video/mp4")},
        )
        assert response.status_code == 201, response.text

        session = SessionFactory()
        project = session.get(VideoProject, response.json()["project_id"])
        session.close()
        assert project.hook_sec == 600.0

    # -- GET / PATCH /api/v2/projects/{pid} -------------------------------

    def test_get_exposes_the_field(self, client, db_session):
        project = _make_project(db_session, status=ProjectStatus.ready, hook_sec=7.0)
        body = client.get(f"/api/v2/projects/{project.id}").json()
        assert body["hook_sec"] == 7.0

    def test_patch_sets_and_persists(self, client, db_session):
        project = _make_project(db_session, status=ProjectStatus.ready)
        resp = client.patch(f"/api/v2/projects/{project.id}", json={"hook_sec": 12})
        assert resp.status_code == 200, resp.text
        assert resp.json()["hook_sec"] == 12.0
        assert client.get(f"/api/v2/projects/{project.id}").json()["hook_sec"] == 12.0

    def test_patch_explicit_null_clears_it(self, client, db_session):
        """`{"hook_sec": null}` resets to the settings default.

        NULL is a value here, not "omitted" — update_project tells the two apart
        via model_fields_set, exactly as it does for max_segment_sec.
        """
        project = _make_project(db_session, status=ProjectStatus.ready, hook_sec=12.0)
        resp = client.patch(f"/api/v2/projects/{project.id}", json={"hook_sec": None})
        assert resp.status_code == 200, resp.text
        assert resp.json()["hook_sec"] is None

        db_session.expire_all()
        assert db_session.get(VideoProject, project.id).hook_sec is None

    def test_patch_without_the_key_leaves_it_untouched(self, client, db_session):
        project = _make_project(db_session, status=ProjectStatus.ready, hook_sec=12.0)
        resp = client.patch(f"/api/v2/projects/{project.id}", json={"name": "renamed"})
        assert resp.status_code == 200
        assert resp.json()["hook_sec"] == 12.0

    @pytest.mark.parametrize("value", [0, -3])
    def test_patch_out_of_range_is_400_and_stores_nothing(
        self, client, db_session, value
    ):
        project = _make_project(db_session, status=ProjectStatus.ready, hook_sec=12.0)
        resp = client.patch(
            f"/api/v2/projects/{project.id}", json={"hook_sec": value}
        )
        assert resp.status_code == 400, resp.text
        assert "hook_sec" in resp.json()["detail"]

        db_session.expire_all()
        assert db_session.get(VideoProject, project.id).hook_sec == 12.0

    @pytest.mark.parametrize(
        "body",
        [
            b'{"hook_sec": NaN}',
            b'{"hook_sec": "nan"}',
            b'{"hook_sec": Infinity}',
            b'{"hook_sec": "inf"}',
            b'{"hook_sec": -Infinity}',
        ],
    )
    def test_patch_nan_and_infinity_are_400(self, client, db_session, body):
        """Same trap max_segment_sec fell into: Pydantic parses these to floats.

        A NaN hook survives every `<` comparison in slice_lines (so the hook
        window silently comes back empty) and poisons the analyze-time clamp.
        The chained-comparison guard rejects it; a `value <= 0 or value == inf`
        guard would not have.
        """
        project = _make_project(db_session, status=ProjectStatus.ready, hook_sec=12.0)
        resp = client.patch(
            f"/api/v2/projects/{project.id}",
            content=body,
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 400, resp.text
        assert "hook_sec" in resp.json()["detail"]

        db_session.expire_all()
        assert db_session.get(VideoProject, project.id).hook_sec == 12.0

    def test_patch_does_not_reanalyze_or_recut_segments(self, spy_client, db_session):
        """hook_sec is consumed at ANALYZE time — a PATCH is inert on an
        already-analyzed project: no re-enqueue, no re-cut. Same contract as
        max_segment_sec, and the reason both docstrings say so.
        """
        client, spy = spy_client
        project = _make_project(
            db_session, status=ProjectStatus.ready, project_type="localisation",
            hook_sec=10.0,
        )
        _make_segment_def(db_session, project.id, 0, start_sec=0.0, end_sec=10.0)
        _make_segment_def(
            db_session, project.id, 1, start_sec=10.0, end_sec=30.0,
            action="keep", has_face=False,
        )
        before = [
            (sd.index, sd.start_sec, sd.end_sec, sd.action)
            for sd in sorted(project.segments, key=lambda s: s.index)
        ]

        resp = client.patch(f"/api/v2/projects/{project.id}", json={"hook_sec": 20})
        assert resp.status_code == 200
        assert resp.json()["hook_sec"] == 20.0

        assert spy["analyze_project"] == []

        db_session.expire_all()
        after_project = db_session.get(VideoProject, project.id)
        after = [
            (sd.index, sd.start_sec, sd.end_sec, sd.action)
            for sd in sorted(after_project.segments, key=lambda s: s.index)
        ]
        assert after == before


class TestTranscribeProject:
    """POST /projects/{pid}/transcribe — 202, status=pending, enqueue."""

    def test_enqueues_and_marks_pending(self, spy_client, db_session):
        client, spy = spy_client
        project = _make_project(
            db_session,
            status=ProjectStatus.ready,
            source_local_path="/tmp/source.mp4",
            transcript_status="failed",
            transcript_error="kie.ai timed out",
        )

        resp = client.post(f"/api/v2/projects/{project.id}/transcribe")
        assert resp.status_code == 202, resp.text
        body = resp.json()
        assert body["status"] == "pending"
        assert body["error"] is None

        assert spy["transcribe_project"] == [project.id]

        db_session.expire_all()
        stored = db_session.get(VideoProject, project.id)
        assert stored.transcript_status == "pending"
        assert stored.transcript_error is None

    def test_rerun_keeps_the_previous_transcript_readable(self, spy_client, db_session):
        """"Распознать заново" must not blank the panel while the job runs."""
        client, _spy = spy_client
        project = _localisation_project(db_session)

        resp = client.post(f"/api/v2/projects/{project.id}/transcribe")
        assert resp.status_code == 202
        assert resp.json()["status"] == "pending"
        assert resp.json()["transcript"]["lines"][0]["text"] == _HOOK_LINES[0]["text"]

    def test_no_local_source_is_409(self, spy_client, db_session):
        """A Drive project before analysis has nothing on disk to transcribe."""
        client, spy = spy_client
        project = _make_project(
            db_session, source_type="gdrive", source_ref="https://drive/x",
            source_local_path=None,
        )

        resp = client.post(f"/api/v2/projects/{project.id}/transcribe")
        assert resp.status_code == 409, resp.text
        assert spy["transcribe_project"] == []

        db_session.expire_all()
        assert db_session.get(VideoProject, project.id).transcript_status is None

    @pytest.mark.parametrize(
        "project_status",
        [
            ProjectStatus.created,
            ProjectStatus.analyzing,
            ProjectStatus.ready,
            ProjectStatus.failed,
        ],
    )
    def test_409_detail_never_leaks_a_python_repr(
        self, spy_client, db_session, project_status
    ):
        """The panel puts this detail straight into a browser alert.

        `f"{project.status!r}"` renders `<ProjectStatus.ready: 'ready'>` — a
        Python repr shown to an operator. The bare value is what belongs there.
        """
        client, _spy = spy_client
        project = _make_project(
            db_session, source_type="gdrive", source_ref="https://drive/x",
            source_local_path=None, status=project_status,
        )

        detail = client.post(
            f"/api/v2/projects/{project.id}/transcribe"
        ).json()["detail"]
        assert "ProjectStatus" not in detail
        assert "<" not in detail
        assert f"project status: {project_status.value}" in detail

    def test_409_before_analysis_says_to_wait(self, spy_client, db_session):
        client, _spy = spy_client
        project = _make_project(
            db_session, source_type="gdrive", source_ref="https://drive/x",
            source_local_path=None, status=ProjectStatus.analyzing,
        )
        detail = client.post(
            f"/api/v2/projects/{project.id}/transcribe"
        ).json()["detail"]
        assert "wait" in detail

    @pytest.mark.parametrize(
        "project_status", [ProjectStatus.ready, ProjectStatus.failed]
    )
    def test_409_after_analysis_points_at_re_analysis_not_at_waiting(
        self, spy_client, db_session, project_status
    ):
        """Telling an operator to "wait for analysis" on a project that is
        already `ready` is advice they can never act on: analysis has finished
        and no file arrived. Re-analysis is what fetches it again."""
        client, _spy = spy_client
        project = _make_project(
            db_session, source_type="gdrive", source_ref="https://drive/x",
            source_local_path=None, status=project_status,
        )
        detail = client.post(
            f"/api/v2/projects/{project.id}/transcribe"
        ).json()["detail"]
        assert "wait" not in detail
        assert f"/api/v2/projects/{project.id}/analyze" in detail

    def test_unknown_project_is_404(self, client):
        assert client.post("/api/v2/projects/nope/transcribe").status_code == 404


class TestGetTranscript:
    def test_never_requested_is_all_null(self, client, db_session):
        project = _make_project(db_session, status=ProjectStatus.ready)
        resp = client.get(f"/api/v2/projects/{project.id}/transcript")
        assert resp.status_code == 200
        assert resp.json() == {"status": None, "error": None, "transcript": None}

    def test_returns_the_stored_transcript_verbatim(self, client, db_session):
        project = _localisation_project(db_session)
        body = client.get(f"/api/v2/projects/{project.id}/transcript").json()
        assert body["status"] == "ready"
        assert body["transcript"] == _transcript(_HOOK_LINES)

    def test_failed_surfaces_the_error(self, client, db_session):
        project = _make_project(
            db_session, transcript_status="failed", transcript_error="upload failed"
        )
        body = client.get(f"/api/v2/projects/{project.id}/transcript").json()
        assert body["status"] == "failed"
        assert body["error"] == "upload failed"

    def test_empty_is_a_status_not_an_error(self, client, db_session):
        """No speech is a legal outcome — status "empty", no error text."""
        project = _make_project(
            db_session,
            transcript_status="empty",
            transcript=_transcript([], source_language="en"),
        )
        body = client.get(f"/api/v2/projects/{project.id}/transcript").json()
        assert body["status"] == "empty"
        assert body["error"] is None
        assert body["transcript"]["lines"] == []

    def test_unknown_project_is_404(self, client):
        assert client.get("/api/v2/projects/nope/transcript").status_code == 404


class TestPatchTranscript:
    """Operator fix-ups. The transcript becomes a Seedance prompt, so a
    malformed edit is refused (400) rather than stored and discovered later."""

    def test_accepts_a_corrected_transcript(self, client, db_session):
        project = _make_project(
            db_session, transcript_status="failed", transcript_error="kie.ai 500"
        )
        fixed = _transcript(
            [_line(1, 0.0, 2.0, "Эй, что ты делаешь?", speaker="the woman")],
            source_language="ru",
        )

        resp = client.patch(f"/api/v2/projects/{project.id}/transcript", json=fixed)
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["status"] == "ready"
        # A hand-fixed transcript clears the failure that prompted the edit.
        assert body["error"] is None
        assert body["transcript"]["lines"][0]["text"] == "Эй, что ты делаешь?"
        assert body["transcript"]["source_language"] == "ru"

        db_session.expire_all()
        stored = db_session.get(VideoProject, project.id)
        assert stored.transcript_status == "ready"
        assert stored.transcript_error is None
        assert stored.transcript["lines"][0]["speaker"] == "the woman"

    def test_no_lines_is_stored_as_empty(self, client, db_session):
        project = _localisation_project(db_session)
        resp = client.patch(
            f"/api/v2/projects/{project.id}/transcript", json=_transcript([])
        )
        assert resp.status_code == 200, resp.text
        assert resp.json()["status"] == "empty"

    def test_envelope_keys_survive_the_round_trip(self, client, db_session):
        """model/prompt_version/created_at record which model produced it —
        an operator edit must not strip them."""
        project = _localisation_project(db_session)
        doc = _transcript(_HOOK_LINES, extra_key="kept")
        body = client.patch(
            f"/api/v2/projects/{project.id}/transcript", json=doc
        ).json()
        assert body["transcript"]["model"] == "gemini-2.5-pro"
        assert body["transcript"]["prompt_version"] == "transcribe/v1"
        assert body["transcript"]["created_at"] == "2026-08-09T12:00:00+00:00"
        assert body["transcript"]["extra_key"] == "kept"

    def test_line_order_is_preserved_not_resorted(self, client, db_session):
        """format_dialogue merges consecutive same-speaker lines, so re-ordering
        would silently change the prompt the operator just proof-read."""
        project = _localisation_project(db_session)
        doc = _transcript(
            [_line(1, 5.0, 6.0, "second half"), _line(2, 0.0, 1.0, "first half")]
        )
        body = client.patch(
            f"/api/v2/projects/{project.id}/transcript", json=doc
        ).json()
        assert [ln["id"] for ln in body["transcript"]["lines"]] == [1, 2]

    def test_on_screen_defaults_to_false_when_omitted(self, client, db_session):
        project = _localisation_project(db_session)
        line = _line(1, 0.0, 1.0, "hi")
        line.pop("on_screen")
        body = client.patch(
            f"/api/v2/projects/{project.id}/transcript", json=_transcript([line])
        ).json()
        assert body["transcript"]["lines"][0]["on_screen"] is False

    def test_on_screen_text_is_validated_and_kept(self, client, db_session):
        project = _localisation_project(db_session)
        doc = _transcript(
            _HOOK_LINES,
            on_screen_text=[{"start": 0.0, "end": 3.0, "text": "Speechify"}],
        )
        body = client.patch(
            f"/api/v2/projects/{project.id}/transcript", json=doc
        ).json()
        assert body["transcript"]["on_screen_text"] == [
            {"start": 0.0, "end": 3.0, "text": "Speechify"}
        ]

    @pytest.mark.parametrize(
        "payload, expect_in_detail",
        [
            ([{"id": 1}], "lines"),                                  # not an object
            ({"source_language": "en"}, "lines"),                    # no lines key
            ({"lines": "nope"}, "lines"),                            # lines not a list
            ({"lines": ["nope"]}, "lines[0]"),                       # line not an object
            ({"lines": [{"start": 0, "end": 1, "speaker": "a", "text": "t"}]}, "id"),
            ({"lines": [{"id": "1", "start": 0, "end": 1, "speaker": "a", "text": "t"}]}, "id"),
            ({"lines": [{"id": 0, "start": 0, "end": 1, "speaker": "a", "text": "t"}]}, "id"),
            ({"lines": [{"id": True, "start": 0, "end": 1, "speaker": "a", "text": "t"}]}, "id"),
            ({"lines": [{"id": 1, "start": 2, "end": 1, "speaker": "a", "text": "t"}]}, "end"),
            ({"lines": [{"id": 1, "start": -1, "end": 1, "speaker": "a", "text": "t"}]}, "start"),
            ({"lines": [{"id": 1, "start": "x", "end": 1, "speaker": "a", "text": "t"}]}, "start"),
            ({"lines": [{"id": 1, "end": 1, "speaker": "a", "text": "t"}]}, "start"),
            ({"lines": [{"id": 1, "start": 0, "end": 1, "speaker": "a", "text": "  "}]}, "text"),
            ({"lines": [{"id": 1, "start": 0, "end": 1, "speaker": "a"}]}, "text"),
            ({"lines": [{"id": 1, "start": 0, "end": 1, "text": "t"}]}, "speaker"),
            ({"lines": [{"id": 1, "start": 0, "end": 1, "speaker": "", "text": "t"}]}, "speaker"),
            ({"lines": [], "on_screen_text": "nope"}, "on_screen_text"),
            ({"lines": [], "on_screen_text": [{"start": 3, "end": 1, "text": "x"}]}, "on_screen_text[0]"),
            ({"lines": [], "source_language": 5}, "source_language"),
        ],
    )
    def test_garbage_is_400(self, client, db_session, payload, expect_in_detail):
        project = _localisation_project(db_session)
        resp = client.patch(
            f"/api/v2/projects/{project.id}/transcript", json=payload
        )
        assert resp.status_code == 400, resp.text
        assert expect_in_detail in resp.json()["detail"]

        # …and the previous transcript is still the one on file.
        db_session.expire_all()
        stored = db_session.get(VideoProject, project.id)
        assert stored.transcript == _transcript(_HOOK_LINES)
        assert stored.transcript_status == "ready"

    def test_preserves_and_strips_video_context_strings(self, client, db_session):
        project = _localisation_project(db_session)
        doc = _transcript(
            _HOOK_LINES,
            video_summary="  A promo for Speechify.  ",
            scene_context="  Woman holds a phone; 'this' is the app.  ",
        )
        body = client.patch(
            f"/api/v2/projects/{project.id}/transcript", json=doc
        ).json()
        assert body["transcript"]["video_summary"] == "A promo for Speechify."
        assert body["transcript"]["scene_context"] == (
            "Woman holds a phone; 'this' is the app."
        )

    def test_omitted_video_context_becomes_empty_string(self, client, db_session):
        """A UI that only edits lines still round-trips; missing keys → ""."""
        project = _localisation_project(db_session)
        doc = _transcript(_HOOK_LINES)
        assert "video_summary" not in doc
        assert "scene_context" not in doc
        body = client.patch(
            f"/api/v2/projects/{project.id}/transcript", json=doc
        ).json()
        assert body["transcript"]["video_summary"] == ""
        assert body["transcript"]["scene_context"] == ""

    @pytest.mark.parametrize(
        "field, value",
        [
            ("video_summary", ["not", "a", "string"]),
            ("video_summary", {"nested": True}),
            ("video_summary", 12),
            ("video_summary", True),
            ("scene_context", ["nope"]),
            ("scene_context", {"x": 1}),
            ("scene_context", 3.14),
            ("scene_context", False),
        ],
    )
    def test_non_string_video_context_is_400(self, client, db_session, field, value):
        project = _localisation_project(db_session)
        doc = _transcript(_HOOK_LINES, **{field: value})
        resp = client.patch(f"/api/v2/projects/{project.id}/transcript", json=doc)
        assert resp.status_code == 400, resp.text
        assert field in resp.json()["detail"]
        assert "must be a string" in resp.json()["detail"]

    def test_duplicate_ids_are_400(self, client, db_session):
        """Translation rejoins on the id — a duplicate loses a line silently."""
        project = _localisation_project(db_session)
        doc = _transcript([_line(1, 0.0, 1.0, "a"), _line(1, 1.0, 2.0, "b")])
        resp = client.patch(f"/api/v2/projects/{project.id}/transcript", json=doc)
        assert resp.status_code == 400
        assert "duplicate" in resp.json()["detail"]

    def test_nan_timestamp_is_400(self, client, db_session):
        """NaN survives every `<` in slice_lines, so the line would vanish from
        the hook window with nothing to show for it."""
        project = _localisation_project(db_session)
        body = (
            b'{"lines": [{"id": 1, "start": NaN, "end": 1.0, '
            b'"speaker": "a", "text": "t"}]}'
        )
        resp = client.patch(
            f"/api/v2/projects/{project.id}/transcript",
            content=body,
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 400, resp.text
        assert "finite" in resp.json()["detail"]

    def test_accepts_a_transcript_on_a_never_transcribed_project(
        self, client, db_session
    ):
        """Pasting one in by hand is the documented fallback (§8)."""
        project = _make_project(db_session, status=ProjectStatus.ready)
        resp = client.patch(
            f"/api/v2/projects/{project.id}/transcript", json=_transcript(_HOOK_LINES)
        )
        assert resp.status_code == 200, resp.text
        assert resp.json()["status"] == "ready"

    def test_unknown_project_is_404(self, client):
        resp = client.patch(
            "/api/v2/projects/nope/transcript", json=_transcript(_HOOK_LINES)
        )
        assert resp.status_code == 404


class TestLocalisationPrompt:
    """POST /projects/{pid}/localisation-prompt (docs/localisation.md §4.4).

    Every test stubs localisation.translate_lines — the real one is a kie.ai
    call, and nothing here may touch the network. build_prompt is NOT stubbed:
    it renders the real app.project_types templates, so a broken template is a
    failing test rather than a surprise in production.
    """

    def _post(self, client, pid, **data):
        payload = {"language": "es"}
        payload.update(data)
        return client.post(f"/api/v2/projects/{pid}/localisation-prompt", data=payload)

    def test_single_segment_hook_puts_everything_in_the_prompt(
        self, client, db_session, fake_translate
    ):
        project = _localisation_project(db_session)
        _make_segment_def(db_session, project.id, 0, start_sec=0.0, end_sec=10.0)
        _make_segment_def(
            db_session, project.id, 1, start_sec=10.0, end_sec=30.0,
            action="keep", has_face=False,
        )

        resp = self._post(client, project.id, language="es")
        assert resp.status_code == 200, resp.text
        body = resp.json()

        assert body["source_language"] == "en"
        assert body["target_language"] == "es"
        assert body["hook_sec"] == settings.LOCALISATION_DEFAULT_HOOK_SEC
        # The usual case: one swap segment, so no per-segment overrides at all.
        assert body["segment_prompts"] == {}
        # Both lines, in the proven "**Speaker:**" dialogue shape.
        assert "<es> Hey, what are you doing?" in body["prompt"]
        assert "<es> I'm just listening to my emails." in body["prompt"]
        assert "**the woman in the red jacket:**" in body["prompt"]
        assert body["warnings"] == []

    def test_returns_translated_lines_with_their_source(
        self, client, db_session, fake_translate
    ):
        project = _localisation_project(db_session)
        body = self._post(client, project.id, language="es").json()

        assert [ln["id"] for ln in body["lines"]] == [1, 2]
        assert body["lines"][0]["text"] == "<es> Hey, what are you doing?"
        assert body["lines"][0]["source_text"] == "Hey, what are you doing?"
        assert body["lines"][0]["speaker"] == "the woman in the red jacket"

    def test_swap_character_picks_a_different_template(
        self, client, db_session, fake_translate
    ):
        project = _localisation_project(db_session)
        swap = self._post(client, project.id, swap_character="true").json()["prompt"]
        keep = self._post(client, project.id, swap_character="false").json()["prompt"]

        assert swap != keep
        # Both still carry the dialogue and the language pair.
        for prompt in (swap, keep):
            assert "<es> Hey, what are you doing?" in prompt
            assert "from english to spanish" in prompt
            assert "lip-sync the translated speech" in prompt.lower()
            assert "facial expressions" in prompt.lower()
            assert "original motion and lip movements" not in prompt.lower()
        assert "Replace the main person" in swap
        assert "Do not replace the character" in keep
        assert "reference image" not in keep.lower()

    def test_passes_cached_video_context_to_translate(
        self, client, db_session, fake_translate
    ):
        project = _localisation_project(
            db_session,
            transcript=_transcript(
                _HOOK_LINES,
                video_summary="Promo for a reading app.",
                scene_context="Woman points at her phone.",
            ),
        )
        resp = self._post(client, project.id, language="es")
        assert resp.status_code == 200, resp.text
        assert len(fake_translate) == 1
        assert fake_translate[0]["video_summary"] == "Promo for a reading app."
        assert fake_translate[0]["scene_context"] == "Woman points at her phone."

    def test_old_transcript_without_context_still_translates(
        self, client, db_session, fake_translate
    ):
        """v1 cached envelopes omit the fields — empty-string defaults apply."""
        project = _localisation_project(db_session)
        assert "video_summary" not in project.transcript
        assert "scene_context" not in project.transcript

        resp = self._post(client, project.id, language="ja")
        assert resp.status_code == 200, resp.text
        assert fake_translate[0]["video_summary"] == ""
        assert fake_translate[0]["scene_context"] == ""
        assert [ln["id"] for ln in resp.json()["lines"]] == [1, 2]

    def test_only_the_hook_window_is_translated(
        self, client, db_session, fake_translate
    ):
        """Lines past the hook belong to the tail, which is never regenerated."""
        project = _localisation_project(
            db_session,
            transcript=_transcript(
                _HOOK_LINES + [_line(3, 20.0, 22.0, "This is in the tail.")]
            ),
        )

        body = self._post(client, project.id).json()
        assert [ln["id"] for ln in body["lines"]] == [1, 2]
        assert "This is in the tail." not in body["prompt"]
        # The translator was never asked to pay for the tail either.
        assert [ln["id"] for ln in fake_translate[0]["lines"]] == [1, 2]

    def test_hook_sec_override_widens_the_window(
        self, client, db_session, fake_translate
    ):
        project = _localisation_project(
            db_session,
            transcript=_transcript(
                _HOOK_LINES + [_line(3, 20.0, 22.0, "This is in the tail.")]
            ),
        )

        body = self._post(client, project.id, hook_sec=25).json()
        assert body["hook_sec"] == 25.0
        assert [ln["id"] for ln in body["lines"]] == [1, 2, 3]

    def test_stored_hook_sec_is_the_default_window(
        self, client, db_session, fake_translate
    ):
        project = _localisation_project(
            db_session,
            hook_sec=3.0,
            transcript=_transcript(_HOOK_LINES),
        )
        body = self._post(client, project.id).json()
        assert body["hook_sec"] == 3.0
        # Line 2 starts at 2.5s, so it overlaps a 3s hook and comes along.
        assert [ln["id"] for ln in body["lines"]] == [1, 2]

    def test_unusable_stored_hook_sec_falls_back_to_the_default(
        self, client, db_session, fake_translate
    ):
        """A 0 written straight into the column (or by an older build) must not
        turn every translate call into a 422 the operator cannot act on."""
        project = _localisation_project(db_session, hook_sec=0.0)
        body = self._post(client, project.id).json()
        assert body["hook_sec"] == settings.LOCALISATION_DEFAULT_HOOK_SEC

    def test_multi_segment_hook_splits_the_dialogue(
        self, client, db_session, fake_translate
    ):
        """A hook longer than the per-clip cap is cut into several swap
        segments, and each is generated as its OWN clip — so each gets only its
        own lines. Handing all of it to every segment would make each clip try
        to say the whole hook.
        """
        project = _localisation_project(db_session, hook_sec=20.0)
        first = _make_segment_def(db_session, project.id, 0, start_sec=0.0, end_sec=10.0)
        second = _make_segment_def(
            db_session, project.id, 1, start_sec=10.0, end_sec=20.0
        )
        _make_segment_def(
            db_session, project.id, 2, start_sec=20.0, end_sec=30.0,
            action="keep", has_face=False,
        )
        project.transcript = _transcript(
            [
                _line(1, 0.0, 4.0, "Front half of the hook.", speaker="the woman"),
                _line(2, 12.0, 16.0, "Back half of the hook.", speaker="the man"),
            ]
        )
        db_session.commit()

        body = self._post(client, project.id).json()

        assert set(body["segment_prompts"]) == {first.id, second.id}
        assert "Front half" in body["segment_prompts"][first.id]
        assert "Back half" not in body["segment_prompts"][first.id]
        assert "Back half" in body["segment_prompts"][second.id]
        assert "Front half" not in body["segment_prompts"][second.id]
        # The run prompt still carries the whole hook: it is the fallback for
        # any segment not in the map.
        assert "Front half" in body["prompt"]
        assert "Back half" in body["prompt"]

    def test_keep_segments_never_get_a_prompt(
        self, client, db_session, fake_translate
    ):
        """Keep segments are never generated, so a prompt for one is dead text."""
        project = _localisation_project(db_session, hook_sec=20.0)
        swap_a = _make_segment_def(db_session, project.id, 0, start_sec=0.0, end_sec=10.0)
        swap_b = _make_segment_def(db_session, project.id, 1, start_sec=10.0, end_sec=20.0)
        keep = _make_segment_def(
            db_session, project.id, 2, start_sec=20.0, end_sec=30.0,
            action="keep", has_face=False,
        )

        body = self._post(client, project.id).json()
        assert keep.id not in body["segment_prompts"]
        assert set(body["segment_prompts"]) == {swap_a.id, swap_b.id}

    def test_a_silent_hook_segment_still_gets_its_own_empty_prompt(
        self, client, db_session, fake_translate
    ):
        """A swap segment with no speech in it gets a prompt with an EMPTY
        dialogue block, not no prompt at all.

        Leaving it out of the map would make it fall back to the run prompt,
        i.e. the WHOLE hook's dialogue — and that clip would try to say all of
        it a second time. Silence is the correct instruction for a silent
        stretch of the hook.
        """
        project = _localisation_project(db_session, hook_sec=20.0)
        speaking = _make_segment_def(
            db_session, project.id, 0, start_sec=0.0, end_sec=10.0
        )
        silent = _make_segment_def(
            db_session, project.id, 1, start_sec=10.0, end_sec=20.0
        )

        body = self._post(client, project.id).json()
        assert silent.id in body["segment_prompts"]
        assert "<es> Hey, what are you doing?" in body["segment_prompts"][speaking.id]
        assert "<es> Hey, what are you doing?" not in body["segment_prompts"][silent.id]
        assert "**" not in body["segment_prompts"][silent.id]  # no dialogue at all

    def test_writes_nothing_to_the_database(
        self, client, db_session, SessionFactory, fake_translate
    ):
        """The whole point of the endpoint: it returns text, it does not record
        anything. The run the operator submits afterwards is the only record.
        """
        project = _localisation_project(db_session, hook_sec=8.0)
        _make_segment_def(db_session, project.id, 0, start_sec=0.0, end_sec=8.0)
        before = {
            "transcript": json.dumps(project.transcript, sort_keys=True),
            "transcript_status": project.transcript_status,
            "transcript_error": project.transcript_error,
            "hook_sec": project.hook_sec,
            "name": project.name,
        }

        assert self._post(client, project.id, language="ja").status_code == 200

        session = SessionFactory()
        after_project = session.get(VideoProject, project.id)
        assert json.dumps(after_project.transcript, sort_keys=True) == before["transcript"]
        assert after_project.transcript_status == before["transcript_status"]
        assert after_project.transcript_error == before["transcript_error"]
        assert after_project.hook_sec == before["hook_sec"]
        assert after_project.name == before["name"]
        # No run, and no new segment, was created behind the operator's back.
        assert session.query(Run).filter(Run.project_id == project.id).count() == 0
        assert (
            session.query(SegmentDef).filter(SegmentDef.project_id == project.id).count()
            == 1
        )
        session.close()

    @pytest.mark.parametrize("status", [None, "pending", "running", "failed", "empty"])
    def test_transcript_not_ready_is_409(
        self, client, db_session, fake_translate, status
    ):
        project = _localisation_project(db_session, transcript_status=status)
        resp = self._post(client, project.id)
        assert resp.status_code == 409, resp.text
        assert "not ready" in resp.json()["detail"]
        assert fake_translate == []

    def test_no_speech_in_the_hook_is_422(self, client, db_session, fake_translate):
        """A wordless hook is a legal transcript, but there is nothing to say."""
        project = _localisation_project(
            db_session,
            transcript=_transcript([_line(1, 30.0, 32.0, "Only speech in the tail.")]),
        )
        resp = self._post(client, project.id)
        assert resp.status_code == 422, resp.text
        assert "No speech" in resp.json()["detail"]
        assert fake_translate == []

    def test_hook_sec_override_of_zero_is_400(self, client, db_session, fake_translate):
        project = _localisation_project(db_session)
        resp = self._post(client, project.id, hook_sec=0)
        assert resp.status_code == 400, resp.text
        assert "hook_sec" in resp.json()["detail"]
        assert fake_translate == []

    @pytest.mark.parametrize("language", ["fr", "klingon", "   "])
    def test_unknown_language_is_400(
        self, client, db_session, fake_translate, language
    ):
        """An unrecognised code is refused, never silently treated as English.

        Whitespace is in here deliberately: it is non-empty, so it reaches the
        handler (unlike "" — see below) and has to be caught there.
        """
        project = _localisation_project(db_session)
        resp = self._post(client, project.id, language=language)
        assert resp.status_code == 400, resp.text
        assert "language" in resp.json()["detail"]
        assert fake_translate == []

    @pytest.mark.parametrize("data", [{}, {"language": ""}])
    def test_absent_language_is_a_422_from_the_form(
        self, client, db_session, fake_translate, data
    ):
        """FastAPI's own validation — the field is required, not defaulted.

        An EMPTY form value counts as absent to FastAPI (it maps ``""`` to a
        missing required Form field), so a blank <select> is a 422 rather than
        the handler's 400. Both refuse it; pinning which one keeps the UI's
        error handling honest.
        """
        project = _localisation_project(db_session)
        resp = client.post(
            f"/api/v2/projects/{project.id}/localisation-prompt", data=data
        )
        assert resp.status_code == 422
        assert fake_translate == []

    def test_translation_failure_is_502(self, client, db_session, monkeypatch):
        """A model outage is not the operator's mistake — and §8 says the
        fallback is writing the prompt by hand, so nothing is stored."""
        def _boom(lines, **kwargs):
            raise api_v2_module.localisation.LocalisationError("kie.ai returned 401")

        monkeypatch.setattr(api_v2_module.localisation, "translate_lines", _boom)
        project = _localisation_project(db_session)

        resp = self._post(client, project.id)
        assert resp.status_code == 502, resp.text
        assert "kie.ai returned 401" in resp.json()["detail"]

    def test_unknown_project_is_404(self, client, fake_translate):
        resp = self._post(client, "nope")
        assert resp.status_code == 404

    # -- warnings ---------------------------------------------------------

    def test_warns_when_a_line_straddles_the_hook_boundary(
        self, client, db_session, fake_translate
    ):
        """slice_lines takes overlapping lines, so this one is translated whole
        — but the hook segment ends mid-sentence."""
        project = _localisation_project(
            db_session,
            hook_sec=10.0,
            transcript=_transcript([_line(1, 8.0, 14.0, "Half in, half out of it.")]),
        )
        body = self._post(client, project.id).json()
        straddle = [w for w in body["warnings"] if "cut mid-sentence" in w]
        assert len(straddle) == 1
        assert "10.0s hook boundary" in straddle[0]

    def test_no_straddle_warning_when_a_line_ends_on_the_boundary(
        self, client, db_session, fake_translate
    ):
        project = _localisation_project(
            db_session,
            hook_sec=10.0,
            transcript=_transcript([_line(1, 8.0, 10.0, "Ends right on it.")]),
        )
        body = self._post(client, project.id).json()
        assert [w for w in body["warnings"] if "cut mid-sentence" in w] == []

    def test_warns_when_the_translation_will_overrun(
        self, client, db_session, fake_translate
    ):
        """The failure that actually shipped: a literal EN→JA hook that took far
        longer to say than the shot allowed."""
        project = _localisation_project(
            db_session,
            transcript=_transcript(
                [
                    _line(
                        1, 0.0, 1.0,
                        "This is a very long sentence for one single second of footage.",
                    )
                ]
            ),
        )
        body = self._post(client, project.id, language="ja").json()
        assert any("will overrun" in w for w in body["warnings"])

    def test_warns_when_the_translation_looks_too_short(
        self, client, db_session, fake_translate
    ):
        project = _localisation_project(
            db_session,
            transcript=_transcript([_line(1, 0.0, 9.0, "Hi.")]),
        )
        body = self._post(client, project.id).json()
        assert any("check nothing was dropped" in w for w in body["warnings"])

    def test_warns_when_source_and_target_match(
        self, client, db_session, fake_translate
    ):
        """Almost always a mis-picked dropdown — a whole generation spent
        reproducing the original."""
        project = _localisation_project(db_session)  # source_language="en"
        body = self._post(client, project.id, language="en").json()
        assert any("no-op" in w for w in body["warnings"])
        # Advisory only: the prompt still comes back.
        assert body["prompt"]

    def test_warns_when_the_hook_override_overruns_the_analyzed_segments(
        self, client, db_session, fake_translate
    ):
        """hook_sec is a per-call override that re-segments nothing, so it can
        ask for more dialogue than the generated clip is able to say — the
        "whole script in one clip" failure the per-segment split prevents."""
        project = _localisation_project(
            db_session,
            hook_sec=10.0,
            transcript=_transcript(
                [
                    _line(1, 0.0, 9.0, "Inside the analyzed hook."),
                    _line(2, 11.0, 18.0, "Past the end of it."),
                ]
            ),
        )
        _make_segment_def(db_session, project.id, 0, start_sec=0.0, end_sec=10.0)
        _make_segment_def(
            db_session, project.id, 1, start_sec=10.0, end_sec=30.0,
            action="keep", has_face=False,
        )

        body = self._post(client, project.id, hook_sec=20.0).json()
        overrun = [w for w in body["warnings"] if "only cover" in w]
        assert len(overrun) == 1
        assert "20.0s" in overrun[0] and "10.0s" in overrun[0]
        # Advisory only — the prompt still comes back with the wider window.
        assert "Past the end of it" in body["prompt"]

    def test_warns_when_no_swap_segment_covers_the_hook(
        self, client, db_session, fake_translate
    ):
        """A project analyzed under another type (or with the hook flipped to
        keep) would receive a prompt no clip is ever handed."""
        project = _localisation_project(db_session)
        _make_segment_def(
            db_session, project.id, 0, start_sec=0.0, end_sec=30.0,
            action="keep", has_face=False,
        )

        body = self._post(client, project.id).json()
        assert any("nothing will be generated" in w for w in body["warnings"])
        assert body["prompt"]

    def test_warnings_never_block_the_response(
        self, client, db_session, fake_translate
    ):
        project = _localisation_project(
            db_session,
            hook_sec=10.0,
            transcript=_transcript([_line(1, 9.0, 12.0, "Hi.")], source_language="en"),
        )
        resp = self._post(client, project.id, language="en")
        assert resp.status_code == 200
        assert len(resp.json()["warnings"]) >= 2
        assert resp.json()["prompt"]


class TestLocalisationSegmentAssignment:
    """Every hook line belongs to EXACTLY ONE swap segment.

    A per-segment prompt REPLACES the run prompt (pipeline_v2 submits
    ``prompt_override if prompt_override else run_prompt``), so the run prompt
    is NOT a fallback for a segment that got one. That makes the split a
    partition problem, and localisation.slice_lines — whose semantics are
    OVERLAP — the wrong tool for it: overlap duplicates a boundary-straddling
    line into both clips and loses a line that falls in a keep gap. The rule is
    the line's MIDPOINT.

    Both shapes below are reachable by hand-splitting the hook in the Segment
    Editor, which is exactly what the new hook editor invites.
    """

    def _post(self, client, pid, **data):
        payload = {"language": "es"}
        payload.update(data)
        return client.post(f"/api/v2/projects/{pid}/localisation-prompt", data=payload)

    def _split_hook_project(self, session):
        """swap[0,5] + swap[5,10], with a line straddling the 5s seam."""
        project = _localisation_project(
            session,
            hook_sec=10.0,
            transcript=_transcript(
                [
                    _line(1, 0.0, 4.8, "Before the seam.", speaker="the woman"),
                    _line(2, 4.8, 5.4, "Right on the seam.", speaker="the woman"),
                    _line(3, 5.4, 9.5, "After the seam.", speaker="the man"),
                ]
            ),
        )
        first = _make_segment_def(session, project.id, 0, start_sec=0.0, end_sec=5.0)
        second = _make_segment_def(session, project.id, 1, start_sec=5.0, end_sec=10.0)
        return project, first, second

    def test_a_straddling_line_lands_in_exactly_one_segment(
        self, client, db_session, fake_translate
    ):
        """The duplication bug: with overlap semantics the 4.8-5.4s line was in
        BOTH overrides, both clips generated it, and the stitched hook said the
        sentence twice."""
        project, first, second = self._split_hook_project(db_session)

        prompts = self._post(client, project.id).json()["segment_prompts"]
        assert set(prompts) == {first.id, second.id}

        occurrences = sum(
            "Right on the seam." in prompts[sid] for sid in (first.id, second.id)
        )
        assert occurrences == 1
        # Midpoint 5.1s falls in [5, 10), so the SECOND clip speaks it.
        assert "Right on the seam." in prompts[second.id]
        assert "Right on the seam." not in prompts[first.id]

    def test_every_line_is_assigned_and_none_are_shared(
        self, client, db_session, fake_translate
    ):
        project, first, second = self._split_hook_project(db_session)
        prompts = self._post(client, project.id).json()["segment_prompts"]

        assert "Before the seam." in prompts[first.id]
        assert "Before the seam." not in prompts[second.id]
        assert "After the seam." in prompts[second.id]
        assert "After the seam." not in prompts[first.id]

    def test_a_straddling_line_is_still_warned_about(
        self, client, db_session, fake_translate
    ):
        """Assigned whole to one clip, but the stitch still cuts through it —
        the same advisory the hook edge has always had."""
        project, _first, _second = self._split_hook_project(db_session)
        warnings = self._post(client, project.id).json()["warnings"]

        seam = [w for w in warnings if "5.0s boundary between two swap segments" in w]
        assert len(seam) == 1
        assert "Line 2" in seam[0]
        assert "cut mid-sentence" in seam[0]

    def test_a_line_ending_on_a_seam_is_not_warned_about(
        self, client, db_session, fake_translate
    ):
        project = _localisation_project(
            db_session,
            hook_sec=10.0,
            transcript=_transcript([_line(1, 2.0, 5.0, "Ends right on the seam.")]),
        )
        _make_segment_def(db_session, project.id, 0, start_sec=0.0, end_sec=5.0)
        _make_segment_def(db_session, project.id, 1, start_sec=5.0, end_sec=10.0)

        warnings = self._post(client, project.id).json()["warnings"]
        assert [w for w in warnings if "cut mid-sentence" in w] == []

    # -- the keep gap ------------------------------------------------------

    def _gapped_hook_project(self, session):
        """swap[0,3] + keep[3,5] + swap[5,10] — the hand-split shape."""
        project = _localisation_project(
            session,
            hook_sec=10.0,
            transcript=_transcript(
                [
                    _line(1, 0.0, 2.5, "In the first swap.", speaker="the woman"),
                    _line(2, 3.2, 4.6, "Stranded in the gap.", speaker="the woman"),
                    _line(3, 5.5, 9.0, "In the second swap.", speaker="the man"),
                ]
            ),
        )
        first = _make_segment_def(session, project.id, 0, start_sec=0.0, end_sec=3.0)
        _make_segment_def(
            session, project.id, 1, start_sec=3.0, end_sec=5.0,
            action="keep", has_face=False,
        )
        second = _make_segment_def(session, project.id, 2, start_sec=5.0, end_sec=10.0)
        return project, first, second

    def test_a_line_in_a_keep_gap_reaches_no_segment_prompt(
        self, client, db_session, fake_translate
    ):
        project, first, second = self._gapped_hook_project(db_session)
        body = self._post(client, project.id).json()

        prompts = body["segment_prompts"]
        assert set(prompts) == {first.id, second.id}
        assert "Stranded in the gap." not in prompts[first.id]
        assert "Stranded in the gap." not in prompts[second.id]
        # It stays in the run prompt so the operator can see what is at stake —
        # they can fix the segmentation, but not text they were never shown.
        assert "Stranded in the gap." in body["prompt"]

    def test_a_line_in_a_keep_gap_is_warned_about_by_name(
        self, client, db_session, fake_translate
    ):
        """The silent failure: nothing generates this line, and the first sign
        used to be a delivered hook missing a sentence."""
        project, _first, _second = self._gapped_hook_project(db_session)
        warnings = self._post(client, project.id).json()["warnings"]

        stranded = [w for w in warnings if "will not be spoken at all" in w]
        assert len(stranded) == 1
        assert "Line 2" in stranded[0]
        assert "3.2s to 4.6s" in stranded[0]

    def test_the_coverage_warning_measures_the_union_not_the_last_end(
        self, client, db_session, fake_translate
    ):
        """max(end_sec) reads 10s here — the hole in the middle was invisible."""
        project, _first, _second = self._gapped_hook_project(db_session)
        warnings = self._post(client, project.id).json()["warnings"]

        coverage = [w for w in warnings if "only cover" in w]
        assert len(coverage) == 1
        assert "10.0s" in coverage[0]      # the hook window
        assert "only cover 8.0s" in coverage[0]  # 3 + 5, not 10
        assert "3.0-5.0s" in coverage[0]   # and it names the gap

    def test_a_gap_free_hook_raises_no_coverage_warning(
        self, client, db_session, fake_translate
    ):
        project = _localisation_project(db_session, hook_sec=10.0)
        _make_segment_def(db_session, project.id, 0, start_sec=0.0, end_sec=5.0)
        _make_segment_def(db_session, project.id, 1, start_sec=5.0, end_sec=10.0)

        warnings = self._post(client, project.id).json()["warnings"]
        assert [w for w in warnings if "only cover" in w] == []
        assert [w for w in warnings if "will not be spoken at all" in w] == []

    def test_a_single_swap_segment_never_strands_a_line(
        self, client, db_session, fake_translate
    ):
        """With one hook segment there are no per-segment prompts at all, so the
        run prompt IS what that clip receives — nothing is lost, and claiming
        otherwise would be a false alarm. The coverage warning still fires."""
        project = _localisation_project(
            db_session,
            hook_sec=10.0,
            transcript=_transcript([_line(1, 0.5, 1.5, "In the uncovered head.")]),
        )
        _make_segment_def(
            db_session, project.id, 0, start_sec=0.0, end_sec=2.0,
            action="keep", has_face=False,
        )
        _make_segment_def(db_session, project.id, 1, start_sec=2.0, end_sec=10.0)

        body = self._post(client, project.id).json()
        assert body["segment_prompts"] == {}
        assert [w for w in body["warnings"] if "will not be spoken at all" in w] == []
        assert any("only cover" in w for w in body["warnings"])


class TestLocalisationHookWindow:
    """Three numbers call themselves "the hook length"; only one is the window.

    1. the per-call ``hook_sec`` form argument — an override for this call only,
       never persisted, re-segments nothing;
    2. the ANALYZED reality — where the project's swap segments actually end.
       This is the DEFAULT window, because it is what a run will really
       generate;
    3. the STORED ``VideoProject.hook_sec`` — the operator's intent. It bounds
       (2), it drives the next analysis, and it is the fallback when there is
       nothing analyzed to measure.

    The bug this pins: a project created with Hook length 15 under the default
    10s segmentation cap is cut ``swap[0,10]`` while the column still says 15,
    so the endpoint used to slice 15s of transcript into a prompt for a 10s
    clip.
    """

    def _post(self, client, pid, **data):
        payload = {"language": "es"}
        payload.update(data)
        return client.post(f"/api/v2/projects/{pid}/localisation-prompt", data=payload)

    def test_the_analyzed_window_beats_the_stored_intent(
        self, client, db_session, fake_translate
    ):
        project = _localisation_project(
            db_session,
            hook_sec=15.0,
            transcript=_transcript(
                [
                    _line(1, 0.0, 4.0, "Inside the clip that will exist."),
                    _line(2, 11.0, 14.0, "Only inside the 15s the operator asked for."),
                ]
            ),
        )
        _make_segment_def(db_session, project.id, 0, start_sec=0.0, end_sec=10.0)
        _make_segment_def(
            db_session, project.id, 1, start_sec=10.0, end_sec=30.0,
            action="keep", has_face=False,
        )

        body = self._post(client, project.id).json()
        assert body["hook_sec"] == 10.0
        assert [ln["id"] for ln in body["lines"]] == [1]
        assert "Only inside the 15s" not in body["prompt"]
        # And the clamped window is fully covered, so no coverage alarm.
        assert [w for w in body["warnings"] if "only cover" in w] == []

    def test_the_stored_intent_is_left_alone(
        self, client, db_session, SessionFactory, fake_translate
    ):
        """The column is the operator's INTENT and still drives the next
        analysis — raise the segmentation cap, re-analyze, and 15 finally
        happens. Reading it as reality is what was wrong, not storing it."""
        project = _localisation_project(db_session, hook_sec=15.0)
        _make_segment_def(db_session, project.id, 0, start_sec=0.0, end_sec=10.0)

        assert self._post(client, project.id).json()["hook_sec"] == 10.0

        session = SessionFactory()
        assert session.get(VideoProject, project.id).hook_sec == 15.0
        session.close()

    def test_the_intent_bounds_the_analyzed_window(
        self, client, db_session, fake_translate
    ):
        """A project analyzed under another type can have swap segments running
        far past the hook; those are not hook, and must not widen the window."""
        project = _localisation_project(
            db_session,
            hook_sec=5.0,
            transcript=_transcript(
                [
                    _line(1, 0.0, 4.0, "Inside the hook."),
                    _line(2, 20.0, 24.0, "Deep in the tail."),
                ]
            ),
        )
        _make_segment_def(db_session, project.id, 0, start_sec=0.0, end_sec=30.0)

        body = self._post(client, project.id).json()
        assert body["hook_sec"] == 5.0
        assert [ln["id"] for ln in body["lines"]] == [1]

    def test_an_unanalyzed_project_falls_back_to_the_intent(
        self, client, db_session, fake_translate
    ):
        """Nothing to measure — the stored value is the only number there is."""
        project = _localisation_project(db_session, hook_sec=4.0)
        body = self._post(client, project.id).json()
        assert body["hook_sec"] == 4.0

    def test_a_hook_flipped_to_keep_falls_back_to_the_intent(
        self, client, db_session, fake_translate
    ):
        project = _localisation_project(db_session, hook_sec=6.0)
        _make_segment_def(
            db_session, project.id, 0, start_sec=0.0, end_sec=30.0,
            action="keep", has_face=False,
        )
        body = self._post(client, project.id).json()
        assert body["hook_sec"] == 6.0
        assert any("nothing will be generated" in w for w in body["warnings"])

    def test_a_hand_split_hook_keeps_its_full_analyzed_width(
        self, client, db_session, fake_translate
    ):
        """swap[0,3] + keep[3,5] + swap[5,10]: the analyzed window is 10s, not
        3s. Shrinking to the first gap would orphan the second swap segment —
        it would drop out of the hook, get no per-segment prompt, and fall back
        to re-speaking the whole hook."""
        project = _localisation_project(db_session, hook_sec=10.0)
        first = _make_segment_def(db_session, project.id, 0, start_sec=0.0, end_sec=3.0)
        _make_segment_def(
            db_session, project.id, 1, start_sec=3.0, end_sec=5.0,
            action="keep", has_face=False,
        )
        second = _make_segment_def(db_session, project.id, 2, start_sec=5.0, end_sec=10.0)

        body = self._post(client, project.id).json()
        assert body["hook_sec"] == 10.0
        assert set(body["segment_prompts"]) == {first.id, second.id}

    def test_an_explicit_override_still_beats_the_analyzed_window(
        self, client, db_session, fake_translate
    ):
        """The override is for retranslating against a different window on
        purpose, so it outranks reality — and keeps its warning for doing so."""
        project = _localisation_project(
            db_session,
            hook_sec=10.0,
            transcript=_transcript(
                [
                    _line(1, 0.0, 4.0, "Inside the analyzed hook."),
                    _line(2, 11.0, 14.0, "Past the end of it."),
                ]
            ),
        )
        _make_segment_def(db_session, project.id, 0, start_sec=0.0, end_sec=10.0)

        body = self._post(client, project.id, hook_sec=15).json()
        assert body["hook_sec"] == 15.0
        assert [ln["id"] for ln in body["lines"]] == [1, 2]
        coverage = [w for w in body["warnings"] if "only cover" in w]
        assert len(coverage) == 1
        assert "10.0-15.0s" in coverage[0]

    def test_an_override_narrower_than_the_analyzed_window_wins_too(
        self, client, db_session, fake_translate
    ):
        project = _localisation_project(
            db_session,
            hook_sec=10.0,
            transcript=_transcript(
                [
                    _line(1, 0.0, 2.0, "Inside the narrowed window."),
                    _line(2, 6.0, 9.0, "Outside it."),
                ]
            ),
        )
        _make_segment_def(db_session, project.id, 0, start_sec=0.0, end_sec=10.0)

        body = self._post(client, project.id, hook_sec=4).json()
        assert body["hook_sec"] == 4.0
        assert [ln["id"] for ln in body["lines"]] == [1]


class TestLocalisationReleaseIntegration:
    """Combined checks for video-context + Fast default + adaptive lips.

    Exercises the three Localisation features together through real application
    helpers / endpoints. Translation stays stubbed (fake_translate); Seedance
    templates are the live project_types strings.
    """

    def _post(self, client, pid, **data):
        payload = {"language": "es"}
        payload.update(data)
        return client.post(f"/api/v2/projects/{pid}/localisation-prompt", data=payload)

    def test_v2_context_translates_into_adaptive_lips_prompt(
        self, client, db_session, fake_translate
    ):
        """Schema-v2 transcript context → adaptive-lips Seedance prompt."""
        project = _localisation_project(
            db_session,
            mute_source=True,
            transcript=_transcript(
                _HOOK_LINES,
                schema_version=2,
                prompt_version="transcribe/v2",
                video_summary="Promo for a reading app.",
                scene_context="Woman points at her phone; 'this' means the app.",
            ),
        )
        _make_segment_def(db_session, project.id, 0, start_sec=0.0, end_sec=10.0)

        for swap_character in ("true", "false"):
            body = self._post(
                client, project.id, language="ja", swap_character=swap_character
            ).json()
            prompt = body["prompt"]
            lowered = prompt.lower()
            assert "{source_language}" not in prompt
            assert "{target_language}" not in prompt
            assert "{dialogue}" not in prompt
            assert "lip-sync the translated speech" in lowered
            assert "facial expressions" in lowered
            assert "emotional reactions" in lowered
            assert "original motion and lip movements" not in lowered
            assert "<ja> Hey, what are you doing?" in prompt

        assert fake_translate[-1]["video_summary"] == "Promo for a reading app."
        assert fake_translate[-1]["scene_context"] == (
            "Woman points at her phone; 'this' means the app."
        )

    def test_new_run_defaults_to_seedance_fast(self, db_session):
        """Registry-driven New Run default under normal project settings."""
        from app.project_types import LOCALISATION, spec_for
        from app.web_v2 import _new_run_defaults

        assert spec_for(LOCALISATION).default_model == "seedance-fast"
        assert spec_for(LOCALISATION).default_audio_mode == "seedance"
        assert spec_for(LOCALISATION).default_mute_source is True

        project = _localisation_project(db_session, mute_source=True)
        ctx = _new_run_defaults(project)
        assert ctx["default_model"] == "seedance-fast"
        assert ctx["default_audio_mode"] == "seedance"
        assert "seedance-2-5" not in ctx["models_without_audio"]
        assert "gemini-omni" in ctx["models_without_audio"]

    def test_multi_segment_context_assigns_dialogue_once_with_adaptive_lips(
        self, client, db_session, fake_translate
    ):
        """Multi-segment partition + adaptive lips + forwarded context."""
        project = _localisation_project(
            db_session,
            hook_sec=20.0,
            mute_source=True,
            transcript=_transcript(
                [
                    _line(1, 0.0, 4.0, "Front half of the hook.", speaker="the woman"),
                    _line(2, 12.0, 16.0, "Back half of the hook.", speaker="the man"),
                ],
                schema_version=2,
                prompt_version="transcribe/v2",
                video_summary="Two-shot street promo.",
                scene_context="Woman then man; each owns their half.",
            ),
        )
        first = _make_segment_def(db_session, project.id, 0, start_sec=0.0, end_sec=10.0)
        second = _make_segment_def(
            db_session, project.id, 1, start_sec=10.0, end_sec=20.0
        )

        body = self._post(client, project.id, swap_character="true").json()
        assert fake_translate[0]["video_summary"] == "Two-shot street promo."
        assert fake_translate[0]["scene_context"] == (
            "Woman then man; each owns their half."
        )
        assert set(body["segment_prompts"]) == {first.id, second.id}
        front = body["segment_prompts"][first.id]
        back = body["segment_prompts"][second.id]
        assert "Front half" in front and "Back half" not in front
        assert "Back half" in back and "Front half" not in back
        for prompt in (front, back, body["prompt"]):
            lowered = prompt.lower()
            assert "lip-sync the translated speech" in lowered
            assert "original motion and lip movements" not in lowered
            assert "{dialogue}" not in prompt

    def test_v1_transcript_builds_adaptive_prompt(
        self, client, db_session, fake_translate
    ):
        """Cached v1 envelopes still produce the adaptive-lips templates."""
        project = _localisation_project(db_session)
        assert "video_summary" not in project.transcript
        assert "scene_context" not in project.transcript
        _make_segment_def(db_session, project.id, 0, start_sec=0.0, end_sec=10.0)

        body = self._post(client, project.id, swap_character="false").json()
        assert fake_translate[0]["video_summary"] == ""
        assert fake_translate[0]["scene_context"] == ""
        prompt = body["prompt"]
        lowered = prompt.lower()
        assert "do not replace the character" in lowered
        assert "lip-sync the translated speech" in lowered
        assert "original motion and lip movements" not in lowered
        assert "{source_language}" not in prompt
