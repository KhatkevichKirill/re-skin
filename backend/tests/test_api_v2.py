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

from app.db import Base, get_db
from app.models import Run, RunSegment, SegmentDef, VideoProject
from app.state_machine import ProjectStatus, RunStatus, SegmentStatus


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

    calls: dict[str, list[str]] = {"analyze_project": [], "process_run": []}

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
        run = _make_run(session, project.id, status=RunStatus.done)
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
        session2.close()
        assert rs_fetched.status == SegmentStatus.pending
        assert rs_fetched.seedance_task_id is None

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
