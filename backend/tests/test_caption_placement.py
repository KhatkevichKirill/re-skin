"""
Tests for AICP-368 — burned captions placed over the original caption area of a
subtitle-removal lineage (app/caption_placement.py, app/captions.py
``placement=``, project-from-run inheritance, the caption snapshot, PATCH and
the derive endpoint).

Own temp SQLite DB, like the other API test modules. The detector scan is
mocked (qa_tools.calibrate_captions); one test renders a real frame through
ffmpeg/libass when the subtitles filter is available.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
from contextlib import contextmanager

_db_fd, _db_path = tempfile.mkstemp(suffix="_caption_placement.db")
os.close(_db_fd)
os.environ["DATABASE_URL"] = f"sqlite:///{_db_path}"
_data_tmp = tempfile.mkdtemp()
os.environ["DATA_DIR"] = _data_tmp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy import create_engine, event  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402

import app.api_v2 as api_v2_module  # noqa: E402
from app import caption_placement as cp  # noqa: E402
from app import captions as captions_mod  # noqa: E402
from app.captions import Placement, build_ass  # noqa: E402
from app.db import Base, get_db  # noqa: E402
from app.models import Run, RunCaption, RunSegment, VideoProject  # noqa: E402
from app.segment_fill_models import RunSegmentFillJob  # noqa: E402
from app.state_machine import ProjectStatus, RunStatus  # noqa: E402

# The real case (project 328341d4 / run 2845eef3).
FILL_SEG4 = [0.2607, 0.5821, 0.7393, 0.6627]
FILL_SEG7 = [0.2256, 0.5821, 0.7763, 0.6627]
CAL_REAL = {
    "frames": 300, "hits": 240, "hit_ratio": 0.8, "calibrated": True,
    "geometry": {"n": 240, "x0_p5": 0.27, "x1_p95": 0.73, "y0_p5": 0.43,
                 "y0_p50": 0.605, "y1_p50": 0.643, "y1_p95": 0.66},
}


# ---------------------------------------------------------------------------
# DB / client fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def engine():
    eng = create_engine(f"sqlite:///{_db_path}", connect_args={"check_same_thread": False})

    @event.listens_for(eng, "connect")
    def _pragmas(dbapi_conn, _):
        cur = dbapi_conn.cursor()
        cur.execute("PRAGMA foreign_keys=ON")
        cur.close()

    Base.metadata.create_all(eng)
    yield eng
    Base.metadata.drop_all(eng)


@pytest.fixture(scope="session")
def SessionFactory(engine):
    return sessionmaker(bind=engine, autocommit=False, autoflush=False,
                        expire_on_commit=False)


@pytest.fixture()
def db_session(SessionFactory):
    s = SessionFactory()
    yield s
    s.rollback()
    s.close()


@pytest.fixture()
def test_get_session(SessionFactory, monkeypatch):
    @contextmanager
    def _gs():
        s = SessionFactory()
        try:
            yield s
            s.commit()
        except Exception:
            s.rollback()
            raise
        finally:
            s.close()

    import app.db as db_mod

    monkeypatch.setattr(db_mod, "get_session", _gs)
    monkeypatch.setattr(api_v2_module, "get_session", _gs)
    return _gs


@pytest.fixture()
def spy(monkeypatch):
    calls = {"analyze": [], "scan": [], "caption": []}
    monkeypatch.setattr(api_v2_module, "enqueue_analyze_project",
                        lambda pid: calls["analyze"].append(pid))
    monkeypatch.setattr(api_v2_module, "enqueue_process_caption",
                        lambda cid: calls["caption"].append(cid))
    import app.tasks as tasks_mod

    monkeypatch.setattr(tasks_mod, "enqueue_caption_placement_scan",
                        lambda pid, rev: calls["scan"].append((pid, rev)))
    return calls


@pytest.fixture()
def client(engine, SessionFactory, spy, test_get_session):
    from app.main import app

    def override_get_db():
        s = SessionFactory()
        try:
            yield s
            s.commit()
        except Exception:
            s.rollback()
            raise
        finally:
            s.close()

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c
    app.dependency_overrides.clear()


def _storage_file(name: str, data: bytes = b"video") -> str:
    from app import storage

    root = storage.data_dir_root()
    d = tempfile.mkdtemp(prefix="cp-", dir=root)
    path = os.path.join(d, name)
    with open(path, "wb") as fh:
        fh.write(data)
    return path


def _project(s, **kw) -> VideoProject:
    defaults = dict(source_type="upload", source_ref="src.mp4", status=ProjectStatus.ready)
    defaults.update(kw)
    p = VideoProject(**defaults)
    s.add(p)
    s.commit()
    return p


def _done_run(s, project, *, republished=False, **kw) -> Run:
    """A done run whose final is the pipeline's own stitch (results/final.mp4)
    — what ``delivery_state`` needs to call a fill ``delivered`` — or, with
    *republished*, a final re-pointed elsewhere."""
    from app import storage

    defaults = dict(project_id=project.id, prompt="p", resolution="720p",
                    status=RunStatus.done, reference_image_urls=[])
    defaults.update(kw)
    r = Run(**defaults)
    s.add(r)
    s.flush()
    if "result_local_path" not in kw:
        if republished:
            r.result_local_path = _storage_file("published.mp4", b"published")
        else:
            final = os.path.join(storage.run_dir(r.id, project.id), "results", "final.mp4")
            os.makedirs(os.path.dirname(final), exist_ok=True)
            with open(final, "wb") as fh:
                fh.write(b"final-bytes")
            r.result_local_path = final
    s.commit()
    return r


def _fill(s, run, box, *, superseded=False, status="applied", idx=0,
          applied_at=None) -> RunSegmentFillJob:
    from datetime import datetime, timedelta, timezone

    if applied_at is None and status == "applied":
        applied_at = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(minutes=5)
    composite = _storage_file(f"composite{idx}.mp4", b"composite-%d" % idx)
    rs = RunSegment(run_id=run.id, segment_def_id=f"def-{idx}-{os.urandom(3).hex()}",
                    index=idx, status="completed",
                    local_result_path=(_storage_file(f"other{idx}.mp4", b"other")
                                       if superseded else composite))
    s.add(rs)
    s.flush()
    job = RunSegmentFillJob(run_id=run.id, run_segment_id=rs.id, project_id=run.project_id,
                            status=status, model="void", prompt="", mask_box=box,
                            box_source="measured", base={"run": {"final_revision": "old"}},
                            composite_path=composite, applied_at=applied_at)
    s.add(job)
    s.commit()
    return job


def _subtitle_lineage(s, *, fills=(FILL_SEG4, FILL_SEG7), width=1080, height=1920,
                      republished=False):
    parent = _project(s, project_type="subtitle_removal", width=width, height=height,
                      source_local_path=_storage_file("source.mp4", b"source"))
    run = _done_run(s, parent, republished=republished)
    for i, box in enumerate(fills):
        _fill(s, run, box, idx=i)
    return parent, run


# ---------------------------------------------------------------------------
# Derivation (pure)
# ---------------------------------------------------------------------------


class TestDerivePlacement:
    def test_real_case_union_centres_on_the_measured_band(self):
        p = cp.derive_placement(fill_boxes=[FILL_SEG4, FILL_SEG7], calibration=CAL_REAL,
                                frame_aspect=1080 / 1920)
        assert p["mode"] == "band"
        # Acceptance: band centred on y ~ 0.60-0.62 (the measured caption geometry).
        assert 0.60 <= p["center_y"] <= 0.625
        x0, y0, x1, y1 = p["band"]
        for fx0, fy0, fx1, fy1 in (FILL_SEG4, FILL_SEG7):
            assert x0 <= fx0 and y0 <= fy0 and x1 >= fx1 and y1 >= fy1
        # The detector median rows, NOT y0_p5 (two-line captions at y~0.43).
        assert y0 == pytest.approx(0.5821)
        assert p["max_rows"] == 3
        assert p["frame_aspect"] == pytest.approx(0.5625)
        src = p["source"]
        assert src["fills"] == 2 and src["fill_boxes"] == [FILL_SEG4, FILL_SEG7]
        assert src["detector"]["geometry"]["y0_p50"] == 0.605
        assert "detector_spread" in src["warnings"]
        assert "fills_detector_disagree" not in src["warnings"]

    def test_fills_only(self):
        p = cp.derive_placement(fill_boxes=[FILL_SEG4, FILL_SEG7])
        assert p["center_y"] == pytest.approx((0.5821 + 0.6627) / 2, abs=1e-4)
        assert "detector" not in p["source"]

    def test_detector_only(self):
        p = cp.derive_placement(fill_boxes=[], calibration=CAL_REAL)
        assert p["center_y"] == pytest.approx(0.624)
        assert p["band"] == [0.27, 0.605, 0.73, 0.643]

    def test_no_evidence_is_the_house_position(self):
        p = cp.derive_placement(fill_boxes=[], calibration=None,
                                source={"kind": "derived", "rev": 2})
        assert p["mode"] == "bottom"
        assert p["source"]["rev"] == 2 and p["source"]["kind"] == "derived"
        assert "no_evidence" in p["source"]["warnings"]
        assert cp.burn_placement(p, width=1080, height=1920) is None

    def test_uncalibrated_detector_is_not_used(self):
        cal = dict(CAL_REAL, calibrated=False, hit_ratio=0.05)
        p = cp.derive_placement(fill_boxes=[FILL_SEG4], calibration=cal)
        assert p["band"] == FILL_SEG4
        assert "detector_uncalibrated" in p["source"]["warnings"]

    def test_disagreeing_and_broad_sources_are_flagged(self):
        p = cp.derive_placement(fill_boxes=[[0.2, 0.30, 0.8, 0.36]], calibration=CAL_REAL)
        assert "fills_detector_disagree" in p["source"]["warnings"]
        assert "broad_band" in p["source"]["warnings"]

    def test_invalid_boxes_are_dropped(self):
        p = cp.derive_placement(fill_boxes=[[0.5, 0.5, 0.4, 0.6], [float("nan")] * 4,
                                            "x", None, FILL_SEG4])
        assert p["band"] == FILL_SEG4 and p["source"]["fills"] == 1

    def test_centre_is_clamped_inside_the_frame(self):
        p = cp.derive_placement(fill_boxes=[[0.2, 0.92, 0.8, 0.99]])
        assert p["center_y"] == cp.CENTER_Y_MAX


class TestValidate:
    @pytest.mark.parametrize("value", [
        {"mode": "band"},                                   # centre missing
        {"mode": "band", "center_y": 0.05},                 # out of range
        {"mode": "band", "center_y": float("nan")},
        {"mode": "band", "center_y": 0.6, "max_rows": 4},
        {"mode": "band", "center_y": 0.6, "max_width": 0.95},
        {"mode": "band", "center_y": 0.6, "band": [0.5, 0.6, 0.4, 0.7]},
        {"mode": "band", "center_y": 0.6, "colour": "red"},  # extra field
        {"mode": "top", "center_y": 0.6},
        "not an object",
    ])
    def test_rejected(self, value):
        with pytest.raises(cp.PlacementError):
            cp.validate(value)

    def test_error_does_not_echo_input(self):
        with pytest.raises(cp.PlacementError) as exc:
            cp.validate({"mode": "band", "center_y": "SECRET-ish"})
        assert "SECRET" not in str(exc.value)

    def test_manual_null_is_a_sticky_bottom(self):
        v = cp.manual(None, current={"mode": "band", "center_y": 0.6,
                                     "source": {"kind": "derived", "rev": 3}})
        assert v["mode"] == "bottom"
        assert v["source"]["kind"] == "manual" and v["source"]["rev"] == 4

    def test_manual_object_replaces_client_source(self):
        v = cp.manual({"mode": "band", "center_y": 0.55, "max_width": 0.6,
                       "source": {"kind": "derived", "rev": 99}}, current=None)
        assert v["center_y"] == 0.55 and v["max_width"] == 0.6
        assert v["source"]["kind"] == "manual" and v["source"]["rev"] == 1


class TestSnapshotForCaption:
    def test_choices(self):
        class P:
            caption_placement = {"mode": "band", "center_y": 0.62}

        assert cp.snapshot_for_caption(P()) == {"mode": "band", "center_y": 0.62}
        assert cp.snapshot_for_caption(P(), None) == {"mode": "band", "center_y": 0.62}
        assert cp.snapshot_for_caption(P(), "bottom")["mode"] == "bottom"
        assert cp.snapshot_for_caption(None) is None
        p = P()
        p.caption_placement = {"mode": "band", "center_y": 0.62, "band": [0.2, 0.5, 0.8, 0.7],
                               "source": {"kind": "derived"}}
        snap = cp.snapshot_for_caption(p)
        p.caption_placement["band"][0] = 0.9
        p.caption_placement["source"]["kind"] = "manual"
        assert snap["band"][0] == 0.2 and snap["source"]["kind"] == "derived"
        with pytest.raises(cp.PlacementError):
            cp.snapshot_for_caption(P(), "top")


class TestBurnPlacement:
    def test_none_and_bottom_are_house_style(self):
        assert cp.burn_placement(None, width=1080, height=1920) is None
        assert cp.burn_placement(cp.bottom(), width=1080, height=1920) is None

    def test_band(self):
        p = cp.burn_placement({"mode": "band", "center_y": 0.62, "max_width": 0.7,
                               "max_rows": 2}, width=1080, height=1920)
        assert p == Placement(center_y=0.62, max_width=0.7, max_rows=2)

    def test_other_aspect_falls_back(self):
        v = {"mode": "band", "center_y": 0.62, "frame_aspect": 0.5625}
        assert cp.burn_placement(v, width=1080, height=1920) is not None
        assert cp.burn_placement(v, width=1920, height=1080) is None

    def test_invalid_stored_value_falls_back(self):
        assert cp.burn_placement({"mode": "band"}, width=1080, height=1920) is None


# ---------------------------------------------------------------------------
# ASS output
# ---------------------------------------------------------------------------


LINES = [
    {"start": 0.5, "end": 2.0, "text": "Hola amigos"},
    {"start": 2.5, "end": 6.0, "text": "Hoy os voy a contar algo que cambió mi vida para siempre, "
                                         "y quiero que lo escuchéis con atención"},
    {"start": 6.5, "end": 8.0, "text": "{\\an8}not an override"},
]


def _style_fields(doc):
    line = next(ln for ln in doc.splitlines() if ln.startswith("Style:"))
    return line[len("Style:"):].strip().split(",")


def _dialogues(doc):
    return [ln for ln in doc.splitlines() if ln.startswith("Dialogue:")]


class TestAssOutput:
    def test_no_placement_is_byte_for_byte_the_house_style(self):
        assert build_ass(LINES, width=1080, height=1920) == build_ass(
            LINES, width=1080, height=1920, placement=None)
        fields = _style_fields(build_ass(LINES, width=1080, height=1920))
        assert fields[18] == str(captions_mod.ALIGNMENT_BOTTOM_CENTER)
        assert fields[19] == fields[20] == str(round(1080 * captions_mod.MARGIN_H_RATIO))
        assert fields[21] == str(round(1920 * captions_mod.MARGIN_V_RATIO))
        assert "\\pos" not in build_ass(LINES, width=1080, height=1920)

    def test_placement_style_and_pos(self):
        doc = build_ass(LINES, width=1080, height=1920,
                        placement=Placement(center_y=0.6219))
        fields = _style_fields(doc)
        assert fields[18] == str(captions_mod.ALIGNMENT_MIDDLE_CENTER)
        assert fields[21] == "0"
        # Everything else in the style is the house style.
        house = _style_fields(build_ass(LINES, width=1080, height=1920))
        assert [f for i, f in enumerate(fields) if i not in (18, 21)] == \
               [f for i, f in enumerate(house) if i not in (18, 21)]
        dialogues = _dialogues(doc)
        assert dialogues
        for d in dialogues:
            text = d.split(",", 9)[9]
            assert text.startswith("{\\pos(540,1194)}")
            # The operator text's own braces are still neutralised.
            assert "{\\an8}" not in text[len("{\\pos(540,1194)}"):]

    def test_placed_captions_allow_three_rows_house_two(self):
        house = _dialogues(build_ass(LINES[1:2], width=1080, height=1920))
        placed = _dialogues(build_ass(LINES[1:2], width=1080, height=1920,
                                      placement=Placement(center_y=0.62)))
        rows = lambda d: d.split(",", 9)[9].count("\\N") + 1  # noqa: E731
        assert max(rows(d) for d in house) <= 2
        assert max(rows(d) for d in placed) <= 3
        assert len(placed) <= len(house)
        assert any(rows(d) == 3 for d in placed) or len(placed) == 1

    def test_max_rows_and_width_are_honoured(self):
        p = Placement(center_y=0.62, max_width=0.5, max_rows=1)
        doc = build_ass(LINES[1:2], width=1080, height=1920, placement=p)
        fields = _style_fields(doc)
        assert fields[19] == fields[20] == str(round(1080 * 0.25))
        limit = captions_mod.usable_width_px(1080, 1920, 0.5)
        font = fields[1]
        size = captions_mod.font_size_px(1920)
        for d in _dialogues(doc):
            text = re.sub(r"^\{\\pos\(\d+,\d+\)\}", "", d.split(",", 9)[9])
            assert "\\N" not in text
            assert captions_mod.measure_text(text, font=font, size_px=size) <= limit + 1

    def test_margin_never_narrower_than_house(self):
        assert captions_mod.margin_h_px(1080, None) == round(1080 * captions_mod.MARGIN_H_RATIO)
        assert captions_mod.margin_h_px(1080, 0.88) == round(1080 * captions_mod.MARGIN_H_RATIO)


def _has_libass() -> bool:
    if not shutil.which("ffmpeg"):
        return False
    out = subprocess.run(["ffmpeg", "-hide_banner", "-filters"], capture_output=True, text=True)
    return " subtitles " in out.stdout


@pytest.mark.skipif(not _has_libass(), reason="ffmpeg with libass not available")
class TestRenderedFrame:
    """Render one frame through libass and measure the box (AICP-368: the
    block centre must sit on center_y for one to three rows)."""

    @pytest.mark.parametrize("text", ["Hola", "Hola amigos de todo el mundo entero hoy",
                                      "una dos tres cuatro cinco seis siete ocho nueve "
                                      "diez once doce trece catorce quince"])
    def test_box_centre_is_on_center_y(self, tmp_path, text):
        import numpy as np

        W, H = 540, 960  # small frame: same ratios, a quarter of the pixels
        doc = build_ass([{"start": 0.0, "end": 1.0, "text": text}], width=W, height=H,
                        placement=Placement(center_y=0.6219))
        ass = tmp_path / "c.ass"
        ass.write_text(doc, encoding="utf-8")
        raw = subprocess.run(
            ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", f"color=white:s={W}x{H}:d=0.5",
             "-vf", f"subtitles={ass}", "-frames:v", "1", "-f", "rawvideo",
             "-pix_fmt", "gray", "-"], capture_output=True, check=True).stdout
        a = np.frombuffer(raw, np.uint8).reshape(H, W)
        rows = np.where((a < 200).any(axis=1))[0]
        assert rows.size
        centre = (rows.min() + rows.max() + 1) / 2 / H
        assert centre == pytest.approx(0.6219, abs=2.5 / H)


# ---------------------------------------------------------------------------
# Lineage, promote, PATCH, derive, scan
# ---------------------------------------------------------------------------


class TestPromote:
    def test_subtitle_removal_parent_gets_provisional_from_fills(self, client, spy, db_session):
        parent, run = _subtitle_lineage(db_session)
        r = client.post(f"/api/v2/runs/{run.id}/to-project", data={"project_type": "localisation"})
        assert r.status_code == 201, r.text
        pid = r.json()["project_id"]
        assert spy["analyze"] == [pid]
        db_session.expire_all()
        p = db_session.get(VideoProject, pid).caption_placement
        assert p["mode"] == "band"
        assert p["center_y"] == pytest.approx(0.6224, abs=1e-4)
        assert p["source"]["kind"] == "derived" and p["source"]["scan"] == "pending"
        assert p["source"]["rev"] == 1 and p["source"]["run_id"] == run.id
        assert p["source"]["project_id"] == parent.id
        assert spy["scan"] == [(pid, 1)]
        assert client.get(f"/api/v2/projects/{pid}").json()["caption_placement"]["mode"] == "band"

    def test_superseded_and_unapplied_fills_are_ignored(self, client, spy, db_session):
        parent, run = _subtitle_lineage(db_session, fills=[FILL_SEG4])
        _fill(db_session, run, [0.1, 0.1, 0.9, 0.2], superseded=True, idx=5)
        _fill(db_session, run, [0.1, 0.8, 0.9, 0.9], status="canceled", idx=6)
        r = client.post(f"/api/v2/runs/{run.id}/to-project", data={"project_type": "localisation"})
        db_session.expire_all()
        p = db_session.get(VideoProject, r.json()["project_id"]).caption_placement
        assert p["band"] == FILL_SEG4

    def test_subtitle_parent_without_fills_waits_for_the_scan(self, client, spy, db_session):
        _, run = _subtitle_lineage(db_session, fills=())
        r = client.post(f"/api/v2/runs/{run.id}/to-project", data={"project_type": "localisation"})
        pid = r.json()["project_id"]
        db_session.expire_all()
        p = db_session.get(VideoProject, pid).caption_placement
        assert p["mode"] == "bottom" and p["source"]["kind"] == "derived"
        assert spy["scan"] == [(pid, 1)]

    def test_parent_placement_is_inherited(self, client, spy, db_session):
        parent = _project(db_session, project_type="face_swap",
                          caption_placement={"mode": "band", "center_y": 0.61,
                                             "source": {"kind": "derived", "rev": 4,
                                                        "scan": "done"}})
        run = _done_run(db_session, parent)
        r = client.post(f"/api/v2/runs/{run.id}/to-project", data={"project_type": "localisation"})
        db_session.expire_all()
        p = db_session.get(VideoProject, r.json()["project_id"]).caption_placement
        assert p["center_y"] == 0.61
        assert p["source"]["kind"] == "inherited" and p["source"]["from_project_id"] == parent.id
        assert p["source"]["origin"] == "derived" and p["source"]["origin_scan"] == "done"
        assert spy["scan"] == []

    def test_plain_parent_gets_nothing(self, client, spy, db_session):
        parent = _project(db_session, project_type="face_swap")
        run = _done_run(db_session, parent)
        r = client.post(f"/api/v2/runs/{run.id}/to-project", data={"project_type": "localisation"})
        db_session.expire_all()
        assert db_session.get(VideoProject, r.json()["project_id"]).caption_placement is None
        assert spy["scan"] == []

    def test_enqueue_failure_does_not_fail_the_promote(self, client, spy, db_session, monkeypatch):
        import app.tasks as tasks_mod

        def boom(pid, rev):
            raise ConnectionError("redis down")

        monkeypatch.setattr(tasks_mod, "enqueue_caption_placement_scan", boom)
        _, run = _subtitle_lineage(db_session)
        r = client.post(f"/api/v2/runs/{run.id}/to-project", data={"project_type": "localisation"})
        assert r.status_code == 201


class TestPromoteGateFixes:
    def test_pending_parent_is_not_frozen_into_the_child(self, client, spy, db_session):
        _, run = _subtitle_lineage(db_session, fills=())
        r = client.post(f"/api/v2/runs/{run.id}/to-project", data={"project_type": "face_swap"})
        mid_id = r.json()["project_id"]
        db_session.expire_all()
        mid = db_session.get(VideoProject, mid_id)
        assert mid.caption_placement["source"]["scan"] == "pending"
        run2 = _done_run(db_session, mid)
        r2 = client.post(f"/api/v2/runs/{run2.id}/to-project", data={"project_type": "localisation"})
        leaf_id = r2.json()["project_id"]
        db_session.expire_all()
        leaf = db_session.get(VideoProject, leaf_id).caption_placement
        assert leaf["source"]["kind"] == "derived" and leaf["source"]["run_id"] == run.id
        assert spy["scan"] == [(mid_id, 1), (leaf_id, 1)]

    def test_pre_feature_parent_without_placement_derives_from_lineage(
        self, client, spy, db_session
    ):
        _, run = _subtitle_lineage(db_session)
        mid = _project(db_session, project_type="localisation", source_run_id=run.id)
        run2 = _done_run(db_session, mid)
        r = client.post(f"/api/v2/runs/{run2.id}/to-project", data={"project_type": "localisation"})
        db_session.expire_all()
        p = db_session.get(VideoProject, r.json()["project_id"]).caption_placement
        assert p["mode"] == "band" and p["source"]["kind"] == "derived"

    def test_pending_parent_hands_its_frozen_fills_to_the_child(self, client, spy, db_session):
        """A fill applied to the subtitle run AFTER the parent copied its final
        is not in the child's video either."""
        _, run = _subtitle_lineage(db_session, fills=[FILL_SEG4])
        mid_id = client.post(f"/api/v2/runs/{run.id}/to-project",
                             data={"project_type": "face_swap"}).json()["project_id"]
        _fill(db_session, run, [0.1, 0.1, 0.9, 0.2], idx=8)  # later re-delivery
        db_session.expire_all()
        run2 = _done_run(db_session, db_session.get(VideoProject, mid_id))
        leaf_id = client.post(f"/api/v2/runs/{run2.id}/to-project",
                              data={"project_type": "localisation"}).json()["project_id"]
        db_session.expire_all()
        leaf = db_session.get(VideoProject, leaf_id).caption_placement
        assert leaf["band"] == FILL_SEG4 and leaf["source"]["fills"] == 1

    def test_inherited_value_keeps_frozen_evidence_for_a_rederive(
        self, client, spy, db_session, test_get_session, monkeypatch
    ):
        from app import qa_tools

        monkeypatch.setattr(qa_tools, "calibrate_captions",
                            lambda *a, **k: dict(CAL_REAL, calibrated=False))
        _, run = _subtitle_lineage(db_session, fills=[FILL_SEG4])
        mid_id = client.post(f"/api/v2/runs/{run.id}/to-project",
                             data={"project_type": "face_swap"}).json()["project_id"]
        cp.scan_and_store(mid_id, 1)  # parent settles
        db_session.expire_all()
        run2 = _done_run(db_session, db_session.get(VideoProject, mid_id))
        leaf_id = client.post(f"/api/v2/runs/{run2.id}/to-project",
                              data={"project_type": "localisation"}).json()["project_id"]
        db_session.expire_all()
        leaf = db_session.get(VideoProject, leaf_id)
        assert leaf.caption_placement["source"]["kind"] == "inherited"
        assert leaf.caption_placement["source"]["fill_band"] == FILL_SEG4
        # The filled segment is replaced on the subtitle run afterwards...
        for job in db_session.query(RunSegmentFillJob).filter_by(run_id=run.id):
            db_session.get(RunSegment, job.run_segment_id).local_result_path = "/elsewhere.mp4"
        db_session.commit()
        cp.request_derivation(db_session, leaf)
        db_session.expire_all()
        # ...but the leaf's video still contains the patch.
        assert db_session.get(VideoProject, leaf_id).caption_placement["band"] == FILL_SEG4

    def test_republished_final_fills_are_not_evidence(self, client, spy, db_session):
        _, run = _subtitle_lineage(db_session, republished=True)
        r = client.post(f"/api/v2/runs/{run.id}/to-project", data={"project_type": "localisation"})
        db_session.expire_all()
        p = db_session.get(VideoProject, r.json()["project_id"]).caption_placement
        assert p["mode"] == "bottom" and p["source"]["fills"] == 0
        assert spy["scan"]  # the detector scan still runs

    def test_audio_patched_final_keeps_its_fills(self, db_session):
        from datetime import datetime, timezone

        from app.models import RunAudioPatch

        _, run = _subtitle_lineage(db_session, fills=[FILL_SEG4])
        pipeline_final = run.result_local_path
        patched = _storage_file("patched.mp4", b"patched")
        db_session.add(RunAudioPatch(
            run_id=run.id, donor_run_id=run.id, target_start_sec=0.0, target_end_sec=1.0,
            donor_start_sec=0.0, crossfade_sec=0.0, base_result_path=pipeline_final,
            status="done", output_path=patched,
            promoted_at=datetime.now(timezone.utc).replace(tzinfo=None)))
        run.result_local_path = patched
        db_session.commit()
        assert [f["box"] for f in cp.delivered_fill_boxes(db_session, run)] == [FILL_SEG4]

    def test_range_attempt_final_keeps_other_segments_fills(self, db_session):
        from datetime import datetime, timezone

        from app.models import RunSegmentRangeAttempt

        _, run = _subtitle_lineage(db_session, fills=[FILL_SEG4])
        attempt_final = _storage_file("range_final.mp4", b"range")
        other = RunSegment(run_id=run.id, segment_def_id="def-range", index=9,
                           status="completed")
        db_session.add(other)
        db_session.flush()
        db_session.add(RunSegmentRangeAttempt(
            run_id=run.id, run_segment_id=other.id, final_local_path=attempt_final,
            promoted_at=datetime.now(timezone.utc).replace(tzinfo=None),
            **_range_attempt_required()))
        run.result_local_path = attempt_final
        db_session.commit()
        assert [f["box"] for f in cp.delivered_fill_boxes(db_session, run)] == [FILL_SEG4]

    def test_assistant_published_final_drops_fills(self, db_session):
        _, run = _subtitle_lineage(db_session, fills=[FILL_SEG4])
        run.result_local_path = _storage_file("assistant.mp4", b"pub")
        db_session.commit()
        assert cp.delivered_fill_boxes(db_session, run) == []

    def test_manual_edit_keeps_the_measured_band_as_evidence(self):
        derived = cp.derive_placement(fill_boxes=[], calibration=CAL_REAL,
                                      source={"kind": "derived", "rev": 1, "run_id": "r1"})
        v = cp.manual({"mode": "band", "center_y": 0.5}, current=derived)
        assert v["source"]["evidence_band"] == derived["band"]
        v2 = cp.manual(None, current=v)
        assert v2["source"]["evidence_band"] == derived["band"]

    def test_inherited_manual_bottom_carries_the_measured_band(self, client, spy, db_session):
        measured = cp.derive_placement(fill_boxes=[], calibration=CAL_REAL,
                                       source={"kind": "derived", "rev": 1, "run_id": "r"})
        parent = _project(db_session, project_type="face_swap",
                          caption_placement=cp.manual(None, current=measured))
        run = _done_run(db_session, parent)
        r = client.post(f"/api/v2/runs/{run.id}/to-project",
                        data={"project_type": "subtitle_removal"})
        db_session.expire_all()
        child = db_session.get(VideoProject, r.json()["project_id"]).caption_placement
        assert child["mode"] == "bottom" and child["source"]["kind"] == "inherited"
        assert child["source"]["evidence_band"] == measured["band"]

    def test_fill_waiting_for_its_restitch_is_not_evidence(self, db_session):
        """Run done but its final is still the one the fill replaced (the copy
        happened before the re-stitch): not in the copied file."""
        from app.assistant_run_result import result_revision

        _, run = _subtitle_lineage(db_session, fills=())
        job = _fill(db_session, run, FILL_SEG4, idx=3)
        job.base = {"run": {"final_revision": result_revision(run)}}
        db_session.commit()
        assert cp.delivered_fill_boxes(db_session, run) == []

    def _second_pass(self, client, db_session, *, parent_placement="keep"):
        """sub1 (fill FILL_SEG4) -> promoted into sub2 (a 2nd removal pass,
        fill UPPER on its own run) -> run2 of sub2, ready to promote."""
        _, run1 = _subtitle_lineage(db_session, fills=[FILL_SEG4])
        sub2_id = client.post(f"/api/v2/runs/{run1.id}/to-project",
                              data={"project_type": "subtitle_removal"}).json()["project_id"]
        db_session.expire_all()
        sub2 = db_session.get(VideoProject, sub2_id)
        sub2.width, sub2.height = 1080, 1920
        sub2.source_local_path = _storage_file("source2.mp4", b"src2")
        if parent_placement != "keep":
            sub2.caption_placement = parent_placement
        db_session.commit()
        run2 = _done_run(db_session, sub2)
        _fill(db_session, run2, self.UPPER, idx=4)
        return run1, sub2, run2

    UPPER = [0.2, 0.30, 0.8, 0.36]

    @pytest.mark.parametrize("parent_placement", [
        "keep",                                                  # pending derived
        {"mode": "band", "center_y": 0.5,                        # manual, no band
         "source": {"kind": "manual", "rev": 2}},
        {"mode": "bottom", "source": {"kind": "manual", "rev": 2}},
    ])
    def test_second_removal_pass_unions_both_passes(self, client, spy, db_session,
                                                    parent_placement):
        run1, sub2, run2 = self._second_pass(client, db_session,
                                             parent_placement=parent_placement)
        r = client.post(f"/api/v2/runs/{run2.id}/to-project", data={"project_type": "localisation"})
        pid = r.json()["project_id"]
        db_session.expire_all()
        p = db_session.get(VideoProject, pid).caption_placement
        assert p["source"]["kind"] == "derived" and p["source"]["run_id"] == run2.id
        assert [c["run_id"] for c in p["source"]["chain"]] == [run2.id, run1.id]
        assert p["band"] == [0.2, 0.30, 0.8, 0.6627]  # both passes' fills
        assert (pid, 1) in spy["scan"]

    def test_earlier_pass_fills_come_from_its_copier_frozen_evidence(self, client, spy,
                                                                      db_session):
        """sub2 froze run1's fill when it copied run1's final; superseding that
        fill afterwards must not drop it from a later pass's union."""
        run1, sub2, run2 = self._second_pass(client, db_session)
        db_session.expire_all()
        assert db_session.get(VideoProject, sub2.id).caption_placement["source"]["fill_band"] \
            == FILL_SEG4
        for job in db_session.query(RunSegmentFillJob).filter_by(run_id=run1.id):
            db_session.get(RunSegment, job.run_segment_id).local_result_path = "/elsewhere.mp4"
        db_session.commit()
        r = client.post(f"/api/v2/runs/{run2.id}/to-project", data={"project_type": "localisation"})
        db_session.expire_all()
        p = db_session.get(VideoProject, r.json()["project_id"]).caption_placement
        assert p["band"] == [0.2, 0.30, 0.8, 0.6627]

    def test_earlier_pass_evidence_survives_a_deleted_link(self, client, spy, db_session):
        run1, sub2, run2 = self._second_pass(client, db_session)
        db_session.delete(db_session.get(VideoProject, run1.project_id))  # pass 1 gone
        db_session.commit()
        r = client.post(f"/api/v2/runs/{run2.id}/to-project", data={"project_type": "localisation"})
        db_session.expire_all()
        p = db_session.get(VideoProject, r.json()["project_id"]).caption_placement
        assert p["band"] == [0.2, 0.30, 0.8, 0.6627]

    def test_earlier_pass_detector_band_survives_a_deleted_link(self, client, spy, db_session):
        run1, sub2, run2 = self._second_pass(client, db_session)
        db_session.expire_all()
        sub2 = db_session.get(VideoProject, sub2.id)
        # sub2's settled value: fills + the first pass's detector band.
        sub2.caption_placement = cp.derive_placement(
            fill_boxes=[], calibration=CAL_REAL,
            source={"kind": "derived", "rev": 2, "run_id": run1.id, "scan": "done"})
        db_session.delete(db_session.get(VideoProject, run1.project_id))
        db_session.commit()
        db_session.query(RunSegmentFillJob).filter_by(run_id=run2.id).delete()
        db_session.commit()
        r = client.post(f"/api/v2/runs/{run2.id}/to-project", data={"project_type": "localisation"})
        db_session.expire_all()
        p = db_session.get(VideoProject, r.json()["project_id"]).caption_placement
        assert p["band"] == [0.27, 0.605, 0.73, 0.643]

    def test_one_failed_pass_fails_the_scan_and_children_rederive(
        self, client, spy, db_session, test_get_session, monkeypatch
    ):
        from app import qa_tools

        run1, sub2, run2 = self._second_pass(client, db_session)
        pid = client.post(f"/api/v2/runs/{run2.id}/to-project",
                          data={"project_type": "face_swap"}).json()["project_id"]
        sub1 = db_session.get(VideoProject, run1.project_id)

        def cal(path, **kw):
            if path == sub1.source_local_path:
                raise RuntimeError("decode failed")
            return dict(CAL_REAL)

        monkeypatch.setattr(qa_tools, "calibrate_captions", cal)
        out = cp.scan_and_store(pid, 1)
        assert out["source"]["scan"] == "failed" and "decode failed" in out["source"]["scan_error"]
        db_session.expire_all()
        run3 = _done_run(db_session, db_session.get(VideoProject, pid))
        spy["scan"].clear()
        leaf_id = client.post(f"/api/v2/runs/{run3.id}/to-project",
                              data={"project_type": "localisation"}).json()["project_id"]
        db_session.expire_all()
        leaf = db_session.get(VideoProject, leaf_id).caption_placement
        assert leaf["source"]["kind"] == "derived" and spy["scan"] == [(leaf_id, 1)]

    def test_failed_parent_without_a_chain_is_still_inherited(self, client, spy, db_session):
        parent = _project(db_session, project_type="face_swap",
                          caption_placement={"mode": "band", "center_y": 0.62,
                                             "source": {"kind": "derived", "rev": 2,
                                                        "scan": "failed"}})
        run = _done_run(db_session, parent)
        r = client.post(f"/api/v2/runs/{run.id}/to-project", data={"project_type": "localisation"})
        db_session.expire_all()
        p = db_session.get(VideoProject, r.json()["project_id"]).caption_placement
        assert p["source"]["kind"] == "inherited" and p["center_y"] == 0.62

    def test_second_pass_scan_reads_every_pass_source(self, client, spy, db_session,
                                                      test_get_session, monkeypatch):
        from app import qa_tools

        run1, sub2, run2 = self._second_pass(client, db_session)
        pid = client.post(f"/api/v2/runs/{run2.id}/to-project",
                          data={"project_type": "localisation"}).json()["project_id"]
        sub1 = db_session.get(VideoProject, run1.project_id)
        seen = []

        def cal(path, **kw):
            seen.append(path)
            if path == sub1.source_local_path:
                return dict(CAL_REAL)
            return dict(CAL_REAL, calibrated=False, hit_ratio=0.0, geometry=None)

        monkeypatch.setattr(qa_tools, "calibrate_captions", cal)
        out = cp.scan_and_store(pid, 1)
        assert seen == [sub2.source_local_path, sub1.source_local_path]
        assert out["source"]["scan"] == "done" and len(out["source"]["detectors"]) == 2
        assert out["band"] == [0.2, 0.30, 0.8, 0.6627]

    def test_many_fills_do_not_fail_the_promote(self, client, spy, db_session):
        _, run = _subtitle_lineage(db_session, fills=[FILL_SEG4] * 60)
        r = client.post(f"/api/v2/runs/{run.id}/to-project", data={"project_type": "localisation"})
        assert r.status_code == 201, r.text
        db_session.expire_all()
        p = db_session.get(VideoProject, r.json()["project_id"]).caption_placement
        assert p["source"]["fills"] == 60 and len(p["source"]["fill_ids"]) == 20

    def test_placement_error_never_fails_the_promote(self, client, spy, db_session, monkeypatch):
        def boom(*a, **k):
            raise RuntimeError("unexpected")

        monkeypatch.setattr(cp, "inherit_on_promote", boom)
        _, run = _subtitle_lineage(db_session)
        r = client.post(f"/api/v2/runs/{run.id}/to-project", data={"project_type": "localisation"})
        assert r.status_code == 201
        db_session.expire_all()
        assert db_session.get(VideoProject, r.json()["project_id"]).caption_placement is None
        assert spy["scan"] == []


class TestLineage:
    def test_walks_back_through_intermediate_projects(self, db_session):
        sub, run1 = _subtitle_lineage(db_session)
        mid = _project(db_session, project_type="face_swap", source_run_id=run1.id)
        run2 = _done_run(db_session, mid)
        leaf = _project(db_session, project_type="localisation", source_run_id=run2.id)
        found = cp.find_lineage(db_session, leaf)
        assert found is not None and found[0].id == run1.id and found[1].id == sub.id

    def test_no_backlink(self, db_session):
        assert cp.find_lineage(db_session, _project(db_session)) is None

    def test_cycle_terminates(self, db_session):
        a = _project(db_session, project_type="face_swap")
        ra = _done_run(db_session, a)
        a.source_run_id = ra.id
        db_session.commit()
        assert cp.find_lineage(db_session, a) is None


class TestPatch:
    def test_null_stores_sticky_bottom(self, client, db_session):
        p = _project(db_session, caption_placement={"mode": "band", "center_y": 0.6,
                                                    "source": {"kind": "derived", "rev": 2}})
        r = client.patch(f"/api/v2/projects/{p.id}", json={"caption_placement": None})
        assert r.status_code == 200, r.text
        v = r.json()["caption_placement"]
        assert v["mode"] == "bottom" and v["source"] == {**v["source"], "kind": "manual", "rev": 3}

    def test_object_is_validated_and_marked_manual(self, client, db_session):
        p = _project(db_session)
        r = client.patch(f"/api/v2/projects/{p.id}",
                         json={"caption_placement": {"mode": "band", "center_y": 0.58}})
        assert r.status_code == 200
        assert r.json()["caption_placement"]["source"]["kind"] == "manual"

    def test_invalid_is_400(self, client, db_session):
        p = _project(db_session)
        r = client.patch(f"/api/v2/projects/{p.id}",
                         json={"caption_placement": {"mode": "band", "center_y": 2}})
        assert r.status_code == 400

    def test_other_fields_in_the_same_patch_are_kept(self, client, db_session):
        p = _project(db_session, name="old")
        r = client.patch(f"/api/v2/projects/{p.id}",
                         json={"name": "new", "mute_source": True, "caption_placement": None})
        assert r.status_code == 200
        db_session.expire_all()
        row = db_session.get(VideoProject, p.id)
        assert row.name == "new" and row.mute_source is True
        assert row.caption_placement["mode"] == "bottom"

    def test_omitted_leaves_it_alone(self, client, db_session):
        p = _project(db_session, caption_placement={"mode": "band", "center_y": 0.6})
        client.patch(f"/api/v2/projects/{p.id}", json={"name": "x"})
        db_session.expire_all()
        assert db_session.get(VideoProject, p.id).caption_placement["center_y"] == 0.6


class TestScanAndDerive:
    def _derived_child(self, db_session, **kw):
        sub, run = _subtitle_lineage(db_session, **kw)
        child = _project(db_session, project_type="localisation", source_run_id=run.id)
        return sub, run, child

    def test_scan_writes_the_union_and_bumps_rev(self, db_session, test_get_session, monkeypatch):
        from app import qa_tools

        sub, run, child = self._derived_child(db_session)
        rev = cp.request_derivation(db_session, child)
        seen = {}

        def fake_cal(path, **kw):
            seen.update(path=path, **kw)
            return dict(CAL_REAL, source=path)

        monkeypatch.setattr(qa_tools, "calibrate_captions", fake_cal)
        out = cp.scan_and_store(child.id, rev)
        assert seen["path"] == sub.source_local_path and seen["fps"] == cp.SCAN_FPS
        assert out["source"]["scan"] == "done" and out["source"]["rev"] == rev + 1
        assert "source" not in out["source"]["detector"]
        assert 0.60 <= out["center_y"] <= 0.625
        db_session.expire_all()
        assert db_session.get(VideoProject, child.id).caption_placement == out

    def test_scan_uses_the_fills_frozen_at_promote(self, db_session, test_get_session,
                                                    monkeypatch):
        """A fill applied to the parent run AFTER the copy is not in the
        child's video, so the scan must not pick it up."""
        from app import qa_tools

        _, run, child = self._derived_child(db_session, fills=[FILL_SEG4])
        rev = cp.request_derivation(db_session, child)
        _fill(db_session, run, [0.1, 0.1, 0.9, 0.2], idx=9)  # later re-delivery
        monkeypatch.setattr(qa_tools, "calibrate_captions",
                            lambda *a, **k: dict(CAL_REAL, calibrated=False))
        out = cp.scan_and_store(child.id, rev)
        assert out["band"] == FILL_SEG4
        assert out["source"]["fills"] == 1 and out["source"]["fill_boxes"] == [FILL_SEG4]

    def test_explicit_derive_without_frozen_evidence_uses_fills_up_to_the_copy(
        self, db_session, test_get_session
    ):
        from datetime import datetime, timedelta, timezone

        _, run, child = self._derived_child(db_session, fills=[FILL_SEG4])
        later = datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(minutes=5)
        _fill(db_session, run, [0.1, 0.1, 0.9, 0.2], idx=7, applied_at=later)
        cp.request_derivation(db_session, child)
        db_session.expire_all()
        assert db_session.get(VideoProject, child.id).caption_placement["band"] == FILL_SEG4

    def test_explicit_derive_reuses_frozen_evidence(self, db_session, test_get_session):
        _, run, child = self._derived_child(db_session, fills=[FILL_SEG4])
        cp.request_derivation(db_session, child)
        _fill(db_session, run, [0.1, 0.1, 0.9, 0.2], idx=6)  # applied before? frozen wins
        rev = cp.request_derivation(db_session, child)
        db_session.expire_all()
        v = db_session.get(VideoProject, child.id).caption_placement
        assert v["band"] == FILL_SEG4 and v["source"]["rev"] == rev

    def test_manual_edit_keeps_frozen_evidence_for_a_rederive(self, db_session, test_get_session):
        _, run, child = self._derived_child(db_session, fills=[FILL_SEG4])
        cp.request_derivation(db_session, child)
        db_session.refresh(child)
        child.caption_placement = cp.manual({"mode": "band", "center_y": 0.5},
                                            current=child.caption_placement)
        db_session.commit()
        assert child.caption_placement["source"]["fill_band"] == FILL_SEG4
        for job in db_session.query(RunSegmentFillJob).filter_by(run_id=run.id):
            db_session.get(RunSegment, job.run_segment_id).local_result_path = "/elsewhere.mp4"
        db_session.commit()
        cp.request_derivation(db_session, child)
        db_session.expire_all()
        assert db_session.get(VideoProject, child.id).caption_placement["band"] == FILL_SEG4

    def test_stale_rev_and_manual_values_are_not_overwritten(
        self, db_session, test_get_session, monkeypatch
    ):
        from app import qa_tools

        monkeypatch.setattr(qa_tools, "calibrate_captions", lambda *a, **k: dict(CAL_REAL))
        _, _, child = self._derived_child(db_session)
        rev = cp.request_derivation(db_session, child)
        rev2 = cp.request_derivation(db_session, child)
        assert rev2 == rev + 1
        assert cp.scan_and_store(child.id, rev) is None  # out of order
        child.caption_placement = cp.manual(None, current=child.caption_placement)
        db_session.commit()
        assert cp.scan_and_store(child.id, rev2) is None  # operator's choice wins
        db_session.expire_all()
        assert db_session.get(VideoProject, child.id).caption_placement["mode"] == "bottom"

    def test_superseded_while_scanning(self, db_session, test_get_session, monkeypatch, SessionFactory):
        from app import qa_tools

        _, _, child = self._derived_child(db_session)
        rev = cp.request_derivation(db_session, child)

        def cal_and_edit(*a, **k):
            s = SessionFactory()
            row = s.get(VideoProject, child.id)
            row.caption_placement = cp.manual({"mode": "band", "center_y": 0.5},
                                              current=row.caption_placement)
            s.commit()
            s.close()
            return dict(CAL_REAL)

        monkeypatch.setattr(qa_tools, "calibrate_captions", cal_and_edit)
        assert cp.scan_and_store(child.id, rev) is None
        db_session.expire_all()
        assert db_session.get(VideoProject, child.id).caption_placement["center_y"] == 0.5

    def test_scan_failure_keeps_fill_placement(self, db_session, test_get_session, monkeypatch):
        from app import qa_tools

        def boom(*a, **k):
            raise RuntimeError("decode failed")

        monkeypatch.setattr(qa_tools, "calibrate_captions", boom)
        _, _, child = self._derived_child(db_session)
        out = cp.scan_and_store(child.id, cp.request_derivation(db_session, child))
        assert out["source"]["scan"] == "failed" and "decode failed" in out["source"]["scan_error"]
        assert out["band"] == [0.2256, 0.5821, 0.7763, 0.6627]

    def test_missing_source_is_a_failed_scan(self, db_session, test_get_session):
        sub, _, child = self._derived_child(db_session)
        os.remove(sub.source_local_path)
        out = cp.scan_and_store(child.id, cp.request_derivation(db_session, child))
        assert out["source"]["scan"] == "failed"

    def test_deleted_lineage_settles_the_pending_scan(self, db_session, test_get_session):
        sub, run, child = self._derived_child(db_session, fills=[FILL_SEG4])
        rev = cp.request_derivation(db_session, child)
        db_session.delete(db_session.get(VideoProject, sub.id))  # cascades to the run
        db_session.commit()
        out = cp.scan_and_store(child.id, rev)
        assert out["source"]["scan"] == "failed"
        assert "lineage deleted" in out["source"]["scan_error"]
        assert out["band"] == FILL_SEG4 and out["source"]["rev"] == rev + 1

    def test_derive_endpoint(self, client, spy, db_session):
        _, _, child = self._derived_child(db_session)
        child.caption_placement = cp.manual(None)
        db_session.commit()
        r = client.post(f"/api/v2/projects/{child.id}/caption-placement/derive")
        assert r.status_code == 200, r.text
        v = r.json()["caption_placement"]
        assert v["source"]["kind"] == "derived" and v["source"]["rev"] == 2
        assert spy["scan"] == [(child.id, 2)]

    def test_derive_endpoint_without_lineage_is_409(self, client, db_session):
        p = _project(db_session)
        assert client.post(f"/api/v2/projects/{p.id}/caption-placement/derive").status_code == 409

    def test_derive_endpoint_503_when_not_enqueued(self, client, db_session, monkeypatch):
        import app.tasks as tasks_mod

        def boom(pid, rev):
            raise ConnectionError("redis down")

        monkeypatch.setattr(tasks_mod, "enqueue_caption_placement_scan", boom)
        _, _, child = self._derived_child(db_session)
        r = client.post(f"/api/v2/projects/{child.id}/caption-placement/derive")
        assert r.status_code == 503
        db_session.expire_all()
        assert db_session.get(VideoProject, child.id).caption_placement["source"]["scan"] == "pending"


class TestCaptionSnapshot:
    def _run(self, db_session, placement):
        p = _project(db_session, project_type="localisation", caption_placement=placement)
        # The run page shows the caption control only next to a Drive link.
        return _done_run(db_session, p, result_gdrive_file_id="drive-file-id")

    def test_default_copies_the_project_placement(self, client, spy, db_session):
        placement = {"mode": "band", "center_y": 0.62, "source": {"kind": "derived", "rev": 1}}
        run = self._run(db_session, placement)
        r = client.post(f"/api/v2/runs/{run.id}/captions", data={"source_tail_edit_id": ""})
        assert r.status_code == 201, r.text
        cap = db_session.get(RunCaption, r.json()["caption_id"])
        assert cap.placement == placement
        listed = client.get(f"/api/v2/runs/{run.id}/captions").json()
        assert listed[0]["placement"]["center_y"] == 0.62

    def test_bottom_opt_out(self, client, spy, db_session):
        run = self._run(db_session, {"mode": "band", "center_y": 0.62})
        r = client.post(f"/api/v2/runs/{run.id}/captions", data={"placement": "bottom"})
        cap = db_session.get(RunCaption, r.json()["caption_id"])
        assert cap.placement["mode"] == "bottom"

    def test_no_project_placement_is_null(self, client, spy, db_session):
        run = self._run(db_session, None)
        r = client.post(f"/api/v2/runs/{run.id}/captions", data={"placement": "project"})
        assert db_session.get(RunCaption, r.json()["caption_id"]).placement is None

    def test_bad_value_is_400(self, client, spy, db_session):
        run = self._run(db_session, None)
        r = client.post(f"/api/v2/runs/{run.id}/captions", data={"placement": "top"})
        assert r.status_code == 400

    def test_retry_keeps_the_frozen_placement(self, client, spy, db_session):
        run = self._run(db_session, {"mode": "band", "center_y": 0.62})
        cid = client.post(f"/api/v2/runs/{run.id}/captions", data={}).json()["caption_id"]
        cap = db_session.get(RunCaption, cid)
        cap.status = "failed"
        project = db_session.get(VideoProject, run.project_id)
        project.caption_placement = cp.manual(None)
        db_session.commit()
        assert client.post(f"/api/v2/captions/{cid}/retry").status_code == 200
        db_session.expire_all()
        assert db_session.get(RunCaption, cid).placement["center_y"] == 0.62

    def test_run_page_offers_the_position_picker(self, client, db_session):
        run = self._run(db_session, {"mode": "band", "center_y": 0.62,
                                     "source": {"kind": "derived", "scan": "pending"}})
        html = client.get(f"/v2/runs/{run.id}").text
        assert 'id="captions-placement"' in html
        assert "Over original captions (y&nbsp;0.62, scan pending)" in html

    def test_run_page_without_placement_has_no_picker(self, client, db_session):
        run = self._run(db_session, None)
        assert 'id="captions-placement"' not in client.get(f"/v2/runs/{run.id}").text


def _range_attempt_required() -> dict:
    return dict(requested_start_sec=0.0, requested_end_sec=1.0,
                base_segment_snapshot=[], candidate_segment_snapshot=[])
