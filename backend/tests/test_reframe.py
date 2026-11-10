"""
Reframe workflow (AICP-251): the pure helpers of app/reframe.py, the render on
real synthetic media, and the schema (migration vs ORM, drain check).
The flow with the database and the provider layer is in test_reframe_flow.py.
"""

from __future__ import annotations

import importlib.util
import math
import os
import re
import sys
import tempfile

_db_fd, _db_path = tempfile.mkstemp(suffix=".db")
os.close(_db_fd)
os.environ.setdefault("DATABASE_URL", f"sqlite:///{_db_path}")
os.environ.setdefault("DATA_DIR", tempfile.mkdtemp())
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

import pytest

from app import project_types as pt
from app import reframe as rf
from app.reframe_models import ReframeJob, ReframeRender
from reframe_media import frame_luma, make_flac, make_raw, make_source

REPO = os.path.join(os.path.dirname(__file__), "..", "..")


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_reframe_project_type_registered():
    spec = pt.spec_for(pt.REFRAME)
    assert pt.REFRAME in pt.VALID_PROJECT_TYPES
    assert spec.supports_runs is False
    assert spec.segmentation == pt.SEG_BLANK
    assert "FAL" in spec.hint.upper()
    assert all(pt.spec_for(k).supports_runs for k in pt.PROJECT_TYPES if k != pt.REFRAME)


def test_model_enum_lists_reframe():
    from app.models import VideoProject

    assert "reframe" in VideoProject.__table__.c.project_type.type.enums


# ---------------------------------------------------------------------------
# Generation parameters
# ---------------------------------------------------------------------------


def test_target_aspect_and_estimate():
    assert rf.target_aspect_for(1920, 1080) == "9:16"
    assert rf.target_aspect_for(1080, 1920) == "16:9"
    assert rf.target_aspect_for(1080, 1080) is None
    assert rf.estimate_usd("480p", 5.0) == (0.25, 0.5)
    base, reserved = rf.estimate_usd("768p", 26.0)
    assert base == pytest.approx(2.08) and reserved == pytest.approx(4.16)
    assert rf.output_size("9:16", "768p") == (768, 1344)
    assert rf.output_size("16:9", "480p") == (832, 480)


@pytest.mark.parametrize("kw, msg", [
    (dict(aspect="16:9"), "already"),
    (dict(resolution="1080p"), "no Fal price"),
    (dict(window_start=5, window_end=4), "0 <= start < end"),
    (dict(window_end=40), "0 <= start < end"),
    (dict(window_start=0, window_end=0.5), "at least"),
    (dict(window_end=float("nan")), "finite"),
    (dict(source_duration=None), "unknown"),
])
def test_validate_generation_refusals(kw, msg):
    args = dict(window_start=0.0, window_end=10.0, aspect="9:16", resolution="768p",
                source_duration=30.0, source_width=1920, source_height=1080)
    args.update(kw)
    with pytest.raises(rf.ReframeError, match=msg):
        rf.validate_generation(**args)


def test_validate_generation_max_window(monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "REFRAME_MAX_WINDOW_SEC", 120.0)
    rf.validate_generation(window_start=0, window_end=120, aspect="9:16", resolution="768p",
                           source_duration=300, source_width=1920, source_height=1080)
    with pytest.raises(rf.ReframeError, match="at most"):
        rf.validate_generation(window_start=0, window_end=121, aspect="9:16",
                               resolution="768p", source_duration=300, source_width=1920,
                               source_height=1080)


# ---------------------------------------------------------------------------
# Ending math
# ---------------------------------------------------------------------------

AICP252 = rf.JobTimes(input_start=0.0, window_end=25.75, raw_frames=618, has_audio=True,
                      audio_end_max=28.75)


def test_aicp252_ending_frames():
    f = rf.validate_ending(AICP252, rf.Ending(25.75, 0.65, 26.65, 0.25))
    assert (f.np, f.nf, f.nt) == (618, 16, 6)
    assert f.na == round(26.65 * 48000)
    assert f.nl == max(618 + 16, math.ceil(f.na / 2000)) + 6 == 646
    assert f.nl / 24 == pytest.approx(26.917, abs=0.001)


def test_default_ending_is_valid_by_construction():
    for t in (AICP252,
              rf.JobTimes(1.0333, 11.0, 239, True, 14.0),          # raw a frame short
              rf.JobTimes(0.0, 10.0, 241, True, 10.0),             # raw a frame long
              rf.JobTimes(0.0, 10.0, 240, False, 0.0)):
        d = rf.default_ending(t)
        f = rf.validate_ending(t, d)
        assert 1 <= f.np <= t.raw_frames
        if not t.has_audio:
            assert d.audio_end is None and f.na == 0


@pytest.mark.parametrize("e, msg", [
    (rf.Ending(0.0, 0.65, 1.0, 0.25), "within the generated picture"),
    (rf.Ending(26.0, 0.65, 26.65, 0.25), "within the generated picture"),
    (rf.Ending(25.75, 2.5, 26.65, 0.25), "fade_sec"),
    (rf.Ending(25.75, 0.65, 26.65, 6.0), "black_tail_sec"),
    (rf.Ending(25.75, 0.65, 25.0, 0.25), "audio_end_sec"),
    (rf.Ending(25.75, 0.65, 29.0, 0.25), "audio_end_sec"),
    (rf.Ending(25.75, 0.65, None, 0.25), "required"),
])
def test_validate_ending_refusals(e, msg):
    with pytest.raises(rf.ReframeError, match=msg):
        rf.validate_ending(AICP252, e)


def test_audio_end_one_frame_before_picture_end_is_allowed():
    rf.validate_ending(AICP252, rf.Ending(25.75, 0.65, 25.75 - 1 / 24, 0.25))


def test_no_audio_source_ignores_audio_end():
    t = rf.JobTimes(0.0, 10.0, 240, False, 0.0)
    f = rf.validate_ending(t, rf.Ending(10.0, 0.0, 99.0, 0.0))
    assert f.na == 0 and f.nl == 240


@pytest.mark.parametrize("p, want", [
    ({"x": 0, "y": 456, "width": 768, "height": 432}, (0, 456, 768, 432)),
    ({"left": 1, "top": 2, "right": 11, "bottom": 22}, (1, 2, 10, 20)),
    ([4, 5, 6, 7], (4, 5, 6, 7)),
    ({"x": 0}, None),
    ("centre", None),
    (None, None),
])
def test_parse_placement(p, want):
    assert rf.parse_placement({"placement": p}) == want


def test_input_bitrate_bounds():
    cap = 90 * 1024 * 1024
    assert rf.input_bitrate(1.0, cap) == 12_000_000
    r120 = rf.input_bitrate(120.0, cap)
    assert 4_500_000 < r120 < 5_500_000
    assert r120 * 120 / 8 < 0.9 * cap


# ---------------------------------------------------------------------------
# Real media: prepare demux and render
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp(tmp_path):
    return str(tmp_path)


def test_first_frame_and_prepare_demux(tmp):
    from app import media

    src = make_source(os.path.join(tmp, "src.mp4"), sec=6.0)
    s = rf.first_frame_at_or_after(src, 1.01)
    assert s == pytest.approx(31 / 30, abs=1e-3)
    vid, aud = os.path.join(tmp, "input.mp4"), os.path.join(tmp, "audio.flac")
    media._run(rf.prepare_argv(src=src, input_start=s, video_sec=4.0 - s, audio_sec=6.0 - s,
                               fps=30.0, video_dst=vid, audio_dst=aud,
                               bitrate=rf.input_bitrate(3.0, 90 << 20), has_audio=True))
    info = media.probe(vid)
    assert info.duration_sec == pytest.approx(4.0 - s, abs=1 / 30 + 1e-3)
    assert rf.count_video_frames(vid) == round((4.0 - s) * 30)
    has, dur = media.probe_audio(aud)
    assert has and dur == pytest.approx(6.0 - s, abs=0.01)


def test_first_frame_is_on_the_container_timeline(tmp):
    """A video stream that starts 0.5 s after the audio: the value must be what
    an input -ss seeks on (the format's start), not the video stream's start."""
    from reframe_media import ffmpeg

    v = make_source(os.path.join(tmp, "v.mp4"), sec=3.0, audio=False)
    a = make_flac(os.path.join(tmp, "a.flac"), sec=3.5)
    src = os.path.join(tmp, "offset.mp4")
    ffmpeg("-itsoffset", "0.5", "-i", v, "-i", a, "-map", "0:v", "-map", "1:a", "-c:v", "copy",
           "-c:a", "aac", src)
    assert rf.first_frame_at_or_after(src, 0.0) == pytest.approx(0.5, abs=0.02)
    assert rf.first_frame_at_or_after(src, 1.0) == pytest.approx(1.0, abs=1 / 30 + 0.01)


def test_cta_aspect_accepts_nominal_and_native_ratios():
    from app.api_reframe import cta_matches

    assert cta_matches(1080, 1920, "9:16", "768p")      # nominal 9:16
    assert cta_matches(768, 1344, "9:16", "768p")       # the model's native size
    assert cta_matches(1920, 1080, "16:9", "480p")
    assert not cta_matches(1920, 1080, "9:16", "768p")
    assert not cta_matches(1080, 1350, "9:16", "768p")  # 4:5
    assert not cta_matches(None, 1920, "9:16", "768p")


def _times(raw_frames=120, has_audio=True):
    return rf.JobTimes(0.0, raw_frames / 24, raw_frames, has_audio, raw_frames / 24 + 1.0)


def test_render_body_frames_fade_and_audio(tmp):
    from app import media

    raw = make_raw(os.path.join(tmp, "raw.mp4"), frames=120)
    snap = make_flac(os.path.join(tmp, "a.flac"), sec=6.0)
    t = _times()
    f = rf.validate_ending(t, rf.Ending(4.0, 0.5, 4.6, 0.25))
    body = os.path.join(tmp, "body.mp4")
    media._run(rf.render_body_argv(raw=raw, audio=snap, f=f, dst=body, width=768, height=1344))
    facts = rf.media_facts(body)
    assert facts["frames"] == f.nl
    assert (facts["width"], facts["height"]) == (768, 1344)
    assert frame_luma(body, f.np - 1) > 30           # generated picture
    assert frame_luma(body, f.nl - 1) < 20           # black after the fade
    r2 = rf.audio_check(body, snap, f.na, f.nl)
    assert r2["ok"], r2
    assert abs(r2["lag_windows"]) <= 1


def test_render_body_cut_to_black_without_fade(tmp):
    from app import media

    raw = make_raw(os.path.join(tmp, "raw.mp4"), frames=48)
    t = _times(48)
    f = rf.validate_ending(t, rf.Ending(1.0, 0.0, 1.0, 0.5))
    assert f.nf == 0
    body = os.path.join(tmp, "body.mp4")
    snap = make_flac(os.path.join(tmp, "a.flac"), sec=3.0)
    media._run(rf.render_body_argv(raw=raw, audio=snap, f=f, dst=body, width=768, height=1344))
    assert rf.media_facts(body)["frames"] == f.nl
    assert frame_luma(body, f.np - 1) > 30
    assert frame_luma(body, f.np) < 20


def test_render_body_silent_source(tmp):
    from app import media

    raw = make_raw(os.path.join(tmp, "raw.mp4"), frames=48, audio=False)
    t = _times(48, has_audio=False)
    f = rf.validate_ending(t, rf.Ending(2.0, 0.25, None, 0.25))
    body = os.path.join(tmp, "body.mp4")
    media._run(rf.render_body_argv(raw=raw, audio=None, f=f, dst=body, width=768, height=1344))
    facts = rf.media_facts(body)
    assert facts["frames"] == f.nl and facts["has_audio"]
    r2 = rf.audio_check(body, None, 0, f.nl)
    assert r2["ok"] and r2["tail_silent"], r2


def test_audio_check_detects_shifted_audio(tmp):
    """A body whose audio is the wrong sound fails R2 (envelope mismatch)."""
    from app import media

    raw = make_raw(os.path.join(tmp, "raw.mp4"), frames=72)
    ref = os.path.join(tmp, "ref.flac")
    from reframe_media import ffmpeg

    # Reference: tone for 1 s, silence for 1 s, tone for 1 s.
    ffmpeg("-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=3",
           "-af", "volume=enable='between(t,1,2)':volume=0", "-c:a", "flac", ref)
    other = make_flac(os.path.join(tmp, "o.flac"), sec=3.0)
    t = _times(72)
    f = rf.validate_ending(t, rf.Ending(3.0, 0.0, 3.0, 0.0))
    body = os.path.join(tmp, "body.mp4")
    media._run(rf.render_body_argv(raw=raw, audio=other, f=f, dst=body, width=768, height=1344))
    assert not rf.audio_check(body, ref, f.na, f.nl)["ok"]


def test_ssim_and_contact_sheet(tmp):
    src = make_source(os.path.join(tmp, "in.mp4"), w=768, h=432, fps=24, sec=2.0, audio=False)
    raw = os.path.join(tmp, "raw.mp4")
    from reframe_media import ffmpeg

    ffmpeg("-i", src, "-vf", "pad=768:1344:0:456:black", "-c:v", "libx264", "-preset",
           "ultrafast", "-pix_fmt", "yuv420p", raw)
    v = rf.ssim_at(raw, src, 1.0, (0, 456, 768, 432))
    assert v is not None and v > 0.9
    sheet = os.path.join(tmp, "sheet.jpg")
    rf.contact_sheet(raw, 48, sheet)
    assert os.path.getsize(sheet) > 0


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


def _migration():
    path = os.path.join(os.path.dirname(__file__), "..", "alembic", "versions",
                        "n4d0f6b2c538_reframe.py")
    spec = importlib.util.spec_from_file_location("reframe_migration", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_migration_matches_models_and_drain_check():
    mig = _migration()
    assert mig.down_revision == "m3c9e5a1b427"
    src = open(mig.__file__).read()
    jobs_part = src.split('"reframe_jobs",', 1)[1].split('op.create_index("ix_reframe_jobs', 1)[0]
    renders_part = src.split('"reframe_renders",', 1)[1].split("op.create_index", 1)[0]
    assert set(re.findall(r'sa\.Column\("([a-z_0-9]+)"', jobs_part)) == \
        {c.name for c in ReframeJob.__table__.columns}
    assert set(re.findall(r'sa\.Column\("([a-z_0-9]+)"', renders_part)) == \
        {c.name for c in ReframeRender.__table__.columns}
    idx = next(i for i in ReframeJob.__table__.indexes
               if i.name == "uq_reframe_jobs_preparing_per_project")
    assert str(idx.dialect_options["postgresql"]["where"]) == mig.PREPARING_WHERE
    # The revision never USES the new enum label (same-transaction rule).
    body = src.split("def upgrade", 1)[1]
    assert body.count("'reframe'") == 1 and "ADD VALUE IF NOT EXISTS 'reframe'" in body
    sql = open(os.path.join(REPO, "deploy", "drain-check.sql")).read()
    assert "FROM reframe_jobs WHERE status = 'preparing' OR qa_status = 'running'" in sql
    assert "FROM reframe_renders WHERE status IN ('queued', 'processing', 'delivering')" in sql


def test_migration_runs_on_sqlite_and_downgrade_refuses_linked_jobs():
    from alembic.migration import MigrationContext
    from alembic.operations import Operations
    from sqlalchemy import create_engine, inspect, text

    from app.db import Base

    eng = create_engine(f"sqlite:///{tempfile.mkstemp(suffix='.db')[1]}")
    Base.metadata.create_all(eng)
    mig = _migration()
    with eng.begin() as conn:
        with Operations.context(MigrationContext.configure(conn)):
            mig.downgrade()
            assert "reframe_jobs" not in inspect(conn).get_table_names()
            mig.upgrade()
        conn.execute(text("INSERT INTO video_projects (id, project_type, source_type, source_ref, "
                          "status, mute_source) VALUES ('p1', 'reframe', 'upload', 'x', 'ready', 0)"))
        conn.execute(text(
            "INSERT INTO reframe_jobs (id, project_id, status, target_aspect, resolution, "
            "window_start_sec, window_end_sec, estimated_usd_base, estimated_usd_reserved, "
            "provider_task_id) VALUES ('j1','p1','admitted','9:16','768p',0,5,0.4,0.8,'t1')"))
        with Operations.context(MigrationContext.configure(conn)):
            with pytest.raises(RuntimeError, match="provider task"):
                mig.downgrade()


# ---------------------------------------------------------------------------
# Real media, free (design §12): the AICP-312 probe's T1 input and Fal output
# ---------------------------------------------------------------------------

PROBE = os.environ.get("REFRAME_PROBE_DIR", "/probe/run-20261027T183911Z")


@pytest.mark.skipif(not os.path.isfile(os.path.join(PROBE, "out_t1_reframe_9x16.mp4")),
                    reason="probe artifacts not mounted")
def test_probe_t1_prepare_qa_and_render(tmp):
    from app import media
    from app import pipeline_reframe as prf

    src = os.path.join(PROBE, "in_h_5s.mp4")
    raw = os.path.join(PROBE, "out_t1_reframe_9x16.mp4")
    s = rf.first_frame_at_or_after(src, 0.0)
    assert s == pytest.approx(0.0, abs=1e-3)
    inp, aud = os.path.join(tmp, "input.mp4"), os.path.join(tmp, "audio.flac")
    media._run(rf.prepare_argv(src=src, input_start=s, video_sec=5.0, audio_sec=5.0, fps=30.0,
                               video_dst=inp, audio_dst=aud,
                               bitrate=rf.input_bitrate(5.0, 90 << 20), has_audio=True))
    in_dur = media.probe(inp).duration_sec
    raw_frames = rf.count_video_frames(raw)
    assert abs(raw_frames - round(in_dur * 24)) <= 1                   # J2
    # J3 on the centred 16:9 band of the 768x1344 output (design §2 fact 4).
    vals = [rf.ssim_at(raw, inp, t, (0, 456, 768, 432)) for t in (1.25, 2.5, 3.75)]
    assert all(v is not None and v >= rf.J3_SSIM_MIN for v in vals), vals
    _, adur = media.probe_audio(aud)
    times = rf.JobTimes(input_start=s, window_end=5.0, raw_frames=raw_frames, has_audio=True,
                        audio_end_max=s + adur)
    f = rf.validate_ending(times, rf.default_ending(times))
    rdir = os.path.join(tmp, "r")
    os.mkdir(rdir)
    qa = prf._render_files(raw, aud, None, f, rdir, 768, 1344, "tok")
    assert qa["ok"], qa
    assert qa["final"]["frames"] == f.nl and qa["R2_final"]["ok"]
    assert os.listdir(rdir) == ["final-tok.mp4"]  # the body is gone, the final unpublished
