"""Whole-video localisation segmentation (AICP-340).

The rule (pipeline_v2.whole_video_ranges) is pure, so the acceptance case is a
fixture of the cut times PySceneDetect found on project 5b1b9b67's source
(tests/fixtures/scene_cuts_5b1b9b67.json), not the 79 s video itself.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from sqlalchemy.orm import sessionmaker

from app.models import ProjectStatus, VideoProject
from app.pipeline_v2 import (
    HOOK_CLAMP_CAP,
    HOOK_CLAMP_NONE,
    hook_split_plan,
    whole_video_ranges,
    whole_video_target_sec,
)
from app.project_types import LOCALISATION_SCOPE_WHOLE

# Fixtures and helpers of the pipeline suite (the session-scoped engine is
# shared). Only fixtures/helpers are imported, never its Test* classes.
from tests.test_pipeline_v2 import (  # noqa: F401
    _create_project,
    db_engine,
    db_session,
    _cleanup_tmp,
    patch_get_session,
    tmp_path,
)

CUTS = json.loads(
    (Path(__file__).parent / "fixtures" / "scene_cuts_5b1b9b67.json").read_text("utf-8")
)


def _bounds(ranges):
    return [ranges[0][0]] + [e for _s, e in ranges]


def _assert_partition(ranges, *, duration, target):
    assert ranges[0][0] == pytest.approx(0.0)
    assert ranges[-1][1] == pytest.approx(duration)
    for (_s1, e1), (s2, _e2) in zip(ranges, ranges[1:]):
        assert s2 == pytest.approx(e1)
    for s, e in ranges:
        assert e - s <= target + 1e-6
        assert e - s > 0


class TestTarget:
    def test_null_cap_targets_the_seedance_25_truncation_length(self):
        # NOT the universal 10 s default: that exists for Gemini Omni, which
        # cannot localise.
        assert whole_video_target_sec(None) == pytest.approx(14.72)

    def test_a_longer_explicit_cap_does_not_raise_the_target(self):
        assert whole_video_target_sec(30.0) == pytest.approx(14.72)

    def test_an_explicit_smaller_cap_is_honoured(self):
        assert whole_video_target_sec(10.0) == pytest.approx(10.0)

    def test_a_non_positive_cap_degrades_instead_of_dividing_by_zero(self):
        assert whole_video_target_sec(0.0) == pytest.approx(14.72)

    def test_never_above_the_shortest_audio_model_clip(self, monkeypatch):
        from app.config import settings

        monkeypatch.setattr(settings, "LOCALISATION_WHOLE_VIDEO_MAX_SEC", 40.0)
        # Seedance 2.0 family = 15 s, the shortest model that produces audio.
        assert whole_video_target_sec(None) == pytest.approx(15.0)


class TestAcceptance:
    def test_5b1b9b67_reproduces_the_operator_cut(self):
        """AICP-340 acceptance: 0-12.13-26.03-35.9-49.45-57.7-65.95-78.85."""
        target = whole_video_target_sec(None)
        ranges = whole_video_ranges(
            [tuple(r) for r in CUTS["scenes"]],
            duration_sec=CUTS["duration_sec"],
            target_sec=target,
        )
        got = _bounds(ranges)
        expected = CUTS["operator_cut"]
        assert len(got) == len(expected)
        # Every inner boundary matches the hand cut to the frame; the end is
        # the probed duration of THIS source (78.878), the operator's 78.85 is
        # the duration of the de-captioned copy they cut.
        for g, e in zip(got[:-1], expected[:-1]):
            assert g == pytest.approx(e, abs=0.01)
        assert got[-1] == pytest.approx(CUTS["duration_sec"])
        _assert_partition(ranges, duration=CUTS["duration_sec"], target=target)

    def test_the_cleaned_copy_also_partitions_within_the_target(self):
        cleaned = CUTS["cleaned"]
        target = whole_video_target_sec(None)
        ranges = whole_video_ranges(
            [tuple(r) for r in cleaned["scenes"]],
            duration_sec=cleaned["duration_sec"],
            target_sec=target,
        )
        _assert_partition(ranges, duration=cleaned["duration_sec"], target=target)
        bounds = _bounds(ranges)
        # Real scene cuts that fit are kept as boundaries.
        for cut in (12.133, 35.9):
            assert any(abs(b - cut) < 1e-3 for b in bounds)
        assert len(ranges) == 7


class TestRule:
    def test_short_scenes_merge_up_to_the_target(self):
        ranges = whole_video_ranges(
            [(0.0, 5.0), (5.0, 10.0), (10.0, 14.0), (14.0, 20.0)],
            duration_sec=20.0, target_sec=14.72,
        )
        assert ranges == [(0.0, 14.0), (14.0, 20.0)]

    def test_a_long_scene_is_split_evenly(self):
        # Split at the 10 s piece length; equal pieces never re-merge past the
        # target, so a single 40 s shot is 4 x 10 s (one seam more than the
        # minimum — the price of the finer candidates that let short scenes
        # absorb part of a long neighbour, which is what the acceptance needs).
        ranges = whole_video_ranges([(0.0, 40.0)], duration_sec=40.0, target_sec=14.72)
        _assert_partition(ranges, duration=40.0, target=14.72)
        assert ranges == [(0.0, 10.0), (10.0, 20.0), (20.0, 30.0), (30.0, 40.0)]

    def test_no_cuts_is_one_or_more_even_segments(self):
        assert whole_video_ranges([], duration_sec=12.0, target_sec=14.72) == [(0.0, 12.0)]

    def test_a_sub_floor_last_scene_is_rebalanced_with_its_predecessor(self):
        # 14.5 + 1.5: the merge cannot take the 1.5 s tail (16 > 14.72) and
        # Seedance 2.5 refuses clips under 4 s, so the pair is halved.
        ranges = whole_video_ranges(
            [(0.0, 14.5), (14.5, 16.0)], duration_sec=16.0, target_sec=14.72
        )
        assert ranges == [(0.0, 8.0), (8.0, 16.0)]

    def test_a_detector_that_stops_short_is_extended_to_the_duration(self):
        ranges = whole_video_ranges(
            [(0.0, 6.0), (6.0, 11.9)], duration_sec=12.0, target_sec=14.72
        )
        assert ranges[-1][1] == pytest.approx(12.0)

    def test_a_sliver_past_a_full_last_segment_is_not_dropped(self):
        """Gate r1: detected footage ending at 14.72 with a 14.76 duration."""
        ranges = whole_video_ranges([(0.0, 14.72)], duration_sec=14.76, target_sec=14.72)
        _assert_partition(ranges, duration=14.76, target=14.72)

    def test_an_end_past_the_duration_is_clamped(self):
        ranges = whole_video_ranges(
            [(0.0, 6.0), (6.0, 12.4)], duration_sec=12.0, target_sec=14.72
        )
        assert ranges == [(0.0, 12.0)]


class TestHookSplitPlanScope:
    def test_hook_scope_is_unchanged(self):
        plan = hook_split_plan(duration_sec=78.85, hook_sec=88.0, max_segment_sec=10.0)
        assert plan["effective"] == pytest.approx(10.0)
        assert plan["reason"] == HOOK_CLAMP_CAP
        assert plan["scope"] == "hook"

    def test_whole_scope_has_no_clamp(self):
        plan = hook_split_plan(
            duration_sec=78.85, hook_sec=None, max_segment_sec=10.0,
            scope=LOCALISATION_SCOPE_WHOLE,
        )
        assert plan["effective"] == pytest.approx(78.85)
        assert plan["reason"] == HOOK_CLAMP_NONE


class TestAnalyzeWholeVideo:
    def _probe(self, monkeypatch, duration, fps=30.0):
        from app.media import MediaInfo
        import app.pipeline_v2 as pv2

        monkeypatch.setattr(
            pv2.media_mod, "probe",
            lambda _p: MediaInfo(duration_sec=duration, width=1080, height=1920,
                                 fps=fps, aspect_ratio="9:16", has_audio=True),
        )

    def _request(self, db_session, pid, scope):
        db_session.expire_all()
        p = db_session.get(VideoProject, pid)
        p.localisation_scope_requested = scope
        db_session.commit()

    def _localisation_project(self, db_session, source, *, scope, **kw):
        pid = _create_project(db_session, str(source), project_type="localisation", **kw)
        p = db_session.get(VideoProject, pid)
        p.localisation_scope = scope
        db_session.commit()
        return pid

    def test_whole_scope_cuts_all_swap_at_scenes(
        self, db_engine, db_session, tmp_path, monkeypatch
    ):
        import app.pipeline_v2 as pv2
        from app.pipeline_v2 import analyze_project

        source = tmp_path / "source.mp4"
        source.write_bytes(b"src")
        pid = self._localisation_project(db_session, source, scope="whole", hook_sec=10.0)
        self._probe(monkeypatch, CUTS["duration_sec"])
        seen = {}

        def fake_detect(path, *, fps, min_scene_len_sec):
            seen["min"] = min_scene_len_sec
            return [tuple(r) for r in CUTS["scenes"]]

        monkeypatch.setattr(pv2.scene_mod, "detect_scenes", fake_detect)
        analyze_project(pid)

        assert seen["min"] == pytest.approx(4.0)
        Session = sessionmaker(bind=db_engine)
        with Session() as s:
            project = s.get(VideoProject, pid)
            assert project.status == ProjectStatus.ready
            segs = sorted(project.segments, key=lambda d: d.index)
            assert [d.action for d in segs] == ["swap"] * 7
            assert [round(d.end_sec, 2) for d in segs] == [
                12.13, 26.03, 35.9, 49.45, 57.7, 65.95, 78.88,
            ]

    def test_hook_scope_still_cuts_the_hook(
        self, db_engine, db_session, tmp_path, monkeypatch
    ):
        import app.pipeline_v2 as pv2
        from app.pipeline_v2 import analyze_project

        source = tmp_path / "source.mp4"
        source.write_bytes(b"src")
        pid = self._localisation_project(db_session, source, scope="hook", hook_sec=88.0)
        self._probe(monkeypatch, 78.85)
        monkeypatch.setattr(
            pv2.scene_mod, "detect_scenes",
            lambda *a, **k: (_ for _ in ()).throw(AssertionError("detect_scenes called")),
        )
        analyze_project(pid)

        Session = sessionmaker(bind=db_engine)
        with Session() as s:
            segs = sorted(s.get(VideoProject, pid).segments, key=lambda d: d.index)
            assert [(d.action, round(d.start_sec, 2), round(d.end_sec, 2)) for d in segs] == [
                ("swap", 0.0, 10.0), ("keep", 10.0, 78.85),
            ]

    def test_a_requested_scope_is_written_with_the_segments(
        self, db_engine, db_session, tmp_path, monkeypatch
    ):
        import app.pipeline_v2 as pv2
        from app.pipeline_v2 import analyze_project

        source = tmp_path / "source.mp4"
        source.write_bytes(b"src")
        pid = self._localisation_project(db_session, source, scope="hook")
        self._request(db_session, pid, "whole")
        self._probe(monkeypatch, 30.0)
        monkeypatch.setattr(
            pv2.scene_mod, "detect_scenes",
            lambda *a, **k: [(0.0, 12.0), (12.0, 30.0)],
        )
        analyze_project(pid)

        Session = sessionmaker(bind=db_engine)
        with Session() as s:
            project = s.get(VideoProject, pid)
            assert project.localisation_scope == "whole"
            assert project.localisation_scope_requested is None
            assert all(d.action == "swap" for d in project.segments)
            assert len(project.segments) == 3

        # A plain re-analysis re-applies the stored scope.
        analyze_project(pid)
        with Session() as s:
            assert s.get(VideoProject, pid).localisation_scope == "whole"

        # And back: hook re-cuts the hook and stores "hook".
        self._request(db_session, pid, "hook")
        analyze_project(pid)
        with Session() as s:
            project = s.get(VideoProject, pid)
            assert project.localisation_scope == "hook"
            segs = sorted(project.segments, key=lambda d: d.index)
            assert [d.action for d in segs] == ["swap", "keep"]

    def test_a_failed_analysis_keeps_the_old_scope(
        self, db_engine, db_session, tmp_path, monkeypatch
    ):
        import app.pipeline_v2 as pv2
        from app.pipeline_v2 import analyze_project

        source = tmp_path / "source.mp4"
        source.write_bytes(b"src")
        pid = self._localisation_project(db_session, source, scope="hook")
        self._probe(monkeypatch, 30.0)

        def boom(*a, **k):
            raise RuntimeError("scenedetect exploded")

        monkeypatch.setattr(pv2.scene_mod, "detect_scenes", boom)
        self._request(db_session, pid, "whole")
        with pytest.raises(RuntimeError):
            analyze_project(pid)
        Session = sessionmaker(bind=db_engine)
        with Session() as s:
            project = s.get(VideoProject, pid)
            assert project.status == ProjectStatus.failed
            assert project.localisation_scope == "hook"
            # Gate r2: the request survives for the retry.
            assert project.localisation_scope_requested == "whole"

        monkeypatch.setattr(
            pv2.scene_mod, "detect_scenes", lambda *a, **k: [(0.0, 30.0)],
        )
        analyze_project(pid)
        with Session() as s:
            project = s.get(VideoProject, pid)
            assert project.status == ProjectStatus.ready
            assert project.localisation_scope == "whole"
            assert project.localisation_scope_requested is None

    def test_a_newer_request_is_not_erased_by_an_older_analysis(
        self, db_engine, db_session, tmp_path, monkeypatch
    ):
        """Gate r3: a request committed while an analysis is already running
        survives that analysis, so its own job applies it."""
        import app.pipeline_v2 as pv2
        from app.pipeline_v2 import analyze_project

        source = tmp_path / "source.mp4"
        source.write_bytes(b"src")
        pid = self._localisation_project(db_session, source, scope="hook")
        self._request(db_session, pid, "whole")
        self._probe(monkeypatch, 30.0)
        Session = sessionmaker(bind=db_engine)

        def detect_while_a_newer_request_lands(*a, **k):
            with Session() as other:
                other.get(VideoProject, pid).localisation_scope_requested = "hook"
                other.commit()
            return [(0.0, 30.0)]

        monkeypatch.setattr(pv2.scene_mod, "detect_scenes", detect_while_a_newer_request_lands)
        analyze_project(pid)
        with Session() as s:
            project = s.get(VideoProject, pid)
            assert project.localisation_scope == "whole"
            assert project.localisation_scope_requested == "hook"

        monkeypatch.setattr(
            pv2.scene_mod, "detect_scenes",
            lambda *a, **k: (_ for _ in ()).throw(AssertionError("hook must not detect")),
        )
        analyze_project(pid)
        with Session() as s:
            project = s.get(VideoProject, pid)
            assert project.localisation_scope == "hook"
            assert project.localisation_scope_requested is None

    def test_a_scope_on_another_type_is_ignored(
        self, db_engine, db_session, tmp_path, monkeypatch
    ):
        import app.pipeline_v2 as pv2
        from app.pipeline_v2 import analyze_project

        source = tmp_path / "source.mp4"
        source.write_bytes(b"src")
        pid = _create_project(db_session, str(source), project_type="subtitle_removal")
        self._request(db_session, pid, "whole")
        self._probe(monkeypatch, 20.0)
        monkeypatch.setattr(
            pv2.scene_mod, "detect_scenes", lambda *a, **k: [(0.0, 20.0)],
        )
        analyze_project(pid)
        Session = sessionmaker(bind=db_engine)
        with Session() as s:
            project = s.get(VideoProject, pid)
            assert project.status == ProjectStatus.ready
            assert project.localisation_scope == "hook"

    def test_new_projects_default_to_hook_scope(self, db_session, tmp_path):
        source = tmp_path / "source.mp4"
        source.write_bytes(b"src")
        pid = _create_project(db_session, str(source), project_type="localisation")
        assert db_session.get(VideoProject, pid).localisation_scope == "hook"


def test_migration_matches_the_model():
    """The fixed migration copy and the model agree on name/default/CHECK."""
    import importlib.util

    from app.models import VideoProject as VP

    path = (
        Path(__file__).resolve().parents[1]
        / "alembic" / "versions" / "o5e1a7c3d649_localisation_scope.py"
    )
    spec = importlib.util.spec_from_file_location("mig_o5e1", path)
    src = path.read_text("utf-8")
    assert 'down_revision: Union[str, None] = "n4d0f6b2c538"' in src
    assert "localisation_scope IN ('hook', 'whole')" in src
    col = VP.__table__.c.localisation_scope
    assert col.nullable is False
    assert col.server_default.arg == "hook"
    checks = {c.name: str(c.sqltext) for c in VP.__table__.constraints if c.name}
    assert checks["ck_video_projects_localisation_scope"] == (
        "localisation_scope IN ('hook', 'whole')"
    )
    assert VP.__table__.c.localisation_scope_requested.nullable is True
    assert "localisation_scope_requested IS NULL" in src
    assert checks["ck_video_projects_localisation_scope_requested"] == (
        "localisation_scope_requested IS NULL "
        "OR localisation_scope_requested IN ('hook', 'whole')"
    )
    assert spec is not None



def test_prompt_job_fingerprint_moves_with_scope_and_duration():
    """AICP-340 x AICP-338: a scope or duration change makes a queued
    localisation-prompt answer stale (409 on poll)."""
    from types import SimpleNamespace

    from app import localisation_prompt_job as lpj

    def proj(**kw):
        base = dict(transcript={"lines": []}, transcript_status="ready", hook_sec=None,
                    source_local_path="/s.mp4", localisation_scope="hook", duration_sec=30.0)
        base.update(kw)
        return SimpleNamespace(**base)

    segs = [SimpleNamespace(id="s0", start_sec=0.0, end_sec=10.0)]
    base = lpj.project_fingerprint(proj(), segs)
    assert base != lpj.project_fingerprint(proj(localisation_scope="whole"), segs)
    assert base != lpj.project_fingerprint(proj(duration_sec=31.0), segs)

    # hook_sec is not read in whole scope, so it must not cause staleness.
    whole = lpj.project_fingerprint(proj(localisation_scope="whole"), segs)
    assert whole == lpj.project_fingerprint(
        proj(localisation_scope="whole", hook_sec=25.0), segs
    )
    assert base != lpj.project_fingerprint(proj(hook_sec=25.0), segs)
    # Without a probed duration the window falls back to hook_sec, which then counts.
    assert lpj.project_fingerprint(
        proj(localisation_scope="whole", duration_sec=None), segs
    ) != lpj.project_fingerprint(
        proj(localisation_scope="whole", duration_sec=None, hook_sec=25.0), segs
    )
