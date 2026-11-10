"""POST /api/v2/projects/{pid}/recut-at-pauses (AICP-347)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.models import ProjectStatus, RunStatus, SegmentDef, VideoProject

# The API suite's fixtures and helpers (only those — never its Test* classes).
from tests.test_api_v2 import (  # noqa: F401
    SessionFactory,
    _localisation_project,
    _make_run,
    _make_segment_def,
    client,
    db_session,
    engine,
    enqueue_spy,
    spy_client,
)

FIX = Path(__file__).parent / "fixtures"
DATA = json.loads((FIX / "localisation_46cc6443.json").read_text("utf-8"))
CUTS = json.loads((FIX / "scene_cuts_5b1b9b67.json").read_text("utf-8"))
EXPECTED = [0.0, 13.0, 19.3, 33.3, 43.1, 50.1, 64.6, DATA["duration_sec"]]


def _project(db_session, tmp_path=None, **kw):
    source = None
    if tmp_path is not None:
        source = tmp_path / "source.mp4"
        source.write_bytes(b"video")
    defaults = dict(
        localisation_scope="whole",
        transcript=DATA["transcript_v2"],
        duration_sec=DATA["duration_sec"],
    )
    if source is not None:
        defaults["source_local_path"] = str(source)
    defaults.update(kw)
    p = _localisation_project(db_session, **defaults)
    for seg in DATA["segments"]:
        _make_segment_def(
            db_session, p.id, seg["index"], start_sec=seg["start_sec"],
            end_sec=seg["end_sec"], action="swap", has_face=False,
        )
    return p


def _bounds(db_session, pid):
    db_session.expire_all()
    segs = sorted(db_session.get(VideoProject, pid).segments, key=lambda s: s.index)
    return [segs[0].start_sec] + [s.end_sec for s in segs], [s.id for s in segs]


def _url(pid, **q):
    qs = "&".join(f"{k}={v}" for k, v in q.items())
    return f"/api/v2/projects/{pid}/recut-at-pauses" + (f"?{qs}" if qs else "")


class TestRecut:
    def test_dry_run_plans_and_writes_nothing(self, client, db_session):
        p = _project(db_session)
        before, ids = _bounds(db_session, p.id)
        resp = client.post(_url(p.id, dry_run="true"))
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["dry_run"] is True and body["applied"] is False
        assert body["feasible"] and body["changed"]
        assert [b["new_sec"] for b in body["boundaries"]] == pytest.approx(EXPECTED[1:-1])
        assert body["target_sec"] == pytest.approx(14.72)
        assert body["scene_cuts_used"] is False
        # Without a scene-cut cache the plan says so.
        assert any("scene cuts" in w for w in body["warnings"])
        assert _bounds(db_session, p.id) == (before, ids)

    def test_apply_moves_bounds_in_place(self, client, db_session):
        p = _project(db_session)
        _before, ids = _bounds(db_session, p.id)
        preview = client.post(_url(p.id, dry_run="true")).json()
        resp = client.post(_url(p.id, expected_fingerprint=preview["fingerprint"]))
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["applied"] is True
        after, after_ids = _bounds(db_session, p.id)
        assert after == pytest.approx(EXPECTED)
        assert after_ids == ids  # same rows, same order
        assert [s["end_sec"] for s in body["segments"]] == pytest.approx(EXPECTED[1:])
        db_session.expire_all()
        project = db_session.get(VideoProject, p.id)
        assert all(s.action == "swap" for s in project.segments)
        assert project.localisation_scope == "whole"
        assert project.transcript_status == "ready"
        # Acceptance: every segment <= 14.72 s, no boundary inside 5/10/16/20.
        by_id = {ln["id"]: ln for ln in DATA["transcript_v2"]["lines"]}
        for s, e in zip(after, after[1:]):
            assert e - s <= 14.72 + 1e-6
        for b in after[1:-1]:
            for lid in (5, 10, 16, 20):
                assert not (by_id[lid]["start"] < b < by_id[lid]["end"])

    def test_second_apply_is_a_no_op(self, client, db_session):
        p = _project(db_session)
        assert client.post(_url(p.id)).json()["applied"] is True
        again = client.post(_url(p.id)).json()
        assert again["applied"] is False and again["changed"] is False

    def test_stale_preview_is_refused(self, client, db_session):
        p = _project(db_session)
        preview = client.post(_url(p.id, dry_run="true")).json()
        db_session.expire_all()
        seg = db_session.get(VideoProject, p.id).segments[0]
        seg.pre_roll_sec = 0.1
        db_session.commit()
        before, _ = _bounds(db_session, p.id)
        resp = client.post(_url(p.id, expected_fingerprint=preview["fingerprint"]))
        assert resp.status_code == 409
        assert "changed since" in resp.text
        assert _bounds(db_session, p.id)[0] == before

    def test_uses_recorded_scene_cuts(self, client, db_session, tmp_path):
        from app import scene as scene_mod
        from app.config import settings
        from app.storage import scene_cuts_cache_path

        p = _project(db_session, tmp_path)
        scene_mod.write_scene_cache(
            scene_cuts_cache_path(p.id), source_path=p.source_local_path,
            ranges=[tuple(r) for r in CUTS["cleaned"]["scenes"]],
            min_scene_len_sec=settings.SCENE_MIN_LEN_SECONDS,
        )
        body = client.post(_url(p.id, dry_run="true")).json()
        assert body["scene_cuts_used"] is True
        assert not any("scene cuts" in w for w in body["warnings"])

    def test_warns_about_unfrozen_runs(self, client, db_session):
        p = _project(db_session)
        _make_run(db_session, p.id, status=RunStatus.failed, segment_snapshot=None)
        body = client.post(_url(p.id, dry_run="true")).json()
        assert any("never froze" in w for w in body["warnings"])

    def test_done_runs_keep_their_snapshot(self, client, db_session):
        p = _project(db_session)
        snapshot = [
            {"id": "x", "index": 0, "action": "swap", "start_sec": 0.0, "end_sec": 12.1333},
        ]
        run = _make_run(db_session, p.id, status=RunStatus.done, segment_snapshot=snapshot)
        assert client.post(_url(p.id)).json()["applied"] is True
        db_session.expire_all()
        from app.models import Run

        assert db_session.get(Run, run.id).segment_snapshot == snapshot


class TestRecutGuards:
    @pytest.mark.parametrize("status", [
        RunStatus.queued, RunStatus.processing, RunStatus.stitching,
        RunStatus.delivering, RunStatus.preprocessing,
    ])
    def test_409_while_a_run_is_active(self, client, db_session, status):
        p = _project(db_session)
        _make_run(db_session, p.id, status=status, name="busy run")
        before, _ = _bounds(db_session, p.id)
        for q in ({}, {"dry_run": "true"}):
            resp = client.post(_url(p.id, **q))
            assert resp.status_code == 409, resp.text
            assert "busy run" in resp.text
        assert _bounds(db_session, p.id)[0] == before

    def test_409_on_hook_scope(self, client, db_session):
        p = _project(db_session, localisation_scope="hook")
        resp = client.post(_url(p.id))
        assert resp.status_code == 409
        assert "whole video" in resp.text

    def test_409_on_another_project_type(self, client, db_session):
        p = _project(db_session, project_type="subtitle_removal")
        assert client.post(_url(p.id)).status_code == 409

    @pytest.mark.parametrize("status", [ProjectStatus.analyzing, ProjectStatus.failed])
    def test_409_unless_ready(self, client, db_session, status):
        p = _project(db_session, status=status)
        assert client.post(_url(p.id)).status_code == 409

    @pytest.mark.parametrize("ts", ["pending", "running", "failed", "empty", None])
    def test_409_without_a_ready_transcript(self, client, db_session, ts):
        p = _project(db_session, transcript_status=ts)
        resp = client.post(_url(p.id))
        assert resp.status_code == 409
        assert "transcript" in resp.text

    def test_409_on_a_broken_partition(self, client, db_session):
        p = _project(db_session)
        db_session.expire_all()
        seg = sorted(db_session.get(VideoProject, p.id).segments, key=lambda s: s.index)[2]
        seg.start_sec += 0.5
        db_session.commit()
        resp = client.post(_url(p.id))
        assert resp.status_code == 409
        assert "not contiguous" in resp.text

    def test_404(self, client):
        assert client.post(_url("nope")).status_code == 404

    def test_reanalyze_then_recut_is_blocked_until_the_transcript_lands(
        self, client, db_session
    ):
        p = _project(db_session)
        assert client.post(f"/api/v2/projects/{p.id}/analyze").status_code == 202
        resp = client.post(_url(p.id))
        assert resp.status_code == 409
        # AICP-350: the re-analysis admission commits `analyzing`, so the
        # project-status guard refuses first; either way nothing is re-cut.
        assert "transcript" in resp.text or "analyzing" in resp.text



class TestTransactionNeutralService:
    """AICP-365 calls the service inside its own transaction: it must never
    commit or roll back, and must keep the caller's pending changes."""

    def test_apply_does_not_commit(self, db_session, SessionFactory):
        from app.speech_recut import recut_project_segments

        p = _project(db_session)
        before, _ = _bounds(db_session, p.id)
        project = db_session.get(VideoProject, p.id)
        outcome = recut_project_segments(db_session, project)
        assert outcome.applied
        assert [s.end_sec for s in outcome.segments] == pytest.approx(EXPECTED[1:])
        # Another session sees nothing: not committed.
        other = SessionFactory()
        try:
            segs = sorted(other.get(VideoProject, p.id).segments, key=lambda s: s.index)
            assert [s.end_sec for s in segs] == pytest.approx(before[1:])
        finally:
            other.close()
        db_session.rollback()
        assert _bounds(db_session, p.id)[0] == pytest.approx(before)

    def test_unchanged_plan_does_not_end_the_transaction(self, db_session, monkeypatch):
        from app.speech_recut import recut_project_segments

        p = _project(db_session)
        project = db_session.get(VideoProject, p.id)
        assert recut_project_segments(db_session, project).applied
        db_session.commit()

        def forbidden(*_a, **_k):
            raise AssertionError("the service must not end the transaction")

        monkeypatch.setattr(db_session, "commit", forbidden)
        monkeypatch.setattr(db_session, "rollback", forbidden)
        monkeypatch.setattr(db_session, "flush", forbidden)
        for dry_run in (False, True):
            again = recut_project_segments(db_session, project, dry_run=dry_run)
            assert not again.applied and not again.plan.changed
        monkeypatch.undo()
        db_session.rollback()

    @pytest.mark.parametrize("what", ["project", "segment", "run", "new_segment"])
    def test_refuses_over_pending_changes(self, db_session, what):
        from app.speech_recut import RecutRefused, recut_project_segments

        p = _project(db_session)
        run = _make_run(db_session, p.id, status=RunStatus.failed)
        project = db_session.get(VideoProject, p.id)
        if what == "project":
            project.name = "pending"
        elif what == "segment":
            project.segments[0].pre_roll_sec = 0.5
        elif what == "run":
            run.status = RunStatus.queued
        else:
            project.segments.append(SegmentDef(index=99, start_sec=1.0, end_sec=2.0, action="swap"))
        for dry_run in (False, True):
            with pytest.raises(RecutRefused, match="unflushed"):
                recut_project_segments(db_session, project, dry_run=dry_run)
        db_session.rollback()

    def test_refusal_raises_before_writing(self, db_session):
        from app.speech_recut import RecutRefused, recut_project_segments

        p = _project(db_session, localisation_scope="hook")
        with pytest.raises(RecutRefused):
            recut_project_segments(db_session, db_session.get(VideoProject, p.id))
