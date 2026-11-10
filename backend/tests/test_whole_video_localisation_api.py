"""Whole-video localisation scope over the REST API (AICP-340)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.models import ProjectStatus, RunStatus, VideoProject

# The API suite's fixtures and helpers (only those — never its Test* classes).
from tests.test_api_v2 import (  # noqa: F401
    SessionFactory,
    _localisation_project,
    _make_project,
    _make_run,
    _make_segment_def,
    _translate_via_job,
    client,
    db_session,
    engine,
    enqueue_spy,
    fake_translate,
    spy_client,
)

DATA = json.loads(
    (Path(__file__).parent / "fixtures" / "localisation_46cc6443.json").read_text("utf-8")
)


class TestCreate:
    def _post(self, client, **data):
        files = {"video_file": ("v.mp4", b"\x00" * 16, "video/mp4")}
        return client.post("/api/v2/projects", data=data, files=files)

    def test_localisation_project_can_be_created_whole(self, client, db_session):
        resp = self._post(client, project_type="localisation", localisation_scope="whole")
        assert resp.status_code == 201, resp.text
        pid = resp.json()["project_id"]
        db_session.expire_all()
        assert db_session.get(VideoProject, pid).localisation_scope == "whole"
        assert client.get(f"/api/v2/projects/{pid}").json()["localisation_scope"] == "whole"

    def test_default_is_hook(self, client, db_session):
        resp = self._post(client, project_type="localisation")
        assert resp.status_code == 201, resp.text
        db_session.expire_all()
        assert db_session.get(VideoProject, resp.json()["project_id"]).localisation_scope == "hook"

    def test_whole_on_another_type_is_400(self, client):
        resp = self._post(client, project_type="face_swap", localisation_scope="whole")
        assert resp.status_code == 400
        assert "only valid on a localisation project" in resp.text

    def test_hook_on_another_type_is_tolerated(self, client):
        resp = self._post(client, project_type="face_swap", localisation_scope="hook")
        assert resp.status_code == 201, resp.text

    def test_unknown_scope_is_400(self, client):
        resp = self._post(client, project_type="localisation", localisation_scope="all")
        assert resp.status_code == 400


class TestScopeChangesOnlyWithReanalysis:
    def test_patch_refuses_the_scope(self, client, db_session):
        p = _localisation_project(db_session)
        resp = client.patch(f"/api/v2/projects/{p.id}", json={"localisation_scope": "whole"})
        assert resp.status_code == 400
        db_session.expire_all()
        assert db_session.get(VideoProject, p.id).localisation_scope == "hook"

    def test_reanalyze_records_a_request_not_the_scope(self, spy_client, db_session):
        """The row keeps its scope until analyze_project applies the request
        together with the new segments (gate r1: a run launched while the job
        is queued must not freeze old segments under a new scope); the job's
        arguments are unchanged (gate r2: old/new worker compatibility)."""
        client, spy = spy_client
        p = _localisation_project(db_session)
        resp = client.post(f"/api/v2/projects/{p.id}/analyze?localisation_scope=whole")
        assert resp.status_code == 202, resp.text
        assert spy["analyze_project"] == [p.id]
        db_session.expire_all()
        row = db_session.get(VideoProject, p.id)
        assert row.localisation_scope == "hook"
        assert row.localisation_scope_requested == "whole"
        body = client.get(f"/api/v2/projects/{p.id}").json()
        assert body["localisation_scope"] == "hook"
        assert body["localisation_scope_requested"] == "whole"

        # AICP-350: the admission committed `analyzing`, so a second click
        # while it is pending is refused; once it is back to ready, an omitted
        # scope leaves a pending request as it is.
        assert client.post(f"/api/v2/projects/{p.id}/analyze").status_code == 409
        db_session.expire_all()
        db_session.get(VideoProject, p.id).status = ProjectStatus.ready
        db_session.commit()
        assert client.post(f"/api/v2/projects/{p.id}/analyze").status_code == 202
        db_session.expire_all()
        assert db_session.get(VideoProject, p.id).localisation_scope_requested == "whole"

    def test_a_refused_reanalysis_does_not_change_the_scope(self, spy_client, db_session):
        client, spy = spy_client
        p = _localisation_project(db_session)
        assert spy["analyze_project"] == []
        _make_run(db_session, p.id, status=RunStatus.processing)
        resp = client.post(f"/api/v2/projects/{p.id}/analyze?localisation_scope=whole")
        assert resp.status_code == 409
        assert spy["analyze_project"] == []
        db_session.expire_all()
        row = db_session.get(VideoProject, p.id)
        assert row.localisation_scope == "hook"
        assert row.localisation_scope_requested is None

    def test_invalid_scope_is_400_before_anything_changes(self, spy_client, db_session):
        client, spy = spy_client
        p = _localisation_project(db_session)
        resp = client.post(f"/api/v2/projects/{p.id}/analyze?localisation_scope=nope")
        assert resp.status_code == 400
        assert spy["analyze_project"] == []

    def test_whole_on_another_type_is_400(self, spy_client, db_session):
        client, spy = spy_client
        p = _make_project(db_session, status=ProjectStatus.ready, project_type="subtitle_removal")
        resp = client.post(f"/api/v2/projects/{p.id}/analyze?localisation_scope=whole")
        assert resp.status_code == 400
        assert spy["analyze_project"] == []


class TestLocalisationPromptWholeVideo:
    """The existing per-segment path gives every whole-video segment a prompt."""

    def _project(self, db_session, scope):
        p = _localisation_project(
            db_session,
            transcript=DATA["transcript_v2"],
            duration_sec=DATA["duration_sec"],
        )
        p.localisation_scope = scope
        db_session.commit()
        self.seg_ids = [
            _make_segment_def(
                db_session, p.id, seg["index"],
                start_sec=float(seg["start_sec"]), end_sec=float(seg["end_sec"]),
                action="swap", has_face=False,
            ).id
            for seg in DATA["segments"]
        ]
        return p

    def test_whole_scope_translates_every_segment(self, client, db_session, fake_translate):
        p = self._project(db_session, "whole")
        # AICP-338: 202 + poll; the inline test queue has already run the job,
        # whose snapshot froze the whole-video window at request time.
        resp = _translate_via_job(client, p.id, {"language": "es"})
        assert resp.status_code == 200, resp.text
        body = resp.json()
        # The window is the whole video, not the 10 s default hook.
        assert body["hook_sec"] == pytest.approx(DATA["segments"][-1]["end_sec"], abs=0.05)
        assert len(self.seg_ids) == 7
        assert set(body["segment_prompts"]) == set(self.seg_ids)
        assert all(v.strip() for v in body["segment_prompts"].values())

    def test_hook_scope_with_the_same_segments_keeps_the_hook_window(
        self, client, db_session, fake_translate
    ):
        p = self._project(db_session, "hook")
        # AICP-338: 202 + poll; the inline test queue has already run the job,
        # whose snapshot froze the whole-video window at request time.
        resp = _translate_via_job(client, p.id, {"language": "es"})
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["hook_sec"] == pytest.approx(10.0)


class TestPromoteRunScope:
    """Q3: POST /runs/{rid}/to-project takes the new project's scope."""

    def _done_run(self, session, tmp_path):
        project = _make_project(session, status=ProjectStatus.ready, name="Parent")
        result = tmp_path / "final.mp4"
        result.write_bytes(b"final-video-bytes")
        return _make_run(
            session, project.id, status=RunStatus.done, result_local_path=str(result),
        )

    def _promote(self, client, rid, **data):
        return client.post(f"/api/v2/runs/{rid}/to-project", data=data)

    def test_whole_scope_is_stored_for_the_first_analysis(
        self, spy_client, SessionFactory, db_session, tmp_path
    ):
        client, spy = spy_client
        run = self._done_run(db_session, tmp_path)
        resp = self._promote(
            client, run.id, project_type="localisation", localisation_scope="whole",
        )
        assert resp.status_code == 201, resp.text
        new_id = resp.json()["project_id"]
        assert spy["analyze_project"] == [new_id]
        s = SessionFactory()
        try:
            p = s.get(VideoProject, new_id)
            assert p.localisation_scope == "whole"
            assert p.localisation_scope_requested is None
        finally:
            s.close()

    def test_default_is_hook(self, spy_client, SessionFactory, db_session, tmp_path):
        client, _ = spy_client
        run = self._done_run(db_session, tmp_path)
        resp = self._promote(client, run.id, project_type="localisation")
        assert resp.status_code == 201, resp.text
        s = SessionFactory()
        try:
            assert s.get(VideoProject, resp.json()["project_id"]).localisation_scope == "hook"
        finally:
            s.close()

    def test_whole_on_another_type_is_400(self, spy_client, db_session, tmp_path):
        client, spy = spy_client
        run = self._done_run(db_session, tmp_path)
        resp = self._promote(
            client, run.id, project_type="cover_change", localisation_scope="whole",
        )
        assert resp.status_code == 400
        assert spy["analyze_project"] == []

    def test_unknown_scope_is_400(self, spy_client, db_session, tmp_path):
        client, _ = spy_client
        run = self._done_run(db_session, tmp_path)
        resp = self._promote(
            client, run.id, project_type="localisation", localisation_scope="all",
        )
        assert resp.status_code == 400


class TestWholeVideoRollWarning:
    """Q4: a whole-video swap clip that rolls push past the target only WARNS."""

    def _project(self, session, scope, *, pre=0.0, post=0.0, start=0.0, end=14.0):
        p = _make_project(
            session, status=ProjectStatus.ready, project_type="localisation",
            duration_sec=60.0, localisation_scope=scope,
        )
        _make_segment_def(
            session, p.id, 0, start_sec=start, end_sec=end, action="swap",
            has_face=False, pre_roll_sec=pre, post_roll_sec=post,
        )
        _make_segment_def(
            session, p.id, 1, start_sec=end, end_sec=60.0, action="keep", has_face=False,
        )
        return p

    def _launch(self, client, pid, model="seedance-2-5"):
        return client.post(
            f"/api/v2/projects/{pid}/runs",
            data={"prompt": "x", "model": model, "audio_mode": "seedance"},
        )

    def test_rolls_past_the_target_warn_but_launch(self, client, db_session):
        p = self._project(db_session, "whole", start=10.0, end=24.0, pre=1.0, post=0.5)
        resp = self._launch(client, p.id)
        assert resp.status_code == 201, resp.text
        warnings = resp.json()["warnings"]
        assert len(warnings) == 1
        assert "#0 (15.50s, segment 14.00s)" in warnings[0]
        assert "launched anyway" in warnings[0]

    def test_the_model_roll_trim_is_applied(self, client, db_session):
        # Seedance 2.0 (15 s cap) trims post-roll first: 14 + 1 pre + 0.5 post
        # becomes a 15 s clip, still over 14.72 -> warn with 15.00 s.
        p = self._project(db_session, "whole", start=10.0, end=24.0, pre=1.0, post=0.5)
        resp = self._launch(client, p.id, model="seedance")
        assert resp.status_code == 201, resp.text
        assert "#0 (15.00s" in resp.json()["warnings"][0]

    def test_within_the_target_no_warning(self, client, db_session):
        p = self._project(db_session, "whole", start=10.0, end=24.0, pre=0.5)
        resp = self._launch(client, p.id)
        assert resp.status_code == 201, resp.text
        assert resp.json()["warnings"] == []

    def test_hook_scope_never_warns(self, client, db_session):
        p = self._project(db_session, "hook", start=10.0, end=24.0, pre=1.0, post=0.5)
        resp = self._launch(client, p.id)
        assert resp.status_code == 201, resp.text
        assert resp.json()["warnings"] == []
