"""POST /api/v2/runs/{rid}/to-project from a run's step (AICP-331).

"Move to new project" may copy the output of one of the run's derived
creatives — a caption job, a tail edit or an overlay — instead of the run's
final video. The run path itself is covered (unchanged) by
tests/test_api_v2.py::TestPromoteRunToProject.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone

import pytest

from app import storage
from app.models import OverlayJob, RunCaption, RunTailEdit, VideoProject
from app.state_machine import (
    CaptionStatus,
    OverlayStatus,
    ProjectStatus,
    RunStatus,
    TailEditStatus,
)

# The API suite's fixtures and helpers (only those — never its Test* classes).
from tests.test_api_v2 import (  # noqa: F401
    SessionFactory,
    _make_caption,
    _make_project,
    _make_run,
    _make_tail_edit,
    client,
    db_session,
    engine,
    enqueue_spy,
    spy_client,
    tmp_path,
)

KINDS = ("caption", "tail_edit", "overlay")


def _project_dirs() -> set[str]:
    root = os.path.join(storage.data_dir_root(), "projects")
    return set(os.listdir(root)) if os.path.isdir(root) else set()


def _project_ids(session) -> set[str]:
    session.expire_all()
    return {p.id for p in session.query(VideoProject).all()}


def _done_run(session, tmp_path, *, name="Redhead woman", **project_kwargs):
    project = _make_project(session, status=ProjectStatus.ready, name="Parent", **project_kwargs)
    final = tmp_path / "final.mp4"
    final.write_bytes(b"run-final-bytes")
    run = _make_run(session, project.id, status=RunStatus.done,
                    result_local_path=str(final), name=name)
    return project, run


def _make_step(session, tmp_path, kind: str, run_id: str, *, done=True, with_file=True,
               payload: bytes | None = None):
    """A caption / tail edit / overlay of *run_id* whose output holds *payload*."""
    payload = payload if payload is not None else f"{kind}-output-bytes".encode()
    out = tmp_path / f"{kind}-out.mp4"
    if with_file:
        out.write_bytes(payload)
    if kind == "caption":
        return _make_caption(session, run_id,
                             status=CaptionStatus.done if done else CaptionStatus.burning,
                             output_path=str(out))
    if kind == "tail_edit":
        return _make_tail_edit(session, run_id,
                               status=TailEditStatus.done if done else TailEditStatus.processing,
                               output_path=str(out))
    job = OverlayJob(source_type="run", source_run_id=run_id, parent_run_id=run_id,
                     status=OverlayStatus.done if done else OverlayStatus.processing,
                     overlays=[], result_local_path=str(out))
    session.add(job)
    session.commit()
    return job


def _post(client, run_id, kind=None, step_id=None, **data):
    if kind is not None:
        data["source_kind"] = kind
    if step_id is not None:
        data["source_id"] = step_id
    return client.post(f"/api/v2/runs/{run_id}/to-project", data=data)


# ---------------------------------------------------------------------------
# Happy path, per kind
# ---------------------------------------------------------------------------


class TestPromoteStep:
    @pytest.mark.parametrize("kind", KINDS)
    def test_done_step_becomes_new_project(
        self, spy_client, SessionFactory, db_session, tmp_path, kind
    ):
        client, spy = spy_client
        _, run = _done_run(db_session, tmp_path)
        payload = os.urandom(4096) + kind.encode()
        step = _make_step(db_session, tmp_path, kind, run.id, payload=payload)

        resp = _post(client, run.id, kind, step.id, project_type="cover_change")
        assert resp.status_code == 201, resp.text
        new_id = resp.json()["project_id"]
        assert spy["analyze_project"] == [new_id]

        session = SessionFactory()
        p = session.get(VideoProject, new_id)
        session.close()
        assert p.project_type == "cover_change"
        assert p.source_type == "upload"
        # Lineage: the OWNING run, plus the step as soft refs.
        assert p.source_run_id == run.id
        assert p.source_kind == kind
        assert p.source_artifact_id == step.id
        label = {"caption": "captions", "tail_edit": "tail edit", "overlay": "overlay"}[kind]
        assert p.name == f"Redhead woman · {label} → Change cover"
        assert p.source_ref == f"Redhead woman ({run.id[:8]}) — {label} {step.id[:8]}"
        # Byte-identical COPY, the step's file left in place.
        src = step.result_local_path if kind == "overlay" else step.output_path
        assert p.source_local_path != src
        with open(p.source_local_path, "rb") as fh:
            assert fh.read() == payload
        assert os.path.exists(src)

    def test_api_response_exposes_the_lineage(self, client, db_session, tmp_path):
        _, run = _done_run(db_session, tmp_path)
        cap = _make_step(db_session, tmp_path, "caption", run.id)
        new_id = _post(client, run.id, "caption", cap.id).json()["project_id"]
        body = client.get(f"/api/v2/projects/{new_id}").json()
        assert body["source_run_id"] == run.id
        assert body["source_kind"] == "caption"
        assert body["source_artifact_id"] == cap.id

    def test_inherits_settings_from_the_owning_runs_project(
        self, spy_client, SessionFactory, db_session, tmp_path
    ):
        client, _ = spy_client
        project, run = _done_run(db_session, tmp_path, mute_source=True)
        project.max_segment_sec = 30.0
        db_session.commit()
        edit = _make_step(db_session, tmp_path, "tail_edit", run.id)

        resp = _post(client, run.id, "tail_edit", edit.id, project_type="cover_change")
        assert resp.status_code == 201, resp.text
        session = SessionFactory()
        p = session.get(VideoProject, resp.json()["project_id"])
        session.close()
        assert p.mute_source is True
        assert p.max_segment_sec == 30.0

    @pytest.mark.parametrize("kind", [None, *KINDS])
    @pytest.mark.parametrize(
        "project_type,expected",
        [("face_swap", False), ("cover_change", True), ("localisation", True)],
    )
    def test_mute_source_from_a_muted_localisation_lineage(
        self, spy_client, SessionFactory, db_session, tmp_path, kind, project_type, expected
    ):
        """Every element of a run (the run itself, captions, tail edit,
        overlay) follows the same rule: a Face changing project starts with
        "Remove original audio" OFF even when the parent is a muted
        localisation project; other types keep their behaviour."""
        client, _ = spy_client
        project, run = _done_run(
            db_session, tmp_path, project_type="localisation", mute_source=True
        )
        step_id = None if kind is None else _make_step(db_session, tmp_path, kind, run.id).id

        resp = _post(client, run.id, kind, step_id, project_type=project_type)
        assert resp.status_code == 201, resp.text
        session = SessionFactory()
        p = session.get(VideoProject, resp.json()["project_id"])
        session.close()
        assert p.mute_source is expected

    def test_unusable_parent_cap_is_not_propagated(
        self, spy_client, SessionFactory, db_session, tmp_path
    ):
        client, _ = spy_client
        project, run = _done_run(db_session, tmp_path)
        project.max_segment_sec = 2.0
        db_session.commit()
        cap = _make_step(db_session, tmp_path, "caption", run.id)
        resp = _post(client, run.id, "caption", cap.id, project_type="cover_change")
        assert resp.status_code == 201, resp.text
        session = SessionFactory()
        assert session.get(VideoProject, resp.json()["project_id"]).max_segment_sec is None
        session.close()

    def test_explicit_name_wins(self, spy_client, SessionFactory, db_session, tmp_path):
        client, _ = spy_client
        _, run = _done_run(db_session, tmp_path)
        job = _make_step(db_session, tmp_path, "overlay", run.id)
        resp = _post(client, run.id, "overlay", job.id, name="Cover v3")
        assert resp.status_code == 201, resp.text
        session = SessionFactory()
        assert session.get(VideoProject, resp.json()["project_id"]).name == "Cover v3"
        session.close()

    def test_step_of_a_run_that_is_no_longer_done_still_moves(
        self, spy_client, db_session, tmp_path
    ):
        """The step's file does not depend on the run's current result."""
        client, _ = spy_client
        _, run = _done_run(db_session, tmp_path)
        cap = _make_step(db_session, tmp_path, "caption", run.id)
        run.status = RunStatus.failed
        db_session.commit()
        assert _post(client, run.id, "caption", cap.id).status_code == 201

    def test_overlay_burned_from_a_caption_is_owned_by_the_run(
        self, spy_client, SessionFactory, db_session, tmp_path
    ):
        client, _ = spy_client
        _, run = _done_run(db_session, tmp_path)
        cap = _make_step(db_session, tmp_path, "caption", run.id)
        out = tmp_path / "ov.mp4"
        out.write_bytes(b"ov")
        job = OverlayJob(source_type="caption", source_caption_id=cap.id, parent_run_id=run.id,
                         status=OverlayStatus.done, overlays=[], result_local_path=str(out))
        db_session.add(job)
        db_session.commit()
        resp = _post(client, run.id, "overlay", job.id)
        assert resp.status_code == 201, resp.text
        session = SessionFactory()
        p = session.get(VideoProject, resp.json()["project_id"])
        session.close()
        assert (p.source_run_id, p.source_kind, p.source_artifact_id) == (run.id, "overlay", job.id)

    def test_legacy_overlay_without_parent_stamp_resolves_through_its_source(
        self, spy_client, db_session, tmp_path
    ):
        client, _ = spy_client
        _, run = _done_run(db_session, tmp_path)
        edit = _make_step(db_session, tmp_path, "tail_edit", run.id)
        out = tmp_path / "legacy-ov.mp4"
        out.write_bytes(b"ov")
        job = OverlayJob(source_type="tail_edit", source_tail_edit_id=edit.id,
                         parent_run_id=None, status=OverlayStatus.done, overlays=[],
                         result_local_path=str(out))
        db_session.add(job)
        db_session.commit()
        assert _post(client, run.id, "overlay", job.id).status_code == 201

    def test_deleting_the_step_afterwards_leaves_the_project_usable(
        self, spy_client, SessionFactory, db_session, tmp_path
    ):
        client, _ = spy_client
        _, run = _done_run(db_session, tmp_path)
        cap = _make_step(db_session, tmp_path, "caption", run.id, payload=b"captioned")
        new_id = _post(client, run.id, "caption", cap.id).json()["project_id"]

        assert client.delete(f"/api/v2/captions/{cap.id}").status_code == 204
        body = client.get(f"/api/v2/projects/{new_id}")
        assert body.status_code == 200
        assert body.json()["source_run_id"] == run.id
        session = SessionFactory()
        p = session.get(VideoProject, new_id)
        session.close()
        with open(p.source_local_path, "rb") as fh:
            assert fh.read() == b"captioned"


# ---------------------------------------------------------------------------
# Refusals: nothing created (no row, no file)
# ---------------------------------------------------------------------------


class TestPromoteStepRefusals:
    def _assert_refused(self, client, db_session, run_id, kind, step_id, code, needle):
        before_rows, before_dirs = _project_ids(db_session), _project_dirs()
        resp = _post(client, run_id, kind, step_id)
        assert resp.status_code == code, resp.text
        assert needle in resp.json()["detail"]
        assert _project_ids(db_session) == before_rows
        assert _project_dirs() == before_dirs

    @pytest.mark.parametrize("kind", KINDS)
    def test_not_done_is_409(self, client, db_session, tmp_path, kind):
        _, run = _done_run(db_session, tmp_path)
        step = _make_step(db_session, tmp_path, kind, run.id, done=False)
        self._assert_refused(client, db_session, run.id, kind, step.id, 409, "not done yet")

    @pytest.mark.parametrize("kind", KINDS)
    def test_missing_file_is_409(self, client, db_session, tmp_path, kind):
        _, run = _done_run(db_session, tmp_path)
        step = _make_step(db_session, tmp_path, kind, run.id, with_file=False)
        self._assert_refused(client, db_session, run.id, kind, step.id, 409, "no output video")

    @pytest.mark.parametrize("kind", KINDS)
    def test_foreign_run_is_404(self, client, db_session, tmp_path, kind):
        _, run = _done_run(db_session, tmp_path)
        _, other = _done_run(db_session, tmp_path, name="Other")
        step = _make_step(db_session, tmp_path, kind, other.id)
        self._assert_refused(client, db_session, run.id, kind, step.id, 404, "not found on this run")

    @pytest.mark.parametrize("kind", KINDS)
    def test_unknown_step_is_404(self, client, db_session, tmp_path, kind):
        _, run = _done_run(db_session, tmp_path)
        self._assert_refused(client, db_session, run.id, kind, "no-such-id", 404, "not found")

    def test_soft_deleted_overlay_is_404(self, client, db_session, tmp_path):
        _, run = _done_run(db_session, tmp_path)
        job = _make_step(db_session, tmp_path, "overlay", run.id)
        job.deleted_at = datetime.now(timezone.utc)
        db_session.commit()
        self._assert_refused(client, db_session, run.id, "overlay", job.id, 404, "not found")

    def test_upload_overlay_without_parent_run_is_404(self, client, db_session, tmp_path):
        _, run = _done_run(db_session, tmp_path)
        out = tmp_path / "up.mp4"
        out.write_bytes(b"up")
        job = OverlayJob(source_type="upload", parent_run_id=None, status=OverlayStatus.done,
                         overlays=[], result_local_path=str(out),
                         source_local_path=str(out))
        db_session.add(job)
        db_session.commit()
        self._assert_refused(client, db_session, run.id, "overlay", job.id, 404, "not found")

    def test_unknown_kind_is_400(self, client, db_session, tmp_path):
        _, run = _done_run(db_session, tmp_path)
        self._assert_refused(client, db_session, run.id, "assistant_version", "x", 400,
                             "source_kind must be one of")

    def test_step_kind_without_id_is_400(self, client, db_session, tmp_path):
        _, run = _done_run(db_session, tmp_path)
        self._assert_refused(client, db_session, run.id, "caption", None, 400, "source_id is required")

    def test_run_kind_with_id_is_400(self, client, db_session, tmp_path):
        _, run = _done_run(db_session, tmp_path)
        self._assert_refused(client, db_session, run.id, "run", "x", 400, "only valid")

    def test_missing_run_is_404(self, client, db_session, tmp_path):
        _, run = _done_run(db_session, tmp_path)
        cap = _make_step(db_session, tmp_path, "caption", run.id)
        self._assert_refused(client, db_session, "no-such-run", "caption", cap.id, 404, "not found")


# ---------------------------------------------------------------------------
# Failures after the copy: no orphan file
# ---------------------------------------------------------------------------


class TestPromoteNoOrphans:
    @pytest.mark.parametrize("kind", ("run", "caption"))
    def test_commit_failure_removes_the_copy(self, client, db_session, tmp_path, monkeypatch, kind):
        import app.api_v2 as api_v2_module

        _, run = _done_run(db_session, tmp_path)
        step = _make_step(db_session, tmp_path, "caption", run.id) if kind == "caption" else None

        def boom(*a, **k):
            raise RuntimeError("placement exploded")

        # Raised inside the guarded block (after the copy, before the commit).
        monkeypatch.setattr(api_v2_module, "_inherited_max_segment_sec", boom)
        before_rows, before_dirs = _project_ids(db_session), _project_dirs()
        with pytest.raises(RuntimeError):
            _post(client, run.id, None if kind == "run" else "caption",
                  step.id if step else None)
        assert _project_ids(db_session) == before_rows
        assert _project_dirs() == before_dirs

    def test_enqueue_failure_after_commit_keeps_project_and_file(
        self, client, db_session, tmp_path, monkeypatch
    ):
        import app.api_v2 as api_v2_module

        _, run = _done_run(db_session, tmp_path)
        cap = _make_step(db_session, tmp_path, "caption", run.id, payload=b"kept")

        def down(pid):
            raise RuntimeError("redis down")

        monkeypatch.setattr(api_v2_module, "enqueue_analyze_project", down)
        before_rows = _project_ids(db_session)
        with pytest.raises(RuntimeError):
            _post(client, run.id, "caption", cap.id)
        (new_id,) = _project_ids(db_session) - before_rows
        p = db_session.get(VideoProject, new_id)
        with open(p.source_local_path, "rb") as fh:
            assert fh.read() == b"kept"


# ---------------------------------------------------------------------------
# Run path regression: the step fields stay NULL
# ---------------------------------------------------------------------------


def test_run_final_path_records_no_step(spy_client, SessionFactory, db_session, tmp_path):
    client, spy = spy_client
    _, run = _done_run(db_session, tmp_path)
    resp = client.post(f"/api/v2/runs/{run.id}/to-project", data={"project_type": "cover_change"})
    assert resp.status_code == 201, resp.text
    session = SessionFactory()
    p = session.get(VideoProject, resp.json()["project_id"])
    session.close()
    assert (p.source_kind, p.source_artifact_id) == (None, None)
    assert p.name == "Redhead woman → Change cover"
    assert p.source_ref == f"Redhead woman ({run.id[:8]}) — final video"
    with open(p.source_local_path, "rb") as fh:
        assert fh.read() == b"run-final-bytes"
    # Explicit source_kind=run is the same request.
    resp2 = client.post(f"/api/v2/runs/{run.id}/to-project",
                        data={"project_type": "cover_change", "source_kind": "run"})
    assert resp2.status_code == 201, resp2.text
