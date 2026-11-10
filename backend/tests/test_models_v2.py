"""
Tests for v2 ORM models: VideoProject, SegmentDef, Run, RunSegment.

Uses an in-memory SQLite database — no external dependencies required.
"""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker

from app.db import Base
from app.models import (
    InsertExtraction,
    InsertMerge,
    Run,
    RunCaption,
    RunSegment,
    RunTailEdit,
    SegmentDef,
    VideoProject,
)
from app.state_machine import (
    InvalidTransition,
    MergeStatus,
    ProjectStatus,
    RunStatus,
    SegmentStatus,
    can_transition,
    transition,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def engine():
    eng = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(eng)
    yield eng
    Base.metadata.drop_all(eng)
    eng.dispose()


@pytest.fixture()
def session(engine):
    Session = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    sess = Session()
    yield sess
    sess.rollback()
    sess.close()


# ---------------------------------------------------------------------------
# Helper builders
# ---------------------------------------------------------------------------


def make_project(session, source_type="upload", source_ref="s3://bucket/video.mp4"):
    proj = VideoProject(
        source_type=source_type,
        source_ref=source_ref,
        duration_sec=30.0,
        width=1920,
        height=1080,
        fps=24.0,
        aspect_ratio="16:9",
        status=ProjectStatus.created,
    )
    session.add(proj)
    session.flush()
    return proj


def make_segment_defs(session, project_id, count=3):
    defs = [
        SegmentDef(
            project_id=project_id,
            index=i,
            start_sec=i * 10.0,
            end_sec=(i + 1) * 10.0,
            has_face=(i % 2 == 0),
            action="swap" if i % 2 == 0 else "keep",
        )
        for i in range(count)
    ]
    session.add_all(defs)
    session.flush()
    return defs


def make_run(session, project_id, name="Run A", refs=None):
    run = Run(
        project_id=project_id,
        name=name,
        prompt="A redhead woman",
        reference_image_urls=refs if refs is not None else ["https://cdn.example.com/ref1.jpg"],
        resolution="720p",
    )
    session.add(run)
    session.flush()
    return run


def make_run_segments(session, run_id, segment_defs):
    rs_list = [
        RunSegment(
            run_id=run_id,
            segment_def_id=sd.id,
            index=sd.index,
            status=SegmentStatus.pending,
        )
        for sd in segment_defs
        if sd.action == "swap"
    ]
    session.add_all(rs_list)
    session.flush()
    return rs_list


def make_tail_edit(session, run_id, cut_sec=5.0, **kwargs):
    te = RunTailEdit(run_id=run_id, cut_sec=cut_sec, **kwargs)
    session.add(te)
    session.flush()
    return te


def make_caption(session, run_id, **kwargs):
    rc = RunCaption(run_id=run_id, **kwargs)
    session.add(rc)
    session.flush()
    return rc


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestVideoProject:
    def test_create_and_query(self, session):
        proj = make_project(session)
        session.commit()

        fetched = session.get(VideoProject, proj.id)
        assert fetched is not None
        assert fetched.source_type == "upload"
        assert fetched.width == 1920
        assert fetched.fps == 24.0
        assert fetched.status == ProjectStatus.created

    def test_timestamps_set(self, session):
        proj = make_project(session)
        session.commit()
        fetched = session.get(VideoProject, proj.id)
        assert fetched.created_at is not None
        assert fetched.updated_at is not None

    def test_gdrive_source_type(self, session):
        proj = make_project(session, source_type="gdrive", source_ref="https://drive.google.com/file/abc")
        session.commit()
        fetched = session.get(VideoProject, proj.id)
        assert fetched.source_type == "gdrive"

    def test_max_segment_sec_defaults_to_none(self, session):
        """NULL is the default and means ai_models.UNIVERSAL_MAX_SEGMENT_SEC.

        Storing NULL rather than the resolved number is deliberate: every row
        that predates the column reads as "no explicit choice" and follows the
        registry if its universal default ever moves.
        """
        proj = make_project(session)
        session.commit()
        assert session.get(VideoProject, proj.id).max_segment_sec is None

    @pytest.mark.parametrize("value", [2.0, 10.0, 15.0, 30.0, 12.5])
    def test_max_segment_sec_round_trips(self, session, value):
        proj = make_project(session)
        proj.max_segment_sec = value
        session.commit()

        fetched = session.get(VideoProject, proj.id)
        assert fetched.max_segment_sec == value
        assert isinstance(fetched.max_segment_sec, float)

    def test_max_segment_sec_can_be_cleared_back_to_null(self, session):
        """Clearing the cap is a real edit (PATCH sends an explicit null)."""
        proj = make_project(session)
        proj.max_segment_sec = 30.0
        session.commit()

        proj.max_segment_sec = None
        session.commit()
        assert session.get(VideoProject, proj.id).max_segment_sec is None


class TestSegmentDef:
    def test_three_segment_defs_created(self, session):
        proj = make_project(session)
        defs = make_segment_defs(session, proj.id, count=3)
        session.commit()

        fetched_proj = session.get(VideoProject, proj.id)
        assert len(fetched_proj.segments) == 3

    def test_ordered_by_index(self, session):
        proj = make_project(session)
        # Add in reverse order to prove ordering works
        defs = [
            SegmentDef(project_id=proj.id, index=2, start_sec=20.0, end_sec=30.0, has_face=False, action="keep"),
            SegmentDef(project_id=proj.id, index=0, start_sec=0.0, end_sec=10.0, has_face=True, action="swap"),
            SegmentDef(project_id=proj.id, index=1, start_sec=10.0, end_sec=20.0, has_face=False, action="keep"),
        ]
        session.add_all(defs)
        session.commit()

        fetched_proj = session.get(VideoProject, proj.id)
        indices = [sd.index for sd in fetched_proj.segments]
        assert indices == sorted(indices)

    def test_relationship_back_populates(self, session):
        proj = make_project(session)
        defs = make_segment_defs(session, proj.id, count=2)
        session.commit()

        seg = session.get(SegmentDef, defs[0].id)
        assert seg.project.id == proj.id

    def test_pre_post_roll_defaults(self, session):
        proj = make_project(session)
        sd = SegmentDef(project_id=proj.id, index=0, start_sec=0.0, end_sec=5.0, has_face=False, action="keep")
        session.add(sd)
        session.commit()
        fetched = session.get(SegmentDef, sd.id)
        assert fetched.pre_roll_sec == 0.0
        assert fetched.post_roll_sec == 0.0


class TestRun:
    def test_create_two_runs(self, session):
        proj = make_project(session)
        r1 = make_run(session, proj.id, name="Redhead Woman")
        r2 = make_run(session, proj.id, name="Brunette Man")
        session.commit()

        fetched_proj = session.get(VideoProject, proj.id)
        assert len(fetched_proj.runs) == 2

    def test_json_round_trip_reference_image_urls(self, session):
        proj = make_project(session)
        urls = ["https://cdn.example.com/a.jpg", "https://cdn.example.com/b.jpg"]
        run = make_run(session, proj.id, refs=urls)
        session.commit()

        fetched = session.get(Run, run.id)
        assert fetched.reference_image_urls == urls
        assert isinstance(fetched.reference_image_urls, list)

    def test_empty_json_list_default(self, session):
        proj = make_project(session)
        run = Run(project_id=proj.id, prompt="test", reference_image_urls=[])
        session.add(run)
        session.commit()
        fetched = session.get(Run, run.id)
        assert fetched.reference_image_urls == []

    def test_default_status_created(self, session):
        proj = make_project(session)
        run = make_run(session, proj.id)
        session.commit()
        fetched = session.get(Run, run.id)
        assert fetched.status == RunStatus.created

    def test_default_audio_mode_is_original(self, session):
        """Run.audio_mode defaults to 'original' when not specified."""
        proj = make_project(session)
        run = Run(project_id=proj.id, prompt="test", reference_image_urls=[])
        session.add(run)
        session.commit()
        fetched = session.get(Run, run.id)
        audio_mode_val = (
            fetched.audio_mode.value
            if hasattr(fetched.audio_mode, "value")
            else str(fetched.audio_mode)
        )
        assert audio_mode_val == "original"

    def test_audio_mode_seedance_round_trip(self, session):
        """Run.audio_mode can be set to 'seedance' and round-trips correctly."""
        proj = make_project(session)
        run = Run(
            project_id=proj.id,
            prompt="test",
            reference_image_urls=[],
            audio_mode="seedance",
        )
        session.add(run)
        session.commit()
        fetched = session.get(Run, run.id)
        audio_mode_val = (
            fetched.audio_mode.value
            if hasattr(fetched.audio_mode, "value")
            else str(fetched.audio_mode)
        )
        assert audio_mode_val == "seedance"

    def test_run_project_back_populates(self, session):
        proj = make_project(session)
        run = make_run(session, proj.id)
        session.commit()
        fetched_run = session.get(Run, run.id)
        assert fetched_run.project.id == proj.id

    @pytest.mark.parametrize(
        "model",
        ["seedance", "seedance-fast", "seedance-mini", "seedance-2-5", "gemini-omni"],
    )
    def test_model_accepts_every_registered_key(self, session, model):
        """run_model_enum must carry a label for every app.ai_models entry.

        The enum is mirrored by hand in migrations (Postgres cannot drop a
        value), so a spec added without its ALTER TYPE would be rejected in
        production — this pins the two together.
        """
        from app import ai_models

        assert model in ai_models.AI_MODELS

        proj = make_project(session)
        run = Run(
            project_id=proj.id,
            prompt="test",
            reference_image_urls=[],
            model=model,
        )
        session.add(run)
        session.commit()
        assert session.get(Run, run.id).model == model

    def test_model_defaults_to_seedance(self, session):
        proj = make_project(session)
        run = Run(project_id=proj.id, prompt="test", reference_image_urls=[])
        session.add(run)
        session.commit()
        assert session.get(Run, run.id).model == "seedance"


class TestRunSegment:
    def test_run_segments_created(self, session):
        proj = make_project(session)
        seg_defs = make_segment_defs(session, proj.id, count=3)  # indices 0,1,2; swap at 0,2
        run = make_run(session, proj.id)
        rs_list = make_run_segments(session, run.id, seg_defs)
        session.commit()

        # Only swap segments (index 0 and 2) get RunSegments
        fetched_run = session.get(Run, run.id)
        assert len(fetched_run.run_segments) == 2

    def test_run_segments_ordered_by_index(self, session):
        proj = make_project(session)
        # All swap so all get RunSegments
        seg_defs = [
            SegmentDef(project_id=proj.id, index=i, start_sec=i * 5.0, end_sec=(i + 1) * 5.0, has_face=True, action="swap")
            for i in range(4)
        ]
        session.add_all(seg_defs)
        session.flush()
        run = make_run(session, proj.id)
        # Add in reverse order
        rs_list = [
            RunSegment(run_id=run.id, segment_def_id=sd.id, index=sd.index, status=SegmentStatus.pending)
            for sd in reversed(seg_defs)
        ]
        session.add_all(rs_list)
        session.commit()

        fetched_run = session.get(Run, run.id)
        indices = [rs.index for rs in fetched_run.run_segments]
        assert indices == sorted(indices)

    def test_default_status_pending(self, session):
        proj = make_project(session)
        sd = SegmentDef(project_id=proj.id, index=0, start_sec=0.0, end_sec=5.0, has_face=True, action="swap")
        session.add(sd)
        session.flush()
        run = make_run(session, proj.id)
        rs = RunSegment(run_id=run.id, segment_def_id=sd.id, index=0)
        session.add(rs)
        session.commit()
        fetched = session.get(RunSegment, rs.id)
        assert fetched.status == SegmentStatus.pending

    def test_run_segment_back_populates(self, session):
        proj = make_project(session)
        sd = SegmentDef(project_id=proj.id, index=0, start_sec=0.0, end_sec=5.0, has_face=True, action="swap")
        session.add(sd)
        session.flush()
        run = make_run(session, proj.id)
        rs = RunSegment(run_id=run.id, segment_def_id=sd.id, index=0)
        session.add(rs)
        session.commit()
        fetched = session.get(RunSegment, rs.id)
        assert fetched.run.id == run.id


class TestCascadeDelete:
    def test_delete_project_cascades_to_segment_defs_runs_run_segments(self, session):
        proj = make_project(session)
        seg_defs = make_segment_defs(session, proj.id, count=3)
        run1 = make_run(session, proj.id, name="Run 1")
        run2 = make_run(session, proj.id, name="Run 2")
        rs1 = make_run_segments(session, run1.id, seg_defs)
        rs2 = make_run_segments(session, run2.id, seg_defs)
        session.commit()

        proj_id = proj.id
        run1_id = run1.id
        sd_id = seg_defs[0].id
        rs_id = rs1[0].id if rs1 else None

        # Delete the project
        session.delete(proj)
        session.commit()

        assert session.get(VideoProject, proj_id) is None
        assert session.get(Run, run1_id) is None
        assert session.get(SegmentDef, sd_id) is None
        if rs_id:
            assert session.get(RunSegment, rs_id) is None

    def test_delete_run_cascades_to_run_segments_only(self, session):
        proj = make_project(session)
        seg_defs = make_segment_defs(session, proj.id, count=2)
        run = make_run(session, proj.id)
        rs_list = make_run_segments(session, run.id, seg_defs)
        session.commit()

        run_id = run.id
        proj_id = proj.id
        sd_id = seg_defs[0].id

        session.delete(run)
        session.commit()

        assert session.get(Run, run_id) is None
        # Project and SegmentDefs should survive
        assert session.get(VideoProject, proj_id) is not None
        assert session.get(SegmentDef, sd_id) is not None


class TestRunSegmentOverrides:
    """Verify that prompt_override and reference_image_urls_override round-trip correctly."""

    def test_prompt_override_round_trip(self, session):
        proj = make_project(session)
        sd = SegmentDef(
            project_id=proj.id, index=0, start_sec=0.0, end_sec=5.0,
            has_face=True, action="swap"
        )
        session.add(sd)
        session.flush()
        run = make_run(session, proj.id)
        rs = RunSegment(
            run_id=run.id, segment_def_id=sd.id, index=0,
            prompt_override="Swap to redhead for this segment only",
        )
        session.add(rs)
        session.commit()

        fetched = session.get(RunSegment, rs.id)
        assert fetched.prompt_override == "Swap to redhead for this segment only"

    def test_reference_image_urls_override_round_trip(self, session):
        proj = make_project(session)
        sd = SegmentDef(
            project_id=proj.id, index=1, start_sec=5.0, end_sec=10.0,
            has_face=True, action="swap"
        )
        session.add(sd)
        session.flush()
        run = make_run(session, proj.id)
        override_urls = ["/data/refs/a.jpg", "https://cdn.example.com/b.jpg"]
        rs = RunSegment(
            run_id=run.id, segment_def_id=sd.id, index=1,
            reference_image_urls_override=override_urls,
        )
        session.add(rs)
        session.commit()

        fetched = session.get(RunSegment, rs.id)
        assert fetched.reference_image_urls_override == override_urls
        assert isinstance(fetched.reference_image_urls_override, list)

    def test_overrides_default_to_none(self, session):
        proj = make_project(session)
        sd = SegmentDef(
            project_id=proj.id, index=2, start_sec=10.0, end_sec=15.0,
            has_face=True, action="swap"
        )
        session.add(sd)
        session.flush()
        run = make_run(session, proj.id)
        rs = RunSegment(run_id=run.id, segment_def_id=sd.id, index=2)
        session.add(rs)
        session.commit()

        fetched = session.get(RunSegment, rs.id)
        assert fetched.prompt_override is None
        assert fetched.reference_image_urls_override is None


class TestTableCreation:
    """Smoke test — verify all v2 tables exist after Base.metadata.create_all."""

    def test_v2_tables_present(self, engine):
        insp = inspect(engine)
        tables = set(insp.get_table_names())
        assert "video_projects" in tables
        assert "segment_defs" in tables
        assert "runs" in tables
        assert "run_segments" in tables
        assert "insert_extractions" in tables
        assert "insert_merges" in tables

    def test_v1_tables_removed(self, engine):
        # The v1 Job/Segment pipeline was removed; their ORM models no longer
        # exist, so create_all() must not recreate the old tables.
        insp = inspect(engine)
        tables = set(insp.get_table_names())
        assert "jobs" not in tables
        assert "segments" not in tables


class TestInsertExtraction:
    """Tests for PIP insert extraction model."""

    def test_create_insertion_extraction(self, session):
        """Create a parent project, child project, and extraction linking them."""
        parent = make_project(session, source_ref="s3://bucket/parent.mp4")
        child = make_project(session, source_ref="s3://bucket/child.mp4")
        session.flush()

        extraction = InsertExtraction(
            parent_project_id=parent.id,
            child_project_id=child.id,
            x=100,
            y=50,
            width=400,
            height=500,
            start_sec=5.0,
            end_sec=25.0,
            corner_radius_px=10,
        )
        session.add(extraction)
        session.commit()

        fetched = session.get(InsertExtraction, extraction.id)
        assert fetched is not None
        assert fetched.parent_project_id == parent.id
        assert fetched.child_project_id == child.id
        assert fetched.x == 100
        assert fetched.y == 50
        assert fetched.width == 400
        assert fetched.height == 500
        assert fetched.start_sec == 5.0
        assert fetched.end_sec == 25.0
        assert fetched.corner_radius_px == 10

    def test_corner_radius_defaults_to_zero(self, session):
        """Verify corner_radius_px defaults to 0."""
        parent = make_project(session)
        child = make_project(session, source_ref="s3://bucket/child2.mp4")
        session.flush()

        extraction = InsertExtraction(
            parent_project_id=parent.id,
            child_project_id=child.id,
            x=0,
            y=0,
            width=256,
            height=256,
            start_sec=0.0,
            end_sec=10.0,
        )
        session.add(extraction)
        session.commit()

        fetched = session.get(InsertExtraction, extraction.id)
        assert fetched.corner_radius_px == 0

    def test_unique_constraint_on_child_project_id(self, session):
        """Each child project can have at most one extraction."""
        parent1 = make_project(session, source_ref="s3://bucket/p1.mp4")
        parent2 = make_project(session, source_ref="s3://bucket/p2.mp4")
        child = make_project(session, source_ref="s3://bucket/child3.mp4")
        session.flush()

        ext1 = InsertExtraction(
            parent_project_id=parent1.id,
            child_project_id=child.id,
            x=0, y=0, width=100, height=100,
            start_sec=0.0, end_sec=5.0,
        )
        session.add(ext1)
        session.commit()

        # Try to create a second extraction with the same child
        ext2 = InsertExtraction(
            parent_project_id=parent2.id,
            child_project_id=child.id,
            x=0, y=0, width=100, height=100,
            start_sec=0.0, end_sec=5.0,
        )
        session.add(ext2)
        with pytest.raises(Exception):  # UNIQUE constraint violation
            session.commit()


class TestInsertMerge:
    """Tests for PIP insert merge model."""

    def test_create_insert_merge(self, session):
        """Create a merge operation linking a target project and child run."""
        target = make_project(session, source_ref="s3://bucket/target.mp4")
        child = make_project(session, source_ref="s3://bucket/child_merge.mp4")
        run = make_run(session, child.id, name="Child Run")
        session.flush()

        merge = InsertMerge(
            target_project_id=target.id,
            child_run_id=run.id,
            x=100,
            y=200,
            width=300,
            height=400,
            start_sec=1.0,
            end_sec=11.0,
            corner_radius_px=5,
        )
        session.add(merge)
        session.commit()

        fetched = session.get(InsertMerge, merge.id)
        assert fetched is not None
        assert fetched.target_project_id == target.id
        assert fetched.child_run_id == run.id
        assert fetched.x == 100
        assert fetched.y == 200
        assert fetched.width == 300
        assert fetched.height == 400
        assert fetched.start_sec == 1.0
        assert fetched.end_sec == 11.0
        assert fetched.corner_radius_px == 5

    def test_default_status_is_queued(self, session):
        """InsertMerge.status defaults to queued."""
        target = make_project(session)
        child = make_project(session, source_ref="s3://bucket/child_default.mp4")
        run = make_run(session, child.id)
        session.flush()

        merge = InsertMerge(
            target_project_id=target.id,
            child_run_id=run.id,
            x=0, y=0, width=100, height=100,
            start_sec=0.0, end_sec=5.0,
        )
        session.add(merge)
        session.commit()

        fetched = session.get(InsertMerge, merge.id)
        assert fetched.status == MergeStatus.queued

    def test_timestamps_set(self, session):
        """InsertMerge has created_at and updated_at."""
        target = make_project(session)
        child = make_project(session, source_ref="s3://bucket/child_ts.mp4")
        run = make_run(session, child.id)
        session.flush()

        merge = InsertMerge(
            target_project_id=target.id,
            child_run_id=run.id,
            x=0, y=0, width=100, height=100,
            start_sec=0.0, end_sec=5.0,
        )
        session.add(merge)
        session.commit()

        fetched = session.get(InsertMerge, merge.id)
        assert fetched.created_at is not None
        assert fetched.updated_at is not None

    def test_optional_result_fields(self, session):
        """Result fields (output_path, gdrive_url, error_message) default to None."""
        target = make_project(session)
        child = make_project(session, source_ref="s3://bucket/child_result.mp4")
        run = make_run(session, child.id)
        session.flush()

        merge = InsertMerge(
            target_project_id=target.id,
            child_run_id=run.id,
            x=0, y=0, width=100, height=100,
            start_sec=0.0, end_sec=5.0,
        )
        session.add(merge)
        session.commit()

        fetched = session.get(InsertMerge, merge.id)
        assert fetched.output_path is None
        assert fetched.gdrive_url is None
        assert fetched.error_message is None


class TestMergeStatusTransitions:
    """Tests for MergeStatus state machine transitions."""

    def test_can_transition_queued_to_compositing(self):
        """Queued can transition to compositing."""
        assert can_transition(MergeStatus.queued, MergeStatus.compositing)

    def test_can_transition_queued_to_failed(self):
        """Queued can transition to failed."""
        assert can_transition(MergeStatus.queued, MergeStatus.failed)

    def test_cannot_transition_queued_to_done(self):
        """Queued cannot transition directly to done."""
        assert not can_transition(MergeStatus.queued, MergeStatus.done)

    def test_can_transition_compositing_to_delivering(self):
        """Compositing can transition to delivering."""
        assert can_transition(MergeStatus.compositing, MergeStatus.delivering)

    def test_can_transition_compositing_to_failed(self):
        """Compositing can transition to failed."""
        assert can_transition(MergeStatus.compositing, MergeStatus.failed)

    def test_can_transition_delivering_to_done(self):
        """Delivering can transition to done."""
        assert can_transition(MergeStatus.delivering, MergeStatus.done)

    def test_can_transition_delivering_to_failed(self):
        """Delivering can transition to failed."""
        assert can_transition(MergeStatus.delivering, MergeStatus.failed)

    def test_can_transition_failed_to_queued(self):
        """Failed can transition back to queued for retry."""
        assert can_transition(MergeStatus.failed, MergeStatus.queued)

    def test_can_transition_done_to_queued(self):
        """Done can transition back to queued for re-run."""
        assert can_transition(MergeStatus.done, MergeStatus.queued)

    def test_transition_updates_merge_status(self, session):
        """The transition() helper updates InsertMerge.status and updated_at."""
        target = make_project(session)
        child = make_project(session, source_ref="s3://bucket/child_transition.mp4")
        run = make_run(session, child.id)
        session.flush()

        merge = InsertMerge(
            target_project_id=target.id,
            child_run_id=run.id,
            x=0, y=0, width=100, height=100,
            start_sec=0.0, end_sec=5.0,
        )
        session.add(merge)
        session.commit()

        # Transition from queued to compositing
        transition(merge, MergeStatus.compositing)
        session.commit()

        fetched = session.get(InsertMerge, merge.id)
        assert fetched.status == MergeStatus.compositing

    def test_invalid_transition_raises(self, session):
        """An invalid transition raises InvalidTransition."""
        target = make_project(session)
        child = make_project(session, source_ref="s3://bucket/child_invalid.mp4")
        run = make_run(session, child.id)
        session.flush()

        merge = InsertMerge(
            target_project_id=target.id,
            child_run_id=run.id,
            x=0, y=0, width=100, height=100,
            start_sec=0.0, end_sec=5.0,
        )
        session.add(merge)
        session.commit()

        # Try invalid transition: queued → done
        with pytest.raises(InvalidTransition):
            transition(merge, MergeStatus.done)


# ---------------------------------------------------------------------------
# Tests: source-fallback columns
# ---------------------------------------------------------------------------


class TestSourceFallbackColumns:
    """`RunSegment.source_fallback` + `Run.source_fallback_segments`.

    Marks a swap segment whose generation was impossible on the run's model
    (longer than its per-clip ceiling), so the stitch substitutes the ORIGINAL
    footage instead of blocking the run.

    The two defaults differ deliberately, and the difference is load-bearing:
    the segment flag is NOT NULL / false, so no pre-existing row can start
    claiming it delivered original footage; the run count is nullable, so NULL
    ("not recorded" — predates the column or never finalized) stays
    distinguishable from 0 ("finalized and verified fully swapped").
    """

    def test_segment_flag_defaults_to_false(self, session):
        project = make_project(session)
        sds = make_segment_defs(session, project.id)
        run = make_run(session, project.id)
        rs_list = make_run_segments(session, run.id, sds)
        session.commit()
        assert rs_list
        for rs in rs_list:
            session.refresh(rs)
            assert rs.source_fallback is False

    def test_segment_flag_round_trips(self, session):
        project = make_project(session)
        sds = make_segment_defs(session, project.id)
        run = make_run(session, project.id)
        rs = make_run_segments(session, run.id, sds)[0]
        rs.status = SegmentStatus.failed
        rs.source_fallback = True
        rs.error_message = (
            "22.0s > 15s Seedance 2.0 limit; original footage used instead"
        )
        session.commit()
        session.refresh(rs)
        assert rs.source_fallback is True
        # The status stays `failed` — the swap genuinely did not happen; the flag
        # is only what lets the completeness gate pass anyway.
        assert rs.status == SegmentStatus.failed

    def test_run_count_defaults_to_none_not_zero(self, session):
        project = make_project(session)
        run = make_run(session, project.id)
        session.commit()
        session.refresh(run)
        assert run.source_fallback_segments is None

    @pytest.mark.parametrize("value", [0, 1, 3])
    def test_run_count_round_trips(self, session, value):
        project = make_project(session)
        run = make_run(session, project.id)
        run.source_fallback_segments = value
        session.commit()
        session.refresh(run)
        assert run.source_fallback_segments == value


class TestChainedDerivedCreativeSources:
    """`RunTailEdit.source_type`/`source_caption_id` and
    `RunCaption.source_type`/`source_tail_edit_id` (docs/burned-captions.md
    §3.7).

    v1 hung both derived creatives off the Run's result only; these columns
    let a tail edit read a caption job's output (or vice versa) instead.
    `run_id` is always the parent Run regardless of source_type — it is
    provenance (project id, language suffix, sheet Idea, filename), never
    the input video — so it must stay populated and unaffected by these
    columns in every case.
    """

    def test_tail_edit_source_type_defaults_to_run(self, session):
        project = make_project(session)
        run = make_run(session, project.id)
        te = make_tail_edit(session, run.id)
        session.commit()
        session.refresh(te)
        assert te.source_type == "run"
        assert te.source_caption_id is None
        # run_id is the parent run's provenance, independent of source_type.
        assert te.run_id == run.id

    def test_tail_edit_source_caption_round_trips(self, session):
        project = make_project(session)
        run = make_run(session, project.id)
        caption = make_caption(session, run.id)
        te = make_tail_edit(
            session,
            run.id,
            source_type="caption",
            source_caption_id=caption.id,
        )
        session.commit()
        session.refresh(te)
        assert te.source_type == "caption"
        assert te.source_caption_id == caption.id
        # Chaining a caption as the input does not change whose run this is.
        assert te.run_id == run.id

    def test_caption_source_type_defaults_to_run(self, session):
        project = make_project(session)
        run = make_run(session, project.id)
        rc = make_caption(session, run.id)
        session.commit()
        session.refresh(rc)
        assert rc.source_type == "run"
        assert rc.source_tail_edit_id is None
        assert rc.run_id == run.id

    def test_caption_source_tail_edit_round_trips(self, session):
        project = make_project(session)
        run = make_run(session, project.id)
        te = make_tail_edit(session, run.id)
        rc = make_caption(
            session,
            run.id,
            source_type="tail_edit",
            source_tail_edit_id=te.id,
        )
        session.commit()
        session.refresh(rc)
        assert rc.source_type == "tail_edit"
        assert rc.source_tail_edit_id == te.id
        # Chaining a tail edit as the input does not change whose run this is.
        assert rc.run_id == run.id

    def test_no_foreign_key_on_source_ids(self, session):
        """Soft references only — same reasoning as OverlayJob.source_run_id:
        a source row can be deleted independently without breaking the
        derived creative that once pointed at it."""
        project = make_project(session)
        run = make_run(session, project.id)
        te = make_tail_edit(
            session,
            run.id,
            source_type="caption",
            source_caption_id="deadbeef-0000-0000-0000-000000000000",
        )
        rc = make_caption(
            session,
            run.id,
            source_type="tail_edit",
            source_tail_edit_id="deadbeef-0000-0000-0000-000000000001",
        )
        session.commit()
        session.refresh(te)
        session.refresh(rc)
        assert te.source_caption_id == "deadbeef-0000-0000-0000-000000000000"
        assert rc.source_tail_edit_id == "deadbeef-0000-0000-0000-000000000001"
