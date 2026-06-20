"""
Tests for v2 ORM models: VideoProject, SegmentDef, Run, RunSegment.

Uses an in-memory SQLite database — no external dependencies required.
"""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker

from app.db import Base
from app.models import Run, RunSegment, SegmentDef, VideoProject
from app.state_machine import ProjectStatus, RunStatus, SegmentStatus

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

    def test_run_project_back_populates(self, session):
        proj = make_project(session)
        run = make_run(session, proj.id)
        session.commit()
        fetched_run = session.get(Run, run.id)
        assert fetched_run.project.id == proj.id


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


class TestTableCreation:
    """Smoke test — verify all v2 tables exist after Base.metadata.create_all."""

    def test_v2_tables_present(self, engine):
        insp = inspect(engine)
        tables = set(insp.get_table_names())
        assert "video_projects" in tables
        assert "segment_defs" in tables
        assert "runs" in tables
        assert "run_segments" in tables

    def test_v1_tables_still_present(self, engine):
        insp = inspect(engine)
        tables = set(insp.get_table_names())
        assert "jobs" in tables
        assert "segments" in tables
