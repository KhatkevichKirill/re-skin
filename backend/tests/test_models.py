"""
Tests for ORM models using an in-memory SQLite database.

Verifies:
- Job and Segment creation + query-back
- Job.segments relationship + ordering
- Cascade delete (deleting Job removes Segments)
- JSON column round-trip
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Point DATABASE_URL to in-memory SQLite BEFORE importing db/models
os.environ["DATABASE_URL"] = "sqlite:///:memory:"

import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

# Import Base and models *after* setting env var so engine uses :memory:
from app.db import Base
from app.models import Job, Segment
from app.state_machine import JobStatus, SegmentStatus


@pytest.fixture(scope="module")
def engine():
    eng = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
    )

    @event.listens_for(eng, "connect")
    def set_pragmas(dbapi_conn, _):
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    Base.metadata.create_all(eng)
    yield eng
    Base.metadata.drop_all(eng)


@pytest.fixture()
def session(engine):
    SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    sess = SessionLocal()
    yield sess
    sess.rollback()
    sess.close()


def _make_job(**kwargs) -> Job:
    defaults = dict(
        source_type="upload",
        source_ref="video.mp4",
        default_prompt="Replace the character",
        default_reference_image_urls=["https://example.com/ref1.jpg", "https://example.com/ref2.jpg"],
        resolution="720p",
        status=JobStatus.created,
    )
    defaults.update(kwargs)
    return Job(**defaults)


def _make_segment(job_id: str, index: int, **kwargs) -> Segment:
    defaults = dict(
        job_id=job_id,
        index=index,
        start_sec=index * 5.0,
        end_sec=index * 5.0 + 5.0,
        has_face=True,
        action="swap",
        status=SegmentStatus.pending,
    )
    defaults.update(kwargs)
    return Segment(**defaults)


class TestJobCreation:
    def test_create_and_query_job(self, session):
        job = _make_job()
        session.add(job)
        session.commit()

        fetched = session.get(Job, job.id)
        assert fetched is not None
        assert fetched.source_type == "upload"
        assert fetched.source_ref == "video.mp4"
        assert fetched.resolution == "720p"
        assert fetched.status == JobStatus.created

    def test_job_id_is_uuid_string(self, session):
        job = _make_job()
        session.add(job)
        session.commit()
        assert isinstance(job.id, str)
        assert len(job.id) == 36  # UUID4 with dashes

    def test_created_at_populated(self, session):
        job = _make_job()
        session.add(job)
        session.commit()
        assert job.created_at is not None

    def test_nullable_probe_fields_default_none(self, session):
        job = _make_job()
        session.add(job)
        session.commit()
        assert job.duration_sec is None
        assert job.width is None
        assert job.height is None
        assert job.fps is None
        assert job.aspect_ratio is None

    def test_probe_fields_can_be_set(self, session):
        job = _make_job(
            duration_sec=120.5,
            width=1920,
            height=1080,
            fps=29.97,
            aspect_ratio="16:9",
        )
        session.add(job)
        session.commit()

        fetched = session.get(Job, job.id)
        assert fetched.duration_sec == pytest.approx(120.5)
        assert fetched.width == 1920
        assert fetched.fps == pytest.approx(29.97)
        assert fetched.aspect_ratio == "16:9"


class TestJsonColumnRoundTrip:
    def test_default_reference_image_urls_roundtrip(self, session):
        urls = ["https://cdn.example.com/a.jpg", "https://cdn.example.com/b.png"]
        job = _make_job(default_reference_image_urls=urls)
        session.add(job)
        session.commit()
        session.expire(job)

        fetched = session.get(Job, job.id)
        assert fetched.default_reference_image_urls == urls

    def test_segment_reference_image_urls_override_roundtrip(self, session):
        job = _make_job()
        session.add(job)
        session.flush()

        overrides = ["https://example.com/override1.jpg"]
        seg = _make_segment(
            job_id=job.id,
            index=0,
            reference_image_urls_override=overrides,
        )
        session.add(seg)
        session.commit()
        session.expire(seg)

        fetched = session.get(Segment, seg.id)
        assert fetched.reference_image_urls_override == overrides

    def test_json_list_empty(self, session):
        job = _make_job(default_reference_image_urls=[])
        session.add(job)
        session.commit()
        session.expire(job)

        fetched = session.get(Job, job.id)
        assert fetched.default_reference_image_urls == []


class TestSegmentsRelationship:
    def test_job_with_three_segments(self, session):
        job = _make_job()
        session.add(job)
        session.flush()

        segments = [_make_segment(job.id, i) for i in range(3)]
        session.add_all(segments)
        session.commit()

        fetched = session.get(Job, job.id)
        assert len(fetched.segments) == 3

    def test_segments_ordered_by_index(self, session):
        job = _make_job()
        session.add(job)
        session.flush()

        # Add in reverse order to test ordering
        segs = [_make_segment(job.id, i) for i in [2, 0, 1]]
        session.add_all(segs)
        session.commit()

        fetched = session.get(Job, job.id)
        indices = [s.index for s in fetched.segments]
        assert indices == sorted(indices)

    def test_segment_back_ref_to_job(self, session):
        job = _make_job()
        session.add(job)
        session.flush()

        seg = _make_segment(job.id, 0)
        session.add(seg)
        session.commit()

        fetched_seg = session.get(Segment, seg.id)
        assert fetched_seg.job_id == job.id
        assert fetched_seg.job.source_ref == "video.mp4"

    def test_segment_fields(self, session):
        job = _make_job()
        session.add(job)
        session.flush()

        seg = _make_segment(
            job.id,
            index=0,
            start_sec=1.5,
            end_sec=6.5,
            has_face=True,
            action="swap",
            pre_roll_sec=0.5,
            post_roll_sec=0.25,
            prompt_override="Swap with warrior",
        )
        session.add(seg)
        session.commit()
        session.expire(seg)

        fetched = session.get(Segment, seg.id)
        assert fetched.start_sec == pytest.approx(1.5)
        assert fetched.end_sec == pytest.approx(6.5)
        assert fetched.has_face is True
        assert fetched.action == "swap"
        assert fetched.pre_roll_sec == pytest.approx(0.5)
        assert fetched.post_roll_sec == pytest.approx(0.25)
        assert fetched.prompt_override == "Swap with warrior"


class TestCascadeDelete:
    def test_deleting_job_deletes_segments(self, session):
        job = _make_job()
        session.add(job)
        session.flush()

        segments = [_make_segment(job.id, i) for i in range(3)]
        session.add_all(segments)
        session.commit()
        seg_ids = [s.id for s in segments]

        session.delete(job)
        session.commit()

        # Job should be gone
        assert session.get(Job, job.id) is None

        # All segments should be gone too
        for sid in seg_ids:
            assert session.get(Segment, sid) is None
