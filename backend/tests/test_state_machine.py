"""
Tests for state_machine.py: valid transitions succeed, invalid ones raise InvalidTransition.
"""

import pytest
from unittest.mock import MagicMock
from datetime import datetime, timezone

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.state_machine import (
    JobStatus,
    SegmentStatus,
    InvalidTransition,
    can_transition,
    transition,
)


# ---------------------------------------------------------------------------
# can_transition — JobStatus
# ---------------------------------------------------------------------------

class TestJobCanTransition:
    def test_created_to_analyzing(self):
        assert can_transition(JobStatus.created, JobStatus.analyzing) is True

    def test_analyzing_to_review(self):
        assert can_transition(JobStatus.analyzing, JobStatus.review) is True

    def test_analyzing_to_failed(self):
        assert can_transition(JobStatus.analyzing, JobStatus.failed) is True

    def test_review_to_queued(self):
        assert can_transition(JobStatus.review, JobStatus.queued) is True

    def test_review_to_analyzing(self):
        assert can_transition(JobStatus.review, JobStatus.analyzing) is True

    def test_queued_to_processing(self):
        assert can_transition(JobStatus.queued, JobStatus.processing) is True

    def test_processing_to_stitching(self):
        assert can_transition(JobStatus.processing, JobStatus.stitching) is True

    def test_processing_to_failed(self):
        assert can_transition(JobStatus.processing, JobStatus.failed) is True

    def test_stitching_to_delivering(self):
        assert can_transition(JobStatus.stitching, JobStatus.delivering) is True

    def test_delivering_to_done(self):
        assert can_transition(JobStatus.delivering, JobStatus.done) is True

    def test_failed_to_queued_retry(self):
        assert can_transition(JobStatus.failed, JobStatus.queued) is True

    # Invalid transitions
    def test_created_to_done_invalid(self):
        assert can_transition(JobStatus.created, JobStatus.done) is False

    def test_done_to_failed_invalid(self):
        assert can_transition(JobStatus.done, JobStatus.failed) is False

    def test_review_to_done_invalid(self):
        assert can_transition(JobStatus.review, JobStatus.done) is False

    def test_queued_to_done_invalid(self):
        assert can_transition(JobStatus.queued, JobStatus.done) is False

    def test_created_to_stitching_invalid(self):
        assert can_transition(JobStatus.created, JobStatus.stitching) is False


# ---------------------------------------------------------------------------
# can_transition — SegmentStatus
# ---------------------------------------------------------------------------

class TestSegmentCanTransition:
    def test_pending_to_uploading(self):
        assert can_transition(SegmentStatus.pending, SegmentStatus.uploading) is True

    def test_pending_to_skipped(self):
        assert can_transition(SegmentStatus.pending, SegmentStatus.skipped) is True

    def test_uploading_to_submitted(self):
        assert can_transition(SegmentStatus.uploading, SegmentStatus.submitted) is True

    def test_uploading_to_failed(self):
        assert can_transition(SegmentStatus.uploading, SegmentStatus.failed) is True

    def test_submitted_to_generating(self):
        assert can_transition(SegmentStatus.submitted, SegmentStatus.generating) is True

    def test_generating_to_completed(self):
        assert can_transition(SegmentStatus.generating, SegmentStatus.completed) is True

    def test_failed_to_pending_retry(self):
        assert can_transition(SegmentStatus.failed, SegmentStatus.pending) is True

    # Invalid transitions
    def test_completed_to_pending_invalid(self):
        assert can_transition(SegmentStatus.completed, SegmentStatus.pending) is False

    def test_skipped_to_uploading_invalid(self):
        assert can_transition(SegmentStatus.skipped, SegmentStatus.uploading) is False

    def test_pending_to_completed_invalid(self):
        assert can_transition(SegmentStatus.pending, SegmentStatus.completed) is False


# ---------------------------------------------------------------------------
# transition() — applies status and updated_at, raises on illegal moves
# ---------------------------------------------------------------------------

def _make_entity(status):
    entity = MagicMock()
    entity.status = status
    entity.updated_at = None
    return entity


class TestTransitionFunction:
    def test_valid_job_transition_updates_status(self):
        entity = _make_entity(JobStatus.created)
        transition(entity, JobStatus.analyzing)
        assert entity.status == JobStatus.analyzing

    def test_valid_job_transition_updates_updated_at(self):
        entity = _make_entity(JobStatus.created)
        before = datetime.now(timezone.utc)
        transition(entity, JobStatus.analyzing)
        assert entity.updated_at >= before

    def test_valid_segment_transition_updates_status(self):
        entity = _make_entity(SegmentStatus.pending)
        transition(entity, SegmentStatus.uploading)
        assert entity.status == SegmentStatus.uploading

    def test_invalid_job_transition_raises(self):
        entity = _make_entity(JobStatus.created)
        with pytest.raises(InvalidTransition) as exc_info:
            transition(entity, JobStatus.done)
        assert "created" in str(exc_info.value)
        assert "done" in str(exc_info.value)

    def test_invalid_segment_transition_raises(self):
        entity = _make_entity(SegmentStatus.completed)
        with pytest.raises(InvalidTransition):
            transition(entity, SegmentStatus.pending)

    def test_done_is_terminal_raises(self):
        entity = _make_entity(JobStatus.done)
        with pytest.raises(InvalidTransition):
            transition(entity, JobStatus.failed)

    def test_skipped_is_terminal_raises(self):
        entity = _make_entity(SegmentStatus.skipped)
        with pytest.raises(InvalidTransition):
            transition(entity, SegmentStatus.uploading)

    def test_invalid_transition_error_message(self):
        entity = _make_entity(JobStatus.review)
        with pytest.raises(InvalidTransition) as exc_info:
            transition(entity, JobStatus.stitching)
        assert "review" in str(exc_info.value)
        assert "stitching" in str(exc_info.value)
