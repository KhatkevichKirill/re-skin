"""
Tests for v2 state machine: ProjectStatus and RunStatus transitions.

Also confirms existing v1 Job/Segment transitions still work correctly.
"""

from __future__ import annotations

import pytest

from app.state_machine import (
    InvalidTransition,
    JobStatus,
    ProjectStatus,
    RunStatus,
    SegmentStatus,
    can_transition,
    transition,
)


# ---------------------------------------------------------------------------
# Minimal stub entity for transition() calls
# ---------------------------------------------------------------------------


class _Entity:
    """Minimal stub that transition() can mutate."""

    def __init__(self, status):
        self.status = status
        self.updated_at = None


# ---------------------------------------------------------------------------
# ProjectStatus transitions
# ---------------------------------------------------------------------------


class TestProjectStatusTransitions:
    # --- valid transitions ---

    def test_created_to_analyzing(self):
        assert can_transition(ProjectStatus.created, ProjectStatus.analyzing)

    def test_analyzing_to_ready(self):
        assert can_transition(ProjectStatus.analyzing, ProjectStatus.ready)

    def test_analyzing_to_failed(self):
        assert can_transition(ProjectStatus.analyzing, ProjectStatus.failed)

    def test_ready_to_analyzing(self):
        """Re-analyze is allowed from ready."""
        assert can_transition(ProjectStatus.ready, ProjectStatus.analyzing)

    def test_failed_to_analyzing(self):
        """Retry is allowed from failed."""
        assert can_transition(ProjectStatus.failed, ProjectStatus.analyzing)

    # --- invalid transitions ---

    def test_created_to_ready_invalid(self):
        assert not can_transition(ProjectStatus.created, ProjectStatus.ready)

    def test_created_to_failed_invalid(self):
        assert not can_transition(ProjectStatus.created, ProjectStatus.failed)

    def test_ready_to_failed_invalid(self):
        assert not can_transition(ProjectStatus.ready, ProjectStatus.failed)

    def test_ready_to_done_invalid(self):
        # done doesn't exist in ProjectStatus — test with a wrong type caught by can_transition
        # Use analyzing as target to make the test meaningful: ready → created is invalid
        assert not can_transition(ProjectStatus.ready, ProjectStatus.created)

    def test_failed_to_ready_invalid(self):
        assert not can_transition(ProjectStatus.failed, ProjectStatus.ready)

    # --- transition() raises InvalidTransition on bad moves ---

    def test_transition_raises_on_invalid(self):
        e = _Entity(ProjectStatus.created)
        with pytest.raises(InvalidTransition) as exc_info:
            transition(e, ProjectStatus.ready)
        assert exc_info.value.current == ProjectStatus.created
        assert exc_info.value.target == ProjectStatus.ready

    def test_transition_applies_valid_move(self):
        e = _Entity(ProjectStatus.created)
        transition(e, ProjectStatus.analyzing)
        assert e.status == ProjectStatus.analyzing
        assert e.updated_at is not None

    def test_full_happy_path_created_to_ready(self):
        e = _Entity(ProjectStatus.created)
        transition(e, ProjectStatus.analyzing)
        transition(e, ProjectStatus.ready)
        assert e.status == ProjectStatus.ready

    def test_re_analyze_from_ready(self):
        e = _Entity(ProjectStatus.ready)
        transition(e, ProjectStatus.analyzing)
        assert e.status == ProjectStatus.analyzing

    def test_retry_from_failed(self):
        e = _Entity(ProjectStatus.failed)
        transition(e, ProjectStatus.analyzing)
        assert e.status == ProjectStatus.analyzing


# ---------------------------------------------------------------------------
# RunStatus transitions
# ---------------------------------------------------------------------------


class TestRunStatusTransitions:
    # --- valid transitions ---

    def test_created_to_queued(self):
        assert can_transition(RunStatus.created, RunStatus.queued)

    def test_queued_to_processing(self):
        assert can_transition(RunStatus.queued, RunStatus.processing)

    def test_processing_to_stitching(self):
        assert can_transition(RunStatus.processing, RunStatus.stitching)

    def test_processing_to_failed(self):
        assert can_transition(RunStatus.processing, RunStatus.failed)

    def test_stitching_to_delivering(self):
        assert can_transition(RunStatus.stitching, RunStatus.delivering)

    def test_stitching_to_failed(self):
        assert can_transition(RunStatus.stitching, RunStatus.failed)

    def test_delivering_to_done(self):
        assert can_transition(RunStatus.delivering, RunStatus.done)

    def test_delivering_to_failed(self):
        assert can_transition(RunStatus.delivering, RunStatus.failed)

    def test_failed_to_queued_retry(self):
        assert can_transition(RunStatus.failed, RunStatus.queued)

    # --- invalid transitions ---

    def test_created_to_processing_invalid(self):
        assert not can_transition(RunStatus.created, RunStatus.processing)

    def test_done_to_anything_invalid(self):
        for status in RunStatus:
            assert not can_transition(RunStatus.done, status)

    def test_queued_to_done_invalid(self):
        assert not can_transition(RunStatus.queued, RunStatus.done)

    def test_failed_to_done_invalid(self):
        assert not can_transition(RunStatus.failed, RunStatus.done)

    # --- transition() behaviour ---

    def test_transition_raises_on_invalid(self):
        e = _Entity(RunStatus.created)
        with pytest.raises(InvalidTransition):
            transition(e, RunStatus.done)

    def test_transition_applies_valid_move(self):
        e = _Entity(RunStatus.created)
        transition(e, RunStatus.queued)
        assert e.status == RunStatus.queued
        assert e.updated_at is not None

    def test_full_happy_path_to_done(self):
        e = _Entity(RunStatus.created)
        for target in (
            RunStatus.queued,
            RunStatus.processing,
            RunStatus.stitching,
            RunStatus.delivering,
            RunStatus.done,
        ):
            transition(e, target)
        assert e.status == RunStatus.done

    def test_retry_flow(self):
        e = _Entity(RunStatus.processing)
        transition(e, RunStatus.failed)
        transition(e, RunStatus.queued)
        assert e.status == RunStatus.queued


# ---------------------------------------------------------------------------
# v1 regression — Job and Segment transitions still work
# ---------------------------------------------------------------------------


class TestV1JobTransitionsUnchanged:
    def test_created_to_analyzing(self):
        assert can_transition(JobStatus.created, JobStatus.analyzing)

    def test_analyzing_to_review(self):
        assert can_transition(JobStatus.analyzing, JobStatus.review)

    def test_review_to_queued(self):
        assert can_transition(JobStatus.review, JobStatus.queued)

    def test_review_to_analyzing(self):
        assert can_transition(JobStatus.review, JobStatus.analyzing)

    def test_queued_to_processing(self):
        assert can_transition(JobStatus.queued, JobStatus.processing)

    def test_processing_to_stitching(self):
        assert can_transition(JobStatus.processing, JobStatus.stitching)

    def test_stitching_to_delivering(self):
        assert can_transition(JobStatus.stitching, JobStatus.delivering)

    def test_delivering_to_done(self):
        assert can_transition(JobStatus.delivering, JobStatus.done)

    def test_failed_to_queued_retry(self):
        assert can_transition(JobStatus.failed, JobStatus.queued)

    def test_done_to_anything_invalid(self):
        for status in JobStatus:
            assert not can_transition(JobStatus.done, status)

    def test_created_to_done_invalid(self):
        assert not can_transition(JobStatus.created, JobStatus.done)

    def test_transition_applies_job(self):
        e = _Entity(JobStatus.created)
        transition(e, JobStatus.analyzing)
        assert e.status == JobStatus.analyzing


class TestV1SegmentTransitionsUnchanged:
    def test_pending_to_uploading(self):
        assert can_transition(SegmentStatus.pending, SegmentStatus.uploading)

    def test_pending_to_skipped(self):
        assert can_transition(SegmentStatus.pending, SegmentStatus.skipped)

    def test_uploading_to_submitted(self):
        assert can_transition(SegmentStatus.uploading, SegmentStatus.submitted)

    def test_submitted_to_generating(self):
        assert can_transition(SegmentStatus.submitted, SegmentStatus.generating)

    def test_generating_to_completed(self):
        assert can_transition(SegmentStatus.generating, SegmentStatus.completed)

    def test_failed_to_pending_retry(self):
        assert can_transition(SegmentStatus.failed, SegmentStatus.pending)

    def test_completed_to_anything_invalid(self):
        for status in SegmentStatus:
            assert not can_transition(SegmentStatus.completed, status)

    def test_skipped_to_anything_invalid(self):
        for status in SegmentStatus:
            assert not can_transition(SegmentStatus.skipped, status)

    def test_transition_applies_segment(self):
        e = _Entity(SegmentStatus.pending)
        transition(e, SegmentStatus.uploading)
        assert e.status == SegmentStatus.uploading


class TestTypeSafetyError:
    def test_unsupported_type_raises_type_error(self):
        with pytest.raises(TypeError, match="Unsupported status type"):
            can_transition("not_an_enum", "also_not")  # type: ignore[arg-type]
