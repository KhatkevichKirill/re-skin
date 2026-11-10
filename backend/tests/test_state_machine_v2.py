"""
Tests for the state machine: ProjectStatus, RunStatus, and SegmentStatus transitions.
"""

from __future__ import annotations

import pytest

from app.state_machine import (
    CaptionStatus,
    InvalidTransition,
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

    def test_processing_to_done(self):
        # No-stitch runs (stitch_plan == []) finish straight from processing.
        assert can_transition(RunStatus.processing, RunStatus.done)

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

    def test_done_to_queued_allowed_for_segment_rerun(self):
        """done → queued must be allowed so a completed run can be re-queued after a per-segment edit."""
        assert can_transition(RunStatus.done, RunStatus.queued)

    def test_done_to_non_queued_invalid(self):
        """done can only go to queued — all other targets remain invalid."""
        for status in RunStatus:
            if status != RunStatus.queued:
                assert not can_transition(RunStatus.done, status), (
                    f"Expected done → {status.value} to be invalid"
                )

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
# SegmentStatus transitions (used by RunSegment)
# ---------------------------------------------------------------------------


class TestSegmentTransitions:
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


# ---------------------------------------------------------------------------
# CaptionStatus transitions (used by RunCaption — "Add Captions")
# ---------------------------------------------------------------------------


class TestCaptionStatusTransitions:
    # --- valid transitions ---

    def test_queued_to_transcribing(self):
        assert can_transition(CaptionStatus.queued, CaptionStatus.transcribing)

    def test_queued_to_burning(self):
        """Manual lines (lines_source == "manual") skip the transcribe step."""
        assert can_transition(CaptionStatus.queued, CaptionStatus.burning)

    def test_queued_to_failed(self):
        assert can_transition(CaptionStatus.queued, CaptionStatus.failed)

    def test_transcribing_to_burning(self):
        assert can_transition(CaptionStatus.transcribing, CaptionStatus.burning)

    def test_transcribing_to_failed(self):
        assert can_transition(CaptionStatus.transcribing, CaptionStatus.failed)

    def test_burning_to_delivering(self):
        assert can_transition(CaptionStatus.burning, CaptionStatus.delivering)

    def test_burning_to_failed(self):
        assert can_transition(CaptionStatus.burning, CaptionStatus.failed)

    def test_delivering_to_done(self):
        assert can_transition(CaptionStatus.delivering, CaptionStatus.done)

    def test_delivering_to_failed(self):
        assert can_transition(CaptionStatus.delivering, CaptionStatus.failed)

    def test_done_to_queued_rerun(self):
        assert can_transition(CaptionStatus.done, CaptionStatus.queued)

    def test_failed_to_queued_retry(self):
        assert can_transition(CaptionStatus.failed, CaptionStatus.queued)

    # --- invalid transitions ---

    def test_queued_to_done_invalid(self):
        assert not can_transition(CaptionStatus.queued, CaptionStatus.done)

    def test_queued_to_delivering_invalid(self):
        assert not can_transition(CaptionStatus.queued, CaptionStatus.delivering)

    def test_transcribing_to_delivering_invalid(self):
        assert not can_transition(CaptionStatus.transcribing, CaptionStatus.delivering)

    def test_transcribing_to_done_invalid(self):
        assert not can_transition(CaptionStatus.transcribing, CaptionStatus.done)

    def test_delivering_to_transcribing_invalid(self):
        assert not can_transition(CaptionStatus.delivering, CaptionStatus.transcribing)

    def test_delivering_to_burning_invalid(self):
        assert not can_transition(CaptionStatus.delivering, CaptionStatus.burning)

    def test_done_to_non_queued_invalid(self):
        """done can only go to queued — all other targets remain invalid."""
        for status in CaptionStatus:
            if status != CaptionStatus.queued:
                assert not can_transition(CaptionStatus.done, status), (
                    f"Expected done -> {status.value} to be invalid"
                )

    def test_failed_to_non_queued_invalid(self):
        """failed can only go to queued — all other targets remain invalid."""
        for status in CaptionStatus:
            if status != CaptionStatus.queued:
                assert not can_transition(CaptionStatus.failed, status), (
                    f"Expected failed -> {status.value} to be invalid"
                )

    # --- transition() behaviour ---

    def test_transition_raises_on_invalid(self):
        e = _Entity(CaptionStatus.queued)
        with pytest.raises(InvalidTransition) as exc_info:
            transition(e, CaptionStatus.done)
        assert exc_info.value.current == CaptionStatus.queued
        assert exc_info.value.target == CaptionStatus.done

    def test_transition_applies_valid_move(self):
        e = _Entity(CaptionStatus.queued)
        transition(e, CaptionStatus.transcribing)
        assert e.status == CaptionStatus.transcribing
        assert e.updated_at is not None

    def test_full_happy_path_transcribe_then_burn(self):
        e = _Entity(CaptionStatus.queued)
        for target in (
            CaptionStatus.transcribing,
            CaptionStatus.burning,
            CaptionStatus.delivering,
            CaptionStatus.done,
        ):
            transition(e, target)
        assert e.status == CaptionStatus.done

    def test_full_happy_path_manual_skips_transcribing(self):
        """lines_source == "manual" goes straight from queued to burning."""
        e = _Entity(CaptionStatus.queued)
        for target in (
            CaptionStatus.burning,
            CaptionStatus.delivering,
            CaptionStatus.done,
        ):
            transition(e, target)
        assert e.status == CaptionStatus.done

    def test_retry_flow(self):
        e = _Entity(CaptionStatus.burning)
        transition(e, CaptionStatus.failed)
        transition(e, CaptionStatus.queued)
        assert e.status == CaptionStatus.queued

    def test_rerun_flow_from_done(self):
        e = _Entity(CaptionStatus.done)
        transition(e, CaptionStatus.queued)
        assert e.status == CaptionStatus.queued
