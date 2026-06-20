"""
State machine for Job and Segment status transitions.
Defines valid status enums and enforces allowed transition paths.

v2 additions: ProjectStatus, RunStatus (added alongside v1 enums — no v1 changes).
"""

from __future__ import annotations

import enum
from datetime import datetime, timezone
from typing import Union


class JobStatus(str, enum.Enum):
    created = "created"
    analyzing = "analyzing"
    review = "review"
    queued = "queued"
    processing = "processing"
    stitching = "stitching"
    delivering = "delivering"
    done = "done"
    failed = "failed"


class SegmentStatus(str, enum.Enum):
    pending = "pending"
    uploading = "uploading"
    submitted = "submitted"
    generating = "generating"
    completed = "completed"
    failed = "failed"
    skipped = "skipped"


# --- v2 enums ---

class ProjectStatus(str, enum.Enum):
    created = "created"
    analyzing = "analyzing"
    ready = "ready"
    failed = "failed"


class RunStatus(str, enum.Enum):
    created = "created"
    queued = "queued"
    processing = "processing"
    stitching = "stitching"
    delivering = "delivering"
    done = "done"
    failed = "failed"


# Allowed transitions: {current_status -> set of valid next statuses}
JOB_TRANSITIONS: dict[JobStatus, set[JobStatus]] = {
    JobStatus.created:    {JobStatus.analyzing},
    JobStatus.analyzing:  {JobStatus.review, JobStatus.failed},
    JobStatus.review:     {JobStatus.queued, JobStatus.analyzing},
    JobStatus.queued:     {JobStatus.processing},
    JobStatus.processing: {JobStatus.stitching, JobStatus.failed},
    JobStatus.stitching:  {JobStatus.delivering, JobStatus.failed},
    JobStatus.delivering: {JobStatus.done, JobStatus.failed},
    JobStatus.done:       set(),
    JobStatus.failed:     {JobStatus.queued},
}

SEGMENT_TRANSITIONS: dict[SegmentStatus, set[SegmentStatus]] = {
    SegmentStatus.pending:    {SegmentStatus.uploading, SegmentStatus.skipped},
    SegmentStatus.uploading:  {SegmentStatus.submitted, SegmentStatus.failed},
    SegmentStatus.submitted:  {SegmentStatus.generating, SegmentStatus.failed},
    SegmentStatus.generating: {SegmentStatus.completed, SegmentStatus.failed},
    SegmentStatus.completed:  set(),
    SegmentStatus.failed:     {SegmentStatus.pending},
    SegmentStatus.skipped:    set(),
}

# v2 transition tables
PROJECT_TRANSITIONS: dict[ProjectStatus, set[ProjectStatus]] = {
    ProjectStatus.created:   {ProjectStatus.analyzing},
    ProjectStatus.analyzing: {ProjectStatus.ready, ProjectStatus.failed},
    ProjectStatus.ready:     {ProjectStatus.analyzing},
    ProjectStatus.failed:    {ProjectStatus.analyzing},
}

RUN_TRANSITIONS: dict[RunStatus, set[RunStatus]] = {
    RunStatus.created:    {RunStatus.queued},
    RunStatus.queued:     {RunStatus.processing},
    RunStatus.processing: {RunStatus.stitching, RunStatus.failed},
    RunStatus.stitching:  {RunStatus.delivering, RunStatus.failed},
    RunStatus.delivering: {RunStatus.done, RunStatus.failed},
    RunStatus.done:       set(),
    RunStatus.failed:     {RunStatus.queued},
}


class InvalidTransition(Exception):
    """Raised when a state transition is not permitted."""

    def __init__(self, current: enum.Enum, target: enum.Enum) -> None:
        self.current = current
        self.target = target
        super().__init__(
            f"Invalid transition: {current.value!r} -> {target.value!r}"
        )


def can_transition(
    current: Union[JobStatus, SegmentStatus, ProjectStatus, RunStatus],
    target: Union[JobStatus, SegmentStatus, ProjectStatus, RunStatus],
) -> bool:
    """Return True if transitioning from current to target is permitted."""
    if isinstance(current, JobStatus):
        return target in JOB_TRANSITIONS.get(current, set())
    if isinstance(current, SegmentStatus):
        return target in SEGMENT_TRANSITIONS.get(current, set())
    if isinstance(current, ProjectStatus):
        return target in PROJECT_TRANSITIONS.get(current, set())
    if isinstance(current, RunStatus):
        return target in RUN_TRANSITIONS.get(current, set())
    raise TypeError(f"Unsupported status type: {type(current)}")


def transition(
    entity: object,
    target: Union[JobStatus, SegmentStatus, ProjectStatus, RunStatus],
) -> None:
    """
    Apply a status transition to a Job, Segment, VideoProject, or Run ORM instance.

    Updates entity.status and entity.updated_at.
    Raises InvalidTransition if the move is not allowed.
    """
    current = entity.status  # type: ignore[attr-defined]
    if not can_transition(current, target):
        raise InvalidTransition(current, target)
    entity.status = target  # type: ignore[attr-defined]
    entity.updated_at = datetime.now(timezone.utc)  # type: ignore[attr-defined]
