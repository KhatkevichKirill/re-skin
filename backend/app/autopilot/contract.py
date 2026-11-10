"""The flow plug-in contract (docs/autopilot-design.md §4).

A flow is plain Python. It never writes to the database and never calls an
endpoint: ``advance(ctx)`` returns exactly one :data:`Decision`, and the core
acts on it (core.py). Paid or state-changing work only starts through an
:class:`ActionHandler` (actions.py) that the core executes with the launch
protocol of §5.4.

Public names (stable for the flows B / C1 / C2): ``Flow``, ``StepSpec``,
``StepContext``, the decisions ``Complete`` / ``Wait`` / ``Launch`` /
``Compute`` / ``Judge`` / ``Ask`` / ``Fail``, ``Option``, ``ArtifactSpec``,
``BudgetLine``, the views, ``ActionHandler`` and ``registry.register``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Literal, Mapping, Optional, Protocol, Sequence, Union

ApprovalKind = Literal[
    "budget", "regeneration", "overrun", "listening_probe", "listening_final", "handoff",
]
APPROVAL_KINDS: tuple[str, ...] = (
    "budget", "regeneration", "overrun", "listening_probe", "listening_final", "handoff",
)
# Kinds the core creates itself; flows may Ask `regeneration` only WITH options.
CORE_APPROVAL_KINDS = ("budget", "overrun")

STEP_ID_RE = re.compile(r"^[a-z0-9_:.-]{1,96}$")
NAME_RE = re.compile(r"^[a-z0-9_:.-]{1,96}$")
OUTPUT_MAX_BYTES = 64 * 1024


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class ContractViolation(Exception):
    """A flow broke the contract: the autopilot fails, the reason is journaled."""


class FlowParamsError(ValueError):
    """``Flow.validate_params`` refusal (the API answers 422)."""


class ActionInvalid(ContractViolation):
    """A handler refused the params a flow asked for (a flow bug)."""


class ActionRefused(Exception):
    """An explicit admission refusal (409 run busy, 429 daily cap, 422 …).
    Nothing was admitted; the step is retried with backoff."""

    def __init__(self, status: int, detail: str):
        self.status = int(status)
        self.detail = str(detail)
        super().__init__(f"{status}: {detail}")


# ---------------------------------------------------------------------------
# Value objects
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Option:
    """One choice of an ``Ask``. With ``handler``/``params`` the option
    pre-authorises exactly that launch (it is validated and estimated by the
    core when the approval is created)."""

    id: str
    label: str
    handler: Optional[str] = None
    params: Optional[Mapping[str, Any]] = None
    # What choosing it does, in plain words (≤ 120 chars, AICP-376). Presentation
    # only: never part of the action, its params hash or the bound versions.
    consequence: str = ""


@dataclass(frozen=True)
class ArtifactSpec:
    """A file the operator is asked to watch / listen to (§2.5). The producer
    is a terminal run (``run_id``) or a succeeded compute effect of this
    autopilot (``effect_id``)."""

    path: str
    label: str
    kind: Literal["video", "audio", "image"]
    run_id: Optional[str] = None
    effect_id: Optional[str] = None


@dataclass(frozen=True)
class BudgetLine:
    label: str
    credits: float
    step_id: Optional[str] = None


# ---------------------------------------------------------------------------
# Decisions — exactly one per advance()
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Complete:
    output: Mapping[str, Any] = field(default_factory=dict)
    depends_on: Mapping[str, str] = field(default_factory=dict)
    reason: str = ""
    consumes: Sequence[str] = ()


@dataclass(frozen=True)
class Wait:
    reason: str
    recheck_sec: int = 30
    # Names of this step's PENDING flow approvals the flow knows are obsolete
    # (their result / QA round / run version moved on): the core supersedes
    # each one still pending under its row lock, journalled; answered ones are
    # never touched (2026-11-03, docs/segment-fill-batch-design.md §5).
    retire: Sequence[str] = ()


@dataclass(frozen=True)
class Launch:
    name: str
    handler: str
    params: Mapping[str, Any]
    reason: str
    regeneration: bool = False
    expect: Mapping[str, str] = field(default_factory=dict)
    consumes: Sequence[str] = ()


@dataclass(frozen=True)
class Compute:
    name: str
    func: str
    args: Mapping[str, Any]
    reason: str
    timeout_sec: int = 1800


@dataclass(frozen=True)
class Judge:
    name: str
    kind: str
    payload: Mapping[str, Any]
    reason: str


@dataclass(frozen=True)
class Ask:
    name: str
    kind: ApprovalKind
    question: str
    proposal: Mapping[str, Any]
    subject: str = ""
    options: Sequence[Option] = ()
    artifacts: Sequence[ArtifactSpec] = ()
    about: Mapping[str, str] = field(default_factory=dict)
    cost_credits: float = 0.0
    # Names of this step's earlier flow approvals the new one REPLACES: each
    # one still pending (unanswered) is superseded in the same transaction; an
    # answered one makes the core discard this Ask (the flow replans with the
    # answer). Answered approvals are never superseded this way.
    replaces: Sequence[str] = ()
    # As Wait.retire: obsolete pending approvals superseded with this Ask.
    retire: Sequence[str] = ()
    # Short plain-text title for the card, Telegram and the inbox (≤ 80 chars,
    # AICP-376); stored as proposal["_title"] (a reserved key: never set it in
    # ``proposal``). Presentation only, like Option.consequence.
    title: str = ""


@dataclass(frozen=True)
class Fail:
    reason: str
    consumes: Sequence[str] = ()


Decision = Union[Complete, Wait, Launch, Compute, Judge, Ask, Fail]


# ---------------------------------------------------------------------------
# Read-only views handed to advance()
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EffectView:
    id: str
    name: str
    kind: str
    handler: str
    state: str
    params: Mapping[str, Any]
    targets: Mapping[str, Any]
    created_refs: Mapping[str, Any]
    result: Optional[Mapping[str, Any]]
    error: Optional[str]
    created_at: datetime


@dataclass(frozen=True)
class ApprovalView:
    id: str
    name: str
    generation: int
    kind: str
    status: str  # pending | approved | rejected | superseded | consumed
    decision: Optional[str]  # approve | reject once answered (kept after consumption)
    chosen_option: Optional[str]
    comments: Sequence[Mapping[str, Any]]
    answer_comment: Optional[str]
    answered_at: Optional[datetime]

    @property
    def answered(self) -> bool:
        return self.decision is not None


@dataclass(frozen=True)
class RefusalView:
    name: str
    status: int
    detail: str
    count: int
    at: datetime


@dataclass(frozen=True)
class RunSegmentView:
    id: str
    index: int
    status: str
    local_result_path: Optional[str]
    cost_credits: Optional[float]
    take_count: int


@dataclass(frozen=True)
class RunView:
    id: str
    status: str
    is_terminal: bool
    result_local_path: Optional[str]
    result_gdrive_file_id: Optional[str]
    segments: Sequence[RunSegmentView]
    short_id: str
    updated_at: Optional[datetime]


@dataclass(frozen=True)
class BudgetView:
    approved_credits: Optional[float]
    committed_credits: float
    consumed_credits: float
    settled_credits: float
    tolerance: float

    @property
    def headroom_credits(self) -> Optional[float]:
        if self.approved_credits is None:
            return None
        return self.approved_credits * (1 + self.tolerance) - self.committed_credits


# ---------------------------------------------------------------------------
# Steps and flows
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StepSpec:
    id: str
    title: str
    advance: Callable[["StepContext"], Decision]
    needs: tuple[str, ...] = ()
    paid: bool = False
    approvals: tuple[str, ...] = ()
    estimate: Optional[Callable[["StepContext"], float]] = None

    def __post_init__(self):
        if not STEP_ID_RE.match(self.id):
            raise ValueError(f"invalid step id {self.id!r}")
        bad = [k for k in self.approvals if k not in APPROVAL_KINDS]
        if bad:
            raise ValueError(f"step {self.id}: unknown approval kinds {bad}")


class Flow(Protocol):
    id: str
    version: int
    label: str
    project_types: frozenset
    handlers: Mapping[str, int]

    def validate_params(self, project, params: dict) -> dict: ...

    def steps(self, project, params: dict) -> Sequence[StepSpec]: ...

    def budget_lines(self, ctx: "StepContext") -> Sequence[BudgetLine]: ...


# ---------------------------------------------------------------------------
# Action handlers
# ---------------------------------------------------------------------------


class ActionHandler:
    """Base class of an action handler (§4.4). Subclasses override what they need.

    ``execute`` contract: (a) every domain write in ``db`` in one outer
    transaction; (b) commit EXACTLY once, after every check, never on a refusal
    path; (c) no external paid call before that commit; (d) ids it creates go
    into ``effect.created_refs`` (assign a new dict) in that transaction;
    (e) refusals raise :class:`ActionRefused` (HTTPException 4xx is mapped by
    the core); (f) files written before the commit use fresh unique paths;
    (g) NEVER ``db.rollback()`` or ``db.close()`` inside ``execute``, and call
    only transaction-neutral services: the session already holds the core's
    flushed effect row, and a rollback would silently drop it (the admission
    would then commit without its idempotency key). Endpoints that roll back on
    a refusal path (e.g. ``start_fill``) need a transaction-neutral service
    first. The handler may set ``effect.result`` (assign a new dict) as well
    as ``effect.created_refs``; the core merges its return value into
    ``result`` afterwards.
    """

    name: str = ""
    version: int = 1
    paid: bool = False
    always_regeneration: bool = False

    def validate(self, db, project, params: Mapping[str, Any]) -> dict:
        return dict(params)

    def targets(self, db, project, params: Mapping[str, Any]) -> dict:
        """Existing rows the action acts on: ``{"runs": [run ids], ...}``."""
        return {}

    def target_run_ids(self, targets: Mapping[str, Any]) -> list[str]:
        return sorted(set(targets.get("runs", []) or []))

    def estimate(self, db, project, params: Mapping[str, Any]) -> float:
        return 0.0

    def spend_scopes(self, params: Mapping[str, Any], targets: Mapping[str, Any]) -> list[str]:
        """Spend scopes known before execute (registered in the admission tx)."""
        return [f"run:{rid}" for rid in self.target_run_ids(targets)]

    def execute(self, db, project, params: Mapping[str, Any], effect) -> dict:
        raise NotImplementedError

    def reconcile(self, db, effect) -> Literal["launched", "not_launched", "unknown"]:
        # Single-commit handler: the effect row only exists when the admission
        # commit landed.
        return "launched"


class StepContext(Protocol):
    """What ``advance`` / ``estimate`` see (implemented by core.Context)."""

    project: Any
    db: Any
    now: datetime
    params: dict
    step_id: str
    step_generation: int
    hold: bool
    budget: BudgetView

    def output(self, step_id: str) -> Optional[dict]: ...
    def effect(self, name: str) -> Optional[EffectView]: ...
    def effects(self, prefix: str = "") -> list[EffectView]: ...
    def next_round(self, prefix: str) -> str: ...
    def approval(self, name: str) -> Optional[ApprovalView]: ...
    def refusal(self, name: str) -> Optional[RefusalView]: ...
    def run(self, run_id: str) -> Optional[RunView]: ...
    def run_version(self, run_id: str) -> str: ...
    def segments_version(self) -> str: ...
    def transcript_version(self) -> str: ...
    def version(self, key: str) -> str: ...
    def previous_outputs(self) -> list[dict]: ...
    def fills(self, run_id: str) -> list[dict]: ...
    def artifact(self, path: str, *, label: str, kind: str,
                 run_id: Optional[str] = None, effect_id: Optional[str] = None) -> ArtifactSpec: ...
