# Autopilot core — "by Assistant" project mode (AICP-363)

Tracking: AICP-363 (core), parent
AICP-362. Flows that plug into this core:
AICP-364 (subtitle removal, "B"), AICP-365 (localisation generation, "C1"), AICP-366
(localisation QA/finishing, "C2"). UX design is AICP-367 ("D", later).

Status: **rev 7, frozen for phase 2** (2026-10-31; rev 7 = rev 6 + the operator's answers to Q1–Q8, §13). Revisions 1–5 went through a §2 design discussion with GPT-6.1
Terra (`codex exec`, reasoning effort high); §14 records every finding, its resolution, and what
was rejected and why.

## 1. Goal, non-goals, guarantees

A project is created in mode `manual` (today, unchanged) or `assistant`. In `assistant` mode a
server-side **autopilot** runs the project's **flow** (the steps written down in
`docs/assistant-flow-*.md`) and stops only for operator decisions:

* a **budget** approval of the predicted cost before the first paid launch;
* every **regeneration**, and any **overrun** above the tolerance (default 10 %, per project);
* **listening** approvals (probe segment 1 and the final, per language) and a **hand-off** at the end.

Non-goals of the core: the flows themselves (B, C1, C2), the UX (D), the AICP-344 deploy gate and
durable dispatch, a new billed LLM path (§8).

What the core guarantees, precisely:

1. **The autopilot admits each action at most once.** A launch is one database transaction that
   contains both a row with a UNIQUE idempotency key and the admission's domain writes (§5.4).
   Concurrent ticks, crashes, restarts and deploys cannot produce a second admission of the same
   action. When the outcome of a commit is uncertain, the next attempt collides with the key and is
   reconciled instead of repeated.
2. **Paid work only goes through the existing admission paths** (the functions the API endpoints
   run: locks, 409 gates, daily spend cap, reservations); workers, the recovery reaper and Drive
   delivery then handle it exactly as for an operator click. The autopilot never transitions a run,
   never holds a run's Redis lock and never hand-drives a pipeline.
3. **Not claimed:** exactly-once *provider submission* after an admission. That is a property of
   the run pipeline (per-run Redis lock + reaper), which has a known pre-existing race between a
   handler's enqueue and the reaper's liveness snapshot (§5.6). The autopilot does not make one
   admission more exposed to it than an operator click; it does make admissions more frequent.
   The fix belongs to a durable dispatch generation, filed as AICP-360.
4. **The project's own automatic paid call is gated:** for `assistant` projects the transcription
   that analysis starts by itself for localisation projects is not started (§6.5); the flow
   launches it inside the budget. The project budget then governs **autopilot admissions**; manual
   operator actions on the project are not blocked by it, but a manual **paid** action pauses the
   autopilot (§5.7, Q5); their spend is counted (best effort, §6.1) and a later autopilot launch
   asks for an overrun when needed.

## 2. Data model

New migration `r1a7c3e5b9d2_autopilot_core` chained onto the current single head `q8b4d0f2a716`
(prod is at that head, checked 2026-10-31). Status columns are `VARCHAR` + `CHECK`, not Postgres
enums: a new state is a transactional, reversible `CHECK` swap in a later migration (not an
irreversible `ALTER TYPE … ADD VALUE`), and new states are only emitted by code deployed after it.

### 2.1 `video_projects.mode`

`VARCHAR(16) NOT NULL DEFAULT 'manual'`, `CHECK (mode IN ('manual','assistant'))`. Set only at
creation (`POST /projects`, `POST /runs/{rid}/to-project`); never changed by `PATCH`. The server
default keeps any old-image writer on `manual`. It is read by project creation, the transcription
hand-off after analysis (§6.5), the UI and the autopilot — nothing else in the pipeline.

`assistant` is refused (422) when `AUTOPILOT_ENABLED` is false or no flow is registered for the
project type. At merge of the core no production flow is registered (the stub flow is test-only)
and `AUTOPILOT_ENABLED` stays false: **merging and deploying the core changes no behaviour** (§9,
staged rollout).

project-from-run offers the mode; default = the parent project's mode if the target type has a
registered flow, else `manual`.

### 2.2 `project_autopilots` — one record per autopilot attempt on a project

| column | type | notes |
|---|---|---|
| `id` | uuid | |
| `project_id` | FK `video_projects.id` ON DELETE CASCADE | |
| `flow_id`, `flow_version` | varchar, int | frozen at creation (§4.7) |
| `status` | varchar | `active` · `paused` · `completed` · `stopped` · `failed` |
| `status_reason` | text | why paused/stopped/failed |
| `params` | JSON | flow parameters (languages, tails, …); editable only while no paid effect exists |
| `approved_budget_mcredits` | bigint NULL | NULL = no budget approved yet |
| `tolerance` | float NOT NULL DEFAULT 0.10 | `CHECK (tolerance >= 0 AND tolerance <= 0.5)` |
| `version` | int NOT NULL | bumped by **every** write to core-owned state — the autopilot, its steps, approvals, effects, spend-scope registrations (tick, reconciliation, compute/judgement results, operator answer, pause, PATCH); the compare-and-set token of §5.3. All core writes go through one helper that bumps it |
| `lease_token`, `lease_expires_at` | varchar, timestamp | per-autopilot tick lease (work avoidance only) |
| `outstanding` | int NOT NULL DEFAULT 0 | effects not yet terminal; a finished autopilot keeps reconciling while > 0 |
| `created_at`, `updated_at`, `finished_at` | timestamps | |

Partial unique index `uq_project_autopilots_live (project_id) WHERE status IN ('active','paused')`
(+ `sqlite_where`).

### 2.3 `autopilot_steps`

`(id, autopilot_id FK CASCADE, step_id varchar(96), generation int NOT NULL DEFAULT 1, status,
output JSON, depends_on JSON, previous_outputs JSON, reason text, next_check_at, refusals int,
errors int, started_at, completed_at)`, unique `(autopilot_id, step_id)`. Status `pending` →
`active` → `completed` | `failed` | `skipped`; `completed` → `active` again only by invalidation
(§4.6), which increments `generation`. The generation is part of every effect key and approval
identity of the step, so a reopened step starts with a clean namespace; earlier generations stay
for audit.

### 2.4 `autopilot_effects`

| column | notes |
|---|---|
| `id` | uuid (also the judgement invocation id) |
| `autopilot_id`, `step_id`, `step_generation`, `name` | `name` is step-chosen and deterministic (`launch`, `rerun:r2`) |
| `idempotency_key` | `"{autopilot_id}:{step_id}:g{step_generation}:{name}"`, **UNIQUE** (`uq_autopilot_effects_key`) |
| `kind` | `action` · `compute` · `judgement` |
| `handler`, `handler_version` | handler / compute function / judgement kind and its version (§4.7) |
| `params`, `params_sha256` | the normalised request; the sha binds approvals |
| `paid`, `estimate_mcredits` | from the handler, never from the flow (§4.4) |
| `targets` | rows the action acts on, known before execute (`{"run": rid, "run_segments": […]}`) |
| `created_refs` | ids the admission created, written by the handler **inside** the admission transaction |
| `state` | action: `launching` → `launched` / `not_launched` / `unknown`; compute: `queued` → `running` → `succeeded` / `failed`; judgement: `prepared` (never sent) → `sending` (a send may have happened) → `submitted` → `succeeded` / `failed` / `unknown` |
| `target_versions` | the version of every target/`expect` row, computed under the APPLY locks (§5.4) |
| `result`, `error` | |
| `approval_id` | the approval this launch consumed (restored if `not_launched`) |
| `claim_token`, `attempts`, `rq_job_id` | compute recovery (§5.5) |
| `created_at`, `updated_at` | |

### 2.5 `autopilot_approvals`

`(id, autopilot_id, step_id, step_generation, name, generation, kind, subject, status, question,
proposal JSON, options JSON, chosen_option, artifacts JSON, action_sha256, handler_version,
bound_versions JSON, cost_mcredits, committed_at_ask_mcredits, budget_after_mcredits, comments
JSON, answer_comment, answered_at, answered_by, consumed_by_effect_id, consumed_at, created_at)`.

* `kind`: `budget` · `regeneration` · `overrun` · `listening_probe` · `listening_final` · `handoff`.
* `status`: `pending` → `approved` / `rejected` / `superseded`; `approved` / `rejected` →
  `consumed` when a decision acts on the answer (§4.3 `consumes`): exactly one launch, completion or
  failure uses an answer, and only after it was revalidated (§5.4).
* `generation`: 1 for the first ask of a `(step generation, name)`, +1 each time the previous one
  was superseded. Unique `dedupe_key =
  "{autopilot_id}:{step_id}:g{step_generation}:{name}:{generation}"`; a re-ask of the same question
  while the current generation is pending is a no-op.
* `bound_versions`: the versions of the rows the question is about (the reviewed run, the action's
  targets) at ask time, plus for action-bound approvals the locked estimate. An answer is acted on
  only if they still hold (§5.4); otherwise the approval is `superseded` and asked again.
* `subject`: e.g. the language `"es"` or `"segment 4"`.
* `options` (optional): `[{id, label, cost_mcredits, action: {handler, handler_version, params,
  params_sha256} | null}]`, validated and estimated by the core at ask time. Approving requires
  `chosen_option`; an option with an action pre-authorises **exactly** that launch (it is the
  regeneration/overrun approval for that sha + handler version).
* `artifacts` (optional): what the operator watches or listens to. Each artifact belongs to a
  producer: a **terminal** run (`done` / `failed` / `incomplete`: its final or a segment result),
  or a **succeeded** compute effect of this autopilot (a QA sheet in its winning attempt dir,
  bound by `effect:<id>`). At ask time, inside APPLY with that run row locked (no rerun, restore or
  re-delivery can start while it is held, so the file is not being written), the core verifies the
  run's version and **copies** the file into a content-addressed store
  `data/projects/{pid}/autopilot/artifacts/{sha256}.{ext}` (streamed, never loaded whole), and
  records `{label, kind, producer, source_path, sha256, size}` with the producer's version in
  `bound_versions`. The card serves only those immutable copies. An answer is acted on only while
  the producer version still matches **and** the source file's bytes still hash to the recorded
  sha256 (streamed re-hash under the run lock at consumption); otherwise the approval is
  superseded — the operator never accepts a different result than the one they were shown. A source
  that is missing or whose producer is not terminal makes the `Ask` wait (retryable). Copies are
  removed 7 days after the autopilot finishes (disk cleanup sweep).
* `action_sha256`, `handler_version`: for regeneration/overrun approvals.

### 2.6 `project_spend_scopes`

`(project_id, scope_key, registered_at, registered_by_autopilot_id, source)`, primary key
`(project_id, scope_key)`; `source` = `admission` | `scan`. **Project-owned** and append-only
(§6.1), so a restarted autopilot inherits every scope an earlier attempt saw, including those of
rows deleted since. `project_id` is a plain indexed column (no FK): the registry is history.
Deleting the project deletes its rows explicitly.

### 2.7 `autopilot_journal` — append-only

`(id bigserial, autopilot_id, at, step_id NULL, kind, message, data JSON, cost_mcredits NULL,
effect_id NULL, approval_id NULL)`. Kinds: `created`, `step_started`, `decision`, `launch`,
`refused`, `observed`, `reopened`, `approval_requested`, `approval_answered`, `comment`, `budget`,
`warning`, `completed`, `failed`, `paused`, `resumed`, `stopped`. `message` = the human "what and
why". A `Wait` journals only when its reason changes. Index `(autopilot_id, id DESC)`.

### 2.8 `autopilot_gate` — the deploy hold (§9)

Singleton row `id = 1`, created by the migration: `(id, held bool, generation int, held_since,
held_by, note)`. Every admitting transaction takes it `FOR SHARE` first (§5.2); `hold on` flips it
with an `UPDATE`, which waits for every such transaction to end.

## 3. Lifecycle

```
create (mode=assistant) ──► active ──tick──► … steps … ──► completed
                              │  ▲                 │
                     pause    ▼  │ resume          ├── rejection not handled / budget rejected ► stopped
                            paused                 ├── operator Stop ─────────────────────────► stopped
                                                   └── step Fail / contract violation ───────► failed
```

* `active`: the tick advances ready steps. "Waiting for approval" is a step with a pending
  approval, not a status.
* `paused` (operator, or the core on a stall/ambiguity): the tick only **reconciles** effects
  (observes; never submits, launches or asks).
* `stopped` / `completed` / `failed`: nothing new is started; in-flight pipeline work is not
  cancelled (the pipeline has no run cancel). Outstanding effects keep being reconciled.
* The autopilot row is created in the same transaction as an assistant project; its first steps
  `Wait` until the project is `ready`. The core pauses the autopilot when the project is `failed`
  ("analysis failed: re-analyze, then Resume") or has sat in `created`/`analyzing` longer than
  `AUTOPILOT_STALL_SEC` (a lost analysis enqueue: recovery covers only `analyzing`).

## 4. The flow plug-in contract

Module `app/autopilot/contract.py`. A flow is plain Python; it never writes and never calls an
endpoint. It **decides**; the core **acts**.

### 4.1 Flow and steps

```python
ApprovalKind = Literal["budget", "regeneration", "overrun",
                       "listening_probe", "listening_final", "handoff"]

@dataclass(frozen=True)
class StepSpec:
    id: str                         # [a-z0-9_:.-]{1,96}; may carry a subject: "probe:es"
    title: str
    needs: tuple[str, ...] = ()     # step ids that must be `completed` first (a DAG)
    paid: bool = False              # may this step Launch a paid handler at all?
    approvals: tuple[ApprovalKind, ...] = ()   # kinds this step may Ask
    estimate: Callable[["StepContext"], float] | None = None   # credits still to spend (pure)
    advance: Callable[["StepContext"], "Decision"]

class Flow(Protocol):
    id: str                         # "subtitle_removal"
    version: int                    # bump on any change of step ids/semantics (§4.7)
    label: str
    project_types: frozenset[str]
    handlers: Mapping[str, int]     # handler/compute/judgement name -> version this flow needs
    def validate_params(self, project, params: dict) -> dict: ...   # or FlowParamsError -> 422
    def steps(self, project, params: dict) -> Sequence[StepSpec]: ...   # deterministic; may expand per language
    def budget_lines(self, ctx: "StepContext") -> Sequence["BudgetLine"]: ...   # optional extras

def register(flow: Flow) -> None    # app/autopilot/registry.py
```

A step is **ready** when all its `needs` are `completed` and it is not terminal. Steps whose ids
vanish from `steps()` (params changed before any paid launch) become `skipped`.

### 4.2 StepContext

```python
class StepContext:
    project: VideoProject            # loaded in the read-only planning session (§4.5)
    db: Session                      # read-only, DB-enforced (§4.5)
    now: datetime
    params: dict
    step_id: str
    hold: bool                       # gate held (§9): Launch/Compute/Judge/Ask will be refused
    budget: BudgetView               # approved, committed, consumed, settled, tolerance, headroom
    def output(self, step_id: str) -> dict | None
    def effect(self, name: str) -> EffectView | None
    def effects(self, prefix: str = "") -> list[EffectView]
    def next_round(self, prefix: str) -> str               # first unused "rerun:rN"
    def approval(self, name: str) -> ApprovalView | None   # current generation
    def refusal(self, name: str) -> RefusalView | None
    def run(self, run_id: str) -> RunView | None
    def run_version(self, run_id: str) -> str              # §4.6
    def artifact(self, path: str, *, label: str, kind: str) -> ArtifactSpec
```

`EffectView` = `(name, kind, handler, state, targets, created_refs, result, error, created_at)`.
`ApprovalView` = `(name, generation, kind, status, chosen_option, comments, answer_comment,
answered_at)`. `RunView` = status, `is_terminal` (`done`/`failed`/`incomplete`), RunSegments
(status, index, result path, take count, cost), `result_local_path`, `result_gdrive_file_id`.

### 4.3 Decisions — exactly one per `advance`

```python
Complete(output: dict = {}, depends_on: Mapping[str, str] = {}, reason: str = "",
         consumes: Sequence[str] = ())
Wait(reason: str, recheck_sec: int = 30)
Launch(name: str, handler: str, params: dict, reason: str, *,
       regeneration: bool = False, expect: Mapping[str, str] = {}, consumes: Sequence[str] = ())
Compute(name: str, func: str, args: dict, reason: str, timeout_sec: int = 1800)
Judge(name: str, kind: str, payload: dict, reason: str)
Ask(name: str, kind: ApprovalKind, question: str, proposal: dict, *, subject: str = "",
    options: Sequence[Option] = (), artifacts: Sequence[ArtifactSpec] = (),
    about: Mapping[str, str] = {}, cost_credits: float = 0.0)
Fail(reason: str, consumes: Sequence[str] = ())
```

* **Version keys.** `depends_on`, `expect` and `about` map a version key to the version the step
  saw. Only six kinds of key exist, each with a lock that fences every writer of it:

  | key | version | fenced by |
  |---|---|---|
  | `run:<id>` | `ctx.run_version(id)` (§4.6) | the run row lock, **for a terminal run**. Result changes already lock it first (rerun, restore, retry, delete — `_lock_run_for_result_mutation`; range-attempt, audio-patch and finalize promotion, fill, assistant publication — `refresh(run, with_for_update=True)`). Generation **inputs** of a terminal run do not today: phase 2 makes `patch_run_segment` (api_v2.py:5305), `apply_segment_image_job` (:6828), `clear_segment_generated_image` (:6877) and the segment-image worker's apply (pipeline_segment_image.py:302) take the run row lock first. A non-terminal run is changed by workers without that lock, so `Complete` / `Ask` whose keys name a non-terminal run are a violation; a `Launch` against one is refused by the endpoint's own status gate |
  | `effect:<id>` | `(state, claim_token)` of a compute effect of this autopilot — immutable once `succeeded` (its attempt dir is written by that claim only) | the autopilot row lock (every core writer of an effect takes it first, §5.2) |
  | `project_segments:<pid>` | digest of the project's SegmentDefs (id, index, action, start, end, pre-roll, post-roll) | the project row lock — phase 2 makes `update_project_segments` (api_v2.py:1203) and Re-analyze take it first (today they do not) |
  | `step:<id>` | that step's `(generation, status)` | the autopilot row lock |
  | `project_detector:<pid>` | digest of the project's `caption_detector` config (AICP-364 R4) | the project row lock — `PATCH /projects/{pid}` takes it first |
  | `project_transcript:<pid>` | digest of the project's `transcript` + `transcript_status` (AICP-365 request) | the project row lock — `PATCH /projects/{pid}/transcript` and the transcription worker's writes (claim, result, failure) take it first |

  Any other key is a contract violation. A run key whose run does not belong to the project, or an
  effect key of another autopilot, is a violation too.
* **Implicit expectations.** For a `Launch`, the core adds `run:<id>` for every run in the
  handler's `targets`, at the version the planning session saw; for any decision, it adds the
  `depends_on` of every step the deciding step transitively `needs`. A flow can only add keys, never
  drop these.
* **`consumes`** names the approvals (by name, current generation) a decision acts on. A decision
  that reads an `approved` or `rejected` answer must list it — the core rejects a `Complete`,
  `Launch` or `Fail` taken while an answered, unconsumed approval of the step exists and is not
  listed. Consumption revalidates the approval's `bound_versions` under the locks (§5.4).
* `about` binds an `Ask` to the rows it is about (e.g. `run:<probe>`); artifacts add their run
  automatically.
* `output` ≤ 64 KiB JSON.

Rules the core enforces (a violation is journaled and fails the autopilot):

* `Launch` of a paid handler from a `paid=False` step; `Ask` of a kind not in the step's
  `approvals`; flows never Ask `overrun`, and Ask `regeneration` only **with options**.
* Names are deterministic; a new round is a new name (`ctx.next_round`).
* `advance` is a pure function of the context (same state → same decision).
* `advance` raising: `errors` + backoff (30 s · 2ⁿ, max 15 min); 5 in a row → `paused`.

Reacting to an answer — the core never guesses:

```python
def advance(ctx):
    a = ctx.approval("listen")
    if a is None:
        return Ask("listen", "listening_probe", "Listen to segment 1 (es)", {...}, subject="es",
                   artifacts=[ctx.artifact(seg1_path, label="segment 1", kind="video")])
    if a.status == "pending":
        return Wait("waiting for the operator to listen to segment 1 (es)")
    if a.status == "rejected":
        return Fail(f"probe rejected: {a.answer_comment}", consumes=["listen"])  # or a fix round
    return Complete({"probe_run_id": rid}, depends_on={f"run:{rid}": ctx.run_version(rid)},
                    consumes=["listen"])
```

A step still returning `Wait` on a rejected approval after `AUTOPILOT_REJECT_GRACE_SEC` (600 s) →
`stopped` with the rejection as reason. A rejected **budget** approval stops the autopilot directly.

### 4.4 Action handlers — the only way state-changing work starts

Module `app/autopilot/actions.py`. A flow names a handler and params; the core executes it.

```python
class ActionHandler(Protocol):
    name: str
    version: int
    paid: bool
    always_regeneration: bool          # rerun_segments: approval every time
    def validate(self, db, project, params: dict) -> dict:    # normalised params, or ActionInvalid
    def targets(self, db, project, params: dict) -> dict:     # existing rows it acts on
    def lock_rows(self, db, project, params: dict) -> None:   # lock targets in the handler's own order
    def estimate(self, db, project, params: dict) -> float:   # via spend.py, never from the flow
    def execute(self, db, project, params: dict, effect) -> dict:
        """Run the admission in `db`. Contract:
        (a) every domain write in `db`, inside ONE outer transaction;
        (b) commit EXACTLY ONCE, after every check passed; never on a refusal path;
        (c) no external paid call before that commit (enqueue after it is fine);
        (d) record created ids into effect.created_refs (assign a new dict) in that transaction;
        (e) refusals raise ActionRefused(status, detail) — HTTPException 4xx is mapped to it;
        (f) filesystem writes before the commit go to fresh, uniquely named paths (never
            overwrite existing files), so a rolled-back or repeated attempt leaves at most an
            orphan file."""
    def reconcile(self, db, effect) -> Literal["launched", "not_launched", "unknown"]:
        """Decide an effect found in `launching`, from targets / created_refs / domain state."""
```

**Shipped by the core** (phase 2), each a thin wrapper around the existing endpoint function with
every parameter passed explicitly (an omitted FastAPI `Form(...)` default arrives as a `FieldInfo`):

| handler | endpoint function | paid | notes |
|---|---|---|---|
| `create_run` | `api_v2.create_run` | yes | Run id captured in the admission flush (the handler sets `Run.id` before its `db.flush()`); reference/voice files go to the new run's own dir |
| `rerun_segments` | `api_v2.rerun_segments` | yes, `always_regeneration` | targets = run + run segments; locks the run first |
| `restore_takes` | `api_v2.restore_segment_takes` | no (refused by the endpoint when it would bill) | |
| `patch_segments` | `api_v2.update_project_segments` | no | the endpoint's own active-run 409 |

Each has tests: every refusal path commits nothing; success commits once; a crash after the
commit and before bookkeeping reconciles to `launched` with the right refs.

**Not shipped, with the reason** — their owners must make them conform first:

* `retry_run` (B): the endpoint accepts non-terminal runs and says it does not check liveness
  (api_v2.py:6957). An autopilot handler may only accept `failed`/`incomplete` or a delivery-only
  `done` retry verified under the run lock, and a paid retry is a regeneration.
* `segment_fill_start` (B): `start_fill` commits a canceled stale preview before raising 409 and
  maps any commit `IntegrityError` to 409 — needs a single-commit service first.
* `transcribe_project` (C1): commits `transcript_status=pending` before an enqueue no recovery
  covers; a worker killed mid-call leaves `running`. Needs durable dispatch/liveness, and an
  ambiguous paid outcome must park for the operator, not resubmit.
* `localisation_prompt` (C1): reservation committed before an RQ-only job a deploy kills without
  replay; a joined job may never reach the commit. Needs a durable job record.
* `to_project` (not needed by the flows: the operator creates the localisation project from the
  run, AICP-362 decision).

### 4.5 Planning is read-only, enforced by the database

`advance` runs in a **separate planning session** whose transaction is read-only at the database
level (`SET TRANSACTION READ ONLY` on Postgres; `PRAGMA query_only=ON` on the SQLite test
connection), with `flush`/`commit` on that session replaced by functions that raise. The session is
rolled back and closed after `advance`. A flow can therefore not write rows, not even by
`db.execute`, and cannot end a transaction. Filesystem writes are forbidden by contract and by
review (heavy or file-producing work is `Compute`, artifacts are copied by the core); flows read
files only through the core's helpers. Planning reads `project_autopilots.version` = `v` and
records the version of every run/segment key it computes, for the implicit expectations.

### 4.6 Reopening steps when what they depended on changes

Every tick (step A, §5.3), and again inside every APPLY for the deciding step's transitive
`needs` (§5.4), the core recomputes the `depends_on` version keys of completed steps. A changed key
(a manual rerun replaced the probe result, the segments were edited, an upstream step was reopened)
**reopens** the step: `generation` + 1, status `active`, output moved to `previous_outputs`, every
unconsumed approval of the old generation (pending **or** answered) `superseded`, and every step
that transitively `needs` it reopened the same way. The flow then re-decides in a clean namespace
(e.g. captions built from an old final are rebuilt). More than 5 reopens of one step pause the
autopilot (a loop). In APPLY, a changed dependency discards the decision being applied (the next
tick replans after the reopen).

`run_version` is a digest of the result- and generation-relevant state only — status, `prompt`,
`model`, `resolution`, `reference_image_urls`, `reference_audio_urls`, `stitch_plan`,
`result_local_path` with that file's `(size, mtime_ns)`, `result_gdrive_file_id`, and each
RunSegment's `(id, status, prompt_override, reference_image_urls_override,
generated_reference_urls, generated_image_job_id, local_result_path + (size, mtime_ns),
seedance_task_id)` — so a sheet export or a rename does not reopen anything, while any change to
what a rerun would generate from does. `(size, mtime_ns)` is the cheap per-tick signal; consuming an
artifact-bound answer re-hashes the bytes (§2.5). Invalidation runs only while the autopilot is live; after `completed`, later manual
work is the operator's.

What the flow docs need, and where it is in the contract:

| Need | Mechanism |
|---|---|
| "Ap" paid launches (Omni pass, Seedance probe, rest of segments) | `Launch` + budget approval |
| Fixes (Omni rerun, VOID fill, Seedance segment rerun) | `Launch(regeneration=True)` or `Ask(regeneration, options)` |
| Subtitle eye checks (smears, invented words, phone screen) | no automatic visual verdict exists: `Ask(handoff, artifacts=[sheets], options=[accept, VOID, LTX, wider mask, delogo])` until a qualified judgement exists (Q8) |
| Listening (probe, final, per language) | `Ask(listening_probe / listening_final, subject=lang, artifacts=[video])` |
| Probe clip reused via a TEMP library clip, seg-1 voice reference | C1 handlers, ids in `created_refs`, a cleanup step at the end |
| Captions/tails rebuilt after a regeneration | `depends_on` the run version → reopen (above) |
| Languages + CTA tails (owner O) | `autopilot_params` at creation (Q4) |
| "No deploy in progress" before launching | the gate (§9): refused launches wait |

### 4.7 Versioning across deploys

`flow_version` is frozen at creation; every effect records `handler_version`; approvals bound to
an action record it too. The registry may hold several versions. A missing flow version pauses the
autopilot; an outstanding effect whose handler version is not registered is left untouched and the
autopilot paused. Compute jobs carry `handler_version`; a worker without it does not claim the job
(no write) and it is retried later. Changing step ids/semantics, a handler's admission or a compute
result schema bumps the version; the old one stays registered until nothing live uses it.

## 5. The tick

### 5.1 Where it runs

A new slim container **`re-skin-autopilot-worker`** (`python -m worker.autopilot_ticker`, PID 1,
`restart: unless-stopped`, 512 MiB / 0.25 CPU cap; it imports `app.api_v2` — phase 2 measures the
RSS and records it in the compose budget comment). A plain loop, not an RQ worker:

```
loop: wait AUTOPILOT_TICK_INTERVAL_SEC (30 s) or until the wake key is set
    if not AUTOPILOT_ENABLED: continue
    for each autopilot with status in (active, paused) OR outstanding > 0, by next_check:
        tick_autopilot(id)          # own sessions; one failure never stops the loop
```

It never runs inside the run workers, imports no InsightFace and runs no run recovery. Free
**compute** (`Compute`) runs on a new RQ queue `autopilot_compute` served by the existing run
workers after `analyze` and before `process` (memory, pre-warmed InsightFace for `subtitle_loop
review --faces`; the host has no room for another GiB-class cap — Q3).

### 5.2 Locking

* **Lease** (work avoidance, not correctness): `UPDATE project_autopilots SET lease_token=:t,
  lease_expires_at=now()+90s WHERE id=:id AND (lease_token IS NULL OR lease_expires_at < now())`.
  Correctness never depends on it.
* **Lock graph** for every admitting transaction. Every path takes a **subsequence** of it, in
  this order (which is what rules out cycles):
  0. `autopilot_gate` row `FOR SHARE` (the hold barrier; `hold on` takes it exclusively and holds
     nothing else);
  1. `video_projects` row `FOR UPDATE` (taken first by `delete_project`, `start_fill` and — after
     phase 2 — `update_project_segments` / Re-analyze);
  2. `project_autopilots` row `FOR UPDATE` (the approval / pause / stop / PATCH endpoints take 2,
     then 4, nothing else);
  3. the run rows the decision names — the action's `targets`, its `expect` / `about` keys and the
     transitive `depends_on` runs — all `FOR UPDATE` in one statement sorted by id (as
     `_lock_runs_for_result_admission` does); the handler's endpoint later re-locks the same rows
     re-entrantly;
  4. the approval rows the decision consumes;
  5. the effect insert (unique index entry), flushed by the core before any handler code;
  6. the handler's other locks (fill job, provider-task savepoint);
  7. spend-day rows, ascending, inside `spend.reserve` — nothing after them until commit (spend.py's
     rule holds because 4–5 are already flushed).

  **Every core writer** — tick, reconciliation, invalidation, the compute job's claim and result
  writes, the judgement sender, the operator endpoints — takes 2 before any step, approval, effect
  or scope row of that autopilot (the version bump needs it anyway); only the tick also takes 1, and
  always before 2. Manual result-changing admissions lock one run (3) and nothing of 0–2. The reaper and workers
  write run/segment rows without waiting for 0–2. The one path outside the order is
  `delete_project`: 1 → its runs (3) → the DB cascade into `project_autopilots` (2). It holds 1
  first, so no tick (which needs 1 before 2 and 3) can hold anything it waits for except the
  approval endpoints, which hold 2 and then only take 4 — so delete waits for them to commit, and
  they never wait for delete.
* The unique `idempotency_key` is the final guard.

### 5.3 One tick of one autopilot: plan, then apply with compare-and-set

```
A. reconcile — any status, observation only (no new submits / launches / asks):
   - action effects `launching`  -> handler.reconcile -> launched / not_launched / unknown
   - compute effects             -> liveness -> re-enqueue / fail (§5.5)
   - judgement effects `submitted` -> poll; a lost answer -> fence (§8)
   - register spend scopes, recompute the budget view (§6.1); warning if committed > B·(1+t)
   - completed steps' depends_on -> reopen (§4.6); stall checks (§3, §5.6)
   - outstanding := count of non-terminal effects
B. status != active, flow version missing, or project unusable -> stop here
C. steps = flow.steps(project, params); rows for newly ready steps
D. for each ready step with next_check_at <= now, in steps() order:
       PLAN  (read-only session, §4.5): read version v; decision = advance(ctx)
       APPLY (one transaction): lock 0 -> 1 -> 2; if gate held, or autopilot.version != v, or the
             step is no longer active: discard the decision, replan next tick
             else apply it (§5.4 for Launch), bump version, commit
       a Launch admitted -> end this autopilot's tick
E. all steps completed and outstanding == 0 -> completed
```

Because every core-owned write (autopilot, steps, approvals, effects, scope registrations, compute
and judgement results) bumps `version`, any operator action, reconciliation or concurrent tick
between PLAN and APPLY makes APPLY discard the stale decision. Domain state the decision relied on
is fenced by the version keys checked under the run/project locks (§4.3, §5.4). The budget figure
is never taken from planning: APPLY recomputes `C`.

Refusals: `ActionRefused` (an explicit admission refusal: 409 run busy, 429 daily cap, 422) rolled
back everything, so no effect row exists; journaled once per distinct detail; `refusals` + backoff
60 s · 2ⁿ (max 30 min; a 429 waits at least 10 min); 8 in a row → `paused`. Any **other** exception
(programming error, filesystem, DB connection) is not a refusal: see §5.4.

### 5.4 The launch protocol

Inside APPLY, after the fence checks (gate, project, autopilot version, step generation):

```
h  = handler; p = h.validate(params); tg = h.targets(p)
lock 3: every run in tg ∪ expect ∪ about ∪ transitive depends_on, sorted, FOR UPDATE, refreshed
verify every version key against the locked rows (implicit + explicit, §4.3) -> mismatch: discard,
       reopen if a depends_on changed, replan next tick
est = h.estimate(p)        # from the LOCKED state (e.g. how many unfinished segments a rerun bills)
sha = sha256(handler, h.version, p)
lock 4 and revalidate every consumed approval: status, sha + handler version, bound_versions,
       artifact bytes, and for action-bound approvals est ≤ the approval's authorised maximum
       (the cost shown, rounded up to whole credits) and committed growth (§6.3) -> mismatch:
       supersede (new generation), commit, no launch
checks of §6.3 with a fresh C — may create an approval and commit (no launch)
mark the consumed approvals `consumed`
E = AutopilotEffect(idempotency_key, state="launching", params=p, params_sha256=sha, targets=tg,
                    target_versions, paid=h.paid, estimate=est, approval_id, handler_version)
register the spend scopes the handler names (§6.1)
db.add(E); db.flush()                 # controlled flush: a concurrent duplicate collides HERE
h.execute(db, project, p, E)          # its single COMMIT persists E + approvals + scopes +
                                      # created_refs + domain rows atomically (and ends the gate share)
T2: lock 1 -> 2; E.state = "launched"; journal `launch`; commit
```

`Complete`, `Fail` and `Ask` are applied the same way minus the effect: lock 3 for their keys,
verify, revalidate and consume the approvals they name, write, commit. An `Ask` with artifacts
copies them here (§2.5).

Outcome classification after an exception, never by parsing its text:

| Exception | Then |
|---|---|
| `ActionRefused` (or a mapped 4xx/429) | rollback; key lookup in a fresh session: absent → refusal (§5.3); present → reconcile |
| `IntegrityError` or anything else | rollback; key lookup: **present** → reconcile (another tick won, or this commit landed); **absent** → `errors` + backoff, journaled as an error (not a refusal); 5 in a row → `paused` |
| the lookup itself fails (DB down) | the tick aborts; the next tick replans — if the commit had landed, the new attempt's flush collides with the key and is reconciled, never repeated |

Crash cases:

| Moment | Outcome |
|---|---|
| Two ticks race one key | the second's flush blocks on the unique index, then fails; its transaction (and its run) never commits; nothing enqueued |
| Crash before the handler's commit | nothing persisted; the next tick replans and launches once |
| Crash after the commit, before T2 | `E` is `launching` with `targets`/`created_refs` committed → `reconcile` → `launched` |
| Crash after the commit, before the enqueue | run `queued` with no job → the reaper re-enqueues it (subject to the pre-existing race, §5.6) |
| `reconcile` → `not_launched` | the consumed approval goes back to `approved`; the step retries under a new round name |
| `reconcile` → `unknown` | parked; autopilot `paused` ("ambiguous launch of X; check run Y"); never relaunched automatically |

### 5.5 Compute effects (free CPU work)

APPLY inserts the effect (`queued`, `rq_job_id = autopilot-compute-{effect_id}`), commits, then
enqueues `app.autopilot.compute.run(effect_id)` on `autopilot_compute` with that fixed job id.
The job claims the row (`UPDATE … SET state='running', claim_token=:t, attempts=attempts+1 WHERE
state='queued' OR (state='running' AND claim expired)`), runs `func(args, attempt_dir)` with
`attempt_dir = data/projects/{pid}/autopilot/compute/{effect_id}/{claim_token}/`, and publishes
`result` (naming its own attempt_dir) with `WHERE claim_token=:t`. Functions are registered with a
version (`register_compute("subtitle_qa", app.subtitle_loop.qa_job, version=1)`); args JSON;
results ≤ 256 KiB; frames are streamed (never a whole video in RAM).

Liveness (step A) uses a generic `rq_job_is_live(conn, queue, job_id)`: pending in that queue, in
its scheduled registry, or the current job of a worker with a live heartbeat (the
`knowledge_job_is_live` test generalised). Not live for 120 s → re-enqueued with the same id after
deleting the stale job hash, at most 3 attempts, then `failed`. Free and claim-fenced: a duplicate
attempt is waste, never harm. A held gate does not block reconcile re-enqueues of already admitted
compute (free).

### 5.6 Deploys, the reconcile tick, the recovery reaper

* **Autopilot container killed**: all state is in the DB; at most a `launching` effect (reconciled)
  and an expired lease remain.
* **Run workers killed**: runs recovered by the reaper as today; compute re-enqueued by step A.
* **Reaper**: the autopilot never writes a run or a run lock, so the hand-driving hazard (the 180 s
  tick re-billing a run parked in `processing`) does not come from it.
* **Pre-existing race (not fixed here, proposed as a separate issue).**
  `recovery._reconcile_orphaned_runs_impl` takes its liveness snapshot (recovery.py:858), reaps
  derived creatives and reframe jobs, then selects active runs (recovery.py:866) and clears the
  per-run lock of each run it judges orphaned (recovery.py:961). A run whose job was enqueued after
  the snapshot can be judged orphaned: its live lock is deleted and a second submit enqueued. A
  timestamp grace would only narrow this; the fix is a durable dispatch generation shared by
  admission and recovery and checked right before the lock clear/enqueue (AICP-344 stage 2). The
  autopilot inherits the same exposure per admission as an operator click. Filed as **AICP-360**;
  fixed separately before `AUTOPILOT_ENABLED` is turned on.
* **Stall watchdog**: a `launched` effect whose target run's `updated_at` has not changed for
  `AUTOPILOT_STALL_SEC` (3 h), or a project stuck in `created`/`analyzing` as long → `warning` +
  `paused`. Never an automatic relaunch.
* **Manual operator actions** stay allowed. A manual **paid** action pauses the autopilot (§5.7,
  Q5); free manual edits are observed: `depends_on` reopens what they invalidated, `expect` refuses
  a launch built on a superseded state. The card warns that the autopilot is active.

### 5.7 Manual paid actions pause the autopilot (Q5)

Every paid admission goes through `spend.reserve`. Phase 2 adds a hook at its start (after its own
flush, before the day lock, inside a SAVEPOINT so it can never break the admission): when the
reserved credits are > 0, the call is **not** inside an autopilot handler's `execute` (a
`contextvars` flag the core sets around it), and the scope belongs to an `assistant` project
(`run:` / `caption:` / `range:` / `segimg:` → the run's project; `fal:` → the task's
`owner_project_id`; `transcribe:` / `localise:` → the project), it inserts a
`project_manual_spend_events` row `(id, project_id, scope_key, kind, mcredits, at, seen_at)` in the
caller's transaction. It takes no lock on autopilot rows (lock order, §5.2). Step A of the next tick
reads the project's unseen events, pauses a live `active` autopilot with `status_reason = "manual
paid action: <kind> (<scope>, ~N credits) — Resume to continue"`, journals one `paused` entry per
event, and marks them seen. Resume is the operator's. The spend itself is counted like any other
(§6.1). Free manual edits never create an event; they reopen dependent steps (§4.6).

## 6. Budget

### 6.1 What counts as spend

The spend-cap ledger (`kie_spend_reservations` / `kie_spend_draws`, AICP-155), keyed by
**scope_key** (not by `run_id`, which is nullable and missing on overdraft rows). It is the only
source that also counts poll-loop resubmits, replaced takes, GPT Image regenerations, LLM calls and
Fal tasks (reserved in credits at `KIE_USD_PER_CREDIT`).

**Attribution** — a scope counts for the project when registered in `project_spend_scopes`
(project-owned, shared by every autopilot attempt on the project):

* `admission`: in the launch transaction, every scope the handler names (`run:{rid}`, …);
* `scan` (step A): `run:{id}` of the project's runs, `range:` / `segimg:` / `caption:` scopes of
  their rows, `transcribe:{pid}`, `localise:{pid}`, `fal:{task}` for provider tasks with
  `owner_project_id = pid`.

Registration is append-only (a later deletion keeps the spend). No baselines: a registered scope
counts in full, whatever it was charged before it was seen. **Known limitation**: attribution of
manual work is best effort between successful scans — a manual row created **and** deleted before
the next successful scan (normally 30 s; longer while the ticker is down or disabled) is not
counted. The card says "manual spend is counted from the project's rows as last scanned" next to
the figure. Accepted because manual actions are outside the autopilot's promise (§1.4).

**Figures** (credits; USD via `spend.usd`):

* **committed** = Σ `charged_mcredits` over registered scopes, all days (in-flight work counts at its
  reservation; a `release_unused` lowers it, correctly, because the released work was not done);
* **consumed** = Σ draws (estimates + settled); **settled** = Σ draws with `settled_at`.

### 6.2 Prediction and the budget approval (core-generated)

The first time a step returns a paid `Launch` while `approved_budget_mcredits IS NULL`:

```
lines = [("already committed", C)]
      + [(step.id, step.title, step.estimate(ctx)) for every paid step not completed]
      + flow.budget_lines(ctx)
total = ceil(Σ lines)
```

→ `budget` approval (`proposal = {lines, total_credits, total_usd, tolerance, assumptions}`).
Approve → `B = total`; reject → `stopped`. A flow may Ask `budget` again with a new prediction
(more languages, a long video); approving replaces `B`. The budget is a project total (spend so far
included), so a restarted autopilot simply shows what was already spent as its first line.

### 6.3 Checks at every paid launch (inside APPLY, with a fresh `C`)

`B` = approved budget, `t` = tolerance, `C` = committed, `e` = handler estimate:

1. **No budget** → budget approval (6.2).
2. **Regeneration** (`Launch(regeneration=True)` or `handler.always_regeneration`) → needs an
   `approved` regeneration approval (or chosen option) for this sha + handler version; otherwise
   one is created, bound to `e`, `C_ask = C` and the target versions: proposal = handler + params
   + `e` + `C`, authorised maximum `ceil(e)` credits, budget after `B' = max(B + e, C_ask + e)`.
   At consumption (under the locks): the fresh estimate must not exceed the authorised maximum, the
   targets' versions must be unchanged, and `C + e ≤ B'·(1+t)`; otherwise the approval is
   superseded and re-asked with fresh figures.
   Approve → `B := B'` (Q6). One regeneration approval also covers the overrun it causes (B' already
   admits it); there is no second approval for the same launch.
3. **Overrun**: `C + e > B·(1+t)` → `overrun` approval bound to the sha, `e` and
   `budget_after = C_ask + e` and the authorised maximum `ceil(e)`; at consumption the fresh
   estimate must not exceed it and `C + e ≤ budget_after·(1+t)` must hold, else superseded. Approve → `B := budget_after`. Within the tolerance it
   launches without asking.
4. Otherwise launch.

The daily kie.ai cap still applies on top as an **admission** cap, with its own configuration
(`KIE_DAILY_BUDGET_USD`, can be disabled) and its documented limits (admitted work can overshoot it
by its own retries; a failed draw leaves the day understated).

### 6.4 Tolerance

`PATCH /projects/{pid}/autopilot {"tolerance": 0.15}` (0 – 0.5); default 0.10
(`AUTOPILOT_DEFAULT_TOLERANCE`).

### 6.5 No automatic paid work on assistant projects

`tasks._enqueue_transcribe_after_analyze` (tasks.py:326) reserves and enqueues a paid
transcription after every localisation analysis. For `mode='assistant'` it does nothing; the
localisation flow launches transcription through its own handler (C1, §4.4) after the budget
approval. Manual projects are unchanged.

## 7. API and the project-page card

All under `/api/v2`, JSON, existing auth/CSRF. Writes lock the autopilot row only, bump `version`
and journal.

| method | path | what |
|---|---|---|
| `POST` | `/projects` (existing) | new form fields `mode` (`manual` default) and `autopilot_params` (JSON) |
| `POST` | `/runs/{rid}/to-project` (existing) | new field `mode` (default per §2.1) |
| `GET` | `/projects/{pid}/autopilot` | live (or latest) autopilot: status + reason, steps, budget `{approved, committed, consumed, settled, tolerance, headroom}` (credits + USD), pending approvals (options, artifact links), last 50 journal rows; `?journal_before=<id>` pages |
| `PATCH` | `/projects/{pid}/autopilot` | `tolerance`; `params` while no paid effect exists |
| `POST` | `/projects/{pid}/autopilot/pause` · `/resume` · `/stop` | operator switch; `stop` takes `reason` |
| `POST` | `/projects/{pid}/autopilot/start` | a new autopilot on an assistant project whose last one finished |
| `POST` | `/projects/{pid}/autopilot/approvals/{aid}/answer` | `{"decision": "approve"\|"reject", "option": "…", "comment": "…"}`; idempotent for the same answer; 409 for a different answer or a non-pending approval; 422 when options exist and none is picked |
| `POST` | `/projects/{pid}/autopilot/approvals/{aid}/comments` | `{"text": "…"}`; does not resolve |
| `GET` | `/projects/{pid}/autopilot/artifacts/{sha256}` | the immutable artifact copy (only shas referenced by this project's approvals) |

Answers and resumes set the Redis wake key so the ticker reacts within a second.

**Card** (`templates/v2/partials/autopilot_card.html`, included in `project_detail.html` only when
`project.mode == 'assistant'`; HTMX refresh every 10 s): mode + status (+ reason); active step(s)
with their last reason; budget bar `committed / approved (+tolerance)` with USD and the
manual-spend note; pending approvals — kind, subject, question, proposal as a key/value table, cost,
artifacts (inline player), options as radio buttons, Approve / Reject + comment; Pause / Resume /
Stop; journal (latest 20, "older" link). Functional only. The new-project and to-project forms get
a `Mode: Manual / by Assistant` choice, disabled with a hint when the type has no registered flow.

**AICP-369:** once the assistant has created a Run, its work on that Run (approvals, steps, journal,
Pause/Resume/Stop) is shown on the Run's page and this card keeps the project-level part plus
links; manual runs are labelled. See `docs/autopilot-run-view.md`.

## 8. Judgement steps on the assistant runtime

```python
class JudgementPort(Protocol):
    def available(self) -> bool
    def submit(self, invocation_id: str, kind: str, payload: dict, payload_sha256: str) -> JudgementState
    def poll(self, invocation_id: str) -> JudgementState
    def fence(self, invocation_id: str, payload_sha256: str) -> JudgementState

register_judgement(kind, input_model, output_model, version)
```

* APPLY commits the effect in `prepared` (`invocation_id = effect.id`, payload sha); nothing has
  been sent.
* Sending is **new work**: only from an active autopilot with the gate open. A fenced APPLY commits
  `sending`; the send itself then runs inside a second transaction that holds the gate `FOR SHARE`
  (re-checked not held) for the duration of the `submit` call (broker client timeout ≤ 5 s) and
  commits `submitted` — so `hold on` cannot return while a send is in progress, and a send never
  starts after it returned. `submit` is idempotent by `(invocation_id,
  payload_sha256)` (the broker's `PUT /v1/turns/{id}` already is), so a crash around the send just
  repeats the same submit. `submitted` is set only after an acknowledged submit.
* A `sending` effect may already be running at the broker. It is treated as in flight: step A
  (any status, also while held) polls it by id and, when the broker does not know it and the
  autopilot may not send (held/paused), leaves it `sending`; when sending is allowed again it is
  re-submitted with the same id and sha. Polling and fencing `submitted` effects are observation.
  A lost or unparseable answer → `fence(id, sha)` → `unknown`; never a fresh call. The step decides
  (usually `Ask(handoff)`).
* `prepared` effects (never sent) are parked while held. Judgements run in the separate
  `reskin-assistant` stack, which an app deploy does not recreate, so the app's drain does not wait
  for `sending`/`submitted` ones; they are polled again after the deploy.
* Results are validated by `output_model` (bounded) before `succeeded`. `available() == False` →
  journal once, re-check later; there is no billed fallback.

**Production port.** The broker accepts only the Run-editing `PlannerContext`, and its runtime
adapter is qualified per change (docs/assistant-media-agent.md). The core ships the port, the
effect handling, a fake port for tests and a `disabled` production port (`available() == False`).
A broker request kind for autopilot judgements is built and qualified with the first flow that
needs one (C1), as its own reviewed change to the `reskin-assistant` stack. B needs none (its eye
checks are operator hand-offs). See Q2.

## 9. Configuration, deploy and drain

Settings (`config.py`; need `up -d`): `AUTOPILOT_ENABLED` (default `false`),
`AUTOPILOT_TICK_INTERVAL_SEC` (30), `AUTOPILOT_DEFAULT_TOLERANCE` (0.10), `AUTOPILOT_STALL_SEC`
(10800), `AUTOPILOT_REJECT_GRACE_SEC` (600).

**Hold** (until the AICP-344 gate exists): `python -m app.autopilot hold on|off|status` (inside
`re-skin-api`). `hold on` runs `UPDATE autopilot_gate SET held=true, generation=generation+1, …
WHERE id=1` and commits: the `UPDATE` waits until every admitting transaction that holds the gate
`FOR SHARE` has committed or rolled back, and every later one sees `held`. When the command returns,
no autopilot admission is in progress and none can start. It then mirrors the flag in Redis for the
ticker's cheap pre-check. While held, no new admission, compute, judgement send or approval is
created; work admitted before the hold continues in the pipeline, and step A keeps observing it
and may re-enqueue already-admitted **free** compute (it holds no gate share). Operator answers are
still recorded.

**Staged rollout** (old images never meet an assistant project):

1. Deploy the core with `AUTOPILOT_ENABLED=false` (the default). No assistant project can be
   created, so the old→new mixed window of this deploy cannot involve one; the transcription gate
   and the compute queue reach every container.
2. Enable (`AUTOPILOT_ENABLED=true` + `up -d`) only once a flow is merged, every container runs
   an image with the gate, **and AICP-360 (the reaper/enqueue race, §5.6) is fixed and deployed**.
   Later flow merges are ordinary deploys: an assistant project created by a new api is only ever
   analysed by workers that already have the gate.

**Every deploy after enabling** (CLAUDE.md lines added in phase 2):

1. `python -m app.autopilot hold on` before the drain check. `deploy/drain-check.sql` gains
   `autopilot_effects WHERE kind='action' AND state='launching'` (always 0 after the hold barrier
   unless a launch is unreconciled). Compute (free, recoverable) and judgement (`prepared` parked,
   `submitted` lives in the separate broker stack) are not counted.
2. **Migration before the image** (the first deploy): every ORM read of a project selects `mode`.
   Rehearsed in phase 2 on a schema-only copy of prod (`pg_dump -s`, read-only) in a throwaway
   Postgres container: upgrade, downgrade, and the mixed window (old ORM against the new schema).
   The downgrade refuses while any `project_autopilots` row exists.
3. The standard deploy command builds and (re)creates `re-skin-autopilot-worker` (default profile).
4. Run workers also listen on `autopilot_compute`.
5. `python -m app.autopilot hold off` after the nginx restart and the curl check.

## 10. Stub flow and acceptance tests (phase 2)

`tests/autopilot_stub_flow.py` registers `stub` (test-only, project type `cover_change`):
`ready` (free: waits for the project) → `probe` (paid: `create_run`) → `listen` (`Ask
listening_probe` with an artifact) → `rest` (paid: `rerun_segments`, a regeneration) → `handoff`
(`Ask handoff`). SQLite unit tests (enqueue and providers mocked):

* manual project: no autopilot row, tick no-op, create / to-project / auto-transcription unchanged
  (plus the full suite green);
* budget approval first; approve → `probe` launches within budget; `listen` pending → nothing
  launched; approve → `rest` asks regeneration; approve → launch with `B` raised; reject → `stopped`;
* overrun above `B·1.1` → overrun approval; within 10 % → silent launch;
* double launch: a second tick on the same key → no second run; crash after the handler commit
  (T2 made to raise) → next tick reconciles `launched`, one run; crash before commit → one launch;
  every exception class of §5.4 classified as specified;
* each shipped handler: refusal paths commit nothing; refs captured; crash-before-T2 reconcile;
* planning session cannot write (`db.execute(update…)` raises), cannot commit; stale version →
  decision discarded (incl. a reconciliation between PLAN and APPLY); implicit/explicit version-key
  mismatch → no launch; a key naming a non-terminal run in `Complete`/`Ask` → violation;
  `depends_on` change → step and dependents reopened with a new generation (fresh effect/approval
  names), their unconsumed approvals (pending or answered) superseded; a `Complete` that ignores an
  answered approval → violation; consuming a listening approval after its run changed → superseded;
* approval binding: estimate drift, committed growth beyond `B'`/`budget_after`, target version
  change → superseded and re-asked; one regeneration approval covers its overrun;
* `update_project_segments` and Re-analyze take the project row lock (fence for
  `project_segments:` keys);
* hold: `hold on` returns only after an in-flight admission commits; held → nothing admitted;
  pause/resume/stop, wake, missing flow/handler version → paused; error and refusal backoff;
  compute claim + per-attempt dirs + recovery; judgement prepared→submit idempotency, unknown never
  resubmits, prepared parked while held; approval superseded by sha or by a changed artifact (new
  generation); finished autopilot keeps reconciling outstanding effects;
* spend: scope registration survives deleting a run; a restarted autopilot inherits the
  project's registry (deleted run still counted); release lowers committed; the restarted budget
  includes spend so far;
* judgement: `prepared` → `sending` → `submitted`; a crash in `sending` re-submits the same id+sha
  only when sending is allowed; held → `sending` is only polled; `hold on` waits for a send in
  progress;
* generation-input writers (`patch_run_segment`, image-job apply/clear, the image worker's apply)
  take the run lock; an input change after a regeneration approval supersedes it;
* compute artifacts bound by `effect:<id>`; a source whose bytes changed with equal size/mtime
  supersedes the approval (content re-hash);
* Postgres: compute claim/result vs a tick vs delete_project — no deadlock (autopilot row before
  effect rows everywhere).

**Postgres tests** (`tests/pg/`, skipped unless `AUTOPILOT_PG_URL` is set; run in phase 2 against a
throwaway `postgres:16` container with the migration applied): two threads launching one key (the
second blocks on the unique index; its run never commits), the gate barrier, read-only planning
transactions, delete_project racing a tick, the partial unique indexes.

## 11. Files (phase 2)

`backend/app/autopilot/{__init__,contract,registry,core,actions,spend_view,compute,judgement,gate,api,cli}.py`,
`backend/app/models_autopilot.py`, `backend/alembic/versions/r1a7c3e5b9d2_autopilot_core.py`,
`worker/autopilot_ticker.py`, `docker-compose.yml` (service), `worker/worker.py` (queue),
`app/tasks.py` (transcription gate, compute enqueue), `api_v2.py` (mode on create / to-project;
project row lock in `update_project_segments` and Re-analyze; run row lock in `patch_run_segment`
and the segment-image apply/clear), `app/pipeline_segment_image.py` (run row lock before apply),
`templates/v2/partials/autopilot_card.html` + include, `deploy/drain-check.sql`, `CLAUDE.md`,
`tests/test_autopilot_*.py`, `tests/pg/`.

## 12. Risks

* **Endpoint functions as the admission API**: pinned by explicit parameters and per-handler tests
  (signature, single commit). Refactoring handlers into services is a later cleanup.
* **The pre-existing reaper race** (§5.6) remains until a dispatch generation exists.
* **Memory**: the tick container imports `api_v2`; compute uses the run workers (Q3).

## 13. Operator decisions (Operator, 2026-10-31)

| Q | Decision | Effect on the design |
|---|---|---|
| Q1 VOID fill | every VOID fill is a **regeneration** and needs an approval; it is not covered by the budget | B's fill handler is `always_regeneration`; approving adds its cost to `B` (Q6) |
| Q2 source breakdown | speakers / gender / on-off face / actions come from the existing `transcribe/v3`; the judgement port stays defined but **disabled** in production; no broker change in this wave | as designed (§8) |
| Q3 memory | accept `re-skin-autopilot-worker` at 512m / 0.25 CPU without trimming other caps; Compute on the run workers via `autopilot_compute` is OK | as designed (§5.1) |
| Q4 operator inputs | languages and CTA tails are `autopilot_params` at creation, editable until the first paid launch | as designed (§2.2, §7) |
| Q5 manual actions | a **manual paid** action on an assistant project (operator-started run, rerun, fill, a restore that bills, an automation copy, …) **pauses** the autopilot with a journal entry naming the action; the operator resumes. Free manual edits keep the reopen behaviour | **changed**: §5.7 |
| Q6 regeneration budget | an approved regeneration **adds** its cost to the approved budget | as designed (§6.3) |
| Q7 manual → assistant | no switching in v1 | as designed (§2.1) |
| Q8 visual checks | no separate `visual_check` kind; `handoff` with options | as designed (§4.6) |

The reaper/enqueue race (§5.6) is filed as **AICP-360** and is fixed separately, before
`AUTOPILOT_ENABLED` is turned on (§9); this branch does not change recovery.

## 14. Design review (§2, GPT-6.1 Terra, `codex exec`, reasoning effort high)

**Round 1** (rev 1, session `01a19005-9a74-7fe2-9b96-14a66bd7c14d`): 15 findings (6 BLOCKER, 9 MAJOR),
all checked against the code.

| # | Sev | Finding | Resolution |
|---|---|---|---|
| 1 | BLOCKER | auto-transcription spends before the budget approval | accepted: §6.5 |
| 2 | BLOCKER | reaper vs handler enqueue gap can double-submit | verified as **pre-existing** (recovery.py:858/866/961); guarantee reworded (§1.3); fix proposed as its own issue (round 2 showed a timestamp grace is not a fix) |
| 3 | BLOCKER | translate admission incompatible | accepted: not a handler until durable (§4.4) |
| 4 | MAJOR | flush-hook refs incomplete | accepted: `targets` before execute + handler-owned `created_refs` |
| 5 | MAJOR | early commits (start_fill), IntegrityError handling | accepted: single-commit rule, non-conforming handlers listed, classification by key lookup |
| 6 | MAJOR | autoflush/savepoints; SQLite ≠ Postgres | accepted: controlled flush; Postgres tests |
| 7 | BLOCKER | stale decisions applied after lease expiry | accepted: plan/apply with version CAS (rev 4 form, §5.3) |
| 8 | MAJOR | lock graph incomplete; hold race | accepted: full graph; gate row barrier (rev 4 form) |
| 9 | BLOCKER | spend not reconstructible after deletions | accepted: scope-keyed registry (§2.6) |
| 10 | MAJOR | baseline subtraction wrong | accepted: baselines dropped in rev 4 (§6.1) |
| 11 | MAJOR | manual admissions race the budget | **partly rejected** (the budget is a plan for autopilot admissions, not a second hard cap); Terra accepted the policy in round 2 and asked for consistent wording — done (§1.4, §6.1, §6.3) |
| 12 | MAJOR | finished autopilots stop reconciling; wrong liveness queue; shared dirs; failed analysis | accepted: `outstanding`, `rq_job_is_live`, per-attempt dirs, stall/failed pause |
| 13 | MAJOR | no visual checks / choices / artifact binding | accepted: options, content-addressed artifacts, hand-offs (Q8) |
| 14 | MAJOR | handler/compute/judgement versioning; judgement crash boundary | accepted: versions everywhere; `prepared` → idempotent submit → fence |
| 15 | MAJOR | migration compatibility / rollback | accepted: CHECK claim corrected, rehearsal incl. mixed window, refusing downgrade |

**Round 2** (rev 3, same session): Terra accepted the #11 policy and raised 15 open points.

| # | Sev | Finding | Resolution (rev 4) |
|---|---|---|---|
| R2-1 | BLOCKER | R-1 timestamp grace does not close the reaper race | accepted: R-1 removed from phase 2; race documented as pre-existing, durable dispatch generation proposed as its own issue (§5.6) |
| R2-2 | MAJOR | stale snapshot after the check, preprocessing | same as R2-1 |
| R2-3 | MAJOR | lost analysis / transcription enqueue strands the flow | accepted: stall pause for `created`/`analyzing` (§3); `transcribe_project` not a core handler, C1 must make it durable (§4.4) |
| R2-4 | BLOCKER | baseline hides new spend after a release | accepted: no baselines; budget = project total incl. spend so far (§6.1–6.2) |
| R2-5 | MAJOR | scan baselines discard manual spend; create+delete gap | accepted: no baselines; the 30 s gap is an explicit, surfaced limitation (§6.1) |
| R2-6 | MAJOR | guarantees contradict the budget scope; "hard cap" wording | accepted: §1.4, §6.3 reworded |
| R2-7 | BLOCKER | in-transaction advance does not fence domain state | accepted: `expect` checked under the handler's row locks before launch; `depends_on` reopen for completions (§4.3, §4.6, §5.4) |
| R2-8 | MAJOR | empty-ORM assertion is not read-only enforcement | accepted: separate DB-enforced read-only planning session, flush/commit disabled (§4.5) |
| R2-9 | MAJOR | superseded artifact approvals cannot be re-asked; mtime binding weak | accepted: `generation` in the dedupe key; content-addressed artifact copies, re-hash at consumption (§2.5) |
| R2-10 | MAJOR | completed steps cannot reopen | accepted: `depends_on` invalidation with transitive reopen (§4.6) |
| R2-11 | BLOCKER | hold barrier relies on a lease TTL | accepted: `autopilot_gate` row, `FOR SHARE` in admissions vs exclusive `UPDATE` (§2.8, §9) |
| R2-12 | MAJOR | hold vs `submit-if-prepared` | accepted: submitting `prepared` is new work (gate-fenced); polling/fencing is observation; drain counts neither (§8, §9) |
| R2-13 | MAJOR | exception classifier treats any failure as a refusal | accepted: explicit refusals vs errors vs uncertain outcome; filesystem rule (f) (§4.4, §5.4) |
| R2-14 | MAJOR | `retry_run` unsafe as an automatic handler | accepted: not shipped; B's handler may only accept terminal states verified under the run lock (§4.4) |
| R2-15 | MAJOR | rolling deploy: old worker enqueues transcription | accepted: staged rollout with `AUTOPILOT_ENABLED=false` first (§9) |

**Round 3** (rev 4, same session): Terra agreed with tracking the reaper race separately (and asked
that the rollout decision accounts for more frequent unattended admissions — noted in §1.3 and for
the orchestrator below). 2 BLOCKER, 8 MAJOR, 1 MINOR, all accepted:

| # | Sev | Finding | Resolution (rev 5) |
|---|---|---|---|
| R3-1 | BLOCKER | listening/handoff answers not revalidated at `Complete` | `consumes` on every decision that acts on an answer; consumption revalidates `bound_versions` under the run lock; artifacts only from terminal runs, copied under that lock (§2.5, §4.3, §5.4) |
| R3-2 | BLOCKER | fencing optional; estimate/budget computed before the locks | implicit expectations for every target and transitive `depends_on`; APPLY locks the rows first, then estimates and checks the budget from the locked state (§4.3, §5.4) |
| R3-3 | MAJOR | locks do not cover every expectation key / writer | only three key kinds, each with a fencing lock; non-terminal runs not allowed as keys of `Complete`/`Ask`; `update_project_segments` + Re-analyze take the project lock in phase 2 (§4.3) |
| R3-4 | MAJOR | reopening does not fence a same-tick dependent decision | depends_on of transitive needs re-verified inside APPLY (§4.6, §5.4) |
| R3-5 | MAJOR | reopened steps reuse effect/approval identities | step `generation` in effect keys and approval identities; answered-but-unconsumed approvals superseded too (§2.3–2.5, §4.6) |
| R3-6 | MAJOR | restarted autopilot loses predecessor attribution | registry made project-owned (`project_spend_scopes`, §2.6) |
| R3-7 | MAJOR | approval not bound to cost/budget state | approvals bind estimate, `C_ask`, target versions; revalidated at consumption; one regeneration approval covers its overrun (§6.3) |
| R3-8 | MAJOR | version CAS does not cover effects/scopes/budget inputs | every core-owned write bumps `version`; `C` always recomputed in APPLY (§2.2, §5.3) |
| R3-9 | MAJOR | `prepared` judgement may be in flight | `sending` state; polled/kept in flight; drain scope stated (§8) |
| R3-10 | MAJOR | mutable filesystem inputs for artifacts | artifacts only from terminal producer rows, copied and bound under the run lock; changing/missing source → the Ask waits (§2.5). Atomic publication of pipeline outputs in general is out of scope |
| R3-11 | MINOR | "≤ 30 s" and hold wording too strong | reworded (§6.1, §9) |

**Round 4** (rev 5, same session): Terra confirmed the `delete_project` exception is sound and raised
1 BLOCKER + 5 MAJOR, all accepted and resolved in rev 6:

| # | Sev | Finding | Resolution (rev 6) |
|---|---|---|---|
| R4-1 | BLOCKER | generation inputs of a terminal run (segment prompt/refs, generated image) change without the run lock and are not in `run_version` | phase 2 makes those four writers take the run lock; `run_version` covers every generation input (§4.3, §4.6) |
| R4-2 | MAJOR | QA sheets (compute outputs) cannot be bound as artifacts | compute effects are artifact producers, bound by `effect:<id>` (immutable once succeeded) (§2.5, §4.3) |
| R4-3 | MAJOR | size/mtime does not prove the bytes | consumption re-hashes the source bytes under the lock (§2.5, §5.4) |
| R4-4 | MAJOR | lock order for compute/reconciliation writers | every core writer takes the autopilot row before effect/step/approval rows; Postgres race test (§5.2, §10) |
| R4-5 | MAJOR | judgement send could start after the hold returned | the send runs inside a gate-share transaction (§8) |
| R4-6 | MAJOR | 2 % estimate drift could authorise a much larger cost | approvals carry an authorised maximum (`ceil(e)`); a larger fresh estimate supersedes (§5.4, §6.3) |

The design is frozen at rev 6 without a fifth round: the four rounds converged from structural
findings (round 1) to fencing details (round 4). Rev 6's resolutions are re-checked by the
mandatory §1 gate on the phase-2 implementation, which also reviews the code itself.


## 15. Implementation notes (phase 2, 2026-10-31)

Where the code differs from, or makes concrete, the text above. Public contract names are
unchanged (`Flow`, `StepSpec`, `StepContext`, the decisions, `ActionHandler`, `registry.register`).

* **Modules.** `app/models_autopilot.py`; `app/autopilot/{contract,registry,core,actions,
  spend_view,compute,judgement,gate,api}.py`, `__main__.py` (CLI), `flows/` (empty: production
  flows register there). Migration `r1a7c3e5b9d2`. Ticker `worker/autopilot_ticker.py`.
* **Contract additions requested by AICP-365** (all additive): the `project_transcript:<pid>`
  version key (table in §4.3); `StepContext.version(key)` for any key kind,
  `transcript_version()`, `previous_outputs()` (the step's earlier generations, so a replacement
  launch can be told apart), `step_generation`; pre/post roll in the `project_segments` digest;
  `create_run` takes `voice_from_library_clip` (a stitch-library clip uploaded as the voice
  reference, exclusive with `voice_clip_id`); a handler may write `effect.result` itself (the core
  merges its return value). `ApprovalView.decision` (`approve` / `reject`) survives consumption,
  where `status` becomes `consumed`.
* **Contract additions requested by AICP-364**: the `project_detector:<pid>` key (R4);
  `EffectView.params` (R7); `rerun_segments` records the takes it archived in
  `created_refs.takes` (R8); `StepContext.fills(run_id)` — a DB-only snapshot of the run's segment
  fills with their provider-task status (R1; the planning session is read-only SQL, so any other
  read a planner needs is available through `ctx.db`, but nothing that probes media or hashes a
  composite belongs in `advance`). R5 (consuming several answers) was already all-or-nothing after
  gate round 1 (`_revalidate_named` locks and checks the whole set before anything is marked; a
  mismatch rolls the decision back and commits only the supersede); a test pins it. **Declined**:
  R6 `fill:<id>` version key — no single lock fences every writer of a fill row (the prepare and
  composite workers update it without the run lock), so a key on it would break the §4.3 rule;
  depend on `run:<id>` instead, which changes when an applied fill re-points the segment under the
  run lock. R2 (fill preview as an artifact producer) — deferred: the preview is written by a
  worker without a fence; copy it through a compute effect (an `effect:` producer) if B needs it.
* **Handler rule (g)** (AICP-365 finding): `execute` must never `db.rollback()` / `db.close()` and
  calls only transaction-neutral services — a rollback would drop the core's flushed effect row.
  Safety net: if the post-launch bookkeeping finds the admission committed without its effect row,
  the autopilot is paused with "handler contract (g) broken". The four shipped handlers'
  endpoints contain no rollback and commit exactly once.
* **`Run.mute_source` frozen at admission**: not in the core. C1 owns it (it needs a runs column
  and a submit-time change in the pipeline, which only localisation uses).
* **Core-created approvals** are named `<launch name>#budget|#regeneration|#overrun` (`#` cannot
  appear in a flow name) so a flow can read them with `ctx.approval(...)`. A rejected core
  regeneration/overrun approval stops the autopilot (a flow that wants another outcome checks the
  approval before re-issuing the Launch).
* **Hold**: the database gate row only (no Redis mirror); the ticker reads it once per pass and
  every admitting transaction holds it `FOR SHARE`.
* **`create_run` handler**: no GPT Image preprocessing parameters in v1 (no flow needs them).
* **`depends_on` and self-modification**: a step must not `depends_on` a row that a later step of
  the same flow changes on purpose (the change would correctly reopen it, in a loop that pauses
  after 5 reopens). Depend on what must stay as it was approved.
* **Tests**: `tests/test_autopilot_core.py` (SQLite, stub flow in `tests/autopilot_stub_flow.py`)
  and `tests/test_autopilot_postgres.py` (opt in with `AUTOPILOT_TEST_POSTGRES_URL`, a throwaway
  database): concurrent applies of one plan admit once, the unique-key loser's run never commits,
  `hold on` waits for an admission in progress, read-only planning, one live autopilot per
  project, delete_project vs a tick, compute claim vs a tick.
* **Migration rehearsal** (2026-10-31, throwaway `postgres:16-alpine`, schema-only `pg_dump -s`
  of prod): upgrade from `q8b4d0f2a716`, downgrade, upgrade again; `compare_metadata` reports 0
  differences; main's image (old ORM, no `mode`) inserts a project against the new schema and gets
  `manual`; the downgrade refuses while a `project_autopilots` row exists.
* **§1 gate fixes worth knowing for flows**: a step skipped by a params change comes back (new
  generation) when its id reappears; its unconsumed approvals are superseded on skip; completion is
  rechecked under the lock against the current params and every completed step's `depends_on`;
  a stale chosen option / regeneration approval never leaves a raised budget behind.
