# Assistant: deploy/drain gate before paid steps, and restart-safe steps

Tracking: AICP-344 (found in
AICP-332; last gap in
[assistant-flow-localisation.md](assistant-flow-localisation.md)). **Design only, nothing
implemented.** Revision 8, 2026-10-30, **FROZEN WITH OPEN ITEMS** after Terra §2 round 6 (§9):
1 BLOCKER and 1 MAJOR left for the implementation's §1 gate. Operator's decisions: §0.
Background: Appendix A; review history: Appendix B.

"The assistant" is the autonomous flow executor AICP-332 specifies (intake → segments →
probe → runs → captions → tail). It launches work only through `/api/v2`, never through
SQL. The run-scoped chat assistant of AICP-261 already has durable turns, leases and
`assistant_paid_call_intents`; the executor reuses that pattern (§4).

## 0. Decisions and acceptance

Operator, 2026-10-30:

| # | Question | Decision |
|---|---|---|
| Q1 | Worker submit intents part of AICP-344? | **Yes.** Stage 2 (§5) is in the acceptance. |
| Q2 | Gate scope | **All new admissions**: operator UI, automation and assistant (§2). |
| Q3 | Forgotten marker: fail closed + banner + `force-end`, or an expiry? | **Option A, fail closed.** No expiry; the deploy script ends the gate on every exit path; a red banner after a configurable 45 min carries the `force-end` command (§2, §6). |
| Q4 | LLM call with an unknown outcome | **May auto-retry**, bounded as in §5.2. |
| Q5 | Paid derived creative stranded between commit and enqueue | **Manual operator handling accepted** in stage 1; the dispatch outbox stays a later item (§3). |
| Q6 | Review | Rounds 4–6 run; frozen after round 6 with 2 open items for the §1 gate (§9). |

**Acceptance of AICP-344** (both stages are required):
1. While the gate is paused no new work of any origin is admitted (503); admitted work
   finishes; `active` reaches 0 without the assistant blocking it.
2. The assistant checks the gate before each launch and waits durably while paused.
3. After an api/worker restart at any point, the assistant resumes the step it was on;
   no resource is created twice across the assistant ↔ API boundary (§3–4).
4. No paid kie.ai/Fal video or image call is submitted twice blind: an ambiguous submit
   parks for the operator (§5.1). LLM-only calls retry within the §5.2 bound only.
5. Each of the above has a test, including PostgreSQL concurrency tests for the gate and
   the idempotency key, and a restart test per stage-2 site.

## 2. Admission gate (app)

**Marker.** New single-row table `ops_admission_gate`: `paused`, `owner`, `reason`,
`generation`, `paused_at`. Written by plain SQL files in `deploy/ops-gate/`
(`begin.sql`, `status.sql`, `end.sql`, `force-end.sql`, run with `docker exec -i
re-skin-db psql -v …`), so they work while the API is down or on an old image. `begin`
sets `paused`, bumps `generation` and refuses if already paused; `end` is a
compare-and-set on `owner` + `generation`; `force-end` is a compare-and-set on
`generation` alone. DB over file or Redis: only the DB can fence admissions in the same
transaction, and it is not recreated by a deploy.

**Forgotten marker: fail closed (Q3, option A).** The marker **never expires**; it stays
until an explicit `end` or `force-end`. Three things keep that from freezing production:
1. **The deploy script ends the gate itself on every exit path** (§6): a shell `trap` on
   `EXIT`, `INT`, `TERM` and `HUP` runs `end` with the script's own owner + generation,
   including after a failed step. The marker survives only if the process is killed
   outright (`SIGKILL`, OOM kill, host reboot or a lost SSH session without `HUP`).
2. **Banner.** While paused, every dashboard page shows "Deploy in progress since HH:MM
   UTC — <owner>: <reason>". After `OPS_GATE_WARN_AFTER_MIN` minutes (setting, default
   **45**) it turns red and shows the ready-to-run command with the current generation
   filled in:
   `docker exec -i re-skin-db psql -U reskin -d reskin -v gen=<N> < deploy/ops-gate/force-end.sql`.
   The assistant notifies the operator at the same threshold.
3. **`force-end` is an operator recovery step**, taken only after confirming that the
   previous deploy process is gone (no `docker compose`/build process on the host, its
   session ended), exactly as in §6 step 3. The banner text says so next to the command.

**Scope (Q2): every new admission, whatever its origin** — the dashboard, automation and
the assistant alike; the dashboard shows the 503 reason ("deploy in progress") and a
banner while paused.

**Where it is enforced: top-level admissions only, at request entry.** A dependency
`require_admitting(db)` reads the row `FOR SHARE` as the first lock of the admission
transaction. `begin` is a short UPDATE of that row only, so it waits for in-flight
admissions and every later one sees it — no check-then-launch race, no "wait a few
seconds" rule. Paused → **503 + `Retry-After`**, nothing written. **What is gated is a
rule, not a hand list:** every route or action that creates or reactivates a row in a
status `active` counts, or enqueues a worker job — including reframe job/render create
and retry (`api_reframe.py` ~L232, ~L297), the stage-2 operator *resubmit* action (§5.1),
assistant turn/job admission, and the automation tick (skip the tick, consume nothing).
Known members today: project create/analyze, transcribe, localisation-prompt, run
create/copy/copy-batch/retry/approve-preprocess/regenerate-preprocess*, segment
rerun/range-rerun/regenerate-image, captions create/retry, tail edits, overlays, insert
merges, audio patches, reframe. **Enforced by a test:** it enumerates every mutating
FastAPI route and fails unless each is marked `gated` or `ungated("<reason>")`, so a new
launch route cannot slip past Q2.
**Not gated:** anything downstream of admitted work — worker-side reserves (transcribe
after analyze, Fal `admit()` from a reframe job, `release_unused`), polls, stitches,
deliveries. `spend.reserve` is untouched, so its lock contract is unchanged.

**Every gated route must leave counted work.** The gate only works if each admission
commits, before the gate lock is released, a row in a status `active` counts. Two routes
do not today and must be fixed or counted: project create commits `created` and enqueues
analysis afterwards (`api_v2.py` ~L722), and re-analysis enqueues without an active status
(~L1028). RQ queue/started counts are shown as a diagnostic only (restart-sensitive job
types, never the self-rescheduling ticks: reconcile, automation, cleanup, knowledge,
Fal), not as part of the zero rule; if Redis cannot be read the diagnostic says
"unknown", never 0.

**Read endpoint.** `GET /api/v2/ops/admission` → `{admitting, marker, active, parked,
build: {revision, started_at}}`. Read-only, off the `kie_spend` limiter.
- `active` = local work a restart would kill; must reach 0 before a deploy.
- `parked` = durable states that are **never retried blind**: provider
  `needs_reconciliation`, stage-2 parked attempts (§5), assistant flows in `waiting_gate`
  (§4, lease released), and AICP-261 paid intents in `unknown`. Reported, not required to
  be 0; anything else non-terminal is `active`.
- **AICP-261 `unknown` intents: parked only once their sender is proven stopped.** Each
  supervisor process registers a row in `ops_gate_acks` (container id, boot id, pid,
  lease generations held, heartbeat). While the gate is paused it stops claiming and,
  once it has no call in flight, writes `acked_generation = <gate generation>`. An
  `unknown` intent counts as `parked` only if the process that held it has acked, or the
  deploy script has verified on the host that its container is stopped (`docker ps`) —
  independent termination, not a missing heartbeat. Otherwise it is `active`. If a
  supervisor does not ack within the banner threshold, the deployer stops that container
  before treating the drain as done. Operator resolution of an `unknown` intent is
  refused while its former holder is registered and neither acked nor stopped, so a late
  sender cannot race it. Already claimed work is admitted work and stays ungated.
The counts live in `app/ops_status.py`; `deploy/drain-check.sql` gains the same split.

## 3. Idempotent admission (app)

New table `api_idempotency_keys`: `key` (PK, ≤ 128), `route`, `request_sha256`,
`operation` (JSON: `resource_kind`, ordered `resource_ids`, route-specific fields),
`created_at`. Admission routes the executor uses accept an `Idempotency-Key` header;
without it behaviour is unchanged.

**Order inside the admission transaction:** authentication and access → gate row
`FOR SHARE` → `INSERT … ON CONFLICT (key) DO NOTHING RETURNING` (Postgres makes a
concurrent duplicate wait for the first transaction: it inserts if that one rolled back,
returns nothing if it committed) → if nothing returned, read the key: hash mismatch → 409,
else **replay** the recorded operation, skipping only mutable business-state validation
(a retried caption is already `queued` and would 409) → otherwise validation and row
writes → `spend.reserve` (day lock last, as its contract requires) → commit → enqueue. A
rollback or kill removes the key with everything else, so resending executes freshly.
Replay of a deleted resource → 410 with the recorded ids. `request_sha256` = SHA-256 of
canonical JSON `{route template, path params, sorted query, form/JSON body with sorted
keys}`; keys are global and the route is part of the hash. A PostgreSQL concurrency test
covers duplicate keys and the order against other paths that lock the same rows.

A route qualifies only after an audit that it has **one commit** and no irreversible
effect before it:
- **localisation-prompt** does not qualify (commits, then bills). Prerequisite: the
  async translate job already listed as a gap — a row counted in `active`, with a
  persisted attempt record before each LLM call, retried within the §5.2 bound.
- **Pre-commit file effects.** Stage 1 admits only routes whose pre-commit writes land in
  a directory owned by the new row (e.g. run create copying staged frames or a library
  voice into the new run's directory, `api_v2.py` ~L3636/L2815): a rollback leaves
  orphan files for the disk cleanup, never changed bytes under committed state. Routes
  that write in place (segment rerun override, range rerun) or take file parts stay
  **unavailable to the executor** until they get durable staging and recoverable
  publication.
- **Commit then enqueue**: a kill in between strands the row. Runs are re-enqueued by
  recovery. Derived creatives are reaped to `failed`: free ones are retried by the
  executor, paid ones go to the operator — **accepted manual intervention** in stage 1
  (Q5), in exchange for never rebilling blind. A transactional dispatch outbox would make
  them resume automatically; it stays a later item, outside AICP-344.

Retention: a key lives as long as its flow is not terminal, then 30 days.

## 4. Restart-safe steps (assistant)

A **DB** step journal (`flow_id`, `step_id`, `attempt`, immutable request and approval,
state, `operation`), unique on `(flow_id, step_id, attempt)`, with lease + generation
fencing from the AICP-261 job machinery. States: `planned → waiting_gate → launching →
launched → done | interrupted | needs_operator`.

- **Key** = `flow:<flow_id>:<step_id>:<attempt>`, stored in `launching` **before** the
  POST. A resumed or duplicate executor sends the same key and gets the same resource;
  lease expiry cannot launch twice (lesson of the range-attempt lease bug).
- **Gate.** Before each launch, `GET /ops/admission`. Paused, 502/503 or refused
  connection → `waiting_gate`: the executor commits the state, **releases its lease and
  bumps the generation in the same transaction**, and holds no process. A wake-up tick
  re-claims it when `admitting` is true; claims (including the supervisor's own turn/job
  selection, `assistant_supervisor.py` ~L320) are suppressed while the gate is paused. So
  a waiting assistant is `parked` and never blocks its own drain.
- **Stale holders are harmless by construction, not by lease timing.** The executor's
  only side effects are keyed API calls; a stalled process that wakes up after losing its
  lease resends the same key and gets the same resource. The lease (≥ longest step wall
  time + margin, generation-checked on every journal write) only decides who polls and
  writes the journal; a stale write fails the generation check.
- **Unknown outcome** (timeout, 502/504, reset) → stay `launching`, resend the same key
  after the gate opens. Never a new key for an unknown outcome.
- **`launched`**: poll until terminal; a restart only resumes the poll. Run back in
  `queued` → recovery's job; diagnose (live job? lock?) rather than launch anew on elapsed
  time alone.
- **Reaped creative** (`failed` with the orphan message): free steps (burn-only caption
  retry, tail edit) retry once as attempt+1; anything paid, and any ambiguous-billing
  message (range attempt `submit_billed_at` without task id), → `needs_operator`. Status
  alone never proves "nothing was billed" (a range attempt stays `queued` past its paid
  call).
- **Approval** binds one step key and its estimate; resuming the same key needs none;
  attempt+1 of a paid step needs a new one. No SQL surgery, ever (2026-08-12 incident).

## 5. Worker submit intents (stage 2, in the acceptance)

§3–4 make the assistant ↔ API boundary exactly-once. The worker windows of Appendix A item 4 are not
closed by them: a drained deploy avoids them, a crash or OOM does not. Stage 2 closes
them; it changes run-pipeline behaviour for every run, not just the assistant's (Q1).

### 5.1 Paid video and image calls: per-attempt intent, then park

One new table `paid_submit_attempts`: `id`, `owner_kind` + `owner_id` (run segment,
preprocessing image slot, segment image job, LLM job), `attempt_no` (unique per owner),
`request_sha256`, `lease_generation`, `started_at`, `provider_task_id` (NULL until
accepted), `status` (`started → accepted | failed_unbilled | parked → resubmitted |
abandoned`), `predecessor_task_id`. A varchar, not a Postgres enum, so no `ALTER TYPE`.

Before each paid kie.ai/Fal **video or image** call the worker inserts and commits a
`started` attempt — checking its lease generation; it is downstream of an admitted run,
so the gate does **not** apply — then calls the provider, then commits the task id **on that attempt**
and on the owner row, as today. Sites: `RunSegment` submit **and every in-poll resubmit**
(`pipeline_v2` ~L1074, ~L1546–1565), each GPT Image preprocessing call (one attempt per
image, ~L3073), segment image jobs. Range attempts (`submit_billed_at`) and Fal
`provider_tasks` already follow this pattern.

The receipt belongs to the attempt, never to the owner row: during an in-poll resubmit
`RunSegment.seedance_task_id` still holds the **previous** task until the new one
commits, so "owner has a task id" proves nothing. Recovery's rule is: **any `started`
attempt without its own `provider_task_id` is parked** — not resubmitted — whatever the
owner row holds; the previous id is kept as `predecessor_task_id` for the operator. A
provider error that proves nothing was billed (kie.ai 4xx validation before acceptance)
closes the attempt as `failed_unbilled`, which the normal retry path may follow.

A parked attempt shows on the run page with its time and predecessor. The operator
resolves it explicitly: *resubmit* (a new gated attempt, accepting a possible second
charge after checking the kie.ai dashboard) or *abandon*, audited like the Fal
`close|resume` actions. The segment stays in its existing status (no new label in
`segmentstatus`), its run does not finalize, and other segments keep going.

### 5.2 LLM-only calls: bounded automatic retry (Q4)

**Where it applies:** jobs whose only paid calls are LLM calls — the async translate job,
project transcription, and caption ASR (the `transcribing` phase, before burn and
delivery).

**Bound:** at most **2** automatic retries per job after an unknown outcome, so at most 3
attempts. One attempt is one run of the job's LLM phase, which already makes several
calls: the translator's 1:1 re-ask and shorten pass, and the model fallback chain for both
translation and transcription (`localisation.py` ~L673, ~L1125). After the bound the job
parks for the operator (the caption's ASR can then be replaced by captions-from-script).

**Exclusive attempts.** Each attempt is a `paid_submit_attempts` row (§5.1, owner = the
LLM job) claimed by a CAS on the job's lease generation, with a lease of at least the RQ
`job_timeout` + 300 s (the `tasks._RUN_LOCK_TTL_SEC` / range-attempt pattern: RQ kills
the work horse at `job_timeout`, so an expired lease is proof here, unlike for the
supervisor). Recovery starts attempt `n + 1` only after attempt `n`'s lease expired; a
superseded worker fails the generation check before its next call. This is **bounded,
not exclusive**: a stale worker that resumes in the gap before its network I/O can make at
most the remaining calls of its own attempt (each keyed `llm:<attempt_id>:<call_no>`, at
most "max calls per attempt"), and its result is discarded by the generation CAS. Total
calls per job therefore never exceed `3 × max calls per attempt × fallback models`, which
is exactly what admission reserves.

**Spend.** Before each call the worker commits a draw keyed
`llm:<attempt_id>:<call_no>` at the estimate. It **replaces** today's post-call draw
(`localisation.py` ~L705, ~L1169, ~L1324, written after the call under a random key, so
a killed call is never measured) — never both. A duplicate observation of the same key is
idempotent (`spend._apply_draw`). Settling to 0 with `settle_draw` is allowed only for a
response whose provider contract proves no charge (a 4xx validation/unknown-model
rejection before acceptance). The current "model unavailable" category also covers 5xx and
429 (`kie_client.py` ~L699); those, timeouts and resets keep the estimate.
Admission reserves the worst case: `3 attempts × max calls per attempt × fallback models
× llm_credits()`, with max calls per attempt = 3 for translation and 1 for transcription
and ASR, and fallback models = the configured chain length (capped in the code); the job's
terminal transition releases the unused part, so retries never overdraft. A retry is
downstream work, so the gate does not block it (it counts in `active`).

**Not allowed to auto-retry:**
- an LLM call inside a job that also submits a paid video or image task (run/segment
  submit, range attempt, segment image job, preprocessing, reframe): the whole job follows
  §5.1, and an LLM retry must never re-drive it past a video/image submit;
- a call whose output was already consumed (a caption past `transcribing`): what is
  retried then is burn/deliver, not the LLM;
- AICP-261 planner calls (subscription, not kie.ai): the broker never replays a turn, and
  that stays.

The executor side is unchanged: an LLM-only step with an unknown API outcome is resent
with the same idempotency key (§4); the worker bound above is separate.

## 6. Operator procedure (CLAUDE.md deploy section)

Steps 1–5 run inside one script, `scripts/deploy.sh` (new, wrapping today's CLAUDE.md
commands, which do not change):

1. Install the `trap` (§2) **first**, then `begin` with a unique owner token (session +
   random nonce). The trap runs `end` as a CAS on that owner token, so it also cleans up
   when `begin`'s result was lost: if the row carries its token, it ends it; if not, it
   does nothing. A `begin` that is refused (gate already held) exits without ending
   anything.
2. Wait for `active` = 0 (old `drain-check.sql` rows plus assistant rows); `parked` is
   listed, not waited on.
3. Before backup, before migration and before `up -d`: `status` must show your generation;
   otherwise abort. This fences sessions that follow the procedure; it cannot stop a
   compose command already running, so `force-end` is an operator recovery step taken only
   after confirming the previous deployer's process is gone (no `docker compose`/build
   process on the host, its session ended).
4. Backup → migrate → `up -d --build --scale worker=2` (+ overlays/profiles) → restart
   nginx → curl through the proxy (unchanged).
5. `end` with your generation (the trap does it; a normal exit and a failure take the same
   path). A failed deploy therefore reopens admission; if repair needs the pause, the
   operator runs `begin` again by hand before touching anything.

**First rollout:** the old image does not enforce the marker, so the deploy that ships it
uses today's manual drain; the gate protects from the next deploy on. Rolling back to old
code drops enforcement (tables are ignored), so the manual drain is the fallback.

**Rollback barrier for stage 2.** Pre-stage-2 code does not know `paid_submit_attempts`:
its recovery would resubmit a segment whose attempt is parked, and its submit path writes
no intents. So rolling back past stage 2 runs in this order: (1) **while admission is still open**,
the operator resolves every parked attempt — *resubmit* (gated, so it needs the gate open)
or *abandon* — and waits for resubmitted ones to finish; (2) `begin`; (3) drain to
`active` = 0 **and zero open attempts**. *Abandon* is ungated, so a parked attempt that
appears during step 3 is abandoned (or the rollback postponed). The rollback script
refuses while any `started` or `parked` attempt exists.

A run whose only unfinished work is parked attempts counts in `parked`, not `active`
(its RQ job has ended; it waits for the operator), so parking never blocks an ordinary
drain. Because no existing status enum gains a
label, the old code still loads every row.

## 7. Migration

One additive revision on head `n4d0f6b2c538`: `ops_admission_gate` (+ its row),
`api_idempotency_keys`; the journal tables belong to the assistant migration chain. No
enum changes, no backfill; rollback is code only. Stage 2 adds a second additive
revision: `paid_submit_attempts` (all attempt and parked state, varchar status), the async
translate job table, and the project transcript job if transcription does not already
have a row. No new labels in existing enums. Rollback past it is gated by the §6 barrier,
not code only.

## 8. Changes per owner

| Owner | Change |
|---|---|
| App, isolated | `ops_gate` SQL + `require_admitting`, `ops_status` + `GET /ops/admission`, drain-check active/parked split incl. assistant rows, migration |
| App, shared paths | dependency on the admission inventory; counted status for project create/re-analysis; RQ diagnostic; automation skip; dashboard 503 message + banner; `Idempotency-Key` on audited routes; async translate job; stage 2: §5.1 intents + park + operator resubmit/abandon, §5.2 LLM attempt records, pre-call draws and the retry bound |
| Assistant (AICP-261 owners) | flow executor journal, keys, `waiting_gate` with lease release and wake-up, resume rules of §4; supervisor claims suppressed while paused |
| Operator / CLAUDE.md | `scripts/deploy.sh` with the trap; `deploy/ops-gate/*.sql`; `OPS_GATE_WARN_AFTER_MIN`; §6 procedure |

## 9. Status: FROZEN WITH OPEN ITEMS (2026-10-30)

The design is frozen after Terra round 6 (GPT-6.1 Terra, high, read-only, delta
`86a0359..8ca6644`), per the review-gate-bar rule. Verdict: **does not pass**; 3 of
round 5's 4 findings closed, 2 items left open. They are **not** fixed in this document;
the implementation's §1 gate must cover each with a test before the stage that contains it
ships:

1. **BLOCKER — supervisor acknowledgement vs an outstanding submission permit** (§2,
   `parked`). The acknowledgement's "no call in flight" is not enough: an AICP-261
   sender can hold a committed `submitting` permit before its network I/O
   (`assistant_jobs.py` ~L552) and use it after acknowledging. The implementation must
   make the acknowledgement prove that no outstanding permit can still be used (e.g.
   acknowledge only after revoking or consuming every permit the process holds, with a
   generation fence the sender re-checks), and keep operator resolution of an `unknown`
   intent fenced even after the gate reopens.
2. **MAJOR — an overlapping LLM worker can outlive its job** (§5.2). The bounded, not
   exclusive, rule lets a stale worker send after its lease is superseded; if the newer
   attempt finishes first and the job turns terminal, nothing keeps the stale call in
   `active`, so a deploy can read zero and restart during it. The implementation must
   track each potentially live sender (the attempt row stays non-terminal until its own
   worker acknowledges or is independently terminated) and count it in `active`; the
   one-send-per-(attempt, call number) rule that the cost bound relies on must be
   enforced, not assumed.

Everything else is frozen; later edge cases become tracked issues rather than design
revisions. Full review history: Appendix B.

## Appendix A. The incident and what is missing (formerly §1)

On 2026-10-29 at ~19:01 a deploy from another session recreated `api` and the workers
between two localisation steps. Nothing was in flight. What makes the next one expensive:

1. **No signal.** Drain is manual: the deployer reruns `deploy/drain-check.sql` until it
   reads 0. Nothing stops new admissions meanwhile, so a busy assistant can keep the drain
   from ever finishing, and the assistant cannot see that a deploy is coming.
2. **No idempotency on admissions.** Run create, rerun, retry, captions, copy… reserve and
   commit in one request, but a client that loses the response (API recreated, nginx 502)
   cannot know whether it committed. Resend = possible double bill; no resend = lost step.
3. **Drain coverage gaps.** Not counted: AICP-261 `assistant_jobs` / `assistant_turns` in
   `queued|running|cancel_requested`; the synchronous `localisation-prompt` request,
   which commits its reservation and then bills an LLM call for minutes with no row at all
   (`api_v2.py` ~L2158).
4. **Worker-internal submit windows.** A kill after kie.ai accepts `createTask` and before
   `seedance_task_id` is committed (`pipeline_v2.py` ~L1070–1082; the draw at L1077 is
   best-effort and run-scoped, not segment-scoped) resubmits on recovery and bills twice.
   GPT Image preprocessing has the same window, several calls per commit (~L3073).

Reused unchanged: boot recovery + the 180 s reconcile tick re-enqueue orphaned runs and
clear their stale lock; segments with a persisted task id are re-polled and completed ones
not rebilled; range attempts with a persisted task id resume; other derived creatives are
reaped to `failed` ("worker was killed … Press Retry").


## Appendix B. Review record (formerly §9)

Terra (GPT-6.1 Terra, high, read-only), round 1 on rev 1: 5 BLOCKER, 7 MAJOR, 1 MINOR.
All accepted; rev 2 changes: gate moved out of `spend.reserve` to request entry of an
explicit inventory (Fal release, reframe and post-analyze transcribe reserves were
blocked); `active`/`parked` split so a waiting assistant cannot block its own drain; no
auto-expiry, generation CAS, SQL-only gate commands; key claimed before validation with
the day lock last; localisation-prompt and multipart routes need prerequisites; "never
started" reap reason; DB-only journal and key retention tied to the flow; worker submit
windows stated as residual risk with a follow-up instead of being claimed closed.

Round 2 on rev 2: 2 BLOCKER, 6 MAJOR. Resolved in rev 1→2: blanket reserve gate,
auto-expiry, day-lock order, first rollout, range-recovery description. Rev 3 changes:
"never started" auto-retry dropped (Terra showed `queued` range attempts past their paid
call); worker submit intents made stage 2 of the acceptance (or the acceptance narrowed,
Operator's call); every gated route must commit counted work, plus RQ counts; `parked`
requires no live owner, `waiting_gate` releases its lease, supervisor claims suppressed;
`ON CONFLICT DO NOTHING` key protocol, canonical hash, replay still authenticates; file
parts unavailable to the executor in stage 1; translate job with intent; `force-end` as
a manual recovery step.

Round 3 on rev 3 (last allowed round): points 1, 2, 3, 5, 7, 8 resolved (1 pending the
owner's stage-2 decision); still open were 1 BLOCKER (lease expiry is not proof a holder
is gone), 1 MAJOR (URL/id routes copy files before commit; paid commit/enqueue gap only
goes to the operator) and 1 new MAJOR (counting RQ scheduled jobs never reaches zero
because the ticks reschedule themselves). Rev 4, **not re-reviewed**: executor safety
rests on the idempotency key, not on lease timing; AICP-261 `unknown` intents count as
parked only after a supervisor pause acknowledgement; stage 1 admits only routes whose
pre-commit files land in the new row's own directory, and the paid commit/enqueue gap is
accepted as manual intervention; RQ counts became a diagnostic outside the zero rule.
A §2 round on rev 4 (or the §1 gate on the implementation) should confirm these.

**Round 4 (before implementation)** on rev 6, same Terra session. Scope: the rev 4
changes not yet reviewed (above) and the rev 5/6 additions — §0 decisions and acceptance,
Q2 scope, Q3 fail-closed with trap, banner and `force-end`, §5.1 park and operator
actions, §5.2 retry bound and pre-call draws, the stage-2 migration. Bar: passes with no
BLOCKER/MAJOR.

Round 4 on rev 6: 1 BLOCKER, 4 MAJOR, 1 MINOR; rev 4's file and RQ answers held. Rev 7
changes: per-attempt receipts in `paid_submit_attempts`, parking any started attempt
without its own task id (the in-poll resubmit keeps the old id on the segment); exclusive
LLM attempts by lease generation with an RQ-timeout-floored lease; the reservation covers
fallback models and the pre-call draw replaces the post-call one; the gate covers a rule
plus a route-marking test (reframe render create/retry were missing); a stage-2 rollback
barrier and no new enum labels; the trap installed before `begin` with an owner-token CAS.
The supervisor acknowledgement is dropped: an `unknown` intent is never retried, so a
late send from a stalled process is the original call, already parked; worker attempts
and claimed supervisor work stay ungated so admitted work can finish.

Round 5 on rev 7 (last allowed round): per-attempt receipt, route coverage and trap
resolved; still open: 3 MAJOR (LLM attempts not exclusive; 5xx/429 settled to 0; the
rollback barrier needed a gated resubmit while paused and a parked run counted as active)
and 1 new BLOCKER (without the supervisor acknowledgement a stalled supervisor can send
its original call after the drain read zero). Rev 8, **not re-reviewed**: LLM attempts are
specified as bounded rather than exclusive, with the bound equal to the reservation;
settle to 0 only on a contract-proven unbilled 4xx; rollback resolves parked attempts
before `begin` and `abandon` is ungated; a run blocked only on parked attempts counts as
parked; the supervisor acknowledgement is restored with process registration, host-side
container-stopped evidence as the alternative, and operator resolution refused while the
former holder is live. The §1 gate on the implementation must confirm these.

Round 6 on rev 8 (delta `86a0359..8ca6644` only, 2026-10-30, last round; design frozen
after it): resolved — 5xx/429 settlement, rollback barrier, LLM attempt bound (within the
owner-approved cost bound). Not closed: the BLOCKER on the stalled supervisor (now only
partly) and one new MAJOR on overlapping LLM workers. Both are the open items in §9.

