# Fal as a second provider — design contract

**Fal is a reserve provider.** kie.ai stays the primary provider for every existing
generation step, and Fal carries no default traffic. Stage 1 uses Fal only for reframe
(AICP-251), a capability kie.ai does not offer; Stage 2 may use it only as an
operator-selected backup while kie.ai is unavailable. Nothing in this document changes how a
kie.ai run is routed, priced or recovered.

AICP-312, Stage 0. Status: **frozen by the owner on 2026-10-27 at revision 11**, under the
review-gate rule "freeze a design contract; later edge cases become tracked issues", given
that Fal is a reserve provider. Revision 11 has NOT had a clean pass: the third Grok pass and
the eleventh Terra pass left 3 BLOCKER and 5 MAJOR findings that this text does not resolve.
They are the mandatory implementation checklist of a separate backlog issue linked from
AICP-312, and each must be closed, with its test, before any Stage 1 code merges (the last
table of the Review record lists them). (GPT-5.6 Terra drew 6 BLOCKER / 4 MAJOR
/ 1 MINOR on revision 1, 2 / 6 / 2 on revision 2, 1 / 5 / 1 on revision 3, 0 / 5 / 1 on
revision 4, 0 / 4 / 0 on revision 5, 0 / 3 / 2 on revision 6, 0 / 3 / 2 on revision 7 and 0 /
3 / 1 on revision 8; revision 9 drew 0 / 1 / 2 on the ninth Terra pass and 0 / 5 / 8 on the
Grok pass (Cursor Grok 4.6); revision 10 drew 2 / 3 / 1 on the tenth Terra pass and 0 / 6 / 7
on the second Grok pass; revision 11 drew 2 / 4 / 10 on the third Grok pass and 1 / 2 / 0 on
the eleventh Terra pass; see *Review record* at the end). Evidence:
[wiki/decisions/fal-second-provider.md](../wiki/decisions/fal-second-provider.md) (live probe
results, §1 and §5) and `scripts/fal_probe.py`. Baseline: main `5c1080d`. The AICP-293
branches are unmerged; this design is written to merge after them (§9).

**Stage 1 (AICP-330) is implemented on `feat/fal-stage1-main`; see *Stage 1 implementation
(AICP-330)* at the end, which closes the checklist and overrides the text above where they
differ (notably: Fal jobs run on their own `fal` queue and `fal-worker` container, not on
`QUEUE_PROCESS`, and every submit 4xx keeps its reservation).**

## 1. Goals and non-goals

**Goal.** A provider layer that lets Re-skin submit, track, pay for and retrieve Fal jobs
without ever sending a paid request twice by accident, so that AICP-251 (horizontal/vertical
reframe) can be built on it. It is also where an operator-selected backup channel could later
plug in (Stage 2). Fal is a reserve provider throughout: kie.ai remains primary and Fal never
receives default traffic.

**Non-goals.**
- The reframe workflow, UI and project type: AICP-251.
- BytePlus: its design (`wiki/decisions/byteplus-second-provider.md`) stays valid and shares
  the rules below, but it is not built here.
- Any change to default routing, to `Run.model`, to `run_model_enum`, or to a single byte of
  the kie.ai request path. Every kie.ai call must be unchanged by Stage 1.
- Automatic failover. Backup use stays operator-selected and default off.
- A backup for the face-swap step. The probe showed Fal-hosted Seedance 2.5 rejecting our
  persona image (422 `content_policy_violation`, `partner_validation_failed`); that is
  unresolved evidence, not a design input.

## 2. Facts the design rests on

All verified by the probe unless marked docs-only.

1. **Auth and scope.** `Authorization: Key <key>`. The key is API-scoped: pricing works,
   `/v1/models/usage` answers 403, so per-request debits are not readable by the app.
2. **No idempotency.** Two identical submits create two request ids and two different
   generations (T3). Fal documents no idempotency key and no way to look a request up
   without its id.
3. **`COMPLETED` is not success.** A finished request can answer its result GET with 422
   (T4: moderation after 143 s of processing).
4. **A refused submit and a lost submit are different.** A 4xx before queueing is a refusal.
   A timeout, an unreadable 2xx or a 5xx leaves the request's fate unknown.
5. **Billing is by output second** for reframe (480p 0.05, 768p 0.08 USD, overlap windows
   billable; docs-only), by token for Seedance. The pricing endpoint returns only the base
   rate, so tiers need a local table. Server errors (5xx) are not billed; a 422 may be
   (docs-only).
6. **Inputs are fetched from a URL** that must be public, or uploaded to the Fal CDN (probe:
   token plus one POST works). Files and payloads are retained by default and are shortened by
   headers (§6).
7. **Fal re-queues a failed runner itself** up to 10 times, same request id, unless
   `X-Fal-No-Retry: 1`; whether a re-queued request is billed again is undocumented.
8. **Concurrency starts at 2** in-progress requests per account; the rest waits unbilled in
   Fal's queue.
9. **Status-GET 401/403/429/404 on a real request, and cancel, are docs-only.** The probe's
   only non-2xx status GETs were the key check (an invented id: 404 with the real key, 401
   with a wrong one); rule 4.9 rests on the docs.

Facts about existing code this design depends on (checked against main `5c1080d`):

- `spend.reserve` charges a scope `max(reserved, consumed)` (`_recharge`); an unused
  reservation keeps counting until a later `reserve(..., release_unused=True)` lowers it.
- `spend.record_draw` opens its **own** session, commits separately and swallows every error.
  Its inner `_apply_draw(db, ...)` works in the caller's session but does not explicitly
  flush before it takes the day lock, and this repository's sessions are created with
  `autoflush=False` (`db.py`), so nothing is flushed implicitly either. That is why
  `draw_in_session` (§5) flushes explicitly.
- `KieSpendDraw.call_key` is globally unique.
- `recovery._live_run_ids` scans only `QUEUE_OVERLAY`, `QUEUE_ANALYZE`, `QUEUE_PROCESS` and
  `default`, keyed by a job's first positional argument; other queues are invisible to it.
- `state_machine.transition()` only checks and assigns in memory; it is not a compare-and-set.
- `RunSegmentRangeAttempt` is the precedent for a paid call that must not be repeated: a
  committed phase claim (`claim_owner`, `claim_expires_at`, `_is_claim_live`), SQLAlchemy
  version-checked commits (`StaleDataError`), `submit_billed_at` committed only after the
  claim and the uploads and immediately before `createTask`, and `_assert_still_owner` (which
  re-reads only `claim_owner`) just before the call.

## 3. Shape of the layer

Flat modules, as the repo does (`kie_client.py`, `gdrive_client.py`, `sheets_client.py`): no
class hierarchy over providers. What is shared across providers is data, not inheritance:

- a `provider` string (`"kie"`, `"fal"`, later `"byteplus"`) on the task row and on spend rows;
- a small typed error set so callers can tell the outcomes of §2.3-2.4 apart:
  `ProviderRefused` (a clear 4xx, including moderation; carries a category),
  `ProviderSubmitUnknown`, `ProviderTaskFailed` (the result says the job failed) and
  `ProviderResultGone` (the result cannot be obtained; rule 4.10). `KieClient`'s own
  exceptions are untouched; nothing is refactored onto the new set.

New files:

| File | Content |
|---|---|
| `app/fal_client.py` | `FalClient`: `upload`, `submit`, `status`, `result`, `cancel`, `download`. One HTTP attempt per paid call; idempotent GETs may retry. Never logs a header; provider text is scrubbed of the key. |
| `app/fal_models.py` | `FAL_ENDPOINTS`: endpoint id, queue owner/app, input limits, tariff table (USD per output second per resolution), worst-case overlap factor. An unknown endpoint or a resolution without a tariff **raises** (fail closed). |
| `app/pipeline_provider.py` | the task state machine (§4) and the RQ job bodies. |

**Queue addressing, one rule for every URL** (as `scripts/fal_probe.py` does it). The queue
app is the FIRST TWO path segments of the endpoint id (`minimax/h3-max` for
`minimax/h3-max/reframe-video`), kept in `FAL_ENDPOINTS`. Submit is `POST
https://queue.fal.run/{endpoint id}`; status (`GET .../requests/{id}/status`), result (`GET
.../requests/{id}`) and cancel (`PUT .../requests/{id}/cancel`) all live under
`https://queue.fal.run/{owner}/{app}`, where the full endpoint path answers 405. `FalClient`
and `adopt` (rule 4.5) build every URL this way.

`models.py` gains two tables, `provider_tasks` and `provider_task_actions` (§4). `spend.py`
gains two small additive functions (§5). `config.py`, `tasks.py`, `recovery.py` and
`state_machine.py` get small additions.

## 4. The task record, its claim and its states

One row per provider request, owned by a consumer through a soft reference (`owner_kind`,
`owner_id` strings; no FK, so a consumer's hard delete cannot remove the record of a possibly
billed call).

Columns: `id`, `provider`, `endpoint`, `owner_kind`, `owner_id`, **`owner_slot`** (varchar,
default `'main'`; a consumer that legitimately needs parallel tasks for one owner declares
distinct slots), `owner_project_id` (immutable), **`destination_path`** (immutable,
contained), **`operation_key`**, `status`, **`claim_owner`**, **`claim_expires_at`**,
`submit_started_at`, `request_id` (nullable), `accepted_at` (when the `request_id` was first
durably stored on the row; the poll deadline of rule 9 counts from it), `status_url`,
`response_url` (kept so the result can be read again; output media URLs are never stored),
`cancel_url`, `cancel_requested_at`, `reservation_held`, **`draw_pending`** (boolean, default
false: a `close` whose draw could not be written yet; rule 5), **`fetch_attempts`** (int,
default 0) and **`fetch_deadline_at`** (rule 10), **`request_spec`** (the canonical request
without URLs, immutable after admission), **`input_refs`** (per input: local path, size and
content sha256), `request_sha256`, `output_sec`, `resolution`, `estimated_usd` (display and
audit only; §5), `error_code`, `error_message` (scrubbed), `result_path`, `created_at`,
`updated_at`, and `version_id` declared as the mapper's `version_id_col`, so every commit is
version-checked. `status` is `varchar` with a CHECK over exactly the thirteen statuses of the
transition map at the end of this section. `UNIQUE (provider, operation_key)`, a unique
`request_id` per provider when present, and **one active task per owner slot, across
providers**: a partial unique index `uq_provider_tasks_active_per_owner_slot` on
`(owner_kind, owner_id, owner_slot)` WHERE the row is **active** (defined below the status
table), with `postgresql_where` and `sqlite_where` spelled identically, as
`uq_run_segment_range_attempts_active_per_run` does.

**`provider_task_actions`**, one row per operator action (`adopt`, `abandon`, `close`,
`resume`), is the only audit record (`provider_tasks` has no `closed_by` or `closed_at`):
`id`, `task_id` (FK, `RESTRICT`), `action`, `actor`, `at`, `from_status`, `to_status`, `note`
(free text: what the operator saw) and `request_id` (the id entered for `adopt`). It is
written in the SAME transaction as the action's CAS (an `adopt` refused by Fal's answer
records `from_status = to_status`) and is insert-only, as `assistant_run_result_promotions`
is (`assistant_models.py`, migration `j0e5f2a7b914`): ORM `before_update` / `before_delete`
listeners that raise, and `_immutable` / `_no_delete` triggers on both dialects in the
migration itself.

**`operation_key` is the idempotency key we lack at Fal.** The consumer builds it from the
thing the operator is doing (`<owner_kind>:<owner_id>:<owner_slot>:<attempt_no>`). Starting
the same operation twice returns the existing row and reserves nothing new; **Retry
deliberately mints `attempt_no + 1`**, which is a new row and a new reservation, and is
admitted only once the slot's previous task is no longer active: while it is (a held
`refused` row included, until the operator's `close` or `abandon`) the partial unique index
refuses the insert and admission answers 409 (rule 1).

**The request is frozen at admission.** The row stores the canonical request spec (endpoint,
parameters, durations, resolution) and the content hash of every input, and the estimate is
computed from that spec. Dispatch only turns the frozen inputs into CDN URLs; it never reads
the owner's current settings, and a local file whose hash no longer matches is a local
failure before anything is sent (rule 3). Otherwise a cheap request admitted under a small
reservation could be replaced by a long, expensive one before the worker runs.

### States

| Status | Meaning |
|---|---|
| `queued` | admitted and reserved; **no POST has been sent** |
| `submit_started` | the marker is committed; the POST may have been sent |
| `accepted_unrecorded` | Fal accepted it and the `request_id` is durably stored, but the spend draw is not yet committed; **not pollable** until recovery writes the draw (no POST) |
| `submitted` | Fal accepted it; `request_id` and the draw are committed; polled until COMPLETED, a terminal failure or the poll deadline (rule 9) |
| `fetching` | Fal reports COMPLETED; `fal_fetch` reads and downloads the result, bounded by `fetch_attempts` and `fetch_deadline_at` (rule 10) |
| `cancelling` | a cancel was requested (the poll deadline or the operator's `resume`); polling continues until Fal says terminal |
| `needs_reconciliation` | the `request_id` is known but Fal's outcome was not confirmed within the age limit, or a status GET answered 404 or another non-auth 4xx; polling continues slowly |
| `closing` | the operator chose `close`; the intent is durable, a final cancel is being sent; never polled |
| `done` | output verified and stored |
| `refused` | a clear refusal: 4xx at submit, or 422 on the result (carries a category); after a possibly billed submit 4xx it keeps `reservation_held` until the operator's `close` or `abandon` (rule 6) |
| `failed` | the job ran and failed, or timed out and Fal confirmed it ended; also a local failure before the marker, `owner_gone`, `result_gone` and `operator_closed` (each in `error_code`) |
| `submit_unknown` | no usable answer to the POST; possibly billed; nothing resubmits; parked until the operator acts |
| `abandoned` | the operator confirmed nothing exists at Fal, or cancelled the task before anything was sent; the reservation is released |

**Terminal statuses are exactly `done`, `refused`, `failed` and `abandoned`.** The other
nine are non-terminal; `submit_unknown` is parked (no job works on it), and the remaining
eight are the drain list (§7).

**Active** means non-terminal OR (`refused` AND `reservation_held`): a held `refused` row is
terminal for the state machine but may have been billed and still owes a decision, so it
keeps its owner slot. The slot index, Retry and the consumer's delete checks (§6) all use
this one predicate; Retry on that slot needs the operator's `close` or `abandon` of the held
row first.

**Operator actions**, each a CAS under rule 3 with its `provider_task_actions` row in the
same transaction: `adopt(request_id)` (rule 5), `abandon` (rules 5 and 6; for a `queued` task
it is the cancel before send), `resume` (a fresh cancel for `needs_reconciliation`) and
`close`. The transition map at the end of this section lists everything each may do.

`needs_reconciliation` is not a lost submit: its `request_id` is known. Its transitions are
explicit: to `fetching` when the slow poll sees COMPLETED, to `failed` when Fal confirms a
terminal failure or cancellation, to `cancelling` when the operator chooses `resume` (a fresh
cancel), to `closing` when the operator chooses `close`.

The operator's `close` is an auditable abandonment and the **universal escape** for every
task that may have had a POST: it is legal from every non-terminal status except `queued` and
`closing` itself, and from a `refused` row that still holds its reservation (where it does
something narrower, rule 6). A `queued` task has had no POST, so there is nothing to close:
it is cancelled with `abandon` (cancel before send), which releases its reservation and
writes no draw. Its order is crash-safe: (1) the generic CAS of rule 3 moves the row to
**`closing`** only when no other phase claim is live (otherwise the request is rejected and
the operator retries), taking the claim (overwriting any expired `claim_owner`) and inserting
the `provider_task_actions` row as the durable intent in the same transaction; (2) only then
is the final cancel sent, when a `request_id` is known (idempotent; whatever it answers, the
close goes on; rule 5 says what that means for `submit_unknown`); (3) one transaction
attempts the task's draw (`fal:task:<task id>`, at the worst-case estimate, inside a
SAVEPOINT; a no-op if the draw exists) and ends the row `failed` with
`error_code=operator_closed`. **The draw is attempted, not required:** if it fails, the
transaction rolls back to the savepoint and still commits `failed(operator_closed)`, with
`reservation_held` kept true and `draw_pending` set, and the backfill sweep (§7) writes the
draw later. That is safe because `close` never releases the reservation: it was taken at the
worst case, and `spend._recharge` charges `max(reserved, consumed)`, so the daily cap already
carries the worst case whether or not the draw exists yet. Recovery finishes a `closing` row
by repeating steps 2 and 3, and nothing polls it. The `request_id` and the draw (or the
pending draw) are kept, and the confirmation says plainly that Fal may still complete and
bill; the operator accepts that. (On a `queued` row a worst-case draw would only charge the
cap for work that does not exist.)

### Rules

1. **Admission** (one transaction): freeze the request spec and the content hash and size of
   every input (each read once, streaming), validating the spec against the `FAL_ENDPOINTS`
   limits before anything is reserved: input duration; input size within
   `FAL_MAX_INPUT_BYTES` (90 MB per input, the single-POST upload path, multipart being a
   later stage); aspect ratio; a tariffed resolution. Every input path and the destination
   are passed through `storage.ensure_under_data_dir()` (the fail-closed guard for paths that
   came out of the database), at admission and again at dispatch, so a malformed or stale
   path is rejected locally instead of being read and uploaded. The destination is frozen as
   `destination_path`; recovery, staging and `owner_gone` use only that value. No copy of the
   input is kept (see rule 3). Insert the row as `queued` inside a SAVEPOINT. On a unique-key
   conflict, roll back to the savepoint and re-read by `operation_key`. If a row with that
   key exists it is the winner: return it as it is, and stop: nothing is reserved twice and
   the session stays usable (a check-then-insert would race). If no row has that key, the
   conflict was the one-active-task-per-slot index: another task of the same `(owner_kind,
   owner_id, owner_slot)`, of any provider, is still active (§4), and admission answers
   **409** with nothing inserted or reserved. Otherwise `spend.reserve` (§5), which sets
   `reservation_held` in the same transaction, then commit. `BudgetExceeded` rolls everything
   back: no row, no request.
2. **Dispatch** is an RQ job `fal_submit(task_id)`. **Every Fal job of Stage 1** (`fal_submit`,
   the poll, `fal_fetch`, `fal_cancel`, `fal_close`) **is enqueued on `QUEUE_PROCESS`**, task id
   first, which `_live_run_ids` already scans (§2): no worker change, and no duplicate
   re-dispatch of live jobs, which a new queue name would cause. Fal jobs share that queue's
   capacity with run processing. The enqueue can fail or be lost, so the
   reconcile tick re-enqueues any `queued` row that has neither a live claim nor a live RQ job
   (the `_live_run_ids` machinery of `recovery.py`: pending, scheduled, or started on a worker
   with a fresh heartbeat) and is older than a grace window, so a backed-up queue is not
   flooded with duplicate jobs. That is safe because `queued` guarantees no POST was sent. This is what separates
   "waiting to be sent" from "possibly sent".
3. **Claim, then work, then marker, then one POST.** Every step that changes ownership is a
   guarded `UPDATE` that must change exactly one row, and each one also increments
   `version_id`:
   - **Acquire:** `UPDATE ... SET claim_owner=:token, claim_expires_at=:until, version_id=version_id+1
     WHERE id=:id AND status='queued' AND (claim_owner IS NULL OR claim_expires_at < now)`.
   - **Work under the claim:** verify the input hashes against `input_refs`, upload into
     claim-scoped paths.
   - **Marker:** `UPDATE ... SET status='submit_started', submit_started_at=now,
     version_id=version_id+1 WHERE id=:id AND status='queued' AND claim_owner=:token AND
     claim_expires_at > now`. Zero rows changed means the lease was lost (another worker took
     it while this one was uploading): exit benignly, nothing was spent, the row is still
     `queued`. Without the token and status in the condition, a stale worker could mark a row
     the new owner is still preparing, and recovery would later declare an unsent task unknown.
   - **Re-verify** ownership and send the POST exactly once. The range attempt's
     `_assert_still_owner` re-reads only `claim_owner`; the Fal re-check is one fresh SELECT
     of status (`submit_started`), our token AND an unexpired lease (`claim_expires_at >
     now`). Because `close` and every phase claim take the claim by the generic CAS below,
     overwriting `claim_owner`, a stale worker's re-check fails. The ownership re-checks of
     rules 8 and 10 are the same three-part check, on their own phase's status.

   This is the `RunSegmentRangeAttempt` protocol in full, not only its timestamp.

   **One CAS primitive for every phase.** All ownership changes, not only these two, go
   through a single helper: an `UPDATE ... WHERE id=:id AND status=:expected AND (the claim is
   free, expired, or ours) ... SET status=:next, claim_owner=:token, claim_expires_at=:until,
   version_id=version_id+1 RETURNING version_id`, which must return one row. The ORM object is
   then `refresh`ed (or the helper runs in its own short session) before any ORM mutation, so
   SQLAlchemy's `version_id_col` never rejects its own later commit as stale. Result reading,
   download, cancel and the `needs_reconciliation` resume each take their phase claim this way
   and release it on exit; the acceptance commit (rule 7) clears the submit claim, as
   `pipeline_range_attempt` does, so pollers are never blocked behind a finished submit.

   **Two claim conditions; the marker's wins.** Acquire and every phase claim use the generic
   condition (free, expired, or ours); the marker requires status `queued`, our token **and a
   live lease** (`claim_expires_at > now`). An expired lease of ours may be re-taken, never
   used to commit the marker; where the two texts differ, the marker text is the rule.

   **The lease outlasts every job that holds it.** One `FAL_JOB_TIMEOUT` (default 1800 s) is
   defined, once, in the `FAL_*` block of `config.py` (§9), and `tasks.py` takes the RQ
   timeout of EVERY claim-holding Fal job from it (`fal_submit`, `fal_fetch`, `fal_cancel`,
   `fal_close`; the plain status poll holds no claim). `FAL_CLAIM_LEASE_SEC` (default
   `FAL_JOB_TIMEOUT` + 300 s) is defined beside it, and configuration validation rejects
   start-up when the lease is shorter than that timeout plus a margin. (The range attempt
   only derives its floor from a duplicated 3600 s constant; this is a real check.)

   **Input bytes are read once.** Under the claim, the worker reads each input file into
   memory exactly once, at most `FAL_MAX_INPUT_BYTES + 1` bytes (a longer file is refused
   locally, so a file replaced by a huge one cannot exhaust the worker's memory), one input at
   a time, hashes THOSE bytes, compares the hash with the frozen `input_refs`, and uploads
   those same bytes. There is no verify-then-upload window on a mutable path, and
   nothing is stored, capped or purged. A file that was replaced, truncated or deleted after
   admission is a hash mismatch or a missing file: a **local failure before the marker**. The
   row becomes `failed` (`input_changed` or `input_missing`) and its reservation is released
   (and `reservation_held` cleared) in the same transaction, because nothing was sent and
   nothing can have been billed. The same holds for a failed upload to the Fal CDN (a free
   call that precedes the marker) and for a path the containment guard rejects at dispatch.

   **The 90 MB cap is a real limit, not a footnote.** Of the 124 project sources on the
   server today the median is 34 MB, the 90th percentile 92 MB and the largest 167 MB:
   about one source in ten does not fit, and the upload limit of a project is 1 GiB
   (`MAX_UPLOAD_SIZE_MB`). A 33 s reel is 33 MB, so roughly 90 s of this footage is the
   practical ceiling. The consumer therefore exposes eligibility before the operator starts
   the workflow (an ineligible source is offered a smaller re-encode, never silently
   refused), and multipart upload is an explicit follow-up requirement of Stage 1, not an
   open question.
4. **A stale `submit_started` row without a `request_id`** becomes `submit_unknown`. A worker
   that finds one never POSTs again. A stale claim on a `queued` row is simply re-taken.
5. **`submit_unknown` is a stop.** It carries the "a billed generation may already exist"
   warning, and its reservation stays charged until the operator acts. After checking the Fal
   dashboard the operator has three actions, each a CAS under rule 3 whose
   `provider_task_actions` row records who, when and what they saw:
   - `adopt(request_id)`: the request exists. The id is verified first with **one status
     GET**: a 2xx admits it; a 404 refuses the adoption and records why on the row
     (`error_message`; the status is unchanged); any other answer refuses it for now and the
     operator tries again. An admitted adoption sets `request_id`, `accepted_at` and the
     three URLs (built by the queue-addressing rule of §3 from owner/app and the id, never
     from the full endpoint path), writes the task's draw in the same transaction
     (`submitted` means the draw exists, so a failed draw write fails the adoption; `close`
     stays available), moves the row to `submitted`, and enqueues the poll.
   - `abandon`: the operator records that they confirmed NOTHING was created at Fal; the
     reservation is released and `reservation_held` cleared in the same transaction.
   - `close`: the operator cannot confirm either way. It never releases anything and follows
     the `close` paragraph above: its draw is attempted, not required, and the charge stays
     on the daily total either way. A `submit_unknown` row has no `request_id`, so its
     `close` cannot send a cancel: the operator first looks the request up in the Fal
     dashboard and, if it is there, `adopt`s it and then `close`s it, which cancels. `close`
     is also the terminal escape for `accepted_unrecorded` rows whose ledger write
     keeps failing: it does not need that write, the held worst-case reservation carries the
     charge meanwhile, and the backfill sweep retries the draw. Being legal from every
     non-terminal status that can have had a POST, that is all but `queued` (cancelled with
     `abandon`) and `closing` (which recovery finishes without the ledger), it leaves nothing
     active forever.
6. **A provider-proven pre-queue refusal at submit** (400, 401, 403, 404, 429: Fal answered
   at the gateway and nothing can have reached a runner) is `refused` at once and, in the
   same transaction, its reservation is released (§5) and `reservation_held` cleared; no draw
   is written. **A 422, or any other 4xx that may have reached a runner, is also `refused`
   but keeps its reservation** (`reservation_held` stays true: it may have been billed, per
   the docs) until the operator decides. A 422 at submit means the frozen spec was not
   validated against the endpoint's limits, so it is also a bug signal. Such a row is
   terminal but still **active** (§4: it holds its owner slot) and owes a decision; the
   operator has two actions on it, each a CAS under rule 3 with its `provider_task_actions`
   row in ONE transaction:
   - **`close`**, the action for an operator who is not sure: it converts the held
     reservation into the task's draw (`fal:task:<task id>`, at the worst-case estimate;
     nothing is released), clears `reservation_held` and leaves the row `refused`. It is
     strict: if the draw cannot be written nothing changes, and the reservation, still held,
     keeps carrying the charge until the operator tries again.
   - **`abandon`**, only the explicit "I looked at the Fal dashboard and nothing exists"
     action: it clears the claim, sets `reservation_held` false and calls `spend.reserve(db,
     scope, 0, release_unused=True)`, which releases the charge, exactly as from
     `submit_unknown`.
7. **The acceptance commit.** One short, version-checked commit sets `submitted`,
   `request_id`, `accepted_at` and the three URLs **and** inserts the draw (§5) in the same
   session, and clears `reservation_held` and the submit claim. If that commit fails after
   Fal accepted, the worker retries it with the response still in memory; if the draw is what
   keeps failing, it commits the `request_id`, `accepted_at` and URLs alone as
   **`accepted_unrecorded`**, which is a known request: recovery writes its draw (idempotent
   by the call key `fal:task:<task id>`) and clears `reservation_held` in the same
   transaction, moves it to `submitted` without another POST, and only then enqueues the
   poll. An `accepted_unrecorded` row is **not pollable**, so no terminal transition can
   happen before its draw exists. If no local commit succeeds at all (the database is down),
   the worker logs the `request_id` at ERROR as an operator **hint, not an accounting
   record**; the row becomes `submit_unknown` once its claim is stale (rule 4), its
   reservation still held, so the cap still carries the worst case.

   **The acceptance window.** Between Fal accepting the POST and the first durable write
   there is an unavoidable window: if the process dies after Fal's 2xx reached the client but
   before the acceptance commit, nothing durable holds the id, even with a healthy database.
   Recovery then sees only `submit_started` and makes the row `submit_unknown`. That is why
   `submit_unknown` never releases its reservation by itself (rule 5), why `abandon` needs
   the operator's explicit confirmation that nothing exists, and why `close` keeps the
   charge. A database and a third party cannot commit atomically; the Fal dashboard is the
   last source of truth for this window. Stage 1 keeps no journal outside the database (D6):
   it would cover only "Fal accepted and the database failed at that very moment", which then
   falls into this same `submit_unknown` path (dashboard, then `adopt`, `abandon` or
   `close`); for a low-volume first consumer that is the right trade, and that journal drew
   most review findings of the last four passes.
8. **Polling is a self-re-enqueueing RQ job** (the `run_poll_run` pattern, on `QUEUE_PROCESS`
   as rule 2 says), one idempotent status GET per tick. Side effects (cancel, result read,
   download) run only under a fresh claim. The poll never reads the result itself: on
   COMPLETED it commits the CAS to `fetching` and, after that commit, enqueues `fal_fetch`
   (rule 10). **The staging path is claim-scoped** (named by task id and claim token, in the
   directory of `destination_path`, so `os.replace` stays on one filesystem), and
   `os.replace` follows the ownership re-check of rule 3. There is **no sibling `.part` next
   to the final file**, unlike `KieClient.download_result`, so two jobs can never write one
   staging file.
9. **Only the deadline, `resume` or `close` sends a cancel; a status answer never does.**
   Each idempotent status GET, in `submitted`, `cancelling` or `needs_reconciliation`, is
   classified by its response:
   - **2xx**: normal. COMPLETED moves the row to `fetching` and enqueues `fal_fetch`, a
     terminal failure to `failed`; anything else keeps polling. A 2xx clears an
     `auth_rejected` hold.
   - **429, any 5xx, a transport error**: transient; the row is kept and polled again on the
     next tick with backoff.
   - **401, 403**: a hold. No cancel, no status change; an ERROR log and an operator-visible
     `error_code=auth_rejected`. A rotated or missing key says nothing about the request, so
     polling resumes by itself once the key works.
   - **404 and any other 4xx**: `needs_reconciliation` (the `request_id` is known), no
     cancel; a row already there stays there.

   **The poll deadline.** `FAL_POLL_DEADLINE_SEC` (default 3600 s, §9), counted from
   `accepted_at`, bounds `submitted`. Reaching it is not terminal: it moves the row to
   `cancelling`, records `cancel_requested_at`, sends the cancel, and keeps polling until Fal
   reports a terminal state. Only then is the row `failed` (or, if the job completed anyway,
   `fetching` and its draw stays). A cancel that is not confirmed within
   `FAL_CANCEL_CONFIRM_SEC` (default 21600 s, 6 h, from `cancel_requested_at`) moves the row
   to **`needs_reconciliation`**, which keeps the known `request_id`, keeps the draw, and
   keeps polling at a slow interval; it is never treated as a lost submit and never silently
   dropped. The operator can `resume` (a fresh cancel, `cancel_requested_at` set again) or
   `close` it.
10. **`fetching`** is worked by `fal_fetch` only, under its fetch claim: it reads the result
    from `response_url`, downloads at once into the claim-scoped staging path of rule 8,
    checks the file with ffprobe (size > 0, duration within tolerance), and `os.replace`s it
    onto `destination_path` after the ownership re-check. The output media URL is never
    stored; `response_url` is, so the result can be re-read for a fresh one. A 422 on the
    result is `refused` with its category (`moderation` for `content_policy_violation`); the
    draw stays counted, because the request ran. A result reporting the job failed is
    `failed` (`ProviderTaskFailed`). **A result that cannot be obtained ends the task, by
    durable counters:** the CAS that first moves the row into `fetching` sets
    `fetch_deadline_at` (entry + 30 min) and `fetch_attempts = 0`; every failed attempt (a
    result GET without a video, a failed download or ffprobe check) increments
    `fetch_attempts` under the fetch claim, releases the claim and re-enqueues `fal_fetch`
    after a backoff of 1, 2, 4 and 8 minutes, so five attempts span at least 15 minutes. The
    5th failed attempt, or a `fal_fetch` that finds `fetch_deadline_at` passed (which also
    bounds attempts that crash before they can count themselves), raises `ProviderResultGone`
    and ends the row `failed(result_gone)` with its draw kept. A failed write into
    `destination_path` is `failed(owner_gone)` (§6).
11. Transitions are declared in `state_machine.py`: a `ProviderTaskStatus` enum and a
    transition table, added to `can_transition()` (which dispatches on enum classes and would
    otherwise raise `TypeError` for a string status). The column stays `varchar` with a
    CHECK. That table is the transition map below, edge for edge, and it is the single source
    for the CHECK constraint (all thirteen statuses), the one-active-task-per-slot index,
    Retry and the consumer's delete checks (the **active** predicate: the nine non-terminal
    statuses, or `refused` with `reservation_held`), the drain-check list (§7: the eight
    non-terminal statuses other than `submit_unknown`) and the transition tests (§10). The
    non-terminal set is spelled once in Python, as `PROVIDER_TASK_NON_TERMINAL_STATUSES`
    beside the enum, the active predicate once beside it as `PROVIDER_TASK_ACTIVE_WHERE`
    (built from that set), and the index's WHERE clause is kept byte-identical to it, as
    `RANGE_ATTEMPT_ACTIVE_STATUSES` is for range attempts. The guarantee of exclusivity is
    the claim plus the version column, not that function, which only assigns in memory.

### Transition map

Every legal status change of a `provider_tasks` row, and nothing else. "No live claim" means
the CAS of rule 3 finds no phase claim held by anyone else; "our claim" means the CAS names
the token of the job doing the work. Every edge except admission (an insert) is one guarded,
version-checked update, and every edge an operator action drives inserts its
`provider_task_actions` row in the same transaction.

| From | To | Trigger or operator action | Preconditions / CAS |
|---|---|---|---|
| (new row) | `queued` | admission (rule 1) | no row with this `operation_key` and no active row (§4) for the owner slot (else 409); `spend.reserve` succeeds and sets `reservation_held`, one transaction |
| `queued` | `abandoned` | `abandon`, the cancel before send (the `close` paragraph above) | `status='queued'`; no live claim (a live claim means a worker is about to reach the marker: the request is rejected and the operator retries); no POST has ever been sent and no `request_id` can exist; reservation released and `reservation_held` cleared in the same transaction; no draw. A worker whose marker update arrives later finds the status no longer `queued`, changes zero rows and exits benignly; once the marker has landed `abandon` is refused and `close` is the action |
| `queued` | `submit_started` | the marker (rule 3) | `status='queued'`, our `claim_owner`, live lease (`claim_expires_at > now`) |
| `queued` | `failed` (`input_changed`, `input_missing`, a failed CDN upload, a rejected path) | local failure before the marker (rule 3) | our claim; reservation released and `reservation_held` cleared in the same transaction |
| `submit_started` | `submitted` | acceptance commit (rule 7) | our submit claim; `request_id`, `accepted_at`, URLs and the draw in one commit; claim and `reservation_held` cleared |
| `submit_started` | `accepted_unrecorded` | acceptance commit without its draw (rule 7) | our submit claim; `request_id`, `accepted_at` and URLs committed; `reservation_held` kept |
| `submit_started` | `refused` | a submit 4xx (rule 6) | our submit claim; a gateway refusal (400, 401, 403, 404, 429) releases and clears `reservation_held` in the same transaction; a 422 or other 4xx keeps it |
| `submit_started` | `submit_unknown` | no usable answer to the POST (timeout, unreadable 2xx, 2xx without id, 5xx, transport error), or recovery finds the row stale (rule 4), including after an acceptance whose every local commit failed (rule 7) | our submit claim; or, for recovery, the claim stale and no `request_id` |
| `submit_unknown` | `submitted` | `adopt(request_id)` (rule 5) | no live claim; one status GET (§3 URL) answered 2xx; URLs built by the §3 rule; draw written and `reservation_held` cleared in the same transaction; poll enqueued |
| `submit_unknown` | `abandoned` | `abandon` (rule 5) | no live claim; reservation released and `reservation_held` cleared in the same transaction |
| `accepted_unrecorded` | `submitted` | recovery writes the draw (§7) | the draw, the status and the cleared `reservation_held` in one transaction; then the poll is enqueued |
| `submitted` | `fetching` | a status GET reports COMPLETED | CAS on `submitted`, no live claim; `fetch_deadline_at` set, `fetch_attempts` 0; `fal_fetch` enqueued after the commit |
| `submitted` | `failed` | a status GET reports a terminal failure | CAS on `submitted` |
| `submitted` | `cancelling` | `FAL_POLL_DEADLINE_SEC` reached, counted from `accepted_at` (rule 9) | cancel phase claim; `cancel_requested_at` set; cancel sent |
| `submitted` | `needs_reconciliation` | a status GET answers 404 or another 4xx other than 401, 403 and 429 (rule 9) | CAS on `submitted`; no cancel |
| `fetching` | `done` | output verified and moved onto `destination_path` (rule 10) | our fetch claim, re-checked before `os.replace` |
| `fetching` | `refused` | a 422 on the result (rule 10) | our fetch claim; the draw stays |
| `fetching` | `failed` (`result_gone`, `owner_gone`, or the job failed) | `ProviderResultGone` (the 5th failed attempt or `fetch_deadline_at` passed), a failed write into `destination_path` (§6), or a result reporting the job failed (`ProviderTaskFailed`) (rule 10) | our fetch claim; the draw stays |
| `cancelling` | `failed` | Fal reports a terminal state other than COMPLETED | CAS on `cancelling` |
| `cancelling` | `fetching` | Fal reports COMPLETED anyway | as `submitted → fetching`; the draw stays |
| `cancelling` | `needs_reconciliation` | the cancel is unconfirmed after `FAL_CANCEL_CONFIRM_SEC`, or a status GET answers 404 or another 4xx other than 401, 403 and 429 | CAS on `cancelling` |
| `needs_reconciliation` | `fetching` | the slow poll sees COMPLETED | as `submitted → fetching` |
| `needs_reconciliation` | `failed` | Fal confirms a terminal failure or cancellation | CAS on `needs_reconciliation` |
| `needs_reconciliation` | `cancelling` | `resume` | cancel phase claim; a fresh cancel; `cancel_requested_at` set again |
| every non-terminal status except `queued` and `closing`: `submit_started`, `submit_unknown`, `accepted_unrecorded`, `submitted`, `fetching`, `cancelling`, `needs_reconciliation` | `closing` | `close` | no live claim (for `submit_started`, the submit claim has expired; for `fetching`, between attempts); the generic CAS takes the claim, overwriting `claim_owner` |
| `closing` | `failed` (`operator_closed`) | the close's last transaction, or recovery finishing it (§7) | final cancel sent when a `request_id` is known; draw attempted in a SAVEPOINT; if it fails, `draw_pending` set and `reservation_held` kept |
| `refused` | `refused` | `close` (rule 6) | `reservation_held` true; no live claim; draw written and `reservation_held` cleared in one strict transaction |
| `refused` | `abandoned` | `abandon` (rule 6) | `reservation_held` true; no live claim; reservation released and `reservation_held` cleared in one transaction |

No other pair is legal. `done`, `failed`, `abandoned`, and a `refused` row without
`reservation_held`, have no outgoing edge, and nothing ever moves a row back to `queued` or
`submit_started`. Three events touch a row without changing its status, and the tests treat
them as such: an auth hold (401 or 403 on a status GET: `error_code=auth_rejected`, cleared
by the next 2xx); a failed fetch attempt (`fetch_attempts` incremented, rule 10); and the
draw backfill (§7), which writes a missing draw and clears `draw_pending` and
`reservation_held` on a row that owes one, terminal rows included.

## 5. Spend

The cap must hold across providers: "switching provider must not bypass the cap".

- **One daily total** in USD, the existing one (`KIE_DAILY_BUDGET_USD`, default 250), in the
  existing `kie_spend_days` row, which stays provider-free because it is the shared lock and
  the cap. The reservation and draw tables gain `provider VARCHAR(16) NOT NULL DEFAULT 'kie'`
  (additive; informational, for the readout).
- **One unit formula, at every reserve and draw site.** `reserve` and `draw_in_session` take
  credits and store milli-credits themselves (`spend._mc`, ×1000), so every Fal amount is
  `fal_credits = estimated_usd / settings.KIE_USD_PER_CREDIT`, passed as `credits=`; no Fal
  call site converts further. `estimated_usd` stays on the row for display and audit only.
- **Admission.** `spend.reserve(db, scope_key="fal:<task id>", credits=fal_credits,
  kind="fal_<family>")` inside the admission transaction (rule 4.1), which also sets
  `reservation_held`. The worst case is `estimated_usd = output_sec × tariff × overlap
  factor` (2.0 for any source until the windowing threshold is measured). A resolution
  without a tariff (reframe 1080p: "no rate listed") is refused before anything is reserved.
  `reserve` flushes before taking the day lock, as it does today, and nothing may take
  another lock between it and the commit.
- **Release on refusal, only when provably nothing ran.** A gateway refusal (rule 4.6: 400,
  401, 403, 404, 429) calls `spend.reserve(db, scope_key, 0, kind=..., release_unused=True)`,
  which lowers the reservation to what was consumed (the same path Retry Run uses). Without
  it such a refusal would count its whole worst case forever. **Every release** (a gateway
  refusal, `abandon` of a `queued`, `submit_unknown` or held `refused` row, a local failure
  before the marker) runs in the SAME transaction as its status CAS: `reserve` works in the
  caller's session and never commits. Only `record_draw` has a separate session, and it stays
  for the existing kie.ai callers. A 422 or other possibly-billed refusal keeps the
  reservation until the operator's `close` converts it into the task's draw or `abandon`
  releases it (rule 4.6).
- **Strict draw.** New `spend.draw_in_session(db, scope_key, call_key, credits, kind,
  provider)`, called with the same `fal_credits` as the reservation: it flushes the caller's
  pending changes first (so the task row's lock is taken before the day lock, the order
  `reserve` already uses), then calls `_apply_draw` and **raises** on failure. `record_draw`
  is unchanged for existing callers. **One task, one draw, one key:** the call key is
  `fal:task:<task id>`. A task is exactly one POST, so it has exactly one draw, and its key
  is known before Fal ever answers; it does not depend on the `request_id`. Every path that
  writes a draw (the acceptance commit, the `accepted_unrecorded` recovery, `adopt`, `close`,
  the `close` of a held `refused` row, the backfill sweep) uses this same key, so
  `_apply_draw`'s existing deduplication by `call_key` makes a second draw for the same task
  impossible, whichever order those paths run in. The prefix keeps it clear of kie.ai task
  ids under the global unique constraint; the `request_id` is stored on the row, never in the
  key. `close` of a non-terminal row is the one path that **attempts** the draw without
  requiring it (rule 4.5): it never releases, so the worst-case reservation carries the
  charge until the backfill sweep writes the draw. Every other path is strict.
- **The ledger is the source of truth; `reservation_held` and `draw_pending` are copies.**
  `reservation_held` is set by the admission `spend.reserve` and only ever changed in the
  same transaction as the `spend.reserve(..., release_unused=True)` call or the draw write
  that justifies it; `draw_pending` is set only by a failed draw attempt and cleared only by
  the draw write. Whether a draw exists is always read from the ledger by call key.
- **Never settled.** The key cannot read usage, so draws stay estimates (`settled_at` NULL)
  and the readout labels them so. An optional reconciler with an ADMIN key may settle later,
  as a separate process with its own credential outside the api/worker environment (D4).
- A failed, timed-out, result-422 or result-gone request keeps its draw: Fal may have billed
  it. `abandon` of a `submit_unknown` row, of a held `refused` row, or of a `queued` task
  (cancel before send; no draw exists to keep), releases its reservation; `adopt` writes its
  draw; `close` writes it or leaves it pending, and never releases.

## 6. Inputs, outputs and data handling

- **Input delivery.** The worker uploads the local file to the Fal CDN (token + one POST) and
  passes the returned URL. Never a Drive link and never the public `/public/` route: Fal
  fetching from our nginx would re-open the stale-upstream window of CLAUDE.md mid-deploy.
- **No input copies are kept.** Bytes are read once, hashed and uploaded from memory (rule
  4.3), so the layer owns no input storage to cap, purge or race. (Revision 5 kept
  task-owned copies of the inputs; revision 6 removed them for exactly that reason.) The only
  local files a task writes are its downloaded result and the claim-scoped staging beside it
  (rule 4.8), both under the **consumer-provided destination**, a directory the consumer
  already owns and already purges with its own rows; it is frozen at admission as
  `destination_path` and nothing else is ever written to.

  **Consumer contract (AICP-251 and any later consumer).** A "409 on delete" rule is only
  real if it is a lock, and today's project and run deletion (`api_v2.py`) look only at `Run`
  rows, so they would never see a provider task. A consumer therefore must: (a) have a
  durable owner row for each task, with the destination under it, scoped to a project or to a
  run; (b) take **one lock object per scope**, the same in admission and in deletion: for a
  project-scoped owner `SELECT ... FROM video_projects WHERE id = :pid FOR UPDATE`, taken by
  admission and by `delete_project`; for a run-scoped owner the `runs` row
  (`_lock_run_for_result_mutation`), taken by admission (after the project row, so that
  `delete_project`'s project lock covers its runs' tasks too) and by `delete_run`; (c) under
  that lock, admission inserts the task and deletion refuses with 409 while any **active**
  task (§4: `submit_unknown` and a held `refused` row included; a possibly billed request
  still owns its destination) exists for the project or the run. That serializes the two, so
  a deletion cannot slip between an admission check and its insert. After the lock is
  released the fetch phase can still find the destination gone (an operator deleting files by
  hand, a disk fault): a failed write into `destination_path` ends the task
  `failed(owner_gone)` and the draw stays, because Fal accepted the request.
  `owner_project_id`, immutable on the row, is what the audit trail keys on. **Lock order:
  the project row first, then `runs` rows in ascending id**, the order
  `_lock_runs_for_result_admission` already uses. It inverts nothing: no code locks a
  `video_projects` row today (checked on `5c1080d` and main), `_lock_run_for_result_mutation`
  locks only the run, the assistant locks its conversation and then a run, and AICP-328's
  sweep uses `SKIP LOCKED`; a Fal admission never runs while holding a run lock.

  The two delete paths that exist today are **required call sites** of (b) and (c):
  - **`delete_project`** today locks only the project's `Run` rows
    (`_lock_runs_for_result_admission`), 409s only for `ProjectStatus.analyzing` or a run in
    `_ACTIVE_RUN_STATUSES`, then `db.delete(project)`, commit,
    `shutil.rmtree(project_dir(pid))`. It must lock the project row first and, under that and
    the run locks, 409 while any active task has `owner_project_id = pid`, **before
    `db.delete` and before `rmtree`**.
  - **`delete_run`** has the same shape (`_lock_run_for_result_mutation`, `run_dir`). Under
    that lock it must 409 while an active task's owner row is that run or belongs to it,
    **before `db.delete` and before `rmtree`**.

  Two further obligations: (d) **the destination is never regenerable media.** The disk
  auto-cleanup that main gained after `5c1080d` (AICP-328, `disk_cleanup.py`, run by
  `tasks.run_disk_cleanup_tick`; its paths verified on main, since this worktree predates it)
  deletes the whole `projects/<project>/runs/<run>/clips/` directory of a `done` run
  untouched for `RUN_CLIPS_RETENTION_DAYS` (default 30), and
  `projects/<project>/tail_edits/<edit>/body.mp4` of a finished tail edit. A
  `destination_path` must not be, or lie under, either path, or the sweep would delete a paid
  output as a free re-cut; its docstring lists `results/` among what it never touches. (e)
  **Retry only from a task that is no longer active**; while the slot's task is active the
  index refuses the admission with 409 anyway (rule 4.1). Parallel tasks for one owner need
  distinct `owner_slot`s.
- **Headers, by request type** (as exercised by the probe and by `fal-client` 1.0.3):
  uploads send `X-Fal-Object-Lifecycle` and `X-Fal-Object-Lifecycle-Preference`
  (`{"expiration_duration_seconds": 3600}`); every submit sends `X-Fal-Store-IO: 0`,
  `X-Fal-Object-Lifecycle-Preference` and `X-Fal-No-Retry: 1`. Stage 1 includes a free
  retention check: upload a tiny file with a 60 s lifecycle and confirm it is gone at 90 s.
- **Outputs** are downloaded as soon as the result is read. The local copy is the record;
  Fal's output media URL is never stored for later use. `response_url` is stored, so a
  download that failed can read the result again for a fresh media URL (rule 4.10).
- **Customer media leaves the box.** Admission is per project: `FAL_ENABLED` (default false)
  and `FAL_PROJECT_IDS` (explicit allowlist; empty means none). The owner accepted Fal's
  terms on the same footing as kie.ai (D3); the protection is `X-Fal-Store-IO: 0` plus the
  short lifecycle above. No statement on training use was found.
- **`FAL_ENABLED=false` gates admission only.** Existing tasks keep being polled, fetched and
  recovered while `FAL_API_KEY` is present; turning the flag off in an incident must not
  orphan paid work.
- **Fal's internal retries are requested off** (`X-Fal-No-Retry: 1`). The guarantee is "at most
  one client submit per row"; the debit itself stays unobservable and is conservatively
  reserved. A transient runner failure surfaces as `failed` and the operator presses Retry, a
  new row (D2).

## 7. Recovery and the reconcile tick

`recovery.py` gets one more entry for `provider_tasks`. Every job it enqueues goes to
`QUEUE_PROCESS` with the task id as the first positional argument (rule 4.2), so the
"no live RQ job" checks below are `_live_run_ids` lookups that can actually see Fal jobs.

| Found | Action |
|---|---|
| `queued`, no live claim, no live RQ job, older than grace | re-enqueue `fal_submit` (no POST was sent) |
| `submit_started`, no `request_id`, claim stale | `submit_unknown` with the "may be billed" message |
| `submitted` / `cancelling`, no live RQ job | re-enqueue the poll (idempotent GET) |
| `fetching`, no live claim, no live RQ job | re-enqueue `fal_fetch`, which honors `fetch_attempts` and `fetch_deadline_at` (rule 4.10): past either limit it ends the row `failed(result_gone)`, draw kept |
| `needs_reconciliation`, no live RQ job | re-enqueue the slow poll; the operator's `resume` / `close` ends it |
| `accepted_unrecorded` | write the draw and clear `reservation_held`, move to `submitted`, then enqueue the poll (no POST); never polled before that |
| `closing` | finish the close: repeat the idempotent cancel when a `request_id` is known, then the last transaction (draw attempted, `failed(operator_closed)`, `draw_pending` if the draw failed); never polled |
| `submit_unknown` | nothing: parked for the operator's `adopt`, `abandon` or `close`; never re-dispatched, no draw written |
| `refused` with `reservation_held` | nothing: parked for the operator's `close` or `abandon` |
| any status, terminal included, with `draw_pending` set, or with a `request_id` and no draw under `fal:task:<task id>` in the ledger (other than `accepted_unrecorded`, above) | backfill the draw (idempotent by call key) and clear `draw_pending` and `reservation_held` in the same transaction; the status is unchanged |
| `done`, `failed`, `abandoned`, `refused` without `reservation_held` | nothing beyond the backfill row above |
| anything | **never a POST** |

**Do not add `provider_tasks` to `_DERIVED_CREATIVE_SPECS`.** That reaper marks an orphaned
row `failed` after 300 s with "Press Retry to run it again", which for a provider task would
invite a second paid POST for a request that may be running at Fal. Provider tasks are
recovered only by the entry above.

Deploy drain: `deploy/drain-check.sql` gets a `provider_tasks` line (`queued`,
`submit_started`, `accepted_unrecorded`, `submitted`, `fetching`, `cancelling`,
`needs_reconciliation`, `closing`, all of which must read 0 before containers are stopped),
with a test per status. These are exactly the non-terminal statuses of the transition map
(§4) except `submit_unknown`, which no job works on and which can wait for the operator for
days; a terminal row (a held `refused` one included) never counts. Without the line the
existing all-zero output would let an operator stop workers mid-submit. **A
`needs_reconciliation` row can park for a long time** (a cancel Fal never confirmed, a 404
on a status GET) and keeps the line non-zero until the operator acts: that is not a stuck
deploy. `close` clears it once the close has finished; `resume` only once Fal confirms the
fresh cancel (`cancelling` is drained too). Webhooks are **not** in Stage 1: a
`fal_webhook` URL carrying our task id would let a lost-response request report its own
`request_id` back and resolve `submit_unknown` without the operator, at the cost of a public
endpoint, HMAC handling and a dedupe path (D1).

## 8. Fail-closed registry (Stage 2 prerequisite)

Stage 1 puts no Fal key into `run_model_enum`; reframe jobs never pass through `Run.model`.
Before a Fal model can be a run model (Stage 2): a new `require_spec` raises on an unknown
key at the recipe entry (`spec_for` keeps its fallback for display paths only); the provider
is derived from the key and bound from the Run at job entry, never from a resubmit recipe; and
the moderation question has an answer.

## 9. Rollout, migration and ordering

- Everything is behind `FAL_ENABLED=false`; with it off, admission is closed and the
  migration is inert.
- **One additive Alembic revision**: `provider_tasks` and `provider_task_actions` (with its
  insert-only triggers, §4), plus the `provider` column on the two spend tables.
  `provider_tasks` has every column of §4, including `owner_slot` (`VARCHAR NOT NULL DEFAULT
  'main'`), `accepted_at`, `draw_pending` (`BOOLEAN NOT NULL DEFAULT false`),
  `fetch_attempts` (`INTEGER NOT NULL DEFAULT 0`) and `fetch_deadline_at`, the status CHECK
  over the thirteen statuses, the two unique constraints, and
  `uq_provider_tasks_active_per_owner_slot` on `(owner_kind, owner_id, owner_slot)` over the
  active predicate, its WHERE clause byte-identical in the migration, `models.py` and
  `PROVIDER_TASK_ACTIVE_WHERE`. **Downgrade refuses** while ANY `provider_tasks` row exists,
  terminal or not: an accepted, refused or failed task may have been billed, and the row
  (with its `fal:task:<task id>` ledger link, its operator actions and the `provider` labels)
  is the only record tying the request to its endpoint and owner. A downgrade is therefore
  possible only for a feature that was never used; once tasks exist, the tables and columns
  stay and the code is rolled back by flag, not by schema.
- **Ordering with AICP-293.** `feat/knowledge-importer` already contains
  `feat/knowledge-storage`, and its only migration, `k1a7c3e9f205`, descends from the
  production head `j0e5f2a7b914`. The Stage 1 implementation branch is therefore cut from the
  final AICP-293 branch, and this revision's `down_revision` is `k1a7c3e9f205` (or whatever
  the head is when AICP-293 lands). The metadata import in `models.py`, the router in
  `main.py` and the reconcile-tick area of `tasks.py` are resolved on that branch, not at
  release.
- **`config.py`**: a `FAL_*` block after the `KIE_*` settings, away from the `ASSISTANT_*`
  and `KNOWLEDGE_*` blocks. `.env.example` gets `FAL_API_KEY=`. The block holds:

  | Setting | Default | Meaning |
  |---|---|---|
  | `FAL_API_KEY` | empty | API-scoped key; polling, fetching and recovery run while it is present (§6) |
  | `FAL_ENABLED` | false | gates admission only (§6) |
  | `FAL_PROJECT_IDS` | empty (none) | explicit per-project allowlist for admission (§6); since 2026-10-29 a `reframe` project is admitted without it (`project_allowed()`, AICP-251) |
  | `FAL_MAX_INPUT_BYTES` | 90 MB | per-input cap of the single-POST upload path (rule 4.1, rule 4.3) |
  | `FAL_JOB_TIMEOUT` | 1800 s | RQ timeout of every claim-holding Fal job (rule 4.3) |
  | `FAL_CLAIM_LEASE_SEC` | `FAL_JOB_TIMEOUT` + 300 s | phase-claim lease; start-up validation rejects a lease shorter than the timeout plus a margin (rule 4.3) |
  | `FAL_POLL_DEADLINE_SEC` | 3600 s | bound on `submitted`, counted from `accepted_at`; reaching it moves the row to `cancelling` (rule 4.9) |
  | `FAL_CANCEL_CONFIRM_SEC` | 21600 s (6 h) | bound on an unconfirmed cancel, counted from `cancel_requested_at`; reaching it moves `cancelling` to `needs_reconciliation` (rule 4.9) |
- Release follows CLAUDE.md: Drive backup, migrate first, `--scale worker=2`, assistant
  overlays, nginx restart. Any paid canary after release is a separate authorization.

## 10. Verification plan

Unit (no network, mocked transport): the six submit outcomes (accepted, 2xx unreadable, 2xx
without id, 5xx, 4xx with an echoed key, transport error); every edge of the transition map
(§4) legal and every other pair illegal; the CHECK, the index's WHERE clause,
`PROVIDER_TASK_NON_TERMINAL_STATUSES`, `PROVIDER_TASK_ACTIVE_WHERE` and the drain line
compared against the map; the drain line counting `queued`, `submit_started`,
`accepted_unrecorded`, `submitted`, `fetching`, `cancelling`, `needs_reconciliation` and
`closing`, never `submit_unknown`, `done`, `refused`, `failed` or `abandoned`; two concurrent
starts of the same `operation_key` (one row, one reservation, no 500); a new `operation_key`
for an owner slot whose task is active (409, nothing reserved), admitted once that task is no
longer active, also across providers; a changed owner setting between admission and dispatch
(the frozen spec is what is sent); a local file whose bytes changed, or that was deleted,
between admission and dispatch (local failure, reservation released, nothing sent); an input
longer than `FAL_MAX_INPUT_BYTES`, read with a bounded read; an input path outside the data
directory (rejected at admission and at dispatch); a consumer admission racing a deletion of
the same owner, project- and run-scoped (one wins, the other 409s or is refused, never a paid
task with no destination; no deadlock with the project-then-runs lock order); a result
fetched into a vanished destination (`failed(owner_gone)`, draw kept); an acceptance whose
every local commit fails (the `request_id` logged at ERROR as a hint, the row
`submit_unknown` once its claim is stale, the reservation still held); two workers racing the
claim (one wins, the loser exits benignly, nothing sent twice); the lease expiring during an
upload and immediately before the POST (the stale worker's marker update changes zero rows,
and its pre-POST re-check of status, token and lease fails after a `close`); `queued`
re-dispatch skipped while the original RQ job is still pending; reservation and draw
idempotency; `BudgetExceeded` leaves no row; a refusal releases its reservation in the
transaction of its status change; unknown endpoint and untariffed resolution raise; the queue
URLs of §3 built from owner/app, never the full endpoint path.

Polls, operator actions and consumers (mocked transport): 429, 5xx and transport errors back
off without a cancel; 401/403 hold with `error_code=auth_rejected` and no status change until
a 2xx clears it; 404 and other 4xx move `submitted` and `cancelling` to
`needs_reconciliation` without a cancel; a cancel only on `FAL_POLL_DEADLINE_SEC` (from
`accepted_at`, so a late adoption is not cancelled at once), `resume` or `close`;
`FAL_CANCEL_CONFIRM_SEC` moving `cancelling` to `needs_reconciliation`; every COMPLETED edge
enqueueing `fal_fetch` after its commit; `failed(result_gone)` on the 5th failed attempt and
on a passed `fetch_deadline_at`, never earlier, draw kept; a stale claim cannot `os.replace`
its staging file. `close` from each of the seven closable non-terminal statuses (rejected
under another live claim), including with the draw write failing (`failed(operator_closed)`,
`draw_pending`, day total unchanged, exactly one draw after the backfill), and of a
`submit_unknown` row (no cancel sent); `close` of a held `refused` row (one draw, nothing
released, still `refused`; a failing draw changes nothing); `abandon` of a held `refused`
row, of `submit_unknown` and of a `queued` task (reservation released, no draw, the worker's
later marker update changes zero rows; `abandon` after the marker is refused and `close` is
the action; `close` is not offered on `queued`); `adopt` with a 2xx and with a 404 (refused,
reason recorded); every operator action writing exactly one `provider_task_actions` row in
its own transaction, and an UPDATE or DELETE of that table rejected on both dialects; the
flags never change without their ledger write. `delete_project` and `delete_run` 409 under
their lock while an active task (`submit_unknown` and a held `refused` row included) exists,
before any delete; Fal jobs on `QUEUE_PROCESS` are seen by `_live_run_ids`;
`_DERIVED_CREATIVE_SPECS` has no `provider_tasks` entry.

Crash windows (the real risk), each ending in `queued` re-dispatched, `submit_unknown`, a
re-polled `submitted`, or a re-fetched `fetching`, never a second POST: kill after the
admission commit and before the enqueue; after the claim; after the marker and before the
POST; after the POST and before the acceptance commit (including a failing local commit);
after COMPLETED and before `fal_fetch` is enqueued; during a download; during `cancelling`;
after the CAS to `closing` and before its last transaction (recovery finishes it, with at
most one draw).

Spend: the draw and `request_id` land in one commit; a missing draw is backfilled; call keys
cannot collide with kie.ai task ids; the daily cap still refuses with Fal and kie.ai draws
mixed; **units**: a 1.00 USD Fal estimate, reserved and drawn as `1.00 / KIE_USD_PER_CREDIT`
credits, raises the day's charged total by 1.00 USD (±1 cent) in `spend.today_summary` when
mixed with kie.ai draws.

Regression: the existing suite stays green with the flag off; the kie request path is
byte-identical (the migration default makes existing spend rows `provider='kie'`).

Live: `scripts/fal_probe.py` stages 0-1 are the free smoke test; the retention check above is
free; paid canaries remain manual and capped.

### Known residuals (not fixed by this document)

- **The overlap factor 2.0 is unmeasured** and overlap billing is docs-only (§2.5); nothing
  yet shows that 2.0 bounds the real cost.
- **Draws are never settled** (D4); every Fal draw stays an estimate.
- **Cancel and the status-GET 401/403/429/404 behavior are docs-only** (§2.9): the probe
  never sent a cancel, and saw only the key check's 404 and 401 on an invented id.
- **An empty Fal balance is unmodelled**: credits are prepaid and no balance endpoint or
  low-balance behavior was found.
- **A database outage at the acceptance commit leaves `submit_unknown` for the operator**
  (rule 4.7, D6): with no receipt journal, the only trace of the accepted `request_id` is an
  ERROR log hint; the reservation stays held, and the operator checks the Fal dashboard, then
  `adopt`s, `abandon`s or `close`s.
- **The spend readout says kie.ai.** In `templates/v2/partials/credit_balance.html` the daily
  total's tooltip begins "kie.ai spend committed today", and `spend.BudgetExceeded`'s message
  begins "Daily kie.ai budget reached:"; Fal estimates count in that total and are refused
  with that message, so they show under a kie.ai label until those strings are relabelled.

## 11. Decisions for the owner

| | Question | Recommendation |
|---|---|---|
| D1 | Add the webhook recovery path later? | Not in Stage 1; revisit if `submit_unknown` ever happens in production. |
| D2 | Ask Fal to disable its internal retries (at most one client submit per row; the debit stays unobservable and is conservatively reserved)? | Off, with an operator Retry that creates a new task. |
| D3 | Fal retention, region and training terms for customer media. | Decided by the owner on 2026-10-27: no separate terms review, the same standard as kie.ai. Every submit sends `X-Fal-Store-IO: 0`, uploads and outputs get a short CDN lifecycle, and outputs are downloaded as soon as the result is read. |
| D4 | Who holds an ADMIN key for settling draws? | Nobody in Stage 1; estimates only. |
| D5 | Generic `provider_tasks` table, or columns on each consumer row? | Generic table: AICP-251 is the first of several consumers, and the double-billing machinery lives in one place. |
| D6 | A durable receipt journal for the acceptance window? | Not in Stage 1 (removed in revision 11; rule 4.7); revisit only if `submit_unknown` with a database-outage cause ever happens in production. |

## Review record

The rows below are kept as history. **Revision 11 removed the receipt journal** that
revisions 7 to 10 built (rule 4.7, D6), so every receipt-related resolution in the earlier
tables is superseded.

**Revision 1 → 2** (GPT-5.6 Terra, Codex, reasoning effort high; each finding checked against
the code before accepting).

| Finding | Resolution |
|---|---|
| BLOCKER: no operation uniqueness; two starts reserve and POST twice | `operation_key`, `UNIQUE (provider, operation_key)`; Retry mints a new key (§4). |
| BLOCKER: `submitting` conflated "queued" and "POST may have happened" | split into `queued` and `submit_started`; the tick re-dispatches `queued` safely (§4.2, §7). |
| BLOCKER: a refusal never releases its reservation (`_recharge` keeps `max(reserved, consumed)`) | explicit release through `reserve(..., 0, release_unused=True)` in the refusal transaction (§5). |
| BLOCKER: `record_draw` is a separate, error-swallowing session, so `request_id` and the draw are not atomic | `draw_in_session` (flush first, then the day lock, raises) committed with `request_id`; recovery backfills a missing draw (§4.7, §5). |
| BLOCKER: `transition()` is not a compare-and-set | lease claim + version column + claim-scoped staging, the full range-attempt protocol (§4.3, §4.8, §4.11). |
| BLOCKER: downgrade drops the only record of billed work | downgrade refuses while any row is non-terminal or unknown (§9). |
| MAJOR: best-effort cancel then terminal | `cancelling` state; keep polling until Fal is terminal (§4.9). |
| MAJOR: `FAL_ENABLED=false` would stop polling of paid tasks | the flag gates admission only (§6). |
| MAJOR: `provider` column does not protect the globally unique `call_key` | call key namespaced; since revision 9 it is `fal:task:<task id>` rather than a key built from the `request_id` (§5). |
| MAJOR: AICP-293 ordering stale (importer contains storage; one migration) | branch from the final AICP-293 branch, `down_revision=k1a7c3e9f205` (§9). |
| MAJOR: the `submit_billed_at` comparison omitted the claim | rule 4.3 adopts the full protocol. |
| MINOR: lifecycle headers contradicted the probe | headers specified per request type; a free retention check added (§6). |

**Revision 2 → 3** (second Terra pass; each finding again checked against the code).

| Finding | Resolution |
|---|---|
| BLOCKER: admission stored only a hash, so the dispatched request could differ from what was reserved | the request spec, input hashes, duration, resolution and estimate are frozen at admission; dispatch only materializes URLs; a changed file is a refusal (§4). |
| BLOCKER: downgrade still allowed dropping terminal rows, the only receipt of possibly billed work | downgrade refuses while any `provider_tasks` row exists (§9). |
| MAJOR: the claim `UPDATE` did not bump `version_id` and the marker was not conditioned on the claim token | acquire and marker are guarded `UPDATE`s that change exactly one row and increment `version_id`; the marker requires status `queued`, the token and a live lease (§4.3). |
| MAJOR: no lease duration tied to the RQ job timeout | `FAL_CLAIM_LEASE_SEC` at least the `fal_submit` timeout plus a margin, enforced at startup; tests for expiry during upload and before the POST (§4.3, §10). |
| MAJOR: the 6 h cancel path turned a known `request_id` into `submit_unknown` | new `needs_reconciliation` state keeps the id and the draw and keeps polling; `resume` / `close` (§4). |
| MAJOR: `queued` re-dispatch checked only the claim, not RQ job liveness | re-dispatch requires no live claim AND no live RQ job (`_live_run_ids`) (§4.2, §7). |
| MAJOR: check-then-insert races on the unique key | insert under a savepoint; on conflict roll back to it and return the winner (§4.1). |
| MAJOR: `deploy/drain-check.sql` has no `provider_tasks` line | a line per active status, with a test each (§7). |
| MINOR: `transition()` dispatches only on enum classes | a `ProviderTaskStatus` enum and transition table in `state_machine.py` (§4.11). |
| MINOR: "one POST, one bill" overstated the evidence | restated as at most one client submit per row, debit unobservable and conservatively reserved (§6). |

**Revision 3 → 4** (third Terra pass; each finding again checked against the code).

| Finding | Resolution |
|---|---|
| BLOCKER: every submit 4xx released its reservation and left no evidence, but a 422 may be billed | only gateway refusals (400, 401, 403, 404, 429) release; a 422 or other possibly-billed 4xx keeps the reservation until `abandon` (§4.6, §5). |
| MAJOR: a known accepted request could end as `submit_unknown` when the draw failed | new `accepted_unrecorded` state stores the `request_id` and URLs, recovery writes the draw and moves on without a POST; `submit_unknown` only for responses never seen (§4.7, §7). |
| MAJOR: only acquire and marker had guarded updates; the submit claim was not cleared at acceptance | one CAS primitive with `RETURNING` for every phase claim; the acceptance commit clears the submit claim (§4.3). |
| MAJOR: raw `version_id+1` updates vs the ORM identity map | every CAS refreshes the object (or runs in its own session) before any ORM mutation (§4.3). |
| MAJOR: the hash verified a mutable path, not the bytes uploaded | inputs are snapshotted into task-owned storage at admission and only the snapshot is uploaded (§4.1). |
| MAJOR: the lease floor was tied to an undefined timeout and mis-described the precedent | `FAL_SUBMIT_JOB_TIMEOUT` defined once in the `FAL_*` config block, used by `tasks.py`, with start-up validation of the lease (§4.3). |
| MAJOR: `needs_reconciliation` had no transitions and "close" could abandon unpolled paid work | explicit transitions to `fetching`, `failed`, `cancelling`; `close` is an auditable operator abandonment with a final cancel and an explicit acceptance of the risk (§4). |
| MINOR: D2 repeated "one POST, one bill" | reworded to match §6 (§11). |

**Revision 4 → 5** (fourth Terra pass: no BLOCKER; each finding again checked against the code).

| Finding | Resolution |
|---|---|
| MAJOR: `accepted_unrecorded` could be polled to a terminal state before its draw existed | the state is not pollable; recovery writes the draw, moves to `submitted`, then enqueues the poll (§4.7, §7). |
| MAJOR: snapshot retention was not a storage contract (refusals, failures, `submit_unknown`, quota, deleter) | `storage.provider_task_dir`, atomic copy with rollback, per-task and global byte limits, retention per status, a durable `snapshot_state`, a recovery purge, named deleters (§6). |
| MAJOR: a 422-retained reservation had no legal `abandon` | `abandon` is allowed from such a `refused` row as well as `submit_unknown`, one transaction, audited (§4.6). |
| MAJOR: lease validated against the submit timeout only | one `FAL_JOB_TIMEOUT` for every claim-holding Fal job, lease validated against it (§4.3). |
| MAJOR: `close` raced a poller and had a crash window | durable `closing` state taken by CAS only with no live claim, intent recorded before the cancel, recovery finishes it, never polled (§4). |
| MINOR: drain list omitted `accepted_unrecorded` | added, with `closing` (§7). |

**Revision 5 → 6** (fifth Terra pass: no BLOCKER, 4 MAJOR, all in the snapshot store that
revision 5 had introduced to answer the "hash verified a mutable path" finding).

| Finding | Resolution |
|---|---|
| MAJOR: purging row-less directories races admission, which copies before the row commits | removed with the store: nothing is copied (§4.3, §6). |
| MAJOR: the global byte check is not an atomic capacity reservation | removed with the store: no input storage remains to bound (§6). |
| MAJOR: no durable purge deadline or `purge_pending` state | removed with the store (§6). |
| MAJOR: a generic soft owner cannot establish project deletion; media is not required to live in a purgeable task directory | immutable `owner_project_id`; outputs live in the consumer's own destination; consumers 409 on deleting an owner or destination while a task is active; the fetch phase checks the destination and discards into `failed(owner_gone)` (§6). |

The simpler answer to the original finding (revision 4, "the hash verified a path, not the
uploaded bytes") is to read each input once, hash those bytes, compare with the frozen hash
and upload them from memory (§4.3). It closes the same window with no storage, quota, purge
state or recovery pass, at the cost of a 90 MB per-input cap for Stage 1 (the single-POST
upload path of `fal-client`); larger inputs need multipart and are a later stage. Rows
"Hashing a mutable input" and "Snapshot retention" in the 4 → 5 table above record the
intermediate design and are superseded by this one.

**Revision 6 → 7** (sixth Terra pass: no BLOCKER; each finding checked against the code).

| Finding | Resolution |
|---|---|
| MAJOR: the destination was not an immutable task attribute, so recovery could not know where to write | `destination_path`, frozen and contained at admission; recovery, staging and `owner_gone` use only it (§4, §4.1). |
| MAJOR: "consumers 409" was not an implementable lock; deletion only looks at `Run` rows | an explicit consumer contract: a durable owner row, locked `FOR UPDATE` by both admission and deletion in the same transaction, deletion checking non-terminal tasks under the lock; residual destination loss ends as `failed(owner_gone)` with the draw kept (§6). |
| MAJOR: accepted response with both commits failing left only a log line | an fsync'd, append-only receipt file under `data/provider_receipts/` written before any commit; recovery reconciles it; `abandon` is refused while a receipt exists; the dual-failure residual is stated as unavoidable (§4.5, §4.7). |
| MINOR: the 90 MB cap vs real sources; unbounded read | measured against 124 real sources (median 34 MB, p90 92 MB, max 167 MB; one in ten does not fit), eligibility exposed before the workflow, multipart an explicit follow-up, bounded read of at most the cap plus one byte, sequential uploads (§4.3). |
| MINOR: `input_refs` paths lacked the containment guard | `storage.ensure_under_data_dir()` at admission and at dispatch (§4.1). |

**Revision 7 → 8** (seventh Terra pass: no BLOCKER; every remaining finding was in the
receipt journal that revision 7 had added).

| Finding | Resolution |
|---|---|
| MAJOR: the receipt was not a defined crash-safe journal (writer, serialization, framing, torn tail) | `write_receipt` is the sole writer; length + CRC framed records; exclusive `flock`; `fsync` of file and directory; a torn tail is ignored and reported; tests for concurrent appends and every truncation point (§4.7, §10). |
| MAJOR: "only a dual DB and disk failure loses a billed request" was false | the POST-to-first-durable-write window is stated as unavoidable; `submit_unknown` never releases by itself; `abandon` needs an explicit confirmation; `close` keeps the charge as a task-keyed provisional draw (§4.5, §4.7). |
| MAJOR: refusing `abandon` while a receipt exists had no terminal escape | `close` is available for `submit_unknown`, `accepted_unrecorded` and receipt-bearing rows; it never releases, writes a provisional draw, cancels when an id is known, and ends the row (§4.5). |
| MINOR: receipt retention and rotation absent | day segments, immutable after rollover, `receipt_applied_at`, a segment deleted only when every record is applied or terminal (§4.7). |
| MINOR: receipt directory ownership and modes unspecified | `10001:10001`, `0700`, files `0600`, verified at deploy; the media root's modes are not copied (§4.7). |

**Revision 8 → 9** (eighth Terra pass: no BLOCKER; the remaining findings were the seams
between `close`, receipt recovery and the ledger).

| Finding | Resolution |
|---|---|
| MAJOR: `close` wrote a task-keyed draw while recovery wrote one keyed by the `request_id`, so one task could end with two draws | one canonical key per task, `fal:task:<task id>`, used by every path that writes a draw; `_apply_draw`'s deduplication by `call_key` then makes a second draw impossible in any order (§5). |
| MAJOR: receipt reconciliation had no rule for a row already `closing` or terminal | explicit per-status application rules under the CAS of rule 3; a `closing` or terminal row is never reopened or polled, the receipt is attached as audit data only; `close` applies any pending receipt first (§4.5, §4.7). |
| MAJOR: a segment could be deleted because its task was terminal, destroying the only request id | the terminal exception is removed; a segment is deleted only when every record is applied (id on the row, draw disposition settled) and at least 30 days old (§4.7). |
| MINOR: the repair of a torn tail was not specified | under the append lock, truncate to the last valid frame and `fsync` before appending; tests append after every truncation point (§4.7, §10). |

**Amendment to revision 10 (owner's request, before the tenth-revision review passes).**

| Finding | Resolution |
|---|---|
| `close` from `queued` would write a worst-case draw for a task that never sent anything, charging the daily cap for work that does not exist | `close` is not offered on `queued`; `abandon` is the cancel before send (reservation released, no draw), with a new `queued → abandoned` edge in the transition map and a race rule against the worker's marker (§4). |

Note: Terra also reported that its own attempt to run the CLAUDE.md second-opinion CLIs from
inside its sandbox failed (read-only filesystem, no Cursor login). That is a property of its
sandbox, not a finding about this design.

**Revision 9 → 10** (ninth Terra pass, 0 / 1 / 2, and an independent Cursor Grok 4.6 pass,
0 / 5 / 8; no BLOCKER; every finding checked against the code before accepting).

| Finding | Resolution |
|---|---|
| MAJOR (Terra): `close` was the terminal escape for a failing ledger write, yet required that write to succeed | `close` attempts the draw inside a SAVEPOINT and does not require it; it never releases, so the worst-case reservation keeps the cap whole; a failed draw leaves `failed(operator_closed)` with `reservation_held` true and the new `draw_pending`, and the backfill sweep writes it (§4, rule 4.5, §5, §7). |
| MAJOR (Grok): every status 4xx moved a billed in-flight task to `cancelling` | status GETs classified by response: 2xx normal; 429, 5xx, transport transient; 401/403 an `auth_rejected` hold; 404 and other 4xx `needs_reconciliation`; a cancel only for the deadline, `resume` or `close`; the docs-only scope noted (§2.9, rule 4.9). |
| MAJOR (Grok): the consumer contract was not tied to the real delete paths | `delete_project` and `delete_run` named as required call sites that 409 on a non-terminal task under the lock, before `db.delete` and `rmtree`; destinations kept out of the paths AICP-328's disk cleanup deletes (§2, §6). |
| MAJOR (Grok): Retry while a task was active admitted a second POST | `owner_slot` (default `'main'`) and a partial unique index on `(provider, owner_kind, owner_id, owner_slot)` over the non-terminal statuses; a second admission for an active slot is a 409; Retry only from a terminal task (§4, rule 4.1, §6, §9, §10). |
| MAJOR (Grok): `submitted` and `fetching` had no deadline, no universal escape, and `ProviderResultGone` no transition | `FAL_POLL_DEADLINE_SEC` (3600 s from the new `accepted_at`); `close` legal from every non-terminal status except `closing`; `response_url` kept, media URLs never stored; bounded download attempts over at least 10 minutes end `failed(result_gone)` (§3, §4, rules 4.9 and 4.10, §9). |
| MAJOR (Grok): a submit 422 kept its reservation but offered only `abandon`, which releases it | `close` of a held `refused` row converts the reservation into the task's draw and leaves the row `refused`; `abandon` stays the explicit "nothing exists" release (rule 4.6, §5). |
| MINOR (Terra): `ProviderTaskStatus` transition table required but never given | an exact transition map with every status, operator action and terminal status; the CHECK, the index, the drain list and the tests derive from it (§4 transition map, rule 4.11, §7, §10). |
| MINOR (Terra): "takes the day lock without flushing first" was imprecise | `_apply_draw` does not explicitly flush, and sessions are `autoflush=False`; `draw_in_session` keeps its explicit flush (§2). |
| MINOR (Grok): two recovery-table rows fused on one line | split; every table checked for a constant column count (§7). |
| MINOR (Grok): a new queue name would be invisible to `_live_run_ids` | every Fal job on `QUEUE_PROCESS`, sharing capacity with run processing (rule 4.2, §7). |
| MINOR (Grok): nothing kept `provider_tasks` out of the derived-creative reaper | an explicit warning against adding it to `_DERIVED_CREATIVE_SPECS` (§7). |
| MINOR (Grok): the generic CAS and the marker disagreed on an expired lease; download staging was described as a sibling `.part` | the marker's live-lease condition wins; staging is claim-scoped with `os.replace` after an ownership re-check and no sibling `.part` (rules 4.3, 4.8, 4.10). |
| MINOR (Grok): `adopt` trusted the operator's id | one status GET verifies it; a 404 refuses and records why; the poll is enqueued (rule 4.5). |
| MINOR (Grok): `FAL_JOB_TIMEOUT` and `FAL_CLAIM_LEASE_SEC` had no defaults | 1800 s and `FAL_JOB_TIMEOUT` + 300 s, in a `FAL_*` settings table (rule 4.3, §9). |
| MINOR (Grok): `reservation_held` duplicated ledger state | the ledger is the source of truth; the flag (and `draw_pending`) change only in the transaction of the ledger write that justifies them (§5). |
| MINOR: residual risks were scattered or unstated | a *Known residuals* list: the overlap factor, unsettled draws, docs-only cancel and status GETs, the empty balance, the kie.ai labels (§10). |

Also corrected while applying these: a `queued` row cannot carry a receipt (rule 4.7 said it
could), the `operation_key` now includes `owner_slot` so two slots of one owner cannot
collide, and a hash mismatch at dispatch is called a local failure, as rule 4.3 already
treated it.

**Revision 10 → 11** (tenth Terra pass, 2 / 3 / 1, and the second Grok pass, 0 / 6 / 7; every
finding checked against the code; the owner chose to simplify first).

| Item | Finding | Resolution |
|---|---|---|
| A | owner's decision: the receipt journal covered only "Fal accepted and the database failed at that moment" and drew most findings of the last four passes | journal removed from Stage 1: no `write_receipt`, `receipt_applied_at`, receipt edges or tests; a failed acceptance commit is retried, then logged at ERROR as a hint and left to become `submit_unknown`, reservation held (rule 4.7, §4, §7, §10, D6, *Known residuals*) |
| B1 | BLOCKER (both): §5 converted Fal USD to milli-credits and passed USD as `credits=` | one formula, `fal_credits = estimated_usd / KIE_USD_PER_CREDIT`, at every reserve and draw site; `estimated_usd` display only; a 1.00 USD units test (§5, §10) |
| B2 | BLOCKER (Terra): a held `refused` row was terminal, so the slot index, Retry and the delete check ignored it | "active" = non-terminal OR (`refused` AND `reservation_held`), one predicate for the index, Retry and deletes (§4, rules 4.6 and 4.11, §6, §9, §10) |
| B3 | MAJOR (Grok): admission and `delete_project` named two different lock objects | one lock per scope: the `video_projects` row (project-scoped) or the `runs` row (run-scoped, after the project row); order project then runs; both deletes 409 on an active task before `db.delete` and `rmtree` (§2, §6) |
| B4 | MAJOR (both): recovery re-enqueued the poll for `fetching`, but `fal_fetch` reads and downloads | `fetching` goes to `fal_fetch` in the recovery table; every COMPLETED edge enqueues `fal_fetch` after its commit (rules 4.8 to 4.10, map, §7) |
| B5 | MAJOR (both): bounded fetch attempts had no durable state | `fetch_attempts` and `fetch_deadline_at`; 5 attempts with backoff or the deadline end `failed(result_gone)` (§4, rule 4.10, map, §7, §9, §10) |
| B6 | MAJOR (Grok): `adopt` built URLs "from the endpoint and the id" | the queue-addressing rule (owner/app = first two segments; full path 405s) written once and used by `adopt` (§3, rule 4.5) |
| B7 | MAJOR (Grok): §5 released a refusal "in its own transaction", the map in the same one | every release runs in the transaction of its status CAS; `record_draw`'s own session only for kie.ai callers (§5) |
| B8 | MAJOR (Terra): audit promised for four actions, schema had only `closed_by` / `closed_at` | insert-only `provider_task_actions`, written in each action's CAS transaction, replaces both columns (§4, map, §9, §10) |
| (Grok) | MAJOR: `abandon` vs an unapplied receipt | removed with the receipt journal |
| (Terra) | MAJOR: a late receipt after `close` sends no cancel | removed with the receipt journal |
| m1 | "eight closable statuses" | seven closable non-terminal statuses plus the held-`refused` close (§10) |
| m2 | the 6 h cancel limit had no name | `FAL_CANCEL_CONFIRM_SEC` (21600 s) in the settings table and where used (rule 4.9, map, §9) |
| m3 | the slot index keyed on `provider` | `provider` removed: one active task per owner slot across providers (§4, rule 4.1, §9) |
| m4 | `_assert_still_owner` checks only the token | the Fal re-check reads status, token and an unexpired lease; `close` takes the claim by the generic CAS (§2, rule 4.3, §4) |
| m5 | a long `needs_reconciliation` blocks drain-check | stated: not a stuck deploy; `close` clears it, `resume` once Fal confirms the fresh cancel (§7) |
| m6 | AICP-328 paths unverified against this worktree | verified on main, noted in the consumer contract (§6) |

**Revision 11 (frozen, not resolved here).** Third Grok pass (2 BLOCKER / 4 MAJOR / 10 MINOR)
and eleventh Terra pass (1 BLOCKER / 2 MAJOR). The owner froze the contract on 2026-10-27
because Fal is a reserve provider; every row below is an open requirement of the Stage 1
implementation checklist (separate backlog issue linked from AICP-312) and must be closed, with
its test, before Stage 1 code merges.

| Finding | Required before Stage 1 merges |
|---|---|
| BLOCKER (both): `fal_fetch` waits on the lowest-priority `QUEUE_PROCESS` behind 1-3 h jobs, the 30 min `fetch_deadline_at` runs from entry to `fetching`, and outputs expire after the 1 h lifecycle we set, so a paid result can be lost unread | a Fal queue served ahead of `process` and scanned by `_live_run_ids`; the deadline counted from the first started attempt and `result_gone` only after at least one; an output lifecycle longer than the worst queue delay; a saturated-queue test |
| BLOCKER (Grok): a `provider` column mapped without a default makes today's `spend.py` inserts send NULL, and `record_draw` swallows the error, so kie.ai draws are silently lost (reproduced on SQLAlchemy 2.0.54) | `default="kie"` and `server_default="kie"` on the mapped column, `provider` passed explicitly by every insert, a test that a kie.ai draw still lands after the migration |
| BLOCKER (Terra): releasing the reservation on a submit 400/401/403/404/429 assumes they are pre-queue and unbilled, which nothing proves | keep the reservation for every submit 4xx; only the operator's audited `abandon` releases it |
| MAJOR (Grok): the backfill can draw for a row whose reservation the operator already released | backfill only rows with `draw_pending` or in `accepted_unrecorded` |
| MAJOR (Grok): `delete_run`'s 409 ("a task that belongs to the run") is not expressible in columns | 409 when an active task has `owner_id` = the run or a `destination_path` under its `run_dir` |
| MAJOR (Grok): a crash between `submitted → cancelling` and the cancel request leaves no cancel for 6 h | recovery of `cancelling` re-enqueues `fal_cancel` (idempotent) as well as the poll |
| MAJOR (Grok): the last transaction of `close` takes the day lock before the task row, the inverse of every other path | stage the status update first, then `draw_in_session` inside the savepoint |
| MAJOR (Terra): an admission unique-key conflict is re-read by `operation_key` alone, so a task of another provider in the same slot can be returned as the "winner" | re-read by `(provider, operation_key)`; if absent, return the active-slot 409 |
| MINOR (Grok, 10) | named in the checklist issue: day attribution of a parked hold after midnight, unnamed grace, ffprobe tolerance and slow-poll interval, `adopt` id conflict as 409, `reservation_held NOT NULL`, the run-lock wording, the assistant coupling of `_lock_run_for_result_mutation`, reuse of the reserved amount instead of a recomputation, AICP-328 paths verified only on main |

## Stage 1 implementation (AICP-330)

Branch `feat/fal-stage1-main`, cut from main `a9f3869` (AICP-293 released there; first built on
its integration commit `488d88c`); migration `m3c9e5a1b427` on `l2b8d4f0a316`. The sections above stay frozen at revision 11; this section
records how each open requirement of the last Review-record table was closed, and every place
the code deliberately departs from the frozen text. **Where the two disagree, this section and
the code win.** Design check before coding: GPT-6.1 Terra (Codex, effort high), two rounds; its
findings are folded in below. Code gate (CLAUDE.md §1, GPT-6.1 Terra, effort high, review-gate
bar): round 1 on the whole diff drew 3 MAJOR (input duration vs `output_sec`; fetch attempts
burnt by a missing/rejected key; deadline cancel under an auth hold); rounds 2-4 on the deltas
drew 3, 1 and 1 MAJOR on those fixes (pricing on the longer duration; the fetch window
restarted while the key is missing; the deadline applied only after a non-auth GET, guarded in
the CAS and by the version the poll read); round 5 NO FINDINGS. Each has a `test_gate*` test.

| Item | Resolution | Tests (`tests/test_pipeline_provider.py`, `_pg.py`) |
|---|---|---|
| B1 fetch can lose a paid result | Every Fal job runs on its own `fal` queue (`tasks.QUEUE_FAL`, replacing rule 4.2's `QUEUE_PROCESS`), served ONLY by the dedicated `fal-worker` container (RQ worker with scheduler + a supervised ticker); the run workers never take `fal` jobs, so a Fal fetch never waits behind a run job and Fal work never starves runs. `_live_run_ids(..., queue_names=["fal"])` sees Fal jobs. The fetch claim counts the attempt and starts `fetch_deadline_at` (`COALESCE`, from the FIRST started attempt) atomically before any network call; the attempt after the 5th, or one after the window, ends `failed(result_gone)`. An already installed, ffprobe-valid destination is `done` before that rule applies (destinations are unique per task and must not exist at admission). Output retention `FAL_OUTPUT_LIFECYCLE_SEC` = 24 h, validated to cover the fetch window + 3 h. | `test_b1_*` |
| B2 `provider` column | `default="kie"` and `server_default='kie'` on both spend tables; `spend.py` passes `provider` explicitly on every insert. The migration's ALTERs run under `lock_timeout = '5s'`. | `test_b2_*`; PG: `test_b2_live_migration_with_old_writers_and_lock_timeout` |
| B3 every submit 4xx | Every submit 4xx is `refused` with `reservation_held` kept; rule 4.6's gateway release and §5's release-on-refusal are removed. Only the operator's audited `abandon` releases; `close` converts into the draw. | `test_b3_*` |
| M1 backfill | The sweep draws only `draw_pending` rows (and recovery of `accepted_unrecorded`); the "request id but no draw" criterion of §7 is removed. | `test_m1_*` |
| M2 delete paths | `delete_project` locks the project row first, then its runs, and 409s on an active task with `owner_project_id`; `delete_run` (run lock) 409s on an active task with `owner_run_id` or `owner_id` = the run, or a destination under its `run_dir`. Admission locks the project row, then the run row, refuses a deleted owner, and requires the destination under that project (outside `runs/`) or under that run (outside `clips/`), never under `tail_edits/`. New column `owner_run_id`. | `test_m2_*`, `test_admission_refusals`; PG: `test_admission_racing_delete_project_never_both` |
| M3 lost cancel | Durable `cancel_delivered_at` (set only on 2xx or 400 "already completed"; any other answer retried by the poll and recovery) and `cancel_attempted_at`. Recovery enqueues `fal_cancel` for every undelivered `cancelling` row regardless of poll liveness. | `test_m3_*` |
| M4 lock order | Every draw for an existing row UPDATEs the task row first, then takes the day lock: acceptance, adopt, `close` of a held `refused` row, the close's last transaction (status UPDATE, then the draw in a SAVEPOINT, then `draw_pending`), recovery of `accepted_unrecorded`, and the backfill (a version-bumping UPDATE on `draw_pending` first). | `test_m4_*`; PG: `test_m4_close_and_backfill_on_one_row_do_not_deadlock` |
| M5 admission conflict | Re-read by `(provider, operation_key)`; absent means the slot (or the destination) is taken → 409. | `test_m5_*` |

**Minors taken:** named `FAL_QUEUED_GRACE_SEC` (120 s), `FAL_SLOW_POLL_SEC` (600 s),
`FAL_OUTPUT_DURATION_TOLERANCE_SEC` (1.5 s), `FAL_POLL_INTERVAL_SEC` (15 s); a second `adopt`
of an id already on another row is 409 (unique `(provider, request_id)`);
`reservation_held` / `draw_pending` NOT NULL; the reserved amount is stored on the row
(`reserved_mcredits`) and every draw uses it; a parked hold drawn after midnight lands on its
admission day (single `fal:<task id>` scope, `_draw_target_day`); the AICP-328 paths were
checked on this branch (`disk_cleanup.py`: `runs/<run>/clips/`, `tail_edits/<edit>/body.mp4`).
The run lock taken by admission is a plain `SELECT ... FOR UPDATE` of the `runs` row, NOT
`api_v2._lock_run_for_result_mutation`, which also consults the assistant's unresolved
publication intents (`ensure_no_unresolved_intent`); `delete_run` keeps using that helper, and
both lock the same row, so they still serialize.

**Other departures from the frozen text**

- The `fal-worker` service is behind the compose profile `fal` (owner's choice, 2026-10-28: no
  cost while Fal is unused). Admission refuses with 503 while no live worker serves the `fal`
  queue; every deploy passes `--profile fal` while Fal is enabled or any task is unfinished.
- Operator `close` releases the claim in its CAS (instead of keeping it) so the `fal_close` job
  can take its own; a stale worker's re-check still fails, because the status changed.
- Inputs are uploaded with `FAL_INPUT_LIFECYCLE_SEC` = 8 h, validated to cover
  `FAL_JOB_TIMEOUT` + `FAL_POLL_DEADLINE_SEC` + `FAL_CANCEL_CONFIRM_SEC` (Fal may read an
  input late); the frozen 1 h is gone.
- `FAL_FETCH_WINDOW_SEC` = 3600 s (validated ≥ 2 × `FAL_JOB_TIMEOUT`), not 30 min.
- A result GET that is neither 200 nor 422 is a failed fetch attempt (counted toward
  `result_gone`), since Fal documents no distinct "job failed" answer; a Fal status of
  `CANCELLED`/`FAILED`/`ERROR` (undocumented) ends the row `failed`.
- Extra columns: `owner_run_id`, `cancel_delivered_at`, `cancel_attempted_at`,
  `completed_seen_at`, `poll_failures`, `reserved_mcredits`; `UNIQUE (destination_path)`.
- `result_meta` (JSON, AICP-251 ask Q1, owner-approved 2026-10-28): the whitelisted, URL-free
  fields of Fal's result (`placement`, `seed`, `frame_count`, `frames_per_second`, `timings`,
  and the video's type/size/dimensions), written with `done` for the consumer's QA. Output
  media URLs are still never stored. Operator-action HTTP endpoints (Q3) are AICP-251's.
- Operator actions are Python functions plus `python -m app.pipeline_provider
  list|adopt|abandon|close|resume`; HTTP endpoints and UI come with the first consumer
  (AICP-251).

**Residuals added by the implementation**

- The bound from Fal's COMPLETED to our download is "a free `fal-worker` plus backoff", not a
  proof. The ticker's watchdog logs ERROR when a `fetching` row is unread half-way to its output
  retention, when tasks are active and no `fal`-queue worker has a fresh heartbeat, and when the
  oldest pending `fal` job is older than 15 min. `completed_seen_at` is when OUR poll saw
  COMPLETED (up to one poll interval late, more while polls back off).
- Fal reading an input after the 8 h input lifecycle fails the job at Fal.
- On SQLite, pysqlite's default makes the outermost `RELEASE SAVEPOINT` a commit; production
  is PostgreSQL and the tests use SQLAlchemy's documented BEGIN recipe.
