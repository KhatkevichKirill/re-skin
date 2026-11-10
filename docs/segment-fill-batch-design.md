# Batch VOID fills, auto VOID within an allowance, fewer eye checks (2026-11-03)

Status: v3, implemented on feat/batch-void-fewer-asks after two §2 design reviews (GPT-6.1 Terra, §9). Builds on `docs/segment-caption-fill-design.md`
(AICP-358, "the fill design") and `docs/autopilot-subtitle-flow.md` ("the flow doc").

## 0. Why

Live: project 81d0afc0, autopilot ff218831, run d46bb1b8. After the free re-stitch the QA
(effect c858675b) found Omni leftovers in 7 of the 11 swap segments (0, 2, 3, 4, 5, 7, 10;
10 accepted). Today each segment costs: a free preview, a `void:` question, a fill that must
composite, re-stitch and re-deliver (run back to `done`) before the next preview can even be
made (a preview is bound to the delivered final; the next fill's re-stitch makes it stale),
then one `eye:` check per composite and, for a dirty composite, a `stop:` question. 7 segments
took a whole afternoon of the operator's clicks.

Operator decisions (Operator, 2026-11-03): (1) batch fills, manual and autopilot, one paid start,
one promote, one re-stitch, one re-delivery; (2) VOID inside an operator-approved allowance
runs without a question; (3) auto-accept a composite the detector calls clean whose mask does
not touch a face; one end-of-run eye-check card for the doubtful ones; (4) one automatic retry
with a grown mask before the stop question; (5) stale cards are retired by the system;
(6) design hook only: a mode "VOID from the source for every segment, Omni as fallback".

## 1. Batch fill (platform)

### 1.1 Data model (migration `u5e1b7c9d3f4`, down_revision `t4d9f2b7c315`)

New table `run_segment_fill_batches` (append-only):
`id, run_id -> runs ON DELETE CASCADE, project_id, created_at, updated_at, status,
basis ('model_result'), approved_usd, discard_audio_patch, base (json: run status, stitch plan
sha, snapshot sha, audio mode, final_revision, drive file id, audio_patch_id, source identity
at ADMISSION), claim_token, claim_expires_at, attempts, applied_at, stitch_requeued_at,
error_code, error_message, summary (json: per member outcome)`.

Status (VARCHAR + CHECK): `active -> promoting -> applied | failed`; `failed` = nothing to apply
(every member failed) or a run gate refused the promote. Partial unique index on `run_id` where
status in (`active`, `promoting`) = **one active batch per run**.

`run_segment_fill_jobs` gains:
* `batch_id` VARCHAR(36) NULL, FK -> `run_segment_fill_batches` ON DELETE SET NULL, indexed;
* `mask_grow_px` INTEGER NULL (extra dilation of the per-frame glyph mask, 0..24; §4);
* status `composited` (CHECK recreated): the member's composite is built and verified, waiting
  for the batch promote. `FILL_ACTIVE` = `admitted, compositing, composited`.
* The partial unique index "one active fill per run" is recreated with
  `AND batch_id IS NULL`: it now covers only legacy rows (admitted before the deploy).
  New starts always create a batch, so no new legacy row can appear.

`video_projects.void_auto_usd_per_run` FLOAT NULL (§3; NULL = `settings.AUTOPILOT_VOID_AUTO_USD`,
default 1.00; 0 = always ask).

No backfill. `deploy/drain-check.sql`: `segment_fills` counts `composited` too (a worker phase
in waiting), `segment_fill_batches` counts `active`/`promoting`.

### 1.2 Binding a preview to its segment, not to the final

A preview made from now on freezes `base.bind = "segment"` (+ `run.drive_file_id`).
`base_mismatch` for such a base ignores `run.final_revision` and `run.status`; it still compares
the segment (result path, identity, task id, status, bounds), the stitch plan / snapshot sha,
`audio_mode`, the Drive id, the source identity and the audio patch. The composite is built only
from the source clip + its mask, and it replaces only its own segment, so another segment's
fill re-stitching the final does not invalidate it. The run gates (`_require_fillable`: `done`,
original audio, active swap segment, finished result) are re-checked at start and promote as
before, and the reconstruction gate refuses runs with published assistant edits. Rows written
before (no `bind`) keep the full AICP-358 comparison, so an in-flight preview behaves exactly as
it did. Delivery receipts move from the preview's `final_revision` to the batch's (frozen at
admission).

### 1.3 Start — `POST /api/v2/runs/{rid}/segments/fill/batch/start`

Body `{fill_ids: [..], approved_usd, discard_audio_patch}`. Service
`psf.admit_batch_start(db, run, fill_ids, ...)` (never commits / rolls back; the
`admit_start` contract): lock project -> run (`_admit_segment_rerun`) -> fill rows in id order.
Refuses (nothing written) unless: 1..12 distinct fill ids of this run, one per segment, every one
`ready`, `model_result` basis, not `source_as_is`, base matches (`StalePreview(fid, part)`); no
active batch and no active legacy fill (the run lock serialises; the batch index is the
backstop at a savepoint flush); `_check_segment_rerun_conflicts`; audio-patch ack; **no paid
re-stitch** — every active swap segment outside the batch is finished; `approved_usd >=
Σ estimated_usd`. Then a batch row (`active`, base frozen), and for each member
`admit_provider_task` (each reserves its own spend in this transaction; one `BudgetExceeded` /
`AdmissionRefused` refuses the WHOLE batch — the caller rolls back) and the member ->
`admitted`, `batch_id`. Caller commits, then enqueues `fal_submit` for every created task.
`admit_start` (one fill, the existing endpoint and autopilot handler) becomes a batch of one.

### 1.4 Members run concurrently

The Fal ticker submits/polls each task as today (no change). `reconcile_fill_jobs` enqueues
`segment_fill_composite` for every admitted member whose task is `done` (unchanged). For a batch
member, the composite job builds and verifies the composite exactly as today and then, instead
of promoting, CASes the member `compositing -> composited` (composite path, size, sha256; claim
cleared) and enqueues `segment_fill_batch_promote(batch)`. Failures are per member: provider
failed/refused-and-released -> member `failed provider_<status>` (unchanged rule), composite
error -> `failed`. A member whose Fal task only the operator can resolve (refused while holding
its reservation, `submit_unknown`, `needs_reconciliation`) stays `admitted` and **holds the
batch** (an uncertain charge is not a failure; the flow's provider-hold message names the
commands). Everything else proceeds without waiting for the others.

### 1.5 One promote — `segment_fill_batch_promote(batch_id)`

Triggered by each composite and by the reconcile tick (`batch active`, no member in
`admitted`/`compositing`). Claim: CAS `active` (or `promoting` with an expired claim) ->
`promoting` + token + lease. One transaction, lock order project -> run -> batch row -> member
rows (id order): run gates (`_lock_run_for_result_mutation(own_batch_id)`, reconstruction
allowed, status `done`, `_check_segment_rerun_conflicts`, audio-patch ack); for each
`composited` member: its base still matches (segment-bound) — else that member -> `failed
base_changed` (its composite is not applied); the no-paid-re-stitch check over every active
swap segment except the members being applied. Then per applied member exactly today's promote
(`_archive_current_take` -> `replaced_take_id`, the segment -> composite, task id / result url
cleared, member `applied`), batch `applied` with a per-member summary, ONE
`_invalidate_final_and_requeue(kind="segment fill batch")`: one stitch, one in-place delivery
(same Drive id). If no member is applicable: batch `failed`, nothing requeued. A run-gate
refusal fails the batch and its `composited` members (`gate_<code>`); their composites are
removed by the terminal cleanup; Fal already charged — the autopilot asks (§4.3 stop rule).

### 1.6 Reverse gates

Unchanged in meaning: every path that refuses while a fill is active refuses while any member is
in `FILL_ACTIVE` (now including `composited`) — `active_fill()`, `assistant_run_result`'s
publication check, `drive_relink`, the run page's busy flag, `Context.fills()`. The legacy
promote keeps `own_fill_id`; the batch promote passes `own_batch_id` (excludes its members).

### 1.7 Manual UI (run page, subtitle_removal)

A "Batch VOID fill" panel: a checkbox per active swap segment with a finished result,
"Select all", "Select segments with leftovers" (from the latest caption QA of the run when the
autopilot made one), "Preview selected (free)" (`POST .../fill/batch/preview
{run_segment_ids}` = one preview per segment, prepared in parallel on the run workers), and
"Start N ready (reserve $N x 0.10, ~$N x 0.05)" over the newest ready, fresh preview of each
selected segment. A batch status line lists members (`admitted`/`composited`/`applied`/`failed`).
Per-segment buttons stay; a single start is a batch of one. A manual paid start on an assistant
project still pauses its autopilot (core §5.7, the `fal:<task>` scope).

## 2. Autopilot — one batch question or none

`plan_clean` (fill part) changes:
1. **In flight**: only an active batch (or a legacy active fill) waits. A preview `preparing`
   no longer blocks planning: previews are made one per tick (free) for every segment that needs
   one, so they prepare in parallel. The batch decision waits until no preview is preparing.
2. **Candidates** = every `propose_start` item (ready, fresh, default prompt, measured box,
   estimate within the reserve, not held by an operator answer) + the flow's own grown previews
   (§4) + ready previews made for an operator-chosen widened mask.
3. **Inside the allowance** (§3) and no promoted audio patch -> `Launch("autovoid:<rv>:rN",
   "segment_fill_batch_start", {run_id, fill_ids, approved_usd: Σ, discard_audio_patch: false},
   regeneration=False)` — no question. The core's own budget rule still applies (committed +
   estimate over the approved budget x (1 + tolerance) -> the core's `overrun` approval).
4. **Otherwise** ONE question `vbatch:<rv>:<digest>` (`regeneration`): "Segments 0, 2, 3, 4, 5, 7
   still show captions ... rebuild all 6 with VOID: $0.60 reserved, about $0.30 charged ...",
   one combined contact sheet (per member: delivered | original at its leftover frames + the
   mask preview), options: `start` (the only action-bound one: the batch start with exactly these
   fill ids), `per_segment` (ask me segment by segment: today's `void:` questions, each a batch
   of one), `accept` (ship all as they are), `manual`. `accept`/`manual` bind to every member's
   result token (`Answer.members`, read from the approval's proposal), so they hold per segment.
5. The handler `segment_fill_batch_start` is paid, NOT `always_regeneration`; the flow sets
   `regeneration=True` for the asked path (authorised by the `start` option's action hash).
   `segment_fill_start` (one fill) keeps `always_regeneration` (operator per-segment path).

The combined sheet is made in the planning-free part: a new free compute
`subtitle_batch_sheet` (`autopilot_compute`, streamed, ffmpeg + cv2 on JPEGs only) stacks each
member's QA crop sheet and preview JPEG into one image in its attempt dir; the ask waits for it
(bounded retries, then asks without it).

## 3. VOID within an allowance (policy change)

Q1 ("every VOID fill is a regeneration approval") is replaced by: **a VOID batch inside the run's
allowance runs without a question; above it, the batch is asked.** Allowance = the project's
`void_auto_usd_per_run` (default $1.00 = 10 reservations, editable on the project page;
0 = always ask) minus the reservations of every VOID start the autopilot already launched on this
run without a question (`autovoid:` effects, `approved_usd`). Accounting stays honest: each member
is a provider task with its own reservation in the spend ledger (`pipeline_provider.admit`), the
daily budget cap refuses the whole batch, the autopilot registers a `fal:<task>` scope per member
so its `committed` includes them, and the core's approved budget must cover it (else `overrun`).
The start-of-run budget approval gains a real line: "VOID rebuilds without a question: up to
$X (N x $0.10 reserved, ~$0.05 charged each)", credits = min(allowance, swap segments x $0.10).
Omni reruns, a paid retry of the run, and any VOID over a promoted audio patch stay
operator-approved.

## 4. Fewer eye checks, and the stop rule

### 4.1 Auto-accept
The QA compute adds, for every VOID composite whose QA row is detector-clean, InsightFace face
boxes sampled at 1 fps from the SOURCE inside the segment (one streamed pass over the source,
only when such a row exists; `qa.fill_faces[index] = {samples, overlap_samples}`). The planner
accepts a composite on its own when: verdict clean with 0 result hits, its re-delivery is
confirmed, `fill_faces` exists with >= 1 sample and no face box (padded 15 %) intersects the
mask box (padded like the crop), the box is at most 0.30 of the frame tall, and it is a VOID
composite (not "source as is", whose eye check is about missed captions). There is no phone-screen
detector; the handoff question still asks for the phone screen (the box-height rule keeps a
big repaint out of the automatic path). Auto-accepts are listed in the step output and in the
hand-off question.

### 4.2 One eye-check card
Doubtful composites are asked together, only when nothing else of the clean step is in
progress: `eyes:<qa8>` with every doubtful segment's fill sheet and mask-box clip, options
`accept_all` / `per_segment` (today's `eye:<fid>` questions) / `manual`. `accept_all` confirms
exactly the fill ids in its proposal (`Answer.members`), in this and later QA rounds.

### 4.3 Stop rule with one automatic retry
A delivered VOID composite the detector still calls dirty, not itself an automatic retry, whose
leftovers lie inside the segment: `Launch("grow:<rsid>:<fid8>:rN", "segment_fill_preview",
{mask_box: old box ∪ in-segment leftovers, padded +0.04 x / +0.03 y, mask_grow_px: 8})` (free),
then it joins the next batch (inside the allowance without a question). Only if that composite
also leaves captions (or no grown box exists) is `stop:<fid>` asked (today's options).

## 5. Stale cards

* **Core sweep**: on every Wait / Ask the core re-checks the step's PENDING approvals' bound
  versions (`about`, artifact producers) and that each artifact source still exists with its
  size; one that fails is superseded ("no longer current: <key> changed"), journalled.
* **Flow retirement**: `Wait.retire` / `Ask.retire` (new contract field) — pending flow asks the
  planner knows are obsolete: a per-result ask (`eye:/stop:<fid>`, `void:/hand:/operator:/sync:/
  edge:/widen_failed:<rsid>:<tok>`, `fp:/keepcap:s<N>:<tok>`) whose fill / result token is no
  longer the segment's current one, a per-QA ask (`inconclusive:/timing:/eyes:<qa8>`) of an
  older QA, a per-version ask (`restitch_*:/best:..:/redeliver_failed:/stuck:<rv>`) of an older
  run version, `vbatch:` asks whose members changed. The core supersedes them in the same
  transaction ("no longer asked: the result changed"), journalled. Answered ones are untouched.

## 6. In-flight on deploy (ff218831)

Migration first (the image reads `batch_id`, `mask_grow_px`, `void_auto_usd_per_run`). Legacy
fills `admitted`/`compositing` finish on the old promote path; a batch start refuses while one is
active. Legacy ready previews (no `bind`) are compared as before: a stale one is canceled by the
flow and remade segment-bound. Pending `void:` / `start:` asks: the flow replaces them with the
batch path (`replaces`), answered ones are honoured (a batch of one). Pending stale cards are
retired on the first tick.

## 7. Mode "VOID from the source" (scope added 2026-11-03, approved by Operator)

**Why.** Trial on 514067f2 / d8105ab5 (`/root/void-trial`): VOID from the source on every
segment left 0 caption leftovers in all 6 segments (post_run_qa + a frame-by-frame scan),
frame-exact with the source (1165/1165), $0.30 for the video vs ~$4.20 for Omni. Omni left
captions, drifted, and rewrote a phone screen ("Delete Account" instead of app stats at 26.88 s)
-- a change the detector cannot see. VOID's one weakness: captions over phone / iPad screens or
app UI leave grey patches and smeared icons (segments 0, 1, 3, 4, 5); clean over skin, books,
table, pavement (segment 2).

### 7.1 Project setting
`video_projects.caption_removal_method` VARCHAR(16) NULL + CHECK in (`void_source`, `omni`):
NULL reads as `omni` (every existing project keeps its behaviour); a NEW `subtitle_removal`
project is created with `void_source`. Editable on the project page (subtitle_removal only) and
PATCH `/projects/{pid}`; a change applies to the NEXT run (a run freezes its mode at creation).
Same migration (`u5e1b7c9d3f4`).

### 7.2 How a run represents it -- `runs.generation_mode = 'void_source'`
`runs.generation_mode` already says how a run generates video (`v2v`, `images_only`); its CHECK
gains `void_source`. The run keeps a real video `model` (the project type's, Omni) as the
per-segment FALLBACK model, and `run_segments.model_fallback` BOOLEAN NOT NULL DEFAULT false
marks the segments the operator sent to that model. Chosen over a new run status or a fake
model so the run state machine, the reaper, the reconcile tick, the drain check and delivery
stay as they are:
* **Creation** (`create_run` for a `void_source` project, manual or the autopilot's
  `create_run`): the segment snapshot is frozen and a `pending` RunSegment is created for every
  active swap segment in the admission transaction; the run is moved `created -> incomplete`
  (a PARKED status: not counted by the drain check, never reaped or requeued by recovery) with
  `error_message` "VOID from the source: ...", reserves nothing (its estimate is 0), is NOT
  enqueued, and the free source previews of every swap segment are queued (`basis=source`).
* **Fill**: `_require_fillable(basis=source)` accepts, on a `void_source` run that is
  `incomplete` / `failed`, every active swap segment without a finished result that is not
  `model_fallback` (plus, as before, a content-blocked segment on any run). The batch admission
  accepts `source` members (one basis per batch; no paid-re-stitch check -- a source promotion
  never re-stitches an unfinished run, see below); the run gate for a source batch is
  `incomplete` / `failed`. A source preview that finds no caption is `source_as_is`: applied for
  free (`apply_source_as_is`), never a batch member.
* **Promote**: the batch promote applies every source member exactly as `_apply_source_result`
  does, then ONE requeue check: only when every active swap segment has a finished result, the
  estimate is 0 and there is no preprocess hole is the run requeued (`incomplete -> queued`,
  zero reservation, one `process_run`): the submit phase finds every segment `completed`,
  submits nothing, the run stitches and delivers through the normal path (Drive, sheet row,
  Job ID). Otherwise the run stays `incomplete`.
* **Never a silent model call**: the submit phase of a `void_source` run submits ONLY
  `model_fallback` segments; any other unfinished segment is left as it is (the completeness
  gate ends the run `incomplete`). `spend.estimate_run_credits` counts only `model_fallback`
  segments of such a run, so every reservation, the source-promotion "free requeue" check and
  the autopilot budget see 0 for VOID-only work.
* **Omni fallback**: the segment rerun endpoint (and the autopilot's `rerun_segments`, always a
  regeneration with its cost) sets `model_fallback = true` on the listed segments of a
  `void_source` run before the reset; `retry_run` submits only fallback segments. A
  `model_fallback` segment's later VOID rebuild is a model-result fill as today.

### 7.3 Autopilot in `void_source` mode
* `launch`: `create_run` as before (estimate 0: nothing goes to Omni); the start-of-run budget
  approval carries the VOID line = swap segments x $0.10, and the run's VOID allowance is
  max(project allowance, that line) -- the operator approves the whole VOID cost once.
* `clean` on the `incomplete` void_source run (instead of the retry route): source previews for
  every segment (free, parallel; the creation already queued them), each `source_as_is` one
  applied for free, then ONE batch of the VOID ones (inside the allowance without a question,
  else one `vbatch:` question). A deterministic preview refusal (`segment_too_short`: VOID needs
  69 frames) goes to one question per segment (Omni fallback / ship the source as is / by hand).
  Long segments (> 197 frames) are subsampled by `void_sampling` as today (the trial's 233-frame
  segment had no flicker); no split.
* After the requeue and delivery: the normal QA round. A source composite is a `void_fill`;
  leftovers -> one grown retry (§4.3), then the stop rule, whose options now include the Omni
  fallback for a source composite that was NOT content-blocked (paid, cost asked).

### 7.4 Targeted eye check (§4.1 extended)
Auto-accept a composite unless the detector still finds text, a sampled face touches the mask
box, or the mask box overlaps a SCREEN / UI region of the source. Screen heuristic (in the
`subtitle_qa` compute, streamed, 2 fps over the segment, inside the padded mask box, caption
glyph pixels (the fill's own `text_mask`, dilated) excluded): the share of fine high-contrast
edge pixels (text-like structure) and the share of bright, low-saturation, low-texture pixels
(lit screen). Thresholds are calibrated on the trial's six segments (0, 1, 3, 4, 5 flagged;
2 not) and recorded in the compute result for the operator. Doubtful segments go into the one
end-of-run eye card, whose options gain the per-segment Omni fallback.

## 8. (superseded) Design hook: "VOID from the source for every segment"

A future project mode would make the Omni pass optional: every swap segment gets a `source`-basis
fill and Omni only runs for segments where VOID refuses or leaves captions. What it needs:
`basis=source` previews on a run with no model results (a run kind with segments but no Omni
submit, or `_require_fillable(source)` admitting any swap segment, not only content-blocked
ones); `admit_batch_start` accepting `source` members (the table has `basis`; the no-paid-
re-stitch check is replaced by `_apply_source_result`'s complete-and-free requeue rule applied
once per batch); the promote calling `_apply_source_result` per member and one requeue check;
the flow's `launch` step choosing the mode; cost lines (VOID per segment instead of Omni).

## 9. Review record (GPT-6.1 Terra, `codex exec`, reasoning effort high)

**§2 round 1** (sections 1-6, before code): 4 BLOCKER, 8 MAJOR -- all accepted:
R1-1 source single fills keep the legacy unbatched path (a model-result single start is a
batch of one); R1-2 member tasks admitted with `pp.admit(reserve=False)`, then ONE
`spend.reserve_many` (one day lock, the sum checked against the daily cap) as the last write;
R1-3 the reconcile tick re-enqueues an `active` batch whose members settled and a `promoting`
one whose claim expired, bounded (3 attempts -> `failed promote_crashed`); R1-4 the legacy
`promote()` refuses a batch member, the drain check counts `composited` members and live
batches, never roll back while one exists (an old worker cannot see a batch row in practice:
one is created only by the new API and reaches `compositing` minutes later); R1-5 the allowance
is enforced in the admission under the run lock from durable `auto` batch rows (per run, no
refund); R1-6 a member's provider price must equal its preview's estimate, sums in 1e-5 USD,
scopes registered in the admission transaction; R1-7 unmeasured / failed face sampling is
doubtful, faces padded 25 %, the hand-off card names and shows every auto-accepted segment,
`AUTOPILOT_AUTO_ACCEPT_FILLS=false` turns it off; R1-8 durable lineage `retry_of`, one grown
retry per composite, any failure of it -> the stop question; R1-9 a held member is detached
after `SEGMENT_FILL_BATCH_HOLD_SEC` once another member is composited (its task, reservation
and charge stay with the provider task; a late result is never applied); R1-10 the sweep uses
the full `_revalidate` under the approval row lock, answered rows untouched, legacy answers
keep their own single action; R1-11 at most `SEGMENT_FILL_BATCH_MAX` (12) members, a preview
preparing for more than 15 min does not hold the ready ones; R1-12 every write before
`_invalidate_final_and_requeue`'s commit, nothing cleaned after it, a composite's claim dir
kept when its member committed `composited`.

**§2 round 2** (section 7 + a pass over 1-6): 2 BLOCKER, 8 MAJOR, 2 MINOR:
* R2-1 (BLOCKER) Copy / automation would turn a void_source run into a paid v2v run --
  accepted: `_build_copied_run` refuses a void_source source run (409); automation templates
  are face-swap runs in practice and get the same refusal.
* R2-2 (BLOCKER) the fallback segment priced at 0 -- accepted: `_segment_rerun_need` adds a
  listed non-fallback segment of a void_source run; the rerun endpoints set `model_fallback`
  before the reset, so the reservation counts it (test: exactly one segment's credits).
* R2-3 `created -> incomplete` added to `RUN_TRANSITIONS`.
* R2-4 the run, its segments and every preview row commit together; the recovery tick
  re-enqueues never-claimed `preparing` rows; a run without a swap segment is refused (422).
* R2-5 more than 12 segments: sequential batches of at most 12, one active at a time; a source
  promotion requeues only when the run is complete, so there is still one final requeue.
* R2-6 "ship the source as is" after a `segment_too_short` refusal dropped: the question offers
  Omni for that segment or by hand; as-is stays only for a preview that found no caption.
* R2-7 a failed model fallback is asked (`vfallback_failed:`: Omni again / back to VOID / by
  hand); `_require_fillable(source)` accepts any unfinished segment of an incomplete/failed
  void_source run (no model task is in flight then), and the source promotion clears the flag.
* R2-8 a void_source run refuses preprocessing, references, a voice, a custom stitch plan and
  a non-original audio mode (422) -- no reference upload, no GPT Image phase.
* R2-9 the `create_run` action and the flow's launch estimate are 0 for a void_source project;
  the allowance stays the project's per-run allowance (default $1.00), the budget line shows
  min(allowance, segments x $0.10); a grown retry consumes the same allowance.
* R2-10 the screen heuristic only ever ADDS doubt: unmeasured, failed or flagged -> the eye
  card; six calibration segments are named as such (no independent validation set exists);
  the Omni fallback is chosen per segment on the `per_segment` eye questions (member-bound).
* R2-11 (MINOR) the run page shows a "VOID from the source" card for a parked void_source run
  instead of the Retry card, the batch panel in source mode and per-segment source previews.
* R2-12 (MINOR) the migration was never applied anywhere, so section 7's columns are folded
  into `u5e1b7c9d3f4`; its downgrade refuses while void_source runs exist; rollout and rollback
  are in §6 / the deploy notes.

Calibration of the screen heuristic (`subtitle_compute.fill_screen`, this code, on the trial's
source, read-only): frames over the threshold 2/8, 2/8, 0/9, 2/16, 8/16, 6/6 for segments 0-5
(maximum share 0.39, 0.48, 0.25, 0.72, 0.88, 0.47; threshold 0.34) -- 0, 1, 3, 4, 5 flagged,
2 not.
