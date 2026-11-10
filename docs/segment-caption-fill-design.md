# Segment caption fill — design (AICP-358)

Status: v3, after two design reviews (GPT-6.1 Terra, 2026-10-31: 13 + 15 findings, all
accepted, its first-release cuts taken) and the operator's decisions of 2026-10-31.
Tracking: AICP-358; flow `docs/assistant-flow-subtitle-removal.md` step 5.

## 1. Goal and scope

A `subtitle_removal` run is `done` after one Gemini Omni pass; QA finds segments that
still show caption words. For one such segment the operator (later the assistant,
AICP-359) rebuilds it as **source frames + a VOID inpaint pasted back only inside a
caption mask**: a free preview, then one approved paid call. The composite becomes the
segment's result, the previous result becomes a take (AICP-342), and the run re-stitches
and re-delivers through the normal path (same Drive id via `_deliver_in_place`).

Operator decisions (2026-10-31):
- **VOID only** (`fal-ai/void-video-inpainting`); LTX-2.3 later.
- **Measured inclusive box + free preview** before the paid start; the operator can
  resend the preview with an edited box.
- **Fal admission**: `subtitle_removal` projects are admitted automatically **only for an
  endpoint whose registry family is `inpaint`**. Reframe projects keep today's exact rule
  (all of them, any endpoint); everything else keeps `FAL_PROJECT_IDS`.

First-release limits (refused in preview with a clear message): run `audio_mode` must
be `original` (all 39 subtitle_removal runs are); source must be 9:16, CFR, rotation 0,
square SAR; segment ≥ 69 frames (VOID's minimum; 2.3 s at 30 fps).

## 2. Data model (one migration)

`run_segment_fill_jobs` (append-only history):
`id, run_id → runs ON DELETE CASCADE, run_segment_id → run_segments ON DELETE CASCADE,
project_id, created_at, updated_at, status, model ('void'), prompt, mask_box (json),
base (json, §4), media (json: clip frame count, fps, w/h, frames_masked, void_frames,
void index map), prep_dir, preview_path, estimated_usd, approved_usd,
provider_task_id (soft ref, no FK — provider rows outlive runs by design),
phase_claim_token, phase_claim_expires_at, phase_attempts, composite_path,
replaced_take_id (soft ref), applied_at, stitch_requeued_at, error_code, error_message,
discard_audio_patch`.

Status (VARCHAR + CHECK): `preparing → ready → admitted → compositing → applied`;
terminal `failed`, `canceled`. Partial unique index on `run_id` where status in
(`admitted`, `compositing`) = the **active** fill. `preparing`/`ready` are not active.

No new columns on `run_segments` / `run_segment_takes` (finding v2-2: a fill id there
would have to be cleared by every other result writer). Identity is fixed instead by
**clearing `rs.seedance_task_id` and `rs.seedance_result_url` when the composite is
applied**: `_existing_take_for_current_result` then matches the composite only by its
path, never by the Omni task it was built from; a restored composite take carries
`seedance_task_id = NULL` back, so the rule holds after restore too. Provenance lives on
the fill row (`composite_path`, `replaced_take_id`).

## 3. Flow

All work files live in **immutable claim directories**
`runs/{rid}/segment_fills/{job}/{phase}-{token}/`; the winning claim's paths are written to
the row under the row lock + claim check. Nothing is renamed to a fixed path. A worker
that loses its claim deletes only its own directory; unreferenced directories are removed
by the terminal cleanup (§7).

### 3.1 Preview (free) — `POST /api/v2/runs/{rid}/segments/{rsid}/fill/preview`
Body `{mask_box?, prompt?}`. Checks (no locks needed beyond a read): project is
`subtitle_removal`, run `done`, `audio_mode == original`, segment is an active swap
segment, `completed`, result on disk, not `source_fallback`. Inserts `preparing` (claim
NULL) with `base` frozen (§4), commits, enqueues `segment_fill_prepare`, 202.

`segment_fill_prepare(job_id)` (process queue, streamed — never a whole video in RAM):
1. Claim: CAS `preparing` with (token NULL or expired) → new token, lease, attempts+1.
   Missing row / run → stop (deletion = cancellation; never recreate a deleted run dir).
2. Re-derive the segment clip **with the pipeline's own cut** (`media.cut_clip` with the
   same arguments the run's submit used, from `project.source_local_path`), so the
   composite has exactly the frame grid an Omni result would be stitched on; verify the
   source identity equals `base.source`. Refuse unsupported media (§1).
3. Box: request box (finite, ordered, within [0,1], ≥ 2 % per side) else measured over
   every frame of the clip: union of caption-glyph hit boxes + padding. No hit →
   `failed` `no_captions_detected`.
4. Per-frame mask (white glyph ringed ≥ 55 % by black stroke inside the box, closed,
   dilated, OR t±1) to `masks.npy` memmap. VOID inputs at 720×1280 (9:16 only): the
   largest valid `69 + 8k ≤ min(n, 197)` frames sampled evenly over the clip; each
   sampled frame's mask = union over its source interval; the index map is stored.
   Mask video inverted (VOID: 0 = remove, 255 = keep).
5. `preview.jpg`: up to 8 frames where the mask is largest + first/last masked frame,
   mask outlined on the source; counts.
6. Publish: under the row lock, if the claim is still ours and unexpired, set `prep_dir`,
   `preview_path`, `media`, `mask_box`, `estimated_usd`, status `ready`. Else delete own dir.

### 3.2 Start (paid) — `POST /api/v2/runs/{rid}/segments/fill/{job}/start`
Body `{approved_usd, discard_audio_patch}`; `approved_usd` finite, ≥ 0. One transaction,
lock order **project → run → fill row** (the Fal admission order):
- run gates exactly as a segment rerun: `_admit_segment_rerun` (Run lock,
  `_require_segment_reconstruction_allowed`, status `done`), `_check_segment_rerun_conflicts`,
  `_require_audio_patch_discard_ack`; no active fill (index);
- job `ready`; `base` equals live state (§4) else `canceled` `stale_preview`, 409;
- **no paid re-stitch** (v2-1): every other active swap segment of the stitch plan is
  `completed`, not `source_fallback`, result on disk — the `restore_segment_takes`
  completeness check; else 409;
- `approved_usd ≥ estimated_usd` (the reservation), else 409 with the estimate;
- `pipeline_provider.admit(...)` (reserves spend in this transaction; `BudgetExceeded` →
  rollback, 409); job `admitted`, `provider_task_id`; commit; enqueue `fal_submit` if
  created (a lost enqueue is recovered by the Fal ticker).

### 3.3 Provider
Unchanged state machine. Uncertain states (`submit_unknown`, `needs_reconciliation`,
fetch/auth holds) keep the fill `admitted` and the run blocked; the run page shows the
provider status and points at `python -m app.pipeline_provider adopt|resume|close|abandon`.
Never a second POST.

### 3.4 Reconcile — `_reconcile_fill_jobs` in the recovery tick (the only trigger)
Each rule re-reads under the row lock and CASes; RQ liveness + a grace period like
`_reconcile_reframe_jobs`.
- `preparing`, claim NULL older than the grace or expired, no live job → re-enqueue
  prepare; after 3 attempts → `failed` `prepare_crashed`.
- `admitted` + task `done` → enqueue `segment_fill_composite`.
- `admitted` + task not active (`PROVIDER_TASK_ACTIVE_WHERE` false, i.e. reservation
  released) and not `done` → `failed` `provider_<status>`. A `refused` task that still
  holds its reservation is active → the fill stays `admitted` until the operator closes
  it (v2-6).
- `compositing`, claim expired, no live job → re-enqueue; after 3 attempts → `failed`.
- `ready` older than 7 days → `canceled`.

### 3.5 Composite + promote — `segment_fill_composite(job_id)`
1. Claim: CAS (`admitted`) or (`compositing` with an expired claim) → `compositing`,
   new token.
2. Fence: the provider task is `done` and its `result_path` is the frozen destination.
3. Decode the VOID output **by frame count** (it tags 12 fps); count must equal the
   submitted frames; output frame k ↔ stored source interval.
4. For each clip frame i: resize the matching VOID frame to the clip size, blend it into the
   clip frame only where `masks[i]` is set (4 px feather); encode at the clip's fps, exact
   clip frame count, no audio, into the claim dir; verify the count.
5. Promote, one transaction, project → run → fill row: claim ours and unexpired; `base`
   equals live state (else `failed` `base_changed`); run gates + no-paid-re-stitch check
   again; `_archive_current_take` → `replaced_take_id`; segment → composite (path set,
   task id + result url cleared, status completed, cost unchanged); job `applied`,
   `composite_path`, `applied_at`, `stitch_requeued_at`;
   `_invalidate_final_and_requeue(kind="segment fill")`. From here normal run recovery
   owns the stitch.
6. Delivery receipt (v2-13): the fill reports "delivered" only when the run's
   `result_revision` differs from `base.final_revision`, the run is `done`, and its
   `updated_at` is after `stitch_requeued_at` with no newer result-mutating event —
   checked on read; otherwise "applied, re-delivery pending/failed" with the run status.

## 4. The frozen base

`base = {segment: {local_result_path, file identity, seedance_task_id, status,
snapshot bounds}, run: {status, stitch_plan sha, segment_snapshot sha,
final_revision = assistant_run_result.result_revision(run)}, source: {path, file
identity}, audio_patch_id}` — captured at preview, compared under the locks at start and
promote. Any difference → refuse; nothing changes.

## 5. Reverse gates

While a fill is active (`admitted | compositing`), these 409 after the Run lock: single +
multi rerun, restore, retry, range-attempt admission, audio-patch admission, segment image
regeneration, segment prompt/reference override, assistant result publication, run delete,
project delete, project source replacement. Run/project delete while `preparing`/`ready` is
allowed: prepare treats a missing row as cancel (3.1.1) and never recreates directories.
Copy, captions, tail edits and overlays build on the delivered final and stay allowed.

## 6. Fal layer changes (minimal, fail closed)

- `FalEndpoint`: new optional `flat_usd` (when set, `worst_case_usd` = `flat_usd`),
  `media_check` ("duration" default | "frames"), `send_resolution` (default True; VOID
  False — `fal_submit` adds `resolution` today), `required_inputs`. Reframe's entry is
  unchanged.
- VOID: family `inpaint`, `flat_usd=0.10` reservation (first release 0.25; the real charge
  is $0.05 per request, Fal logs 2026-10-31, so the reserve keeps 2x headroom; was: record the actual charge on the first jobs
  and lower it), params `{prompt, num_frames, enable_pass2_refinement}`, required inputs
  `video_url` + `quad_mask_video_url` (both video, 720×1280, same frame count =
  `num_frames`, same duration), input duration bounds 2.3–60 s.
- `admit` freezes `request_spec.verify = {"kind": "frames", "frames": N}` for VOID.
  `_verify_media` (both fetch and installed-destination paths) dispatches on it; the field
  **absent** → duration (every existing reframe task); present but unknown → fail closed.
- `project_allowed(project_id, project_type, endpoint_spec)`: reframe → True (as today);
  `subtitle_removal` and `endpoint_spec.family == "inpaint"` → True; else the allowlist.
  The family comes from the registry, never from the request.

## 7. Storage

Idempotent terminal cleanup in the reconciler: for `applied | failed | canceled` jobs whose
provider task is not active, remove every claim dir except the one holding
`composite_path` (applied) and `preview_path`'s small JPEG; delete `masks.npy`, VOID inputs,
`raw.mp4`. The composite stays while it is the segment's current result; once replaced, it
was already copied into `segment_takes/` by `_archive_current_take`, and it is removed with
the run dir. Disk: prepare refuses `insufficient_disk` below 3 GiB free (a guard, not a
reservation; one prepare at a time per run is enforced by the row's claim).

## 8. UI

Run page, `subtitle_removal` run, per swap segment: "Fill captions (VOID)" → preview
(sheet, box, frames masked, estimate) → "Start (reserve $0.10)" with the audio-patch ack
when needed → status incl. provider state → applied / delivered (§3.5.6).

## 9. Tests

Start vs every gated path; restore after a fill (identity by path); claim takeover in
prepare and composite; every provider state incl. refused-with-reservation; crash after
each commit/enqueue (NULL claim, expired claim); base changed (same-path final republish,
source replaced, snapshot/plan changed) between preview/start/promote; other segment
missing → no paid re-stitch; legacy reframe tasks verify by duration, unknown verify kind
fails; project_allowed matrix (reframe, allowlisted, subtitle_removal+inpaint,
subtitle_removal+other, unknown endpoint); frame map first/last frame, n < 69 refused,
wide box; run deleted during prepare.

## 10. Rollout

Migration first; drain incl. Fal `provider_tasks`; deploy with `--profile fal` (the fal-worker
must run the new `_verify_media`). Verification on 328341d4 / 2845eef3 segments 4 and 7:
preview, start, composite frame count = clip, final frame count = source, detector 0 hits,
eye check, re-delivered to Drive <DRIVE_FILE_ID>; record the real VOID
charge from the Fal dashboard.

## 11. Fill from the source: a segment the provider refused (2026-11-02)

Operator decision (Operator, 2026-11-02, project 514067f2 / run d8105ab5 segment 1: Omni
refused it 6x with "Request blocked: The input content was flagged by safety filters as
potentially dangerous" -- a phone screen showing book text with the word "suicide"):
a swap segment the video provider refuses with a content filter is produced by a VOID fill
straight from the **source**. VOID inpaints only the caption mask of the source clip; the
rest of the frame stays original and the model is not involved (so the refused content, e.g.
the phone screen, is left exactly as it was). It is the by-Assistant flow's standard path
(`docs/autopilot-subtitle-flow.md` §1.3.4); a retry with the model stays a secondary option.

**Why it is small.** The fill already builds every composite on the SOURCE clip (§3.1 step 2:
`cut_clip` of `project.source_local_path`); the model result was only the thing being
replaced. A source fill is the same media path with other preconditions and another promote.

**Classification** (`app/content_block.py`): a run segment is *content-blocked* when it is
`failed`, not a source fallback, and its `error_message` (pipeline_v2 stores the provider's
`failMsg`) matches one of the narrow, production-observed texts: "flagged by safety filters"
(Omni / Google: dangerous content, a prominent public figure), "may contain sensitive
information" (Seedance output moderation, e.g. a celebrity photo on a phone), "...Sensitive
ContentDetected" (Seedance input audio) and "copyright restrictions" (Seedance). Anything else
(a bare "Request blocked", a timeout, an internal error) keeps the ordinary retry route. One
blocked pipeline cycle (its 3 internal attempts) is enough: the block is deterministic.
`pipeline_v2` itself is unchanged (it still resubmits 3x; blocked attempts were not billed).

**Basis.** `base["basis"]`: `model_result` (default; every row written before this change
reads as it) or `source`; an unknown value fails closed. No migration. A source base also
freezes the segment's `error_message` and block category and `source.detector_sha` (the
caption detector the mask / the no-caption verdict is measured with; prepare refuses
`base_changed` when the project's detector differs from the frozen one). A model-result base is
exactly what AICP-358 froze, so a fill in flight at the deploy does not go stale.

**Preview** `POST /runs/{rid}/segments/{rsid}/fill/preview {basis: "source", mask_box?}`:
`_require_fillable(..., basis="source")` wants a `subtitle_removal` project, a run that is
`incomplete` or `failed`, `audio_mode = original`, an active swap segment that is
content-blocked. Prepare measures the box FIRST (before VOID's frame minimum):
* captions found -> the AICP-358 VOID inputs, mask and preview, `ready`, $0.10 reserved;
* no detector hit in the clip, or an empty glyph mask inside the box -> **source as is**:
  `ready` with `media.source_as_is` (+ reason), `estimated_usd = 0`, no VOID inputs, a sheet of
  10 evenly spaced source frames with the detector band outlined.

**Start** (VOID, paid) is `admit_start` as before; for a source fill the "no paid re-stitch"
check is not needed (below). A source-as-is job is refused there (409) and applied instead by
`POST /runs/{rid}/segments/fill/{fid}/apply-source` (free, synchronous,
`psf.apply_source_as_is`): locks project -> run -> fill row, the same gates and base check,
copies the prepared clip into its own claim dir `apply-<token>/`, verifies the frame count,
promotes. No provider task, no worker, never `admitted`/`compositing`.

**Promote** (`_apply_source_result`, under the promote locks): the segment becomes `completed`
with the composite (task id / result url / error cleared, `replaced_take_id = None`: there was
no result to keep), the job `applied`. Then, under the run lock: **only if every active swap
segment now has a finished result, the submit estimate (including preprocessing) is 0 and there
is no preprocess hole** is the run requeued (`_invalidate_final_and_requeue`: incomplete ->
queued, a zero reservation, one `process_run` -> stitch + first delivery). Otherwise the run
stays `incomplete` with its error message recomputed and the commit is explicit: another
blocked segment gets its own fill; a transient one waits for the operator's Retry, which never
resubmits the filled segment (it is completed). The stitch's completeness gate is not used as a
spending guard. A failed promotion of a free apply removes its copied clip; a crash leaves it
for the job's terminal cleanup.

**Afterwards** the composite is an ordinary applied fill: `_fill_context` classifies it as
`void_fill` (with `basis` / `source_as_is` in the fill info), the QA eye check applies to it
(§ the flow), a source-as-is fill without a mask box gets its fill sheet / mask-box clip over the
detector band. The run page shows "VOID from source" / "source as is" badges, the
"VOID fill from source -- preview (free)" button for a refused segment of an incomplete run,
and "Use the source clip as is (free)" for a source-as-is preview.

**Fal refusal.** VOID may refuse too (`content_policy_violation` -> provider task `refused`
holding its reservation): the fill stays `admitted` until the operator resolves the task
(`python -m app.pipeline_provider close|abandon`, see `docs/fal-provider-design.md`), then it
ends `failed provider_refused`. Nothing retries it on its own.

**Rollback.** Before rolling the image back, list the source-basis fill rows
(`base->>'basis' = 'source'`) and pending/answered `blocked:` / `blocked_failed:` approvals: the
old image reads a source `ready` job as a model-result one and refuses its start (run not done),
and an `admitted`/`compositing` source fill would fail at its promote (old code wants a done run)
after Fal charged it -- do not roll back while one is active.

**Reviews (GPT-6.1 Terra, `codex exec`, reasoning effort high).** §2 design: round 1 —
4 BLOCKER / 8 MAJOR / 1 MINOR, round 2 — 1 BLOCKER / 13 MAJOR; accepted: a separate free,
synchronous source-as-is path (no taskless `compositing`), the box measured before VOID's frame
minimum, the Fal-hold wait before the generic in-flight wait with state-specific commands,
`Ask.replaces` under row locks (an answer that raced the planning discards the Ask), the refusal +
detector frozen in a source base and checked in prepare, the no-bill requeue check incl.
preprocessing, blocked segments first, previews bounded per refused task, a narrow classifier, the
audio-refusal note, no Omni rerun for a source composite. Declined: a `fill:<id>` core version
key (the planner, `admit_start` and `apply_source_as_is` recheck status + base; a newer ask
replaces the old one; a completed step supersedes its pending asks); a caller-owned split of the
promotion commit (the free handler writes its `created_refs` before the service's single commit,
the established pattern of every endpoint-backed handler). §1 gate over `main..HEAD`: round 1 —
1 MAJOR (the retry option named one segment, `retry_run` resubmits every unfinished one: fixed),
round 2 — 1 MAJOR (a failed run with a preprocessing hole could get free delivery retries: now the
operator's `stuck:` ask, and `retry_delivery` refuses a hole), round 3 — NO FINDINGS.
