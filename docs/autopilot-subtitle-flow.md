# Autopilot flow B — subtitle removal end to end (AICP-364)

Tracking: AICP-364, parent
AICP-362. Runs on the autopilot core
(AICP-363, `docs/autopilot-design.md` rev 7 — "the core" below; section numbers `§x` refer to
it). The flow itself is `docs/assistant-flow-subtitle-removal.md` ("the flow doc"); its
deterministic half is `app/subtitle_loop.py` (AICP-353 + AICP-359).

Status: **phase 2** (2026-10-31) — wired onto the core (AICP-363 tip 959e724, merged): the flow
`app/autopilot/flows/subtitle_removal.py` (adapter + registration), the pure planners
`subtitle_removal_plan.py`, the handlers `subtitle_handlers.py`, the computes
`subtitle_compute.py`; see §9.

Operator decisions this flow follows (AICP-363 §13): **Q1** (changed 2026-11-03, Operator: see
§1.7) VOID fills inside the run's VOID allowance start WITHOUT a question, as one batch; above
it the batch is ONE regeneration approval; **Q2** the judgement port is disabled in production — the
flow never Judges: every eye check is an operator `handoff` approval with options (**Q8**, no
`visual_check` kind); **Q3** heavy free work is `Compute` on the run workers (`autopilot_compute`);
**Q5** a manual paid action pauses the autopilot; **Q6** an approved regeneration adds its cost to
the budget.

## 1. What the flow does

```
segmentation ──► launch ──► clean ──► handoff
 (free)          (paid)     (free + regenerations)   (free)
```

| # | Step id | Flow-doc steps | Paid | Approvals it may Ask | Done when |
|---|---|---|---|---|---|
| 1 | `segmentation` | 0 intake, 2 analyze, calibration, review | no | `handoff` (segmentation eye check) | keeps / detector band applied or declined |
| 2 | `launch` | 3 estimate + launch | yes (budget) | `handoff` (a segmentation re-check when the segments changed after step 1); the core asks `budget`/`overrun` | the Omni run is admitted |
| 3 | `clean` | 4 per-segment QA, 5 VOID fill, 6 fill QA + eye check, 7–9 assembly/deliver (the pipeline's own re-stitch), stop rule, best take | regenerations only | `regeneration` (fill start, retry, Omni rerun), `handoff` (eye checks, stop rule, inconclusive QA) | QA clean after the operator's confirmations |
| 4 | `handoff` | 10 acceptance + hand-off | no | `handoff` (final + "create the localisation project") | the operator accepts the final |

"Pre-launch check: no deploy in progress" (flow-doc step 1) is the core's gate (§9): a held gate
refuses every admission and the step simply waits.

The project is analysed by the normal pipeline at creation (PySceneDetect cuts,
`project_types.SUBTITLE_REMOVAL`); the autopilot waits for `ready` and the core pauses it on a
failed or stalled analysis (§3).

### 1.1 `segmentation` — calibration, keep proposals, detector band

1. Project not `ready` → `Wait`.
2. A swap segment whose billed clip (`end − start + rolls`) is longer than Omni's
   `max_clip_sec` (10 s) → `Fail("segment N is X s; Omni takes at most 10 s: split it in the
   segment editor, then start the autopilot again")`. (Analysis caps segments at the project's
   `max_segment_sec`; a project created with a 15/30 s cap would otherwise have segments the
   pipeline skips — captions included.)
3. `Compute("review:rN", "subtitle_review", {project_id, segments_sig, detector_sha,
   faces: true})` — `subtitle_loop.review_segmentation` on the source with InsightFace face
   samples at 1 fps (`--faces`), contact sheets into the compute's attempt dir. A review
   counts only while its `segments_sig` (id, index, bounds, action, rolls of the live SegmentDefs)
   and `detector_sha` (the project's `caption_detector`) still match; otherwise a new round.
   3 failed rounds for the same inputs → `Fail` with the last error.
4. **Where are the captions?** (caption locator, 2026-11-02 — §1.5). A project without a band of
   its own (`caption_detector.band_y` absent) gets `qa_tools.locate_captions` on its source in the
   same compute, and the review measures with the band `subtitle_loop.effective_band` picks
   (`band.effective_band_y`): the default band when it calibrates and every caption position the
   locator found lies inside it; otherwise the located band, re-measured with the normal 10 fps
   detector. `band.verdict` then decides:
   * `located` (confirmed: locator `located`, the located band calibrates, the detector hits at
     ≥ 70 % of the locator's caption moments) → `Launch("locate:<review8>:rN",
     "patch_caption_detector", {caption_detector: config with band_y, expect_detector_sha,
     expect_no_band: true})` **without an ask** (`AUTOPILOT_AUTO_CAPTION_BAND`, default true),
     then a new review on the saved band (the digest changed) — the normal path from 5 on;
   * `located_uncertain` (a calibrating candidate that is weak, brief, tall, cut, has a competitor,
     or the flag is off, or a located band was saved before in this autopilot and the operator
     removed it) → `Ask("locate:<review8>", "handoff")` with the sheet `review_location.jpg`
     (green = located band, amber = default band) and the option `apply_band` (save it, then
     review again), plus `apply_main` (the strongest position alone) when the candidate is the union
     of several positions; reject stops;
   * not calibrated and `not_found` (nothing caption-like the pixel rule sees, or located text the
     normal detector does not confirm) → `Fail("the caption detector does not see this video's
     captions in its band y … (caption-like hits in X % of the frames, it needs 20 %); <what the
     whole-frame search found> -- the captions may be another colour, outline or size: open the
     project page, set Band y (and Text / Outline …) under QA caption detector, Save, then start
     the autopilot again")`. A project with its own band is never searched: the message says so.
5. Proposals from the review: **keep** for every swap segment the review proposes `keep`
   (calibrated zero in-band hits, or an end card: static text ≥ 2 s over ≥ 60 % of the hits, faces
   in ≤ 20 % of the samples), and the **detector band** when `band.verdict == "widen"`
   (`proposed_band_y` from the measured caption geometry, p5/p95 ± 0.03). Nothing proposed and no
   band warning → `Complete`. A default band that calibrates while the locator also found
   caption-like (or long-held) text outside it — or located text there the detector does not
   confirm — sets `needs_visual_check` (the note names the heights; `review_location.jpg` shows
   that text's own frames with the band in green) → the eye check below.
6. Otherwise one eye-check approval `Ask("segmentation:<review8>", "handoff")`: question, proposal
   (per-segment reasons, band geometry, launch cost current vs proposed), the contact sheets of the
   keep candidates as artifacts (producer: the review compute effect), `about =
   {project_segments:<pid>}` (an edit of the segments supersedes it), options:
   * `apply` — "keep segments 8; detector band y 0.41–0.67" (whatever is proposed);
   * `keep_only` — only when a band change is proposed: apply the keeps, leave the band;
   * `as_is` — launch the segments as analysed.

   Reject = stop the autopilot (the rule for every flow approval: Reject stops, the options choose).
7. Approved `apply` / `keep_only` → `Launch("keep:<review8>:rN", "patch_segments", {updates:
   [{id, action: "keep"}…]}, expect = {project_segments:<pid>: version seen})`, then (for `apply`
   with a band) `Launch("band:<review8>:rN", "patch_caption_detector", {caption_detector:
   current config with band_y replaced, expect_detector_sha})`, then `Complete`. The planner knows
   the patch landed when the live segments equal the review's segments with the approved keeps
   applied (structural comparison), so a landed patch does not trigger a second review. A review
whose keep patch was admitted is **spent**: if the segments are later edited back to what it saw,
the old approval does not cover that state and a new review round is asked (Terra R2-1); the same
holds for a landed band patch whose detector is PATCHed back (R3-4). The eye
check, the patches and the completion are also fenced on `project_detector:<pid>` (R4).

`segmentation` completes **without** `depends_on` on `project_segments` (a reopen would reopen
`launch` after its run exists and admit a second paid run, §4.6) but records the `segments_sig` it
completed on, and the detector digest. Edits *while the eye check is pending* supersede it (`about`), and a review whose
`segments_sig` no longer matches is redone. Edits *after* it completed are fenced by `launch`
(§1.2).

### 1.5 Caption locator (2026-11-02, after project 514067f2)

**Why.** 514067f2's word-by-word captions sit at y 0.76–0.80; the detector's default band is
0.50–0.72, so the review saw 0 hits, `calibrated=false`, and the flow failed with "PATCH
caption_detector" although its own widened scan had measured the captions. Captions sit at the top,
middle or bottom depending on the video; the operator asked that the detector find them itself and
ask only when unsure.

**One locator** — `qa_tools.locate_captions(path, config)`: the detector's own pixel rule
(`_edge_mask` / `_text_clusters`, shared with `caption_score`, which stays bit-identical) applied to
the WHOLE frame, streamed at 2 fps (one 540×960 frame in memory; ≤ 1200 frames, longer sources get
a lower rate). Every text-shaped cluster is merged into lines; a line counts when its centre x is
within 0.5 ± 0.15; clusters held still (IoU ≥ 0.9 for ≥ 2 s) are counted as static (UI, end card,
scene text — reported, not dropped). Centre y of the lines in 0.01 bins → contiguous **positions**
(gaps ≤ 0.03). Captions stay at one baseline, so they pile into one narrow position; scene text that
moves with the camera spreads thin, and book covers / phone screens are rarely centred or of
caption glyph size. Per position: support (frames with a non-static centred line / frames), static
frames, coverage of 5 time windows, p5/p95 of the line boxes, sample times. The **candidate band**
is the union of the caption-like positions with support ≥ max(0.10, main/3), p5/p95 ± 0.03, at
least 0.16 tall (room for a second line and a few pixels of drift). `located` needs ≥ 20 sampled
frames, main support ≥ 0.30, hits in ≥ 3 of 5 windows, height ≤ 0.30, ≤ 5 % of the candidate's
lines crossing its edges, no other caption-like position with support ≥ 0.05 and no centred text
held still elsewhere in ≥ 20 % of the frames (a hook bubble or a held caption track — those are
competitors, shown on the eye-check sheet); otherwise
`ambiguous` (a candidate exists) or `none`. `none` never means "no captions": a caption colour /
outline / size the rule does not match is invisible to it.

**One band, not several.** `band_y` stays a single range (config, UI, `detector_sha`, fill mask,
placement all read it). Two distant caption positions (e.g. d6a97b0a: words at the bottom, boxed
captions mid-top) make the candidate taller than 0.30 → ambiguous → eye check, never a silent union
that would take in the phone screen between them. Multiple scan bands are a follow-up if such
videos turn out to be common.

**Location is evidence, not calibration.** The review re-scans the candidate with the normal
detector (best cluster per frame, 10 fps) and uses it only when that calibrates.

**Where it is used.** The `subtitle_review` compute (above) and the CLIs: `python -m app.qa_tools
locate SOURCE [--project PID] [--sheet OUT.jpg]` (prints the band to type into the project page) and
`python -m app.subtitle_loop review|qa` (locate unless `--no-locate` or the config names a band).
`subtitle_qa` / `subtitle_loop qa` on a project **without** a band add `qa.band_location` and a
warning when caption-like source text lies outside the QA band (a warning only — it does not change
`clean`). The VOID fill mask (`pipeline_segment_fill`) and caption placement (AICP-368) read the
stored band: in the autopilot the located band is stored before the launch; in manual mode the
operator saves it on the project page (nothing writes a located band silently: not
`analyze_project`, not the QA — Terra §2: a silent guess would steer QA, fill masks and placement
and be indistinguishable from an operator's choice).

**Operator's band wins.** Any stored `band_y` (typed on the project page, PATCHed, or saved by the
flow) is never relocated over; `patch_caption_detector` with `expect_no_band` refuses (409) once
the project has a band. `CaptionDetectorConfig.stored()` now keeps an explicitly given `band_y`
even when it equals the default, so typing 0.50 / 0.72 locks the default band; **Reset to
defaults** (`null`) opts back into locating.

**Deploy / in-flight autopilots.** No migration, no new worker. The compute keeps version 1 (same
function, result gains `review.location_checked`, `band.location`, `band.effective_band_y`). An
old result without `location_checked` that is **calibrated** is used as before (a pending eye
check stays valid); an old **uncalibrated** one on a project without a band is not a verdict any
more: a new review round (new effect name) runs with the locator. An autopilot that already
**failed** on the old message stays failed: start it again. A completed segmentation step is not
reopened. `AUTOPILOT_AUTO_CAPTION_BAND=false` (`.env`, needs `up -d`) turns every located band into
an eye check.

**Review record (GPT-6.1 Terra, `codex exec`, effort high).** §2 design: Terra advised against any
silent persistence (accepted: nothing writes a located band outside the flow's own fenced patch;
not from `analyze_project`) and against default-on auto-apply until a labelled corpus exists
(rejected per the operator's stated preference; kept behind `AUTOPILOT_AUTO_CAPTION_BAND`, with a
strict gate). Accepted: location as evidence re-measured by the normal detector, locate even when
the default calibrates, explicit-default `band_y` kept by `stored()`, old-schema results,
patch-back guard across steps, competitors and line containment. §1 gate, 6 rounds: R1 unconfirmed
outside text cleared the eye check, QA warning ignored competitors (fixed); held captions dropped
as static (not changed: the normal calibration excludes the same static hits — named in the Fail
reason instead); R2 held text elsewhere, eye check without a sheet, QA recommendation (fixed); R3
sheet without competitors, index past the kept positions (fixed); R4/R5 competitors in the
no-candidate branch (fixed); R6 NO FINDINGS.

### 1.2 `launch` — the Omni pass inside the budget

* `estimate(ctx)` = Σ `spend.video_credits("gemini-omni", resolution, clip_sec)` over the live
  swap segments — exactly what `create_run` reserves (42 credits/s, ceil). The real bill is lower
  (Omni charges a flat 168 per clip; 2845eef3: 8 × 168 = 1344 billed vs 2268 reserved) and the
  settled draw lowers `committed` (§6.1); the budget is a plan for the reservation, so the launch
  never asks for an overrun of its own estimate.
* `budget_lines(ctx)` adds two **zero-credit** lines to the core's prediction (§6.2):
  "VOID fills: each is approved on its own (Q1), ~$0.05 charged / $0.10 reserved per segment" and,
  for a source over 60 s or more than 8 swap segments, the long-video warning of flow-doc rule 3
  ("some segments will very likely need a VOID fill").
* **Fenced on the reviewed segments and detector.** When the live `segments_sig` or the
  project's detector digest differs from what `segmentation` completed on (an edit while the
  budget approval was open — Terra R1-3, R2-2), `launch` runs the
  segmentation planner again *inside its own step* (review compute, eye check, patches, the
  too-long and calibration refusals) and launches only once that completes for the live
  segments. The `Launch` carries `expect = {project_segments:<pid>, project_detector:<pid>: the
  versions it was planned on}`, re-checked by the core under the project row lock, so an edit between planning and the
  admission is refused, not billed (Terra R1-3).
* `Launch("launch:rN", "create_run", {prompt: the project type's default prompt, model:
  "gemini-omni", resolution: params or the model default, audio_mode: "original", language:
  params or EN, name: "autopilot"})`. One Omni pass per segment (rule 1); never Seedance (rule 2).
* `launched` → `Complete({run_id})` (no `depends_on`: a non-terminal run is not a valid key);
  `not_launched` → the next round; `launching` → `Wait`.

### 1.3 `clean` — QA, VOID fills, eye checks, stop rule

Inputs each tick: the run (`ctx.run`), `ctx.run_version`, the step's QA compute effects, its
action effects, its approvals (including answered ones already consumed — `ApprovalView.decision`
and `chosen_option` survive consumption), and a **fresh fill snapshot** read in planning
(`subtitle_loop._fill_context` minus the ffprobe: the API gates run read-only, the fill history,
the takes), §4.

Order of decisions (`plan_clean`, first match wins):

1. Run not terminal → `Wait`. Any fill job `preparing` / `admitted` / `compositing` in the fresh
   snapshot → `Wait` — checked **before** a retry, so a failed run never competes with a fill that
   is still settling (Terra R1-6).
2. Run `failed` / `incomplete`:
   * every active swap segment has a finished result (completed, not a source fallback, the file
     exists — the free handler's own predicate) → `Launch("redeliver:rN", "retry_delivery")` (free:
     stitch + delivery only; the handler refuses when the retry would bill); at most 3 per delivery
     episode — every free retry since the last QA of a `done` run, not per run version, which a
     failing re-stitch moves each time (R3-6) — then `Ask("redeliver_failed:<rv>", "handoff", options=[manual])` (flow-doc
     step 7's manual assembly) and a wait for the run to change.
   * a segment the provider refused with a content filter (a Google safety block, sensitive
     content, copyright -- `app.content_block`) → **VOID fill from the source**, §1.3.4. Handled
     FIRST, one segment at a time, whatever else is unfinished.
   * otherwise (transient failures, a segment row that was never created) →
     `Ask("retry:<runver8>", "regeneration", options=[retry → retry_run])`; approved →
     `Launch(... regeneration=True)`. An approved `retry` option of any retry / blocked ask is
     honoured first (also a legacy retry-only ask answered before the blocked route replaced it).
   * every segment has a result but the run is still `incomplete` (a source promotion refused a
     requeue that would bill) → `Ask("stuck:<rv>", "handoff", options=[manual])`.
3. Run `done` and no QA for the current `(run_version, detector_sha)` →
   `Compute("qa:<rv>:<det>:rN", "subtitle_qa", {run_id, run_version, detector_sha})`; refusals
   and failure counts are scoped to that pair, so a corrected detector gets its own retries
   (Terra R2-9). A QA whose own
   evidence check saw the run change (`actionable: false`) counts as stale. 3 failed rounds for the
   same version → `Fail`. A QA refusal (final not on the source timeline, custom plan) →
   `Fail` with its message.
4. Apply the operator's **durable answers** to the QA (pure, `apply_answers`): accepted VOID
   composites (by fill job id — `--confirm-fill` semantics, only for a composite whose re-delivery
   is confirmed), confirmed false positives and accepted leftovers / shipped keep captions (by
   segment + result token: the segment's result path, kie.ai task id, file identity, fill id and
   run-local bounds — a rerun can rewrite the same path, Terra R2-3 — and the source file's
   identity, R3-5; a QA made on another source identity is redone), an accepted inconclusive
   QA. A composite's answer is read from the question that matches its current verdict
   (`stop:<fid>` when it is dirty, `eye:<fid>` when it is detector-clean, Terra R2-8). `clean` is recomputed with `post_run_qa`'s own formula. Then
   `subtitle_loop.fill_proposal(run_id, effective_qa, segments, context=fresh snapshot)` gives
   the per-segment fill states.
5. Anything in flight (a fill `preparing`/`admitted`/`compositing`, an action effect
   `launching`, a QA running) → `Wait`.
6. **Act on answers** once every operator ask of the current batch (§1.3.1) is answered: the
   chosen option's action is launched under `do:<approval name>:rN` (`restore_takes`,
   `segment_fill_preview` with a widened box, `rerun_segments` for the Omni option), and an
   approved `start:<fid>` / `retry:<rv>` launches its own action. Every `Launch` / `Complete` /
   `Fail` of the step lists **every** answered, unconsumed approval of the step in `consumes` (the
   core rejects a decision that ignores one; the answers stay readable after consumption). Batch
   options are **choices without actions**: a paid choice (the Omni rerun) is launched with
   `regeneration=True` and the core asks its own regeneration approval with the exact params and
   estimate. Only `start:` and `retry:` asks carry an action-bound option, each consumed by its own
   launch.
7. `fill.next` (free, one at a time): `stale_preview` → `Launch("cancel:<fid>",
   "segment_fill_cancel")`; `propose_preview` → `Launch("preview:<rsid>:rN",
   "segment_fill_preview", {mask_box: null})` — the platform measures the inclusive box on every
   frame. At most 3 previews per segment per run version, then the segment goes to the operator.
8. A `propose_start` item (a ready, fresh, default-prompt, measured-box preview within the
   reservation), or a ready preview the flow made itself for an operator-chosen widened mask →
   `Ask("start:<fid>", "regeneration", options=[start → segment_fill_start {run_id, fill_id,
   approved_usd: estimated_usd}], about={run:<rid>})`. **Q1**: every VOID fill is asked, each adds
   its reservation to the budget when approved (Q6). The proposal carries the measured box vs QA's
   expected box, frames masked, the preview sheet URL, `$0.10 reserved / $0.05 expected`.
9. The **operator batch** (one `Ask` per tick until all are asked):
   * `eye:<fid>` — a delivered VOID composite the detector calls clean (or still dirty) needs the
     eye check (flow-doc step 6; the white-text detector misses grey ghosts and smears — 2845eef3:
     segments 4 and 7 came back detector-clean with smears over the background books and hard seams
     at the necklace). Artifacts: the fill sheet `qa_seg<N>_fill.jpg` (result vs source cropped to
     the mask box at every source caption word) and a side-by-side clip of the mask box over the
     segment (`qa_seg<N>_fill.mp4`, ffmpeg crop + hstack, streamed), plus the comparison sheet.
     Options: `accept` (few patched frames pass, rule 6) + the stop-rule options of §1.3.2.
   * `stop:<fid>` — a VOID composite that still shows words (flow-doc stop rule).
   * `operator:<rsid>:<token>` — `fill_proposal`'s `operator` / deterministic-`blocked` states
     (a repeat of an already-paid fill, a result that is not a known Omni output, ambiguous
     previews, a preview that failed with `no_captions_detected` …).
   * `fp:s<N>:<token>` — an unconfirmed false positive (static UI / end-card text also in the
     source): `confirm` / `as_caption` (treat its hits as leftovers → fill).
   * `keepcap:s<N>:<token>` — a keep segment whose source has captions: `accept` (they ship).
   * `inconclusive:<qa8>` — (2026-11-02: an uncalibrated detector only, two options; timing is §1.6)
     uncalibrated or the timeline is not verified: `accept` waives only that
     condition of `clean`. The flags stay false, so no segment-mapped fill is made or started on
     an unverified timeline (an already-approved `start:` is not launched either, R2-4 / R3-1),
     and every other blocker is still asked on its own — a leftover becomes
     `operator:<rsid>:<token>` ("no fill can be placed"), eye checks and false positives keep
     their asks (R3-2).
10. Effective QA clean → `Complete({run_id, qa_effect, accepted…}, depends_on={run:<rid>,
    effect:<qa>})`.
11. Otherwise `Wait` with what is pending (transient fill blockers: Fal disabled, a range attempt or
    audio patch on the run, unfinished segments; or a `manual` choice waiting for the run to change).

#### 1.3.1 Why the eye checks are batched

The first `Launch` after a batch consumes every answer of the step (core rule). If an answer were
the authorisation of a paid action, the second paid choice of a batch would lose it (Terra R1-1).
So batch asks carry **choices only**, bound to the QA compute effect (immutable) rather than to the
run: an answer survives the other segment's re-delivery, is read from its `decision` /
`chosen_option` after consumption, and is revalidated against the segment's *current* result (the
fill id, or the result token) before its action is launched. A free action (restore, widened
preview) is launched directly; a paid one gets its own core regeneration approval. Asking the batch
only once no fill of the QA round is still to be previewed, started or composited, and acting only
once the whole batch is answered, lets the operator answer segments 4 and 7 side by side.

#### 1.3.2 Stop-rule options (the flow doc's, from `subtitle_loop._operator_options`)

| Option id | Action | When |
|---|---|---|
| `accept` | — (recorded; the segment ships as it is) | always |
| `widen_mask` | `segment_fill_preview {mask_box: old ∪ in-segment leftovers, padded}` (free), then its own `start:<fid>` regeneration ask (a second paid attempt). The flow owns that preview's lifecycle: stale → canceled; canceled / failed → made again (≤ 3), then `widen_failed:<rsid>:<token>` asks `accept` / `manual` (Terra R2-7) | leftovers inside the segment |
| `restore_previous_take` | `restore_takes {takes: [the take the fill replaced]}` (free; its captions come back) | the fill replaced a take |
| `manual` | — the operator fixes it by hand (LTX-2.3 inpaint on the same mask, `ffmpeg delogo` over the listed box/windows); the step waits for the run to change and re-checks | always |
| `omni_rerun` | `rerun_segments` of that segment (paid, regeneration — the core asks its cost approval at launch; coordinate ↔ default prompt alternation from `rerun_proposal`; the run prompt for a smear) — **only as an operator option**, never proposed on its own | fewer than 3 preserved outputs |

#### 1.3.3 Best take

After an Omni rerun the operator chose, the next QA compares each rerun segment's hits with the
previous QA round's hits for the output it replaced; when the new take is worse
(`choose_best_takes`: fewest hits, newest on a tie; 2845eef3: the coordinate-prompt rerun went
27→48 and 25→54 hit frames and the round-0 takes were restored by hand), the flow **asks**
(`best:<rerun>:<run version>`, `handoff`, `about = {run:<rid>}`, options `restore` /
`keep_new`, both QA rounds' crop sheets as artifacts — bound to this exact result, R2-6): hit counts do not see smears or picture changes, so the restore is the operator's call
(Terra R1-10). `restore` → `restore_takes` (free). The replaced output's take is found by
the take id the rerun's admission archived (`created_refs.takes`, request R8), else its
`seedance_task_id`, else — a VOID composite has none (promotion clears it, R2-5) — its path when
it already was a take (a restore re-uses that row). Never inferred from timestamps (R3-3): without
provenance there is no best-take question. A restored segment then goes the VOID way (rule 1).

#### 1.3.4 A segment the provider refused — VOID fill from the source (2026-11-02)

Operator decision (Operator, after 514067f2 / d8105ab5 segment 1 was refused 6x by Omni's safety
filter for a phone screen showing the word "suicide"): such a segment is produced by a VOID fill
straight from the SOURCE clip (`docs/segment-caption-fill-design.md` §11). VOID inpaints only the
caption mask; the rest of the frame stays original; Omni is not involved. A blind retry is no
longer the default.

`run_info` marks each failed active swap segment with `content_block` (the category, from the
provider's message); the snapshot gives each segment its fill `basis` and its jobs' `basis`,
`block_task` (the refused provider task the preview answers), `source_as_is`, preview / clip
paths (only files under the fill's own directory) and the provider task's `reservation_held`.
For the first content-blocked segment (index order):

1. no live source preview for this refusal → `Launch("bpreview:<rsid>:<task12>:rN",
   "segment_fill_preview", {basis: "source", mask_box: null})` (free; the platform measures the
   box with the project's detector, e.g. 514067f2 `band_y` [0.73, 0.83]). At most
   `MAX_BLOCKED_PREVIEWS` (3) per (segment, refused task) -- a retry makes a new refusal and a new
   budget, not a new run version. Segment / run gates (other than "Fal is disabled") → `Wait`.
2. a ready preview that went stale (the refusal, the detector, the source or the run changed) →
   `Launch(cancel)` (free), then 1.
3. a ready preview → `Ask("blocked:<fid>", "regeneration", options=[primary, retry → retry_run],
   artifacts=[the preview sheet (+ the source clip for "as is"), producer the terminal run],
   about={run, project_detector}, replaces=<every pending retry: / blocked: / blocked_failed:
   ask of the step>)`:
   * captions found: primary `void` → `segment_fill_start` ($0.10 reserved, ~$0.05 charged,
     cost on the approval; Q1/Q6 -- approved one by one, outside the project budget);
   * no caption in the source clip: primary `source` → `segment_fill_apply_source` (free; the
     question says a detector miss is possible and attaches the clip itself);
   * an input-AUDIO refusal says the delivered run keeps the original soundtrack.
   The approved option's own action is launched (`do:blocked:<fid>:rN`; the paid ones with
   `regeneration=True`, authorised by the option's action hash). With Fal disabled a VOID
   proposal waits (the free preview and "as is" do not).
4. the source fill ended at Fal without a result (refused / failed / abandoned), a deterministic
   prepare refusal (`segment_too_short`, `unsupported_media`, `not_swap`, `bad_box`), or 3 previews
   that did not become ready → `Ask("blocked_failed:<rsid>:<task12>", "regeneration",
   options=[retry → retry_run (paid), manual])`. Never a second paid VOID on its own.
5. an active fill whose Fal task only the operator can resolve (`refused` still holding its
   reservation, `submit_unknown`, `needs_reconciliation`) → a `Wait` -- checked BEFORE the generic
   "a fill is in progress" wait -- naming the task and the state-specific audited commands
   (`close` keeps the held reservation as the charge, `abandon` only once nothing was billed,
   `adopt` / `resume` where they apply).

The promotion requeues the run only once every active swap segment has a result and the requeue
bills nothing; the run then stitches and delivers (first delivery) and becomes `done`, and the
normal QA round runs: the composite is a `void_fill` result (basis `source`) and gets the
`eye:<fid>` eye check (wording: "made from the SOURCE"; for "as is": "the source clip was shipped
as is; a detector miss is possible"), or the stop rule when the detector still sees words. The
`omni_rerun` option is never offered for a source composite (the provider refused those frames).

**`Ask.replaces` (core).** A new flow Ask may name earlier approvals of its step it replaces:
under the approval row locks, a still-pending one is superseded in the same transaction ("replaced
by <name>"); an answered one discards the Ask (the planner replans with the answer, which it
honours or consumes; a stale one is superseded by the core's normal revalidation). This is how the
live retry-only card of 514067f2 is retired without restarting the autopilot. And a completed
step's still-pending approvals are superseded at completion (they could only be answered into the
void).

### 1.6 Timing, the free re-stitch and the fix questions (2026-11-02, after 514067f2 / d8105ab5)

**What happened.** Segment 1 of d8105ab5 was rebuilt from the source (§1.3.4); the caption QA then
found Omni's leftovers in segments 3 and 4, but the alignment check said the final was up to 9 frames
off the source, so `timeline_verified=false`, `fill_proposal` refused every fill, and the flow asked
a one-option "inconclusive" card ("continue as if it were?") and then, per segment, "caption leftovers,
but no fill can be placed" with accept / manual / Omni and **no VOID option**. The operator answered
manual twice; the delivered final kept its captions.

**Root cause of the drift: the stitch, not Omni's picture.** Per-point alignment of that final:
segment medians 0 | −3 | −2.5 | −5.5 | −8.5 | −9.5 frames (−12 at the very end), constant inside
each segment, stepping at the cuts. Omni returns 24 fps clips whose length is rounded UP to whole
4-frame groups: 108 / 144 / 188 / 188 / 100 frames = 4.5 / 6.0 / 7.833 / 7.833 / 4.167 s for slots of
4.4 / 5.9 / 7.767 / 7.767 / 4.167 s, i.e. +3, +3, +2, +2, 0 frames at 30 fps. `media._stitch_into`
concatenated every clip at its own length and `-shortest` cut at the end of the source sound, so each
long clip pushed all later segments late (cumulative 0, 3, 3, 6, 8, 10 — the measurement). Even the
segment-1 composite (built from the source, exact length) sat 3 frames late behind segment 0's long
clip. The picture inside each Omni clip is not shifted (residual 0 ± 0.5 frame). With
`audio_mode=original` that is lip sync drifting to 0.3–0.4 s at the end of a 39 s video; a prod sample
shows the same rounding on other Omni runs (and Seedance 2.5 clips coming back ~0.2 s SHORT).

**1. The stitch conforms each clip to its slot** (`pipeline_v2.stitch_frame_counts`,
`media.stitch(frame_counts=…)`): on the default timeline with the source soundtrack and no rolls
(prod: 0 of 433 segment defs have rolls), segment *i* gets `round(end_i·fps) − round(start_i·fps)`
output frames — counted at the boundaries, so the budgets sum to the source's own frame count and
rounding never accumulates. Each clip is put on the output grid first (`setpts=PTS-STARTPTS`, scale,
pad, `fps`) and then cut (`trim=end_frame=N`), so the cut is in delivered frames whatever the clip's
own rate. A longer clip loses its tail; a shorter one is left as it is (never padded with invented
picture — a short result is already flagged; Seedance's short clips are a follow-up). Custom plans,
`audio_mode=seedance` and rolled segments keep today's stitch. The range-attempt candidate stitch
uses the same conform (a promoted candidate must not bring the drift back). Verified on the real
clips of d8105ab5: re-stitched with the conform, every 10 s window has median offset 0, 1165 frames
= the source, `in_sync=true`.

**2. QA measures timing per segment** (`subtitle_loop.segment_timing`, `post_run_qa`): the alignment
search reaches 15 frames (was 9, the search limit d8105ab5 sat on), and each segment gets the median
offset of the points ≥ 0.4 s inside it: `in_sync` (±1 frame, the flow doc's final-QA rule), `off`
(`beyond_search` when most points sit at the limit) or `unknown` (< 3 points). `qa.timing` holds the
rows, `qa.timing.verified` = the final is as long as the source, no swap segment is off, and where a
segment is too short to measure the run-wide windows agree. `clean` needs it (an off segment is a
lip-sync defect, not a warning). The `subtitle_qa` compute adds `qa.timing.clips`: each swap
result's length against its slot (ffprobe), from which the planner predicts the shift an
unconformed stitch causes (`stitch_drift`; `explained` only when EVERY measured segment matches the
prediction within 1.5 frames).

**3. Leftovers are attributed per segment** (`subtitle_loop.attribution`): a span belongs to its
segment unless it lies within the picture shift (+0.1 s) of a cut — at the start the segment's own
shift decides where the cut sits in the final, at the end the next segment's. A segment whose every
span is ambiguous is not filled (`ambiguous`); the other segments go first; when nothing else can
move the operator is asked `edge:<rsid>:<tok>` ("Caption words at … sit right at the cut between
segment 3 and segment 4, and the picture there is shifted by 8 frames …": rebuild this segment /
ship as is / by hand). The run-wide "not verified → no fill" rule is gone (it stays only for a QA
without timing, which the planner never uses).

**4. The planner order in `plan_clean`** (after the in-flight / retry / QA rules of §1.3):
* a QA result without `qa.timing` (written before this change) is done again (free; the same
  compute version, the result just gains fields);
* a segment off, or the windows off with nothing measurable, and no `restitch_run` launched for this
  run in the step, and no approved fill start waiting → `Launch("restitch:<rv>:rN", "restitch_run",
  {run_id})` — FREE, no question: the run's final is invalidated and the run requeued; the stitch
  (now conformed) replaces the bytes behind the run's Drive link; then QA again. Once per run:
  * still off by exactly the predicted frames (the conform did not take) →
    `Ask("restitch_failed:<rv>")` (ship as is / by hand);
  * still off otherwise (the model's picture itself is shifted) → the segment is **rebuilt from the
    original** like a dirty one (`fill_proposal(rebuild=[…])`: a VOID fill's composite is the source
    clip with the captions inpainted, so it is exactly on the timeline); a segment whose source shows
    no caption has nothing to mask → `Ask("sync:<rsid>:<tok>")` (Omni again / ship / by hand);
  * nothing measured off but the timeline not proven (segments too short / still, or another length)
    → `Ask("timing:<qa8>")` (lip sync looks right / I fix it) — a real question with two options;
  * the run's current result is a promoted **audio patch**: a re-stitch takes the original sound
    again and drops it, so the flow asks `restitch_patch:<rv>` (re-stitch anyway — the handler gets
    `discard_audio_patch` only from this choice / ship as is / by hand) instead of a restitch the
    handler would refuse forever (gate R2-1); a rolled run is never re-stitched (the conform does
    not apply): its shifted segments go the rebuild way at once (gate R1-2).
* `inconclusive:<qa8>` is asked ONLY for an uncalibrated detector, with two options (I looked: no
  captions left / not clean: I fix the band or the video).

**5. The fix question leads with the fix.** After the free preview the flow asks
`void:<rsid>:<tok>:<fid>` (`regeneration`, replaces the old single-option `start:<fid>`; a pending
legacy `start:` card is replaced, an answered one is honoured): what is wrong in plain words, the
proposal (VOID rebuild from the original, the cost, what happens next), and the queue ("Next in line:
segment 4 (asked after this one: one rebuild at a time)"). Options, in order:
`start` (the only action-bound one: `segment_fill_start`, $0.10 reserved / ~$0.05), `omni_rerun`
(a choice: launched with `regeneration=True`, the core asks its cost), `accept` (ship as is), `manual`
(by hand). Artifacts: the QA crop sheet of exactly those leftover frames (delivered | original) and
the preview sheet (the area VOID repaints). One question per segment, not a combined "rebuild 3 and
4": each preview is bound to the final it was made on and the first fill's re-stitch makes the
second stale, so a combined approval could not be bound to previews (Terra §2). `accept` / `manual`
are bound to the segment's RESULT token (not to the preview), so they hold whichever preview asks
next; `manual` keeps the segment out of the automatic fixes until its video changes. Legacy
`operator:<rsid>:<tok>` answers count only for `accept` (a `manual` there was the only way out of a
question without a fix option).

Over a promoted audio patch every clean-step question also says that rebuilding, restoring or
re-stitching drops the audio fix, and the actions they lead to carry `discard_audio_patch` (they
were refused forever with a hard `False`, gate R3/R4); an answered legacy `start:<fid>` is not
launched over a patch — the `void:` question, which discloses it, is asked instead. The no-fix
question of a segment is named `hand:<rsid>:<tok>` (not the legacy `operator:` name, whose
`manual` answers must not silence it).

**6. "Ask again".** `core.reopen` (card: "Your answers — ask again"; API
`POST /api/v2/projects/{pid}/autopilot/approvals/{aid}/reopen`) takes back an answered choice that
launched nothing (no action on the chosen option), in the current generation of a step that is still
running, latest generation of its name; never a core approval or an answer that authorised a launch.
The approval is superseded (journalled, the answer stays in the record), the autopilot version moves
(a plan in flight is discarded) and the flow asks the same name afresh with its current proposal.

**7. Plain language.** Every clean-step question says what is wrong, what the assistant proposes,
what it costs and what each option does next; "timeline verified", "calibration", "alignment",
"actionable", "mask box" are gone from the questions (they stay in the proposal data).

**Live autopilot e6fd6ac7 — what changed while this was built.** At 18:42–18:59 the operator
rebuilt segment 4 by hand (VOID fill 3beedbec, applied, final re-stitched — still unconformed) and at
19:23 made a preview for segment 5 (925c657f, ready). The VOID fill was a manual paid action, so the
autopilot is **paused** ("manual paid action: fal_inpaint … Resume to continue"). The re-stitched
final still runs late from segment 1 on (0 | −3 | −2 | −5.5 | −8 | −9): the hand-made composite of
segment 4 is exact, the long Omni clips before it are not — a rebuild by hand cannot fix the lip
sync, only the conformed re-stitch does.

**After the deploy and Resume (no restart):** the run changed, so a new QA (free) → `restitch_run`
(free; consumes the answered inconclusive / eye / two operator answers — they stay readable, the
segment-1 eye check stays accepted) → QA on the resynced final → free preview of segment 3 →
`void:<rs3>:ec33ffbf:<fid>` (segment 4 is no longer in the queue: it is a VOID composite now) → in
the batch, the eye check of the operator's segment-4 composite (`eye:3beedbec…`). The segment-5
preview is not the flow's: the re-stitch makes it stale; the operator makes a new one if still
wanted. Replayed in `test_replay_e6fd6ac7_as_it_stands_after_the_operators_manual_fill`.

**Had nothing changed (the state at 18:20, replayed in `test_replay_e6fd6ac7_after_the_deploy`).** Its stored QA has no timing → a new QA
(free, ~4 min) → `restitch_run` (free; consumes the answered inconclusive / eye / two operator
answers — they stay readable, the segment-1 eye check stays accepted) → the stitch conforms →
QA on the resynced final: every segment in sync, leftovers in 3 (21.2–22.1, 22.7–24.2 s) and 4
(29.9–30.4, 32.2–33.8 s) → free preview of segment 3 → `void:<rs3>:ec33ffbf:<fid>`; after its fill and
re-delivery → QA → the eye check of the rebuilt segment 3 and the preview + `void:` question of
segment 4. Replayed from the real data in `tests/test_autopilot_subtitle_replay_514067f2.py`.

**Review record (GPT-6.1 Terra, `codex exec`, effort high).** §2 design review (before code): 3
BLOCKER, 6 MAJOR, 1 MINOR against the first proposal — accepted: the conform is counted in output
frames after the `fps` filter, never pads, is tested frame by frame (24→30 fps, real clips); no
restitch over a promoted audio patch without the operator; range-attempt candidates conformed; the
re-stitch is bounded per run with an operator fallback; a sync-only rebuild only where the source
segment has captions (else a question); reopen limited to actionless choices in the running step;
uncertain timing is `unknown`, not zero, and the planner re-stitches before it rebuilds. Rejected:
the "recovery trap" (the core restores only the first consumed approval) — `_apply_launch` puts the
authorising approval first in `consumed`, so `eff.approval_id` is the authoriser. §1 gate, 5 rounds:
R1 unbounded shifts attributed with a 16-frame margin; restitch admitted for rolled runs (fixed);
R2 a restitch over an audio patch refused forever; unbounded shift attributed the opposite half
(fixed); R3 VOID starts over an audio patch refused forever (fixed); R4 legacy `start:` and Omni
choices over an audio patch (fixed); R5 NO FINDINGS.

**Follow-ups (not in this change).** Check each Omni segment's sync when it completes, before the
stitch (with the conform, a clip's own content shift is the only remaining cause, and the replay
found none); short clips (Seedance 2.5 returns ~0.2 s short): a repair decision rather than padding;
rolled segments are stitched with their rolls (pre-existing; 0 in prod).

### 1.7 Batch VOID fills, fewer questions, VOID from the source (2026-11-03)

Design and review record: `docs/segment-fill-batch-design.md`. What changed in this flow:

* **Previews in parallel, one batch.** Every segment that needs a VOID rebuild gets its free
  preview at once (one launch per tick; a preview is bound to its own segment, so another
  fill's re-stitch does not make it stale). When they are ready, ONE `segment_fill_batch_start`
  covers all of them (at most 12): inside the run's VOID allowance (`video_projects.
  void_auto_usd_per_run`, default `AUTOPILOT_VOID_AUTO_USD` = $1.00, project page "Auto VOID up
  to") it is launched without a question (`autovoid:<rv>:rN`, authorization `auto`, re-checked
  in the admission); above it, or over a promoted audio patch, ONE question `vbatch:<rv>:<digest>`
  ("Segments 0, 2, 3, 4, 5, 6, 7 and 10 still show burned-in captions: rebuild all 8 in one batch
  with VOID from the ORIGINAL video ($0.80 reserved, about $0.40 charged). ...") with one combined
  contact sheet (`subtitle_batch_sheet` compute) and the options start / per_segment (today's
  `void:` questions) / accept / manual. One promote, one re-stitch, one re-delivery.
* **Fewer eye checks.** A detector-clean composite is accepted on its own when no sampled face
  (InsightFace, 2 fps over the source segment) and no screen / app UI (heuristic,
  `fill_screen`) is under its mask box; the hand-off names those segments and shows their fill
  sheets. The doubtful ones are ONE end-of-run card `eyes:<qa8>` (accept_all / per_segment /
  manual).
* **Stop rule.** A dirty composite first gets ONE automatic retry with a grown mask (box
  + leftovers, padded, glyph mask dilated 8 px; `retry_of` lineage), inside the allowance; only
  a dirty retry (or a failed one) is asked `stop:<fid>`.
* **Stale cards.** The core supersedes, on every planned decision, the step's pending approvals
  whose bound versions / artifacts no longer revalidate; the flow retires (`Wait/Ask.retire`)
  pending questions of an older QA round, result or run version (81d0afc0's `operator:` and
  `stop:` cards from before the re-stitch).
* **VOID from the source** (project `caption_removal_method = void_source`, the default for new
  subtitle-removal projects): `create_run` makes a parked `incomplete` run that sends nothing to
  Omni; the clean step makes the source previews, applies the caption-less ones as is (free),
  starts ONE batch for the rest (same allowance rule), and the run stitches and delivers once
  every segment has its fill. A segment VOID cannot take is asked (Omni for it, paid, cost asked
  / by hand); Omni is never called on its own. The launch reserves nothing.

### 1.4 `handoff`

`Ask("handoff", "handoff")` with the final and the last QA's comparison sheet as artifacts (the
run is `done`, terminal), `about = {run:<rid>}`. It is also the **picture review** the detector
cannot do (flow-doc step 4: invented words in another style, an unrequested change — border,
outpaint, phone screen — lip sync); the question says so (Terra R1-11). The proposal is proposal = `subtitle_loop.handoff_proposal`'s request (`POST /runs/{rid}/to-project`,
localisation, whole scope), the keep ranges to mark on the new project after its analysis, the
suggested mode (`assistant`, the default of project-from-run, §2.1), warnings (± frame sync).
Approve → `Complete({run_id, keep_ranges, to_project})`; the operator creates the localisation
project from the run page (not a handler, AICP-362 decision). Reject → `Fail` with the comment.

## 2. Replay: project 328341d4 / run 2845eef3

`tests/test_autopilot_subtitle_plan.py::test_replay_2845eef3` drives the planner over synthetic
detector scans shaped like the real ones (9 scene cuts as analysed, all swap; captions at
y 0.60–0.64 changing every 0.5 s; the end card's static "App of the Day" text at y ≈ 0.43 from
49.1 s; round-0 leftovers in segment 4 at 23.5–28.2 s and segment 7 at 43.3–48.0 s) with every
provider mocked:

1. `segmentation`: the review proposes **segment 8 keep** (end card) → eye-check ask → `apply` →
   `patch_segments` with segment 8 keep; launch cost 1512 → 1344 credits billed.
2. `launch`: `create_run` gemini-omni, 8 swap segments (reservation ⌈s⌉ × 42 per clip = 2268
   credits; billed 8 × 168 = 1344).
3. `clean` round 0: QA → leftovers in 4 and 7 → free preview of 4 → `start:<fid4>`
   **regeneration** ask ($0.10 reserved) → start → re-delivery → QA → preview of 7 →
   `start:<fid7>` regeneration ask → start → QA: both detector-clean → `eye:<fid4>`,
   `eye:<fid7>` with their fill sheets. **No Omni rerun is proposed at any point** (only as the
   `omni_rerun` option of an eye-check/stop ask).
4. Both accepted → `Complete` → `handoff` ask with keep range 49.1–54.52.

## 3. Action handlers this flow needs

Shipped by the core and reused: `create_run`, `patch_segments`, `restore_takes`,
`rerun_segments`. B ships the following (phase 2, `app/autopilot/flows/subtitle_handlers.py`),
each with the core's tests: every refusal commits nothing, success commits once, a crash after the
commit and before T2 reconciles to `launched` with the right refs (§4.4).

| Handler | Endpoint | Paid | Regeneration | Conformance |
|---|---|---|---|---|
| `segment_fill_preview` | `api_segment_fill.create_fill_preview` | no | — | conforms: `_require_fillable` raises before its single commit; the prepare enqueue follows the commit and a never-claimed `preparing` row is re-enqueued by the recovery tick. The new job id is assigned in a `before_flush` listener and written to `created_refs.fill_jobs` in the same transaction |
| `segment_fill_cancel` | `api_segment_fill.cancel_fill` | no | — | conforms (one commit, 409 before it) |
| `segment_fill_apply_source` | `pipeline_segment_fill.apply_source_as_is` | no | — | `created_refs.fill_jobs` written BEFORE the service's single commit (its promotion commits, then enqueues `process_run`); refusals before it; `reconcile` → launched when that job is `applied` |
| `segment_fill_start` | new service `pipeline_segment_fill.admit_start` (below) | yes | `always_regeneration` (Q1) | **refactored** |
| `retry_run` | `api_v2.retry_run` | yes | `always_regeneration` | guarded wrapper (below) |
| `retry_delivery` | `api_v2.retry_run` | no | — | guarded wrapper (below) |
| `restitch_run` | `api_v2._invalidate_final_and_requeue` (2026-11-02, §1.6) | no | — | under the run lock (`_admit_segment_rerun`): a `done` run, default plan, source soundtrack, no fill / range attempt / audio patch / GPT Image job in flight, no promoted audio patch as the result, estimate 0, no preprocessing hole; one commit, enqueue after it |
| `patch_caption_detector` | `api_v2.update_project` (caption_detector only) | no | — | conforms (validation 400 before its commit); fenced (below) |

### 3.1 `segment_fill_start` — single-commit service

Today `start_fill` (api_segment_fill.py) **commits a `canceled` stale preview before raising 409**
and maps any commit `IntegrityError` to 409 (the core's reason for not shipping it). Phase 2 splits
it:

* `pipeline_segment_fill.admit_start(db, run, fill_id, *, approved_usd, discard_audio_patch)
  -> (job, task, created)` does every check and write of today's endpoint in the caller's
  transaction and **never commits and never rolls back** the caller's transaction (today's
  `db.rollback()` on `AdmissionRefused` / `BudgetExceeded` moves to the endpoint wrapper: inside
  the handler it would drop the core's pending effect, approval consumption and scopes — Terra
  R1-4); it raises typed `FillStartRefused(status, detail, stale=None)` and the core rolls the
  whole admission back. Order: lock project → run (`_admit_segment_rerun`, re-entrant under
  the core's locks 1 and 3) → fill row → `_require_fillable` → base check → rerun conflicts →
  audio-patch ack → paid re-stitch check → `approved_usd ≥ estimated_usd` → `admit_provider_task`
  (provider-task savepoint + spend reservation) → job `admitted`. The job's status change is
  flushed inside a SAVEPOINT, so the partial unique index "one active fill per run" raises there
  and becomes a refusal (`FillStartRefused(409)`) instead of a commit-time `IntegrityError`.
  A stale base raises `FillStartRefused(409, stale=<part>)` **without** writing.
* The endpoint keeps its behaviour exactly: on `stale` it marks the job canceled, commits and
  answers 409 (operator UX unchanged); on success it commits, enqueues `fal_submit` when created.
* The handler: `validate` (run of this project, fill job of this run), `targets = {runs: [rid],
  run_segments: [rsid], fill_jobs: [fid]}`, `estimate` = the reservation `pp.admit` makes
  (`fal_models` worst case for VOID at `VOID_RESOLUTION` / `KIE_USD_PER_CREDIT`: $0.10 = 20
  credits), `execute` = `admit_start` + `created_refs = {fill_jobs: [fid], provider_tasks:
  [tid]}` + the `fal:<tid>` spend scope registered in the transaction + **one** commit + enqueue
  after it. A stale preview is a refusal with nothing written; the flow cancels it with the free
  handler on the next tick and makes a new preview. `reconcile`: refs present → `launched`
  (single commit: the effect row exists only if the admission landed), else `unknown`.
* Lock order (§5.2): gate (0) → project (1) → autopilot (2) → run (3) → approvals (4) → effect
  (5) → fill row, provider-task savepoint (6) → spend days (7) — the order `start_fill` already
  uses (project → run → fill row).

### 3.2 `retry_run` / `retry_delivery` — terminal runs only

`api_v2.retry_run` accepts `queued/processing/stitching/delivering` and says it does not check
liveness (api_v2.py:6957). Both handlers call it only after re-checking, **under the run row lock
the core already holds** (lock 3, refreshed):

* status ∈ {`failed`, `incomplete`} — never a non-terminal run (the reaper owns orphans);
* both refuse a run with a preprocess hole (`preprocess_hole_index`): that branch moves the run
  to `preprocessing` and reserves GPT Image tasks the estimate below would not include (Terra
  R1-5; the flow's runs never preprocess);
* `retry_run` (paid, `always_regeneration`): `estimate = spend.estimate_run_credits(db, run)` —
  what the endpoint's `reserve_run(release_unused=True)` reserves;
* `retry_delivery` (free): refuses (409, nothing written) unless status is `failed`, the estimate
  is 0 and the run has no preprocess hole (`preprocess_hole_index` — that branch bills GPT Image);
  the re-stitch + delivery reuses every finished segment.

Both: one commit (the endpoint's), enqueue after it; refusals are its 409s before the commit;
`targets = {runs: [rid]}`; `reconcile` → `launched` (the effect exists only when the commit
landed).

### 3.3 `patch_caption_detector`

Params `{caption_detector: <full config>, expect_detector_sha}`. `execute` re-reads the project
(locked by the tick, lock 1), refuses 409 if `sha(caption_detector) != expect_detector_sha`, then
calls `update_project(pid, ProjectUpdate(caption_detector=…))` (its single commit). Phase 2 also
makes `update_project` take the project row `FOR UPDATE` before writing `caption_detector`, so a
manual PATCH cannot interleave between the check and the write. Optional `expect_no_band: true`
(the located band, §1.5): also refuses 409 when the project's detector names a `band_y` by then —
a located band never replaces one of the project's own.

## 4. Compute functions (registered by B, run on `autopilot_compute`)

| Function | Args | Result (≤ 256 KiB) |
|---|---|---|
| `subtitle_review` v1 | `project_id, segments_sig, detector_sha, faces` | `review_segmentation` output (contact sheets in the attempt dir), the input segments (id, index, bounds, action, rolls), `segments_sig` and `detector_sha` re-read at the end (a mismatch marks it stale) |
| `subtitle_qa` v1 | `run_id, run_version, detector_sha` | what `python -m app.subtitle_loop qa --run` prints (`qa`, `fill`, `operator_options`, `handoff`) with sheets in the attempt dir, the QA's segment inputs, each segment's `seedance_task_id` and result path, the source fps, `run_version` read in the QA's REPEATABLE READ snapshot and **re-read in a fresh transaction** after the scans (`actionable: false` when it changed — a re-read inside the same snapshot would see nothing, Terra R1-8; fill rows changing does not matter, the planner re-reads them every tick), plus `qa_seg<N>_fill.mp4` side-by-side clips for VOID composites (ffmpeg crop + hstack over the segment window, streamed, never a decoded video in RAM); alignment windows trimmed (`plan.qa_result`); since 2026-11-02 `qa.timing` (per-segment offsets) and `qa.timing.clips` (each result's length vs its slot), §1.6 |

Both open their own read-only DB session (`SET TRANSACTION … READ ONLY`), never write the DB,
and write files only into their attempt dir. Detector scans stream frames (AICP-343).

## 5. Requests to the core (phase 2 confirms or adapts)

* **R1 — planning reads.** The planner needs a fresh fill snapshot every tick (fill job states,
  the read-only API gates, the run's takes). Phase 2 reads it in the planning session
  (`ctx.db`, read-only) through one helper split out of `subtitle_loop._fill_context`: the gates
  and the job summaries only — no ffprobe (the source fps comes from the QA result) and **no
  composite identification** (`_is_composite` hashes a whole restored composite; the QA compute
  already classified each result, Terra R1-13). It `stat`s files (`os.path.exists`,
  `file_identity`) — reads, not writes.
* **R2 — the fill preview sheet as an artifact.** A `ready` fill job's `preview.jpg` is written
  once by prepare and never again; the start ask would show it best as an artifact. Producers
  today are terminal runs and compute effects only, so phase 1 links it in the proposal
  (`preview_url`); with a `fill:<id>` producer (bound by the job's status + base) it becomes an
  artifact.
* **R3 — consumed answers stay readable.** The planner relies on `ApprovalView.decision` /
  `chosen_option` surviving consumption (in the core's WIP contract) and lists every answered,
  unconsumed approval of the step in each `Launch` / `Complete` / `Fail`.
* **R4 (required) — a `project_detector:<pid>` version key** (the project's `caption_detector`
  digest, fenced by the project row lock; `update_project` takes that lock first, §3.3), like the
  core's `project_transcript:` key. `clean`'s `Complete` depends on it (a detector change reopens
  QA and the hand-off, never the paid launch); the computes record the detector sha too. Without
  it a detector PATCH between planning and APPLY could complete on obsolete QA (Terra R1-9).
* **R5 (core bug) — `_consume` must be atomic.** In the WIP core `_consume` marks approvals
  consumed one by one and commits when a later one fails revalidation, so an earlier answer is
  committed `consumed` with no action (and on a paid launch, budget changes made before it are
  committed too). The flow's "consume every answer" rule makes this reachable. Fix in the core:
  lock and revalidate the whole set first; on a mismatch commit only the supersede (Terra R1-2).
* **R6 — a `fill:<id>` about-key** (the job's status + base digest, fenced by the fill row lock),
  so a pending `start:<fid>` whose preview was canceled or expired (7-day reaper) is superseded at
  the latest when answered. Without it the planner already moves on (a new preview and a new
  `start:` ask; the old answer is consumed unused) but the stale pending ask stays on the card
  (Terra R1-7).
* **R7 (hard prerequisite) — `EffectView.params`.** The planner reads an earlier rerun's params
  (which segments it regenerated); the WIP `EffectView` does not expose them (Terra R2-11). The
  planner fails closed (raises, so the core backs off and pauses) on a rerun effect without
  params rather than skipping the best-take question (R3-7).
* **R8 — `rerun_segments` records the takes it archived** (`created_refs.takes = {rsid:
  take_id}`, captured in the admission transaction like `create_run`'s run id), so the best-take
  comparison is bound to the exact replaced output (R3-3).
* **AICP-360** (the reaper / enqueue race, core §5.6) stays an activation prerequisite: a paid
  `retry_run` is exposed to it like any admission (Terra R1-12).

## 6. Pure step logic (phase 1)

`backend/app/autopilot/flows/subtitle_removal_plan.py` imports nothing from the core (namespace
package, no `__init__.py`, like the AICP-365 flow). It returns plan decisions with the contract's
field names (`Complete`, `Wait`, `Launch`, `Compute`, `Ask`, `Fail`, `Option`, `Artifact`);
phase 2's adapter maps them 1:1 onto `app.autopilot.contract`. Functions:
`validate_params`, `review_result` / `qa_result` (the compute result schemas), `segments_sig`,
`too_long_segments`, `plan_segmentation`,
`launch_params`, `launch_estimate`, `budget_lines`, `plan_launch`, `apply_answers`,
`stop_rule_options`, `best_take_restores`, `plan_clean`, `plan_handoff`.

## 7. Deploy notes

No migration, no new worker: the handlers run inside the core's ticker, the computes on the run
workers' `autopilot_compute` queue (core §5.1). The flow registers itself only in phase 2 and only
takes effect with `AUTOPILOT_ENABLED=true` (core §9, after AICP-360).

2026-11-02 (§1.6): no migration, no new worker, compute versions unchanged; a new free handler
`restitch_run` (registered by the flow, runs in the ticker) and a core `reopen`. **The stitch change
applies to every run** whose final is stitched after the deploy (default plan + source soundtrack +
no rolls: face-swap runs too): a model clip longer than its segment is cut to its slot. Finals
stitched before the deploy are not touched until something re-stitches them (a delivery-only retry
reuses the old final; `_final_is_stale` checks segment mtimes, not the stitch version).

## 8. Design review (§2, GPT-6.1 Terra, `codex exec`, reasoning effort high)

**Round 1** (on the first draft of this doc): 4 BLOCKER, 8 MAJOR, 1 MINOR — all checked against
the code.

| # | Sev | Finding | Resolution |
|---|---|---|---|
| R1-1 | BLOCKER | the first launch consumes every answer of a batch, so a second paid choice loses its authorisation | accepted: batch options are choices without actions; a paid choice gets the core's own regeneration approval at launch; only `start:` / `retry:` options authorise (§1.3, §1.3.1) |
| R1-2 | BLOCKER | the core's `_consume` commits earlier answers `consumed` when a later one fails revalidation | core bug, reported as R5 (§5) — the flow cannot fix it |
| R1-3 | BLOCKER | segment edits after `segmentation` reach the paid launch unreviewed | accepted: `launch` re-runs the segmentation planner when the segments changed and `expect`s the planned segment version (§1.2) |
| R1-4 | BLOCKER | `start_fill`'s `db.rollback()` on refusal inside the handler would drop the core's effect | accepted: `admit_start` never commits or rolls back; typed refusals; the endpoint keeps the rollback (§3.1) |
| R1-5 | MAJOR | the retry estimate omits the preprocess branch's GPT Image tasks | accepted: both retry handlers refuse a run with a preprocess hole (§3.2) |
| R1-6 | MAJOR | a failed run is retried before an in-flight fill is considered | accepted: in-flight fill jobs are checked first (§1.3 step 1) |
| R1-7 | MAJOR | a canceled / expired preview strands its start approval | partly: the planner already moves on (new preview, new `start:` ask, the old answer consumed unused); the stale pending ask on the card needs a `fill:<id>` about-key — R6 |
| R1-8 | MAJOR | a re-read inside the REPEATABLE READ snapshot cannot see a change | accepted: the QA compute re-reads `run_version` in a fresh transaction (§4) |
| R1-9 | MAJOR | detector changes are not fenced at completion | accepted: `project_detector:<pid>` is a required core key (R4); the planner redoes QA on a mismatch and fences `Complete` with it |
| R1-10 | MAJOR | an automatic best-take restore can undo the operator's visual decision | accepted: the restore is asked (`best:<rerun>`) with both crop sheets (§1.3.3) |
| R1-11 | MAJOR | a detector-clean Omni segment reaches hand-off without the picture comparison | partly: the hand-off approval is the picture review (comparison sheet + final, the question lists the checks; Reject stops). A per-segment "invented text" fix has no platform tool (no detector box to mask), so it stays the operator's |
| R1-12 | MAJOR | retry dispatch is exposed to the reaper race | accepted as documented: AICP-360 is an activation prerequisite (§5) |
| R1-13 | MINOR | a fresh fill snapshot per tick hashes restored composites | accepted: the planning helper skips composite identification (R1) |

**Round 2** (on the revised doc + the committed planner, same session): 2 BLOCKER, 9 MAJOR, 1 MINOR;
R1-1, 4, 5, 6, 8, 11, 12 and 13 confirmed sound.

| # | Sev | Finding | Resolution |
|---|---|---|---|
| R2-1 | BLOCKER | segments edited back after the keep patch completed on the old approval | accepted: a review whose patch was admitted is spent; a new round is asked (§1.1) |
| R2-2 | BLOCKER | a detector change after segmentation bypassed the pre-launch review | accepted: segmentation records the detector digest; `launch` re-reviews on a change and `expect`s `project_detector:<pid>`; the eye check and patches are fenced on it (§1.1, §1.2) |
| R2-3 | MAJOR | durable answers bound to a path a rerun can rewrite | accepted: the result token adds the task id, file identity and run-local bounds (§1.3 step 4) |
| R2-4 | MAJOR | accepting an inconclusive QA turned the timeline flag on and enabled fills | accepted: the flags stay; `accept` hands off as it is (§1.3 step 9) |
| R2-5 | MAJOR | best take missed a replaced composite (no task id) | accepted: path, then the newest take archived after the rerun (§1.3.3) |
| R2-6 | MAJOR | a best-take answer reusable after the result changed | accepted: the name carries the run version, `about` the run (§1.3.3) |
| R2-7 | MAJOR | a stale widened preview stranded the flow | accepted: the flow cancels / remakes it, bounded, then asks (§1.3.2) |
| R2-8 | MAJOR | an old `eye:` answer masked a later `stop:` answer | accepted: the answer is read from the question matching the current verdict |
| R2-9 | MAJOR | failure / refusal counts mixed detector inputs | accepted: compute names and counts are scoped to (version, detector) |
| R2-10 | MAJOR | the detector fence was optional in the planner | accepted: `project_id` / `detector_version` are required; R4 is a hard phase-2 dependency |
| R2-11 | MAJOR | `EffectView` has no `params` | accepted as request R7 (§5) |
| R2-12 | MINOR | retry used segment status only and failed after 3 deliveries | accepted: the free handler's finished predicate (`has_result`) and a `redeliver_failed` hand-off (§1.3 step 2) |

**Round 3** (on commit 0126185, same session): 7 MAJOR, all follow-ons of round-2 resolutions;
R2-1, 6, 7, 8, 9, 10 confirmed sound.

| # | Finding | Resolution |
|---|---|---|
| R3-1 | an already-approved `start:` could launch on an unverified timeline | accepted: start asks and start launches (and widened previews) need an actionable fill proposal |
| R3-2 | the inconclusive waiver accepted every other defect | accepted: it waives calibration / timeline only; leftovers that cannot be filled are asked per segment, eye checks and false positives keep their asks |
| R3-3 | the timestamp fallback could pick another take | accepted: provenance from the rerun's `created_refs.takes` (R8), task id or path; no timestamps |
| R3-4 | a band patch PATCHed back completed on the old approval | accepted: a landed band patch spends the review too |
| R3-5 | keep-caption answers not bound to the source | accepted: the source identity is in the token and a QA on another source is redone |
| R3-6 | delivery retries unbounded when each re-stitch moves the version | accepted: counted per delivery episode |
| R3-7 | R7 open: missing effect params silently skip best take | accepted: R7 is a hard prerequisite; the planner fails closed |

The §2 rounds stop here: round 1 was structural, rounds 2–3 refined edge cases of the same
mechanisms. The phase-2 §1 gate reviews the wired code against the real core.

## 9. Phase 2 implementation notes

* **Adapter** (`subtitle_removal.py`): builds the planner inputs from the context — effects from
  `ctx.effects()` (with `EffectView.params`, R7), every flow approval of the step from `ctx.db`
  (core `#budget` / `#regeneration` / `#overrun` approvals excluded), the run's active swap
  segments with `file_identity` (stat only) and `has_result`, the fill snapshot
  (`subtitle_loop.fill_snapshot`: the real read-only API gates and job summaries, no ffprobe, no
  composite hashing — R1), `project_detector:<pid>` through `ctx.version` (R4), and the
  rerun's `created_refs.takes` (R8, a list of take ids) mapped to `{run_segment_id: take_id}`.
  Plan decisions map 1:1 onto `app.autopilot.contract`.
* **Steps**: `segmentation` (free, asks `handoff`), `launch` (paid, `estimate` = what
  `create_run` reserves, asks `handoff` for a re-review), `clean` (paid, asks `regeneration` /
  `handoff`, estimate 0: regenerations are approved one by one, Q1/Q6), `handoff` (free).
  `budget_lines` adds the zero-credit VOID / long-video notes.
* **Handlers** (`subtitle_handlers.py`) as §3. `segment_fill_start` goes through the new
  `pipeline_segment_fill.admit_start` (never commits or rolls back; a stale preview raises
  `StalePreview` with nothing written; the one-active-fill index is checked at a savepoint
  flush); `start_fill` keeps its operator behaviour (stale → canceled + 409, refusals roll back).
  A 5xx from the Fal admission (no live fal worker) is a refusal with backoff, not an error.
  `update_project` already takes the project row first (core).
* **Computes** (`subtitle_compute.py`): `subtitle_review` (InsightFace samples at 1 fps on the run
  workers) and `subtitle_qa` (the CLI's `qa` output, a mask-box clip per VOID composite made by
  ffmpeg crop + hstack, `run_version` re-read in a fresh transaction).
* **Declined core requests**: R6 (`fill:<id>` key) — the planner already depends on `run:` and
  moves on from a canceled / expired preview; R2 (preview as an artifact) — the start ask links
  `preview_url` in its proposal.
* **Tests**: `tests/test_autopilot_subtitle_plan.py` (planners, 2845eef3 replay) and
  `tests/test_autopilot_subtitle_flow.py` (the real core end to end: review compute → eye check →
  keep patch → budget → `create_run` → QA compute → free preview → `start:` regeneration
  approval → `admit_start` + `pipeline_provider.admit` → eye check of the composite → hand-off →
  completed; handler refusal paths commit nothing).

## 10. §1 gate (GPT-6.1 Terra, `codex exec`, reasoning effort high)

Over `df72a9d..HEAD` (phase 2) and `25436e7..34bf02e` (the planner). **Round 1**: 2 MAJOR, both
fixed in 5670655 — a segment at its preview limit held up the later segments (`fill.next` kept
picking it; the planner now takes subtitle_loop's order minus limited segments, a ready preview's
start still first), and an active swap segment without a RunSegment row (a run that failed before
its submit) was left out of the retry assessment, so a paid retry looked delivery-only (it now
counts as unfinished → regeneration approval). **Round 2**: NO FINDINGS.

Full suite in a throwaway `re-skin-api:latest` container (full repo layout, `-n 3`): 4694 passed,
340 skipped, 15 failed — 11 fail identically on `main` (environment baseline) and the 4
`test_gdrive_run_result.py` ones pass in isolation on this branch (xdist ordering).
