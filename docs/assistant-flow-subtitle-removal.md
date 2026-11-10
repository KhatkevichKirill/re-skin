# Assistant flow 1 — subtitle removal

Tracking: AICP-332. Derived from
project 5b1b9b67 (run 657e1f7f, 78.9 s, 11 segments, 2026-10-28) together with the
operator. Localisation is a separate flow ([assistant-flow-localisation.md](assistant-flow-localisation.md))
that starts from the clean result of this one. Revised 2026-10-30 on project 328341d4
(run 2845eef3, 54.5 s, 9 segments): Omni retries were replaced by a masked VOID fill.

Decision owner per step: **A** = assistant alone, **Ap** = assistant proposes, operator
approves, **O** = operator only.

## Rules

1. **One Gemini Omni pass, then VOID for what is left.** Every swap segment gets one Omni
   generation. A segment that still has caption words after QA is not regenerated with Omni:
   it is rebuilt as the **source + a VOID fill inside the caption mask** (step 5). Decided by
   the operator on 2026-10-30 after 328341d4: on segments 4 and 7 three Omni takes each never
   came clean (the coordinate prompt made them worse, 27→48 and 25→54 hit frames, and on
   segment 7 it turned the white mic foam into a black lavalier), Omni 1.1 was only marginally
   better (AICP-354), while the masked fills removed the words.
2. **Seedance is not used for caption removal**: it copied the captions on 2 of 3 clips and
   its output needs repair (truncated tail, length mismatch).
3. **Long video warning at intake.** Over 1 minute (or many segments) the assistant says up
   front that some segments will very likely need a VOID fill, and budgets for it.
4. **Every segment is checked frame by frame against the source** before assembly. A segment
   that is not clean never reaches the final.
5. **Paid steps are announced with a cost estimate** before launch (kie.ai credits for Omni,
   USD for Fal); the operator measures the balance.
6. **A VOID patch may be visible.** The fill is noticeable on close inspection (so is LTX-2.3's);
   the operator accepts it when it covers only a few frames per segment. Many patched frames, or
   a smear over a face or the phone screen, go to the operator instead.

## Steps

| # | Step | Owner | Signal / tool | Cost |
|---|---|---|---|---|
| 0 | Intake | A | source probe (duration, size, resolution); contact sheet 1 frame / 3 s → are there burned captions, where, which colour; risk zones (phone UI text, captions over hands/mic, borders) | free |
| 1 | Pre-launch check | A | no deploy / drain in progress (active statuses on other runs, container uptime) | free |
| 2 | Analyze | A | `subtitle_removal` project → PySceneDetect cuts; every segment ≤ Omni `max_clip_sec` (10 s) | free |
| 3 | Estimate + launch | Ap | Omni = 168 cr per segment, flat (end card / caption-free segments → `keep`, see `subtitle_loop review`); plus ~$0.05 of VOID per segment expected to stay dirty | 168 × N + VOID |
| 4 | Per-segment QA | A | `subtitle_loop qa` (detector calibrated on the source) + visual 2 fps sheet vs the source: leftover words, **invented words**, unrequested changes (border removed, outpaint), phone UI kept. Hits on phone-screen text that is also in the source are false positives: confirm on the crop sheet | free |
| 5 | VOID fill | Ap | for each dirty segment: caption mask on the **source** frames → `fal-ai/void-video-inpainting` on the source clip + mask → paste back **only inside the mask** over the source. The segment becomes source + fill, so it sits exactly on the source timeline (no Omni frame shift) and everything outside the mask is untouched | ~$0.05 per segment |
| 6 | Fill QA + last-word fix | A | detector on the composite, **plus an eye check**: the white-text detector is blind to grey ghosts and smears. Count the frames where the patch is visible; a few per segment pass (rule 6). A short word the mask missed on a plain background → `ffmpeg delogo` over its box and time window | free |
| 7 | Assembly | A | clean Omni segments + VOID composites + keep segments from the source → pipeline stitch; if it OOMs (AICP-334) → manual per-clip normalise + concat demuxer + source audio | free |
| 8 | Final QA | A | detector 0 hits over the whole video; picture vs source alignment ±1 frame at ~2 points/s; size ≤ 100 MB | free |
| 9 | Deliver | A | final.mp4 into the run dir + `result_local_path` → `POST /runs/{id}/retry` (delivery-only, reuses final.mp4) | free |
| 10 | Acceptance | O | operator watches the final | — |

## Stop rule

Omni gets one pass per segment; a dirty segment goes to VOID, not to another Omni take. If the
VOID composite still shows words or a visible smear on many frames, the assistant stops and
offers the operator a choice: LTX-2.3 inpaint on the same mask (`fal-ai/ltx-2.3-quality/inpaint`,
~$0.41–0.54 per segment, judged slightly cleaner by eye in AICP-354), a widened mask, delogo
(if the leftover is small), or accept. An Omni rerun (default ↔ coordinate prompt, AICP-341/342)
remains available only on the operator's request. Retrying a "Google safety review" block on
the same frames does not help — keep the previous take.

## Known failure modes

| Failure | Fix |
|---|---|
| Leftover caption words after the Omni pass | VOID fill of that segment (step 5) |
| Omni rewrites a caption into a word that is not in the source | detector catches it; VOID fill (the fill starts from the source, so the invented word is gone too) |
| VOID smears the phone screen (faces, text) where the mask overlaps it (328341d4 segs 2–3) | eye check; LTX-2.3 on the same mask, or accept if few frames |
| Words wider than the mask box (x 0.15–0.75) keep their edge letters ("INTRODUCTION") | widen the box from the measured caption geometry; the run's own captions can cover a stray letter later |
| Coordinate prompt built from a wrong box (two-line guess y 0.43 on 328341d4; real line y 0.605–0.643) | measure the geometry first (`subtitle_loop review`) |
| Omni removes a frame border / outpaints | operator decides whether to keep |
| Safety block on a segment (phone UI + celebrity name) | keep the earlier take |
| Stitch OOM on many 1080p clips | manual assembly (AICP-334) |

## Reference cost

5b1b9b67: 11 segments, 6150 cr in total (Omni 3360 + a Seedance 2.0 test 2790, which is
now excluded by rule 2). With Omni only the same job would have been ~3700 cr.

328341d4: one Omni pass 9 × 168 = 1512 cr (1344 with the end card as keep), two dirty
segments. Under this flow: + 2 VOID fills ≈ $0.10 instead of the 2 × 2 Omni retakes
(672 cr) that did not clean them.

## Gaps that block autonomy

- ~~No multi-segment rerun~~ — AICP-341: `POST /runs/{rid}/segments/rerun` takes a
  list of segments (optional prompt each) and re-stitches once.
- ~~Rerun overwrites results in place and re-delivers to a **new** Drive id~~ —
  AICP-342: a rerun keeps the replaced output as a take
  (`GET /runs/{rid}/segments/{rsid}/takes`, `POST /runs/{rid}/segments/restore`,
  free) and the re-delivery replaces the content behind the existing Drive id.
- Stitch memory grows with the clip count (AICP-334).
- ~~The caption detector and frame comparison are scratch scripts, not platform tools.~~ AICP-343
  made them platform tools; AICP-345 lets the assistant call them (`detect_captions`,
  `check_alignment`, `compare_frames`, see [assistant-qa-tools.md](assistant-qa-tools.md)).
- **The VOID path is scratch scripts, not a platform tool**: `/root/omni11-eval/fal/`
  `prep_seg.py <seg>` (source clip + caption-mask memmap, `inpaint/caption_mask.py`),
  `submit.py` (one POST per job, `FAL_CAP` budget), `fetch_result.py` (GET only, never
  resubmit), `SEG=<n> post.py composite` (paste back inside the mask, streams frames);
  run c2283d0e was assembled from the composites by
  `/root/re-skin/data/scratch/omni11-eval/make_runs.py`. VOID takes 69 + 8k ≤ 197 frames
  and returns them tagged 12 fps (retime to 197/8 fps). Run heavy host scripts under
  `systemd-run --scope -p MemoryMax=4G`: whole-video numpy hung the VPS twice on 2026-10-30.
  Fal submits need the operator's approval.
- ~~`subtitle_loop rerun` still proposes Omni reruns~~ — AICP-359: `qa` proposes the VOID
  fill (below); an Omni rerun is only an operator option.
- The assistant does not call `app.subtitle_loop` (below) yet, and paid admissions are not
  idempotent yet (AICP-350/352), so the loop still runs by hand from its proposals.

## The loop's free core (AICP-353)

`app.subtitle_loop` turns steps 2–5 and the hand-off into deterministic verdicts and
**proposals**: the exact request for an existing endpoint plus its cost. It never writes to
the DB, calls a model or spends credits; it writes only JPEG sheets into `--out-dir`.

```
python -m app.subtitle_loop review --project PID --out-dir DIR [--faces]   # before launch
python -m app.subtitle_loop qa --run RID --out-dir DIR [--confirm-fill FID]  # after a run
```

(`--source/--final/--segments/--price` run it on files outside the DB, e.g. a take; the
requests are then evidence only: `fill.actionable: false`, no `next`.)

`qa` prints `{qa, fill, operator_options: {omni_rerun}, handoff}`.

- **review**: two source scans (the configured band decides hits; a band 0.10 taller on each
  side measures where the captions sit). It proposes `keep` for a swap segment with no in-band
  hits (only when the detector is calibrated on this source) and for an end card: a text box
  stable (IoU ≥ 0.9) for ≥ 2 s over ≥ 60 % of the segment's hits, with no face (`--faces`
  samples InsightFace at 1 fps). Every keep proposal carries a contact sheet and
  `needs_visual_check`. The band check reports edge-straddling hits per segment and proposes a
  wider `band_y` only when caption boxes in swap segments cross it. Cost = the observed
  per-clip price (median of recent `run_segments.cost_credits`, 168 for Omni) × swap
  segments, current vs proposed, plus the spend-cap reservation (`budget_check_*`).
- **qa**: refuses a final that is not on the source timeline (duration ±0.3 s, alignment
  windows within 9 frames; a custom stitch plan); a swap segment reaches that drift
  (+0.1 s) into an adjacent keep segment for hits the source cannot explain. Calibration
  counts only caption-like (non-static) source hits. Per segment: `clean` / `leftover` /
  `false_positive` (source text is static UI / end card and ≥ 90 % of the hits lie on a
  padded source text box within 1 s; a caption line held still looks the same, so it blocks
  `clean` until confirmed on its crop sheet: `--confirm-fp 8`) / `keep` / `keep_captions`
  (a keep segment whose source has captions: they ship; blocks `clean`) /
  `clean_uncalibrated`. A crop sheet per dirty
  segment shows one result-vs-source crop per **word** (the detector box jumps when the word
  changes); the words are read off it (no OCR in the image). `qa_compare.jpg` is the visual
  check for invented text and picture changes. The ±1-frame sync of step 8 is a warning.
  Every swap row carries `mask_box`: the union of the source caption boxes in the segment,
  padded like the fill's measured box (an estimate from the 10 fps scan). A segment whose
  result is an applied VOID composite (AICP-358, provenance from `run_segment_fill_jobs`)
  keeps the detector's verdict plus `fill_needs_visual_check` and a `qa_seg<N>_fill.jpg`
  (mask-box crops, result vs source): the detector misses grey ghosts and smears, so an
  unconfirmed composite blocks `clean` until `--confirm-fill <fill job id>` (bound to the
  fill id, not the segment index; refused while the fill's re-delivery is not confirmed).
- **fill** (AICP-359): one item per dirty segment, `state` = `propose_preview` (the free
  `POST /runs/{rid}/segments/{rsid}/fill/preview` with an **empty** body, so prepare measures
  the inclusive box on every frame; QA's box is `expected_mask_box`; the paid start is a
  non-executable template, its fill id comes from the preview) / `propose_start` (a ready
  preview that still matches the run: `approved_usd` = its `estimated_usd`) /
  `stale_preview` (cancel it) / `in_progress` / `blocked` (the real API gates, run
  through read-only: `_require_fillable`, rerun conflicts, audio patch, published assistant
  edits, an active fill, a paid re-stitch, Fal disabled) / `operator` (ambiguous previews,
  a custom preview, a repeat of an already-paid fill, a result that is not a known Omni
  output) / `stop`. `next` is at most one request (fills go one at a time; a preview is bound
  to the final it was made on): after it, poll the fill list and run `qa` again. Cost =
  reservation $0.10 (registry `flat_usd`, what `approved_usd` must cover) and expected
  charge $0.05 (observed) — the same call, not a sum; no kie.ai credits. Requests are
  actionable only when the judged files are the live run's and nothing changed during the
  scans (`fill.evidence`).
- **stop rule** (the doc's, above): a VOID composite still dirty → `stop` with operator
  options: widened-mask preview (old box ∪ in-segment leftovers, padded; a second paid
  attempt), LTX-2.3 (not a platform endpoint yet), delogo (box + windows, by hand), restore
  the replaced take (free), accept, or the Omni rerun. The same options are listed under
  `on_visual_reject` for a composite the detector calls clean.
- **operator_options.omni_rerun**: the former rerun proposal, `executable: false`, only on
  the operator's request: `POST /runs/{rid}/segments/rerun` alternating default ↔ coordinate
  prompt; billed = the listed segments + every other unfinished active swap; withheld at ≥ 3
  preserved outputs. `choose_best_takes` ranks re-checked outputs by hits (newest on a tie)
  into a `segments/restore` body.
- **handoff**: only when clean → `POST /runs/{rid}/to-project` (localisation, whole scope).
  The keep ranges are a separate edit of the new project's segments after its analysis.

Verified on project 328341d4 / run 2845eef3 (round-0 final): review proposes segment 8
(49.1–54.5 s, end card, static 4.6 s) as keep — 1512 → 1344 cr — and reports the band
edge touched only by the end card (captions sit at y 0.60–0.64, so no band change; the
y ≈ 0.43 boxes are the end card's "App of the Day" UI). QA finds leftovers in segments 4
(MY, OUT, PDFS, TEXTS, SWAP at 23.5–28.2 s) and 7 (AT, LOOK, WHAT, YOU, APP at
43.3–48.0 s), calls segment 8 a false positive, and proposes a two-segment coordinate-prompt
rerun for 336 cr. The picture there is up to 3 frames off the source (per-clip Omni shift).
That rerun was made and made both segments worse; their round-0 takes were restored
(`segments/restore`, free). Masked fills of segments 0–7 were then built from the source with
both VOID and LTX-2.3 (runs c2283d0e and 00037197, segment 8 keep); the operator compared both
finals and chose VOID as the default (this revision).

AICP-359, checked on a re-stitch of the round-0 clips (segments 4, 7 and 8 from their
preserved round-0 takes, `--final` of that file, so the requests are evidence only):
`qa` calls segments 4 (27 hit frames) and 7 (25) leftover and segment 8 the end-card false
positive, and `fill` proposes a free preview for segment 4 then 7 (`order [4, 7]`, $0.10
reserved / $0.05 expected each, no kie.ai credits) and no Omni rerun. Its `mask_box`
estimates equal the boxes the platform later measured for those fills (seg 4
x 0.2607–0.7393, y 0.5821–0.6627; seg 7 x 0.2256–0.7763).
