# Assistant flow 2 — localisation

Tracking: AICP-332, prompt format
AICP-335. Derived from project 46cc6443
(es, pt, pl, fr of the 78.9 s video 5b1b9b67, 2026-10-29) together with the operator.
The source must be clean: a source with burned captions goes through
[assistant-flow-subtitle-removal.md](assistant-flow-subtitle-removal.md) first.

Decision owner per step: **A** = assistant alone, **Ap** = assistant proposes, operator
approves, **O** = operator only (the assistant cannot hear).

## Rules

1. **Segment 1 first.** It is generated alone as a probe. Only after the operator accepts it
   are the other segments launched. Its full dialogue (both speakers) becomes the voice
   reference for segments 2..N, and the accepted clip is reused through a temporary library
   clip in the stitch plan — never regenerated.
2. **Every segment is checked frame by frame against the source** (actions, props,
   continuity at the seams) and by ear (operator) before it is accepted.
3. **Prompt format (AICP-335)**: speakers with gender and description, per-line timecodes
   from the transcript, strict order, how many lines each secondary speaker says, off-face
   cues, action cues, silent tail, brand pronunciation.
4. **Captions first, CTA tail second** (tail chained off the caption), file ≤ 99 MB.
5. **Paid steps are announced with a credit estimate** before launch.

## Steps

| # | Step | Owner | Signal / tool | Cost |
|---|---|---|---|---|
| 0 | Intake | A | source clean? (else flow 1); duration (> 1 min → warn about regeneration); `mute_source=true` on the project (not applied by project-from-run); transcript ready | free |
| 1 | Languages + tails | O | order of languages, CTA clip per language or none | — |
| 2 | Segments | Ap | swap segments at scene cuts, each ≤ 14.72 s (Seedance 2.5 truncation); the source's own CTA end card is proposed and, when the operator confirms, cut — never generated (AICP-390) | free |
| 3 | Source breakdown | A | per segment from frames + transcript: who speaks, gender, on/off face, **body actions** (e.g. takes glasses off at 4–5 s), overlapping voices to drop | free |
| 4 | Translate | A | localisation prompt per language (needs the long timeout, see gaps) | LLM |
| 5 | Build prompts | A | `build_v2_prompts.py` → AICP-335 format per segment | free |
| 6 | Estimate | Ap | Seedance 2.5 720p ≈ 74 cr/s of clip; probe ≈ 900 cr; whole language ≈ 6 k cr for 79 s | — |
| 7 | Probe segment 1 | Ap | run with segment 1 only, no voice ref | ~900 cr |
| 8 | Probe QA | A + O | frame-by-frame vs source; length; caption detector 0 hits; operator listens (voices, gender, order) | free |
| 9 | Remaining segments | Ap | one run: stitch plan = [TEMP library clip of seg 1] + segs 2..N, `voice_file` = seg-1 dialogue | ~74 cr/s |
| 10 | Segment QA | A | 1 fps sheet vs source per segment (actions, props, mic hand, seams); lengths (−0.2…+0.8 s); detector 0 hits | free |
| 11 | ASR QA | A | burn captions right after the stitch: ASR of the generated audio = every line present, order, brand name | LLM |
| 12 | Listening QA | O | voices consistent, interviewer gender, attribution, pronunciation, accent | — |
| 13 | Fix loop | Ap | multi-segment rerun (takes are kept automatically, AICP-342) with a targeted cue (gender, off-face, action, pronunciation) | ~74 cr/s per segment |
| 14 | Captions | A | fix wrong words via `PATCH /captions/{id}/lines` + burn-only retry; garbled ASR → captions from the script | free |
| 15 | CTA tail | A | tail edit from the caption, `cut_sec` = duration − 0.01 floored to 2 decimals; size ≤ 99 MB | free |
| 16 | Acceptance | O | operator watches the deliverable | — |
| 17 | Cleanup | A | delete the TEMP library clip; record spend per language | free |

After the first language is accepted, further languages can run in parallel: both probes
together, one listening pass for both.

## Known failure modes

| Failure | Fix |
|---|---|
| Secondary speaker changes gender | Speakers block with explicit gender, "never a male voice" |
| Lines swapped / given to the wrong speaker | per-line timecodes + "strictly in this order" + "says ONLY ONE line" |
| Off-face line attributed to the visible person | off-face cue: what the frame shows and whose voice it still is |
| Brand mispronounced ("Listenifyed") | pronunciation line |
| Body action not reproduced (glasses stay on) | action line with time window and "do not put them back on" |
| Garbled ASR (pl) | captions from the script; language goes to a separate check |
| Accent in the target language (pl) | operator decision: deliver, or hold the language |
| Clip shorter than the segment (14.72 s truncation) | keep segments ≤ 14.72 s |
| The source's own CTA end card generated (wrong voice, the person pasted onto the graphic, two CTAs with the tail) | cut it at the scene cut; the language's CTA tail replaces it (AICP-390, autopilot flow v2) |

## Reference cost (79 s, 7 segments, 720p)

| Language | Regeneration rounds | Credits |
|---|---|---|
| es (old prompt format) | 3 | 11 286 |
| pt (AICP-335 format) | 0 | 5 852 |
| pl | 1 segment (glasses) | 6 498 |
| fr | 1 segment (glasses) | 6 460 |

## Gaps that block autonomy

- Translate endpoint times out at 90 s; a whole-video translation takes 180–280 s
  (needs an async job).
- Project-from-run does not apply the localisation default `mute_source=true`.
- Analyze clamps localisation to the hook, so whole-video segments are cut by hand.
- ~~No multi-segment rerun; rerun re-delivers to a new Drive id.~~ Addressed by
  AICP-341 (`POST /runs/{rid}/segments/rerun`, one re-stitch) and AICP-342 (the
  re-delivery replaces the content behind the existing Drive id; replaced outputs
  are kept as takes, `POST /runs/{rid}/segments/restore`). Captions/tails already
  built from the run keep their own files and must be rebuilt after a fix.
- The prompt builder is a scratch script. Frame comparison and captions from the script are
  assistant tools since AICP-345 (`compare_frames`, `script_captions` + an operator-confirmed
  Apply that patches the caption lines and re-burns, see [assistant-qa-tools.md](assistant-qa-tools.md)).
- The assistant must check for a deploy/drain before launching and survive a restart mid-step.
