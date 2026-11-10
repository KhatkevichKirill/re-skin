# Autopilot flow: localisation QA and finishing (AICP-366, "C2")

Tracking: AICP-366, parent
AICP-362. Continues the localisation flow of
[autopilot-localisation-flow.md](autopilot-localisation-flow.md) (AICP-365, "C1") from the
stitched final of `rest:L`, on the core of [autopilot-design.md](autopilot-design.md) rev 7.
Covers steps 10–17 of [assistant-flow-localisation.md](assistant-flow-localisation.md).

Status: **design rev 3** (2026-10-31): rev 1 + the resolutions of two §2 rounds (GPT-6.1 Terra,
§6). Where §1–§4 and §5 disagree, §5 wins; where §5 and §5b disagree, §5b wins.

Operator decisions that bind it: every regeneration needs an approval (the core's
`<launch>#regeneration`, or a flow `regeneration` Ask with action options); an approved
regeneration adds its cost to the budget (Q6); `listening_final` is this flow's, and a rejection
comment becomes the next regeneration proposal with a targeted cue; the Judge port is disabled in
production (Q2) — every check here is deterministic (ffmpeg + numpy + the existing caption ASR), no
new billed LLM path; a manual paid action pauses the autopilot (Q5); no `visual_check` kind
(`handoff` with options, Q8); background device lines are kept and marked BACKGROUND (2026-10-29).

## 1. Steps (appended to C1's step list, per language L)

| step | needs | paid | approvals | estimate |
|---|---|---|---|---|
| `qa:L` | `rest:L` | yes (ASR caption, reruns) | `regeneration`, `listening_final` | `llm_credits` (one ASR caption) |
| `captions:L` | `qa:L` | no | — | 0 |
| `tail:L` | `captions:L` | no | `handoff` | 0 |
| `cleanup:L` | `tail:L` | no | — | 0 |

The flow keeps its id `localisation` and version 1 (nothing of C1 is deployed; no live autopilot
uses the version). Steps are ordered after C1's, so `steps()` order is unchanged for C1.

### 1.1 `qa:L` — segment QA, ASR QA, regeneration proposals, best take, listening final

One step that loops in **rounds**. A round is keyed by the rest run's version
(`tag = r{n}:{run_version[:16]}`, `n` = 1 + number of `rerun:`/`best:` launches of this step
generation), exactly like C1's probe listening round: a rerun, a restore or a manual change makes a
new tag, so every compute/approval of a round is about one take of the run, and answers about an
old take are never applied to a new one (C1 gate R1/R2; `_clear_stale` from C1 runs before any
`Wait`).

```
run not terminal                     -> Wait
run failed/incomplete                -> C1's _failed_run_round (fail:rN:<ver> regeneration Ask with
                                        [rerun unfinished segments | stop])
segqa:<tag>   Compute localisation_segment_qa (free, run workers, streamed) -> per piece of the
              final: clip length vs its window (issue outside -0.2..+0.8 s), 1 fps frame sheet vs
              source (diffs = warnings), caption detector on the clip at 2 fps with the project's
              detector config (issue on any hit), picture drift (qa_tools.match_times; warning),
              voice-gender check per scripted line (issue, §2.1)
asr:<tag>     Launch localisation_create_caption (paid, llm_credits: the existing caption job =
              ASR of the generated audio + burn, AICP-366 "reuse the caption pipeline's
              transcription"); both started in the same round, they run in parallel
              wait for both (compute terminal; caption done/failed — a failed ASR is a warning)
best:<rerun>  first round after a rerun only: per rerun segment, the take with the fewest issues
              over every round of this step generation; if it is not the current one, Launch
              restore_takes (free, core handler) once per rerun name -> new tag, new round
issues = segment issues + ASR issues mapped to segments (unless the ASR is garbled, §2.2)
regenerable = issues on swap pieces of THIS run (the probe segment is a TEMP library clip in the
              plan: its issues are warnings, it is never regenerated here)
regenerable non-empty:
  regen:<tag>  Ask regeneration, options:
                 rerun  = rerun_segments(run, [{run_segment_id, prompt: prompt + QA cues}])  (cost
                          estimated by the core from the handler)
                 accept = continue to the listening approval without regenerating
                 stop
               artifacts: final video + the flagged segments' sheets; about {run: ver}
  approve rerun -> Launch rerun:rK (regeneration=True, consumes=open answers) -> next round
  accept        -> listening (the answer is consumed by the listening decision)
  stop / reject -> Fail(consumes=open answers)
listen:<tag>   Ask listening_final, subject=L, artifacts: final video + every swap segment's
               sheet; proposal: QA summary (issues accepted, warnings, ASR coverage)
  approve -> Complete({run_id, run_version, tag, caption_id, garbled, pieces timeline},
                      depends_on={run:<rest>: ver}, consumes=open answers)
  reject  -> cue:<tag> Launch localisation_listen_feedback (free; records the comment, the
             targeted segments and their corrected prompts; consumes the rejection at once — the
             core stops an autopilot that waits on a rejected answer, C1 §8)
             fix:<tag> Ask regeneration, options [rerun targeted segments with the cue | stop]
             approve -> Launch rerun:rK (regeneration=True) -> next round; stop/reject -> Fail
```

Targeted segments of a rejection: segment numbers named in the comment (`segment 3`, `seg 3`,
`segments 3 and 5`, `s3`, `#3`, 1-based, the project's segment order); none named → the
segments with issues in the current round; none → every regenerable swap segment. The cue is the
comment verbatim under C1's operator header (`apply_operator_cue`) plus the known-failure-mode
lines its keywords select (§2.3).

`qa:L` completes with `depends_on={run:<rest>}`: the later steps never change the run (captions,
tail edits and the TEMP clip deletion do not touch `run_version`), so there is no self-reopen
(core §15); a manual rerun after acceptance reopens `qa:L` and every later step of L (correct: the
captions/tail were built from the old final). `rest:L` keeps no `depends_on` on its run (C1).

### 1.2 `captions:L` — fix wrong words, garbled ASR → captions from the script (free)

Input: the accepted round's ASR caption (`qa:L` output) and the expected script on the final's
timeline (§2.2: the pieces' current prompts, scaled to the measured clip lengths).

* ASR garbled (coverage < 0.5) or the ASR caption failed/has no lines → lines from the script
  (`qa_tools.captions_from_script` semantics, computed in planning from the frozen pieces).
* Otherwise word-level fixes: aligned short `replace` spans (≤ 3 words each side) take the
  script's words (misheard words, brand spellings); ASR extra/missing words are kept as heard
  (the operator accepted the audio).
* No change → `Complete`. Change → `Launch fix` (`localisation_caption_lines`: caption lines +
  `lines_source=manual` + re-queue burn-only, no reservation — the worker skips the transcription
  for manual non-empty lines) on the existing caption, or `script` (`localisation_script_caption`:
  a new caption row with manual lines, no reservation) when there is no usable caption.
* Wait for the caption job; failed → one more attempt (`fix:r2`), then `Fail`.
* `Complete({caption_id, lines_source, changes, garbled})`.

**AICP-368 hook.** The burn is called as it is. `localisation_create_caption` passes AICP-368's
`placement="project"` to `api_v2.create_caption` when that parameter exists (so the project's
inherited placement is frozen onto the job), and `localisation_script_caption` copies
`project.caption_placement` onto the new row when the column exists. Both go through one function,
`localisation_handlers.caption_placement_kwargs()` / `frozen_caption_placement()`; nothing else
changes when 368 merges.

### 1.3 `tail:L` — CTA tail from the caption (free)

* `params.tails[L]` null → no tail; the deliverable is the caption output.
* Else `Launch tail` (`localisation_create_tail`): the handler probes the caption output and calls
  `api_v2.create_tail_edit(run, cut_sec=floor2(duration − 0.01), library_clip_id=tails[L],
  source_caption_id=caption)`; `cut_sec` is in the effect result.
* Wait; failed → one more attempt (`tail:r2`), then `Fail`.
* `size:<id8>` Compute `localisation_deliverable_check` (ffprobe + stat of the deliverable):
  > 99 MB (99 000 000 bytes) → `Ask handoff` [deliver as is | stop]; else
  `Complete({tail_edit_id, cut_sec, size_bytes, duration_sec})`.

### 1.4 `cleanup:L` (free)

* The TEMP library clip of L (`rest:L` output) still exists → `Launch delete_temp`
  (`localisation_delete_temp_clip`: refuses unless the clip is named `TEMP autopilot <pid8> …` and
  untagged; wraps `api_v2.delete_library_clip`, which 409s while an active run or tail edit
  references it → refusal, backoff).
* `Complete({spend})` with the language's spend in the reason (journaled): Σ committed over its
  scopes — `run:<probe>`, `run:<rest>`, `caption:<every caption of those runs>` — plus the
  project-level shared spend (transcription, translations: `transcribe:<pid>`, `localise:<pid>`)
  reported once as "shared".

## 2. Deterministic checks

### 2.1 Pieces of the final and the voice-gender check

The final is the run's stitch plan in order (`audio_mode=seedance`: every clip brings its own
audio). Piece = `library` (the TEMP probe clip: its whole duration), `swap` (the run segment's
`local_result_path`, concatenated whole, clip window = `localisation.clip_window` of the frozen
`segment_snapshot` entry), `keep` (the source cut, its segment length). Offsets are cumulative
clip durations (as `qa_tools.captions_from_script`). Script lines per swap piece =
`qa_tools.parse_dialogue(current prompt)` (`prompt_override` or the run prompt), clip-relative,
scaled by `clip_sec / window_sec`; the probe piece's lines come from the probe run's segment
prompt.

Expected gender per line: the speaker tag → `localisation._speaker_profiles(transcript lines,
transcript)` → `gender` / `kind`. Checked only for `person` speakers with a known gender, not for
BACKGROUND lines, not where another line overlaps.

Measured: the clip's audio inside the line's scaled window (ffmpeg → 16 kHz mono s16le, one window
at a time, never the whole video), autocorrelation pitch per 40 ms frame (hop 10 ms, 70–400 Hz,
voiced when the normalised peak ≥ 0.5 and the frame energy is above 10 % of the window's peak
RMS), median F0 over voiced frames, at least 0.4 s voiced. `≤ 155 Hz` = male, `≥ 175 Hz` =
female, between = unknown. An issue only when expected and measured are both known and differ.

### 2.2 ASR check

Expected = the script lines on the final timeline (2.1). Words normalised (lower case, NFKD
without diacritics, punctuation stripped; characters instead of words for `ja`).
`difflib.SequenceMatcher(autojunk=False)` over the two word sequences (monotonic, so a swapped or
dropped line shows as unmatched). Per expected line: coverage = matched words / words; present
when ≥ 0.5. Overall coverage < 0.5 → **garbled** (Polish in 46cc6443): ASR issues are not mapped
to regenerations, captions come from the script, the listening approval says so.
Issues: `missing_line` (not present), `brand` (an expected line contains a transcript brand token
that is not matched exactly — "Listenifyed" ≠ "Listenify"). Warning: a matched line whose ASR time
is > 2.5 s from its expected start.

### 2.3 Cues (known failure modes table)

Appended to the segment's current prompt under one header ("Corrections from the automatic QA —
follow them exactly:"), each bullet once:

| issue | cue |
|---|---|
| gender | `<TAG> is a female voice in every line of this clip — never a male voice.` |
| missing_line | `Say every dialogue line above, strictly in this order — none may be skipped (missing: "…").` |
| brand | `Pronounce "<name>" exactly as <pronunciation>.` |
| captions | `Do not add any text, captions or subtitles to the picture.` |
| length | none (a plain rerun) |
| comment keywords: gender/male/female/voice | the gender line per speaker of the segment |
| comment keywords: glasses/action/gesture/hand/takes/puts | `Reproduce every body action of the source at its time, and do not undo it afterwards.` |
| comment keywords: off-face/off-screen/off screen | `Off-face lines are spoken by the off-screen speaker, never by the visible person.` |
| comment keywords: pronounc/brand name | the brand line |

## 3. Handlers and compute functions (all registered on import of the flow)

| name | paid | wraps | notes |
|---|---|---|---|
| `localisation_create_caption` | yes (`llm_credits`) | `api_v2.create_caption(run, language=L)` | one commit + enqueue (endpoint); target run; caption id + `caption:<id>` scope registered in a `before_flush` capture; reconcile: caption row exists |
| `localisation_caption_lines` | no | new single-commit service (validate like `PATCH /captions/{id}/lines`, lock run then caption like `retry_caption`, caption done/failed, lines non-empty, `lines_source=manual`, → `queued`) + enqueue after commit | no reservation (no transcription can run) |
| `localisation_script_caption` | no | new single-commit service (`create_caption` minus the reservation, manual lines) | |
| `localisation_create_tail` | no | `api_v2.create_tail_edit(source_caption_id=…)` | cut computed in the handler from the caption output |
| `localisation_delete_temp_clip` | no | `api_v2.delete_library_clip` | name/language guard |
| `localisation_listen_feedback` | no | — | records the rejection (result) |
| `restore_takes`, `rerun_segments` | core | | |
| compute `localisation_segment_qa` v1 | free | §2.1 | streams; sheets in the attempt dir |
| compute `localisation_deliverable_check` v1 | free | ffprobe + stat | |

**Pre-existing bug found (fixed here):** `api_v2.create_caption` reserves under
`caption_scope(caption.id)` before the row is flushed, i.e. `caption:None` (prod: 6 reservations,
420 credits); the worker's draw then lands on `caption:<id>` as an overdraft. Fix: assign the id at
construction. Needed for the autopilot's spend attribution of the ASR caption.

## 4. What can strand a language, and the answer

| Situation | Outcome |
|---|---|
| ASR caption job fails | warning; QA proceeds on the segment checks; captions come from the script |
| Segment QA compute fails (3 attempts, core) | `regen`/`listen` asked without segment results (the proposal says so) |
| Operator reruns / restores by hand | paid → paused (Q5); free → new tag, stale answers cleared, new round |
| Rerun run ends failed/incomplete | `fail:rN` regeneration Ask (C1 helper) |
| Operator rejects the listening twice | each rejection is a new fix round; every rerun is an approval |
| Caption burn / tail fails | one automatic retry (free), then `Fail` with the error |
| TEMP clip referenced by an active run | delete refused (409) → backoff, retried |
| Deliverable > 99 MB | `handoff` [deliver as is, stop] |

## 5. Rev 2 — changes after the §2 pass

1. **Rounds and media (R1-2).** Every result is used only under the round tag it was planned for,
   and the tag carries `run_version`, which includes `(size, mtime_ns)` of the final and of every
   segment clip: a change after the tag was computed changes the version for good (a re-stitch
   never restores an old mtime), so a compute or caption job that read later bytes belongs to a tag
   that is never consulted again. Consuming answers re-verifies the version under the run lock
   (core §5.4). Superseded caption jobs of earlier rounds keep running and deliver their captioned
   copy to Drive (the pipeline has no cancel): accepted clutter, never the deliverable.
2. **ASR spend (R1-3).** The regeneration option's handler is `localisation_rerun_segments`
   (`rerun_segments` + one follow-up ASR caption in its estimate, `always_regeneration`), so the
   approved cost and the budget raise (Q6) include the re-check. An ASR launch is
   `regeneration=False` only when it is covered: the first ASR of generation 1 with no paid work
   bought in an earlier generation (budgeted), or the first ASR after our own approved rerun. Any
   other (after a manual change, a restore, a reopen) is `regeneration=True` — the core asks
   `asr:<tag>#regeneration` with its cost.
3. **ASR provenance (R1-4).** `process_caption` may reuse a same-audio donor's lines, preferring a
   manual edit. The ASR check uses a caption only when its lines are raw ASR: `lines_source =
   transcribe`, or `reused` along a `lines_reused_from` chain (≤ 5 hops) ending in `transcribe`.
   Otherwise: warning "no independent ASR", no ASR issues, listening decides.
4. **Decision table (R1-5).** Every `Launch` / `Complete` / `Fail` of these steps lists every open
   answer of the step generation (C1's `_with_open`); C1's `_clear_stale` runs before any `Wait`.
   `regen` "accept" stays open across the listening `Ask` and is consumed by the next decision
   (Complete, the rejection's `cue` launch, or Fail). The `handoff` of `tail:L` is consumed by its
   Complete / Fail.
5. **Caption/tail fences (R1-6).** `localisation_caption_lines` and `localisation_create_tail`
   carry `expect_caption` = sha of the caption row's `(status, lines, lines_source, output_path)`
   at planning; the handler re-reads the row under its lock and refuses (409 → replan) when it
   changed. No caption key goes into `depends_on` (no self-reopen).
6. **Job boundaries (R1-7).** A failed ASR caption is never retried automatically (an interrupted
   call may have billed; recovery reaps orphaned derived jobs to `failed`): its evidence is
   missing and the captions come from the script (free). Free burns and tails get one automatic
   retry; a failure after the Drive upload can leave a duplicate file, the same as the operator's
   Retry button. Reconcile of every handler: the row it created exists.
7. **Delete boundary (R1-8).** `localisation_delete_temp_clip`: row present → the endpoint (commit,
   then rmtree); row gone → rmtree the clip dir only. The step launches it again (`delete_temp:r2`)
   when the first one's result does not say the directory is gone. A clip still referenced is
   refused (409): backoff, and after 8 refusals the core pauses with the reason.
8. **TEMP clip deletion strands later re-stitches (R1-1).** Deletion is the operator's step 17;
   kept. The consequence is made safe and explicit: `rerun_segments` now refuses (409) a run whose
   stitch plan references a library clip that no longer exists **before** anything is billed
   (before this change it billed the rerun and then failed at the stitch). Listed for the operator.
9. **False proposals (R1-9).** Gender: per (segment, speaker) median F0. An **issue** needs the
   accepted probe as the validated baseline for that speaker (the TEMP clip's lines, measured in the
   same compute): the segment's voice must classify as the opposite gender **and** sit at least
   half an octave from the baseline toward it. Without a baseline, a mismatch is a warning.
   Captions: an issue only when the clip has hits and the same source window (same detector config)
   has none; else a warning. ASR: a scripted line overlapped by a BACKGROUND line is never a
   `missing_line` issue (warning). Round cap: after 6 rounds no more proposals (warning), straight
   to the listening approval. Each proposal lists its evidence (Hz, baseline, hits, coverage).
10. **Best take (R1-10).** Scored on segment-QA measurements only (free; the same compute for every
    take), never on ASR; a take whose gender or caption check did not complete is unknown and never
    restored; a strictly smaller issue count wins; one restore per rerun (`best:<rerun>`), decided
    before that round's ASR. A restore restores the take's prompt, and the next round reads the
    current prompts again.
11. **AICP-368 (R1-11).** An integration dependency, not a contract: the hook mirrors 368's branch
    (6cbd705: `create_caption(placement="project")`, `VideoProject.caption_placement` frozen into
    `RunCaption.placement`). Passing `placement` explicitly is required once 368 merges (an omitted
    `Form` default arrives as a `FieldInfo`). Tests pin both paths with and without the parameter.
12. **Flow version (R1-12).** Rejected with evidence: prod is at `q8b4d0f2a716`, the core's
    migration is not applied and `project_autopilots` does not exist, so no v1 autopilot can exist.

## 5b. Rev 3 — changes after the §2 round 2

1. **ASR re-checks are their own regenerations (R2-1).** The composite rerun estimate is dropped
   (the option stays the core's `rerun_segments`). The first ASR caption of generation 1 is
   budgeted; **every later one** (after a rerun, a restore, a manual change, a reopen) is
   `Launch(regeneration=True)`, so the core asks `asr:<tag>#regeneration` bound to its own cost and
   raises the budget by it (Q6). Rejecting it stops the autopilot (core rule for core approvals).
2. **Re-stitch admissions refuse a missing library clip, serialised with the deletion (R2-2).**
   `api_v2._admit_segment_rerun` (single rerun, multi rerun, restore) and `retry_run` refuse (409)
   after taking the run row lock, before any reservation or mutation, when the run's stitch plan
   names a library clip row that no longer exists. `localisation_delete_temp_clip` takes the row
   lock of **every** run whose stitch plan names the clip (sorted by id, `FOR UPDATE`) before the
   endpoint's active-run check: a rerun that locked first is seen as active (409, retried); a
   rerun that locks second sees the clip gone (409, nothing billed).
3. **What the jobs actually read (R2-3).** The QA compute gets every non-media input frozen in its
   args (prompts, windows, genders) and records `(path, size, mtime_ns)` of every file it read,
   before and after reading, plus `media.audio_fingerprint` of the final (streamed). Planning
   uses a result only when those signatures equal the files' current ones (a stat, as the core's
   own `run_version`). The ASR caption counts only when its `audio_sha256` (computed by the worker
   from the source it actually resolved) equals the round's final fingerprint — so a caption that
   transcribed another take is never evidence.
4. **Caption/tail after admission (R2-4).** The core has no caption version key, so manual edits
   of the autopilot's caption/tail jobs after their admission are not fenced (a known limitation,
   like manual spend in core §6.1). What was delivered is recorded instead: the deliverable check
   streams a sha256 of the delivered file (and of the caption output it was cut from) into
   `tail:L`'s output and the journal.
5. **Raw ASR only (R2-5).** Evidence = `lines_source == "transcribe"` **and** the fingerprint match
   of 5b.3. A reused (donor) or manual caption is never ASR evidence (warning; listening decides);
   its lines can still be the caption text.
6. **BACKGROUND lines (R2-6).** The QA parser (`localisation_qa.parse_script`) keeps the
   BACKGROUND marker of the cue (the design text's `parse_dialogue` drops it); tested through the
   whole path.
7. **Delete reconciliation (R2-7).** The admission transaction persists `created_refs = {clip}` and
   `result = {"deleting": clip}`; only the bookkeeping adds `dir_removed`. Without it the step
   launches `delete_temp:r2`, which removes the directory and **verifies** it is gone (else 409 →
   backoff → the core pauses after 8 refusals with the reason).
8. **AICP-368 pending scan (R2-8).** While the project's `caption_placement` reports a pending
   scan (`source.scan == "pending"`), `qa:L` waits before creating a caption (a job freezes the
   placement at creation, and a burn-only retry keeps it). A failed scan is not waited on (368
   burns its fallback); the listening proposal notes the placement state.
9. **Best take (R2-9).** Each QA result carries a scoring identity (sha of the probe voice
   baseline and the detector config); only takes measured under the current identity are
   compared. A take is restored only when its own round's raw ASR found no issue in that segment
   (a known ASR regression is never restored automatically).

## 5c. Implementation notes (phase 2, 2026-10-31)

* **Modules.** `app/autopilot/flows/localisation_finish.py` (steps; appended by
  `LocalisationFlow.steps`, flow id `localisation` v1 unchanged), `localisation_finish_handlers.py`
  (handlers + the AICP-368 hook), `localisation_qa.py` (pure checks + the two compute functions).
  No migration, no new worker or queue: the compute runs on `autopilot_compute` (run workers),
  the handlers in the autopilot ticker.
* **Regeneration option handler** is the core's `rerun_segments` (§5b.1); best take uses the core's
  `restore_takes`; the failed-run round is C1's `_failed_run_round`; stale answers C1's
  `_clear_stale`.
* **Voice-gender overlap rule** (found by the e2e test on 46cc6443): a line is excluded only when
  another FOREGROUND line overlaps it; the BACKGROUND device line under the woman's lines in
  segment 3 does not exclude them (it plays quietly, the foreground voice dominates the pitch).
* **Picture drift** on a flat or static picture is matcher noise (every frame matches): warnings
  only, never a proposal.
* **api_v2.** `create_caption` assigns the caption id at construction (its reservation was keyed
  `caption:None`); `_require_stitch_library_clips` in `_admit_segment_rerun` (single rerun, multi
  rerun, restore) and `retry_run`.
* **AICP-368 integrated** (merge of `feat/aicp-368-caption-placement` 08b8f8b, which carries
  main 2f43800 + C1): `create_caption` gets `placement="project"` explicitly; the script caption
  row freezes `caption_placement.snapshot_for_caption(project)`; before the ASR caption of a round
  `qa:L` waits while the scan is `pending`, and re-derives (`localisation_rederive_placement`:
  `request_derivation` + `enqueue_scan`, after a read-only lineage check, at most twice per step
  generation) when it is `failed` or pending for more than 30 min; after that it captions with
  the value there is. The listening proposal shows the placement state.
* **Gate R1 fixes**: a failed burn is retried with exactly the lines decided before (not replanned
  from a failed row); a garbled round is never ASR evidence for restoring its take; the language's
  caption spend includes captions this autopilot created and that were deleted since.
* **Tests.** `tests/test_autopilot_localisation_qa.py` (pure pieces; pitch and the whole segment
  QA on ffmpeg-generated tones), `tests/test_autopilot_localisation_finish.py` (the flow on the
  real core from C1's harness, real tone media for every clip, caption/tail workers faked at their
  enqueue): the seeded gender error in segment 3 → a regeneration approval for segment 3 only with
  its cue and cost → approve → mocked rerun → ASR re-check approval → `listening_final` →
  captions (one misheard word fixed, burn only) and the CTA tail without a new reservation → TEMP
  clip and its directory deleted → spend journaled → a later rerun refused before billing; a
  missing line → proposal for its segment, "accept", a rejected final naming segment 5 → a fix for
  segment 5 only with the operator's and the gender cue, both answers consumed; a worse rerun →
  the earlier take restored before its ASR; garbled ASR → captions from the script; a pending
  AICP-368 scan → no caption is created; the `caption:None` fix.

## 6. §2 design review (GPT-6.1 Terra, `codex exec`, reasoning effort high)

Round 1 (session `01a1910d-97aa-7603-9a24-197cae0faaae`, on d047681): 3 BLOCKER, 8 MAJOR, 1 MINOR.
All checked against the code; resolutions in §5 (R1-n = item n). Rejected: R1-12 (premise false,
see §5.12). Partly rejected: R1-1 (the deletion stays — operator decision — with the rerun guard).

Round 2 (same session, on d3f505c): 2 BLOCKER, 7 MAJOR — resolutions in §5b (R2-n = item n).
Terra accepted R1-12 on the stated evidence and the operator's deletion decision (R1-1). Not
taken: a core extension for multi-action authorisations (R2-1: separate approvals instead), a
caption version key in the core (R2-4: recorded evidence instead, core change out of scope).

## 7. §1 gate (GPT-6.1 Terra, `codex exec`, reasoning effort high)

Session `01a1913b-279b-7203-8174-9251daf35dca`, range `3a0ea21..HEAD`.

| Round | Findings | Resolution |
|---|---|---|
| R1 (e78eb60) | MAJOR: a failed burn was replanned from the script · MAJOR: garbled ASR counted as clean restore evidence · MINOR: deleted autopilot captions missing from the spend | retry with the decided lines; garbled = no evidence; captions from the effects' refs |
| R2 (a04f1c0, after the 368 merge) | the three R1 fixes missing (lost in a stash during the merge — true) · MAJOR: re-derivation not fenced under the project lock · MAJOR: an inherited pending placement (`inherited_at`) never went stale · MAJOR: a failed placement without a lineage refused forever | fixes restored; `expect_placement` checked after `refresh(FOR UPDATE)`; `inherited_at` counts; no re-derivation without a lineage (planning checks `removal_chain`, read-only) |
| R3 (73ad240) | MINOR: captions created and deleted by hand missing from the spend | caption reservations of the language's runs by `run_id` |

R3 left no BLOCKER or MAJOR; its MINOR is fixed (cf977de).
