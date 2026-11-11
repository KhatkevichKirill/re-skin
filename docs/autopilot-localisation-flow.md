# Autopilot flow: localisation generation (AICP-365, "C1")

Tracking: AICP-365, parent
AICP-362. Runs on the autopilot core
(AICP-363, [autopilot-design.md](autopilot-design.md) rev 7). Covers steps 0–9 of
[assistant-flow-localisation.md](assistant-flow-localisation.md); the finishing flow
(AICP-366, "C2": segment/ASR QA, listening final, fix loop, captions, CTA tail, cleanup)
continues from the stitched final of `rest:<lang>`.

Status: **implemented** (phase 2, 2026-10-31) on the merged core (AICP-363 959e724) and
AICP-347 (28a2167). §8 records where the code differs from or makes concrete the text below.

Operator decisions that shape it (2026-10-31): Q2 speakers / gender / on-off face / actions
come from `transcribe/v3` — **no Judge step**; Q3 compute runs on the run workers
(`autopilot_compute`); Q4 languages and CTA tails are `autopilot_params`; Q5 a manual paid
action pauses the autopilot; Q6 an approved regeneration adds its cost to the budget. The
`listening_probe` approval is this flow's; `listening_final` is C2's.

## 1. Params (`autopilot_params`, Q4)

```json
{"languages": ["es", "pt"], "tails": {"es": "<library clip id>", "pt": null},
 "model": "seedance-2-5", "resolution": "720p"}
```

`localisation_plan.validate_params`: languages non-empty, unique, in `app/languages.py`, in
generation order; `tails` keys ⊆ languages, each a library clip id (must exist; a clip tagged
with another language is refused, an untagged clip is a language-agnostic tail) or null; the
model must produce audio **and take a voice reference** (only Seedance 2.5 today); resolution
allowed for it. Unknown keys → 422. "Target = source language" is checked once the transcript
exists (`transcribe` step). Tails are consumed by C2.

## 2. Steps

`steps(project, params)` for languages `L1..Lk` (order of `params.languages`):

| step | needs | paid | approvals | estimate (credits) |
|---|---|---|---|---|
| `intake` | — | no | `handoff` | 0 |
| `transcribe` | intake | yes | `regeneration` | `llm_credits` unless a ready v3 transcript exists |
| `segments` | transcribe | no | `handoff` (v2: the source's end card, §10) | 0 |
| `breakdown` | segments | no | — | 0 |
| `translate:L` | breakdown | yes | `regeneration` | 3 × `llm_credits` (= the admission's reservation) |
| `prompts:L` | translate:L | no | — | 0 |
| `probe:L` | prompts:L | yes | `listening_probe`, `regeneration` | `video_credits(probe segment)` |
| `rest:L` | probe:L | yes | `regeneration` | Σ `video_credits(other swap segments)` |

For 46cc6443 (79 s, 7 segments, 720p) one language estimates 60 + 1 040 + 5 520 credits;
it really cost 912 + ~4 950 with no regeneration (the rate is per started second).

The budget approval (core, §6.2) is raised by the first paid `Launch` — `transcribe`, or
`translate:L1` when a ready v3 transcript already exists — and lists every paid step above.
Translations of every language run right after `breakdown` (cheap, and every language's
translation warnings surface before the first probe). The probes of languages after the first
wait for the first language's probe to be **accepted**: one listening pass validates the prompt
format before more video is paid for. That wait is a soft gate (`Wait` until
`ctx.output("probe:L1")` exists), **not** a `needs` edge: a `needs` edge would make a later
change to L1's probe reopen and re-buy every other language's probe (§7 #2).

**Replacement launches are regenerations.** Any paid `Launch` of a step whose generation is
> 1 (it was reopened by a `depends_on` change) or whose earlier round already launched is
`regeneration=True`, so the core asks an action-bound approval before re-buying work that was
already paid for (§7 #2; needs a ctx accessor for the step's generation / previous outputs).

### 2.1 `intake` (free)

* project not `ready` → `Wait` (the core pauses on `failed` / a stall).
* `localisation_plan.intake(project)`: not localisation, or not `localisation_scope=whole`
  (AICP-340), or an `empty` transcript → `Fail` with the remedy (nothing has been spent).
* `mute_source` false → `Launch("mute", "set_mute_source", {"mute_source": true})` (free).
* Source clean? `Compute("captions", "localisation_source_captions", {…})` — `qa_tools.detect_captions`
  on the source at 2 fps (streamed, run workers). A hit ratio above 20 % →
  `Ask("source_captions", "handoff", options=[continue: the source is clean, stop: run the
  subtitle-removal flow first])`; `stop`/reject → `Fail(consumes=["source_captions"])`;
  continue → the `Complete` below lists `consumes=["source_captions"]`.
* `Complete({warnings, duration_sec, caption_scan})`; the > 60 s warning ("expect regenerations")
  goes into the reason, so it is journaled.

### 2.2 `transcribe` (paid)

`transcript_state()`: `ready` only for a ready **schema-3** transcript; a ready v1/v2 is
`stale` and is re-transcribed (it lacks speakers/gender/face/actions, Q2).

* `ready` → check `check_languages_against_source` (problem → `Fail`, params are still
  editable when nothing was launched) → `Complete({transcript_digest, source_language})`.
* `missing`/`stale`/`failed` with no effect of ours → `Launch("launch", "transcribe_project", {})`.
* our job `queued`/`running` → `Wait`.
* our job `failed`/`interrupted` → `Ask("retry:rN", "regeneration", options=[retry
  (transcribe_project, ~20 cr), stop])` → approve retry → `Launch(next_round, …,
  regeneration=True, consumes=[…])`; stop/reject → `Fail(consumes=[…])`. An interrupted call
  may have been billed: it is never resubmitted without that approval.

### 2.3 `segments` (free)

* `Launch("recut", "recut_at_pauses", {})` once — the AICP-347 endpoint (moves boundaries off
  spoken lines, ids/count unchanged, one commit). Its result (`applied`, `feasible`,
  `boundaries`, `warnings`) is the effect result.
* Then `check_partition(segments, duration, whole_video_target_sec)`: contiguous 0 → duration,
  first segment swap, every swap clip incl. rolls ≤ 14.72 s. Not ok → `Fail` with the problems
  (only a hand edit can produce it; the operator fixes the segments and starts a new
  autopilot — the transcript is reused, nothing is re-billed).
* `Complete({segments, recut}, depends_on={project_segments:<pid>: …})`.
* **Flow v2 (AICP-390)** asks about the source's own CTA end card before completing, and
  completes with `end_card` and `depends_on` segments + transcript — §10.

### 2.4 `breakdown` (free)

`source_breakdown(transcript, segments)` — per swap segment: clip window, lines (tag, kind,
gender, on/off face + what the frame shows), persons present with their line counts, actions
inside the window (clip-relative), brands spoken, device lines under other speakers. Those
"overlapping voices" are **kept** in the prompt marked BACKGROUND (operator decision
2026-10-29, codified in `localisation.format_timed_dialogue`) and listed as a warning.
`pick_probe_segment`: the first swap segment with ≥ 2 s of speech (segment 1 on 46cc6443; a
silent intro is skipped — it would give no voice reference). None → `Fail`.
`Complete({breakdown, probe_segment_id}, depends_on={project_segments:…})`; the reason
summarises speakers and warnings (the journal shows it, AICP-365 scope item 3).

### 2.5 `translate:L` (paid, LLM)

* `Launch("launch", "localisation_prompt", {"language": L, "swap_character": false,
  "model": params.model})` → a durable job row (§3.2), id in `created_refs`.
* job `queued`/`running` → `Wait`; `succeeded` and its fingerprint equal to the project's
  current one (`localisation_prompt_job.project_fingerprint`) → `Complete({job_id,
  warnings}, depends_on={project_segments:…})`.
* `failed`/`interrupted`, or a stale fingerprint → `Ask("retry:rN", "regeneration",
  options=[retranslate (~60 cr), stop])`.

### 2.6 `prompts:L` (free)

`build_language_prompts(translated_lines, transcript, segments, probe_segment_id, …)` — the
translate job's construction (midpoint partition, clip windows, speech-only template), with
the probe built **without** and every other segment **with** the voice-reference Speakers
header. `Complete({sha256, probe_segment_id, warnings})`. Prompts are deterministic, so
`probe:L` / `rest:L` rebuild them and check the sha instead of carrying ~2.5 KB per segment
in a 64 KiB output.

Acceptance (tests): for 46cc6443 es/pt/pl/fr the prompts of segments 2–7 are byte-identical
to the AICP-335 fixture's expected prompts (the refs' `segment_prompts` + AICP-335's three
documented deltas), and the probe prompt is the refs' `seg1_probe` with only the WOMAN gender
clause delta.

### 2.7 `probe:L` (paid) — segment 1 alone, then the listening approval

```
launch     Launch create_run(probe_run_spec)    stitch_plan = [probe segment], no voice
           run not terminal -> Wait
           run failed/incomplete -> Ask rerun:rN regeneration, options=[{rerun: rerun_segments(
                         probe run, [probe segment, current prompt]), cost}, {stop}];
                         rerun -> Launch(regeneration=True, consumes=[rerun:rN]);
                         stop/reject -> Fail(consumes=[rerun:rN])
sheet:rN   Compute localisation_probe_sheet     qa_tools.comparison_sheet source window vs
                                                result, 1 fps, time_map=scale (streamed)
listen:rN  Ask listening_probe, subject=L, artifacts=[probe final (video), sheet (image)],
           proposal={run, clip length vs source, per-frame diffs}, about={run:<probe>}
           pending -> Wait
           approved -> Complete({probe_run_id, probe_segment_id},
                                depends_on={run:<probe>: run_version}, consumes=[listen:rN])
                       (earlier rounds were consumed by their rerun Launch)
           rejected -> fix:rN
fix:rN     Ask regeneration, options=[{rerun: rerun_segments(probe run,
           [{run_segment_id, prompt: apply_operator_cue(prompt, comment)}]), cost}, {stop}]
           approved rerun -> Launch rerun:rN (regeneration=True, consumes=[listen:rN, fix:rN])
                             -> back to "run not terminal"; next sheet/listen round
           stop / rejected -> Fail("probe rejected: …", consumes=[…])
```

"Reject with a comment = regeneration approval with the corrected cue" (AICP-365 §6): the
comment is appended verbatim under one fixed header ("Corrections from the operator's
listening check — follow them exactly:"), and the regeneration approval shows the resulting
prompt and its cost before anything is billed. No judgement call rewrites it (Q2). A manual
rerun of the probe by the operator pauses the autopilot (Q5); after Resume the listening
approval is superseded by the new run version (core §2.5) and asked again.

### 2.8 `rest:L` (paid) — segments 2..N in one run

* `Launch("temp_clip", "create_temp_library_clip", temp_clip_spec)` — the accepted probe's
  final as an untagged library clip named `TEMP autopilot <pid8> <L> probe` (never auto-picked
  as a tail); id in `created_refs`.
* `Launch("launch", "create_run", rest_run_spec)` — stitch plan = every segment in order with
  the TEMP clip **in the probe segment's place** (the probe segment is never regenerated);
  `segment_prompts` for every other swap segment; voice reference = the audio of that same
  TEMP clip (the accepted probe's full dialogue, both speakers: "voice B" of the 46cc6443
  journal). Picture and voice come from one immutable copy made under the probe run's lock, so
  they cannot come from two different takes (§7 #12). Both launches carry
  `expect={run:<probe>: the version probe:L completed with}`; if the probe changed since, the
  TEMP clip effect is discarded and both inputs are made again from the newly accepted probe.
* run not terminal → `Wait`; failed/incomplete → `Ask rerun:rN regeneration` [rerun the
  failed segments | stop] (consumed as in §2.7); done → `Complete({run_id, temp_clip_id, probe_run_id},
  depends_on={run:<rest>: …})`. C2 deletes the TEMP clip at the end (step 17).
* **Flow v2 (AICP-390):** a language with a CTA tail leaves the cut end card's segments out of
  its stitch plan (`rest_run_spec(exclude_segment_ids=…)`); a language without one keeps them
  as keep segments (source footage, source audio).

Both launches rely on the core's implicit expectation `run:<probe>` (transitive
`depends_on` of `probe:L`): a manual change to the accepted probe reopens `probe:L` and
`rest:L` instead of reusing a clip the operator did not accept.

## 3. Action handlers

| handler | paid | owner | state |
|---|---|---|---|
| `create_run` | yes | core | shipped; **needs `voice_from_library_clip`** (§3.4) |
| `rerun_segments` | yes, always regeneration | core | shipped as is |
| `transcribe_project` | yes | C1 | **durable job first** (§3.1) |
| `localisation_prompt` | yes | C1 | **durable job first** (§3.2) |
| `recut_at_pauses` | no | C1 | wraps AICP-347's endpoint (§3.3) |
| `set_mute_source` | no | C1 | wraps `update_project` with `mute_source=True` only |
| `create_temp_library_clip` | no | C1 | new service (§3.5) |

Compute functions (C1, registered with a version): `localisation_source_captions`
(`qa_tools.detect_captions` on the source, 2 fps), `localisation_probe_sheet`
(`qa_tools.comparison_sheet` + clip length). Both stream frames; outputs go to the attempt dir.

### 3.1 `transcribe_project` — a durable transcription job

Today `POST /transcribe` commits `transcript_status=pending` + the reservation, then enqueues
an RQ job no recovery covers; a lost enqueue leaves `pending` forever, a killed worker leaves
`running` forever (core §4.4).

New table `project_transcriptions` (migration chained onto the core's head):
`(id, project_id FK CASCADE, status queued|running|succeeded|failed|interrupted, origin
autopilot|manual, claim_token, attempts, rq_job_id, error, created_at, started_at,
heartbeat_at, finished_at)`, partial unique index one live (`queued`/`running`) job per project.

* **Admission** (`app/transcription_jobs.admit`, the handler's `execute`): project row locked;
  refuse (409 → `ActionRefused`) without a source file or with a live job; set
  `transcript_status=pending`, insert the job row, `spend.reserve(transcribe_scope)`, write
  `created_refs`, **one commit**; enqueue `run_transcription_job(job_id)` on `process` with
  RQ id `transcribe-{job_id}` after the commit.
* **Worker**: claim `UPDATE … SET status='running', claim_token=:t WHERE id=:id AND
  status='queued'` (0 rows → a duplicate delivery, exit); the paid call; then ONE commit that
  writes the transcript + `transcript_status` + `succeeded`/`failed` `WHERE claim_token=:t`.
  A heartbeat column is refreshed every 30 s from a thread while the call runs.
* **Dispatch generation** (§7 #8): the row carries `dispatch_gen`; every enqueue (admission,
  recovery) bumps it under the row lock and the RQ job id is `transcribe-{job_id}-g{gen}`. The
  worker claims only `WHERE status='queued' AND dispatch_gen=:gen`, so a stale or duplicate
  delivery claims nothing; the database claim is authoritative, RQ is only the doorbell.
* **Reaper** (recovery tick of the run workers, every 180 s; rows taken `FOR UPDATE SKIP
  LOCKED`, so two reapers never act on one row): `queued` with no live RQ job for 120 s →
  bump `dispatch_gen` and re-enqueue (nothing was sent before the claim); `running` with a
  heartbeat older than 150 s → in one statement `SET status='interrupted', claim_token=NULL
  WHERE id=:id AND status='running' AND claim_token=:observed`, `transcript_status=failed`,
  error "interrupted — the call may have been billed". Heartbeat and final writes require
  `status='running' AND claim_token=:t`, so a worker that comes back after the interrupt
  writes nothing (§7 #4). **Never resubmitted automatically**; the flow asks a regeneration.
* `reconcile(effect)`: the job row named in `created_refs` exists → `launched`, else
  `not_launched`.
* **One admission for every origin** (§7 #5): `POST /transcribe` and the after-analysis
  hand-off go through the same locked durable admission, so the one-live-job index refuses a
  second concurrent paid call whoever started it (last-writer-wins disappears). The hand-off is
  suppressed for assistant projects (core §6.5). Manual change: a second click while a
  transcription is live now 409s instead of starting a parallel one — acceptable because the
  reaper now resolves the stale `running` that the "re-runnable at any time" rule existed for.
  **Operator decision needed** (pending).

### 3.2 `localisation_prompt` — a durable translate job

Today the reservation is committed before an RQ-only job a deploy kills without replay, and a
joined job may never reach the commit (core §4.4).

New table `localisation_prompt_jobs`: `(id, project_id FK CASCADE, language, swap_character,
request_params JSON, snapshot JSON, fingerprint, status queued|running|succeeded|failed|
interrupted, result JSON, error JSON, claim_token, attempts, rq_job_id, heartbeat_at,
timestamps)`.

* **Admission**: `api_v2._prepare_localisation_snapshot` (its 400/404/409/422 →
  `ActionRefused`), localise worker alive (else 503 → refusal, retried with backoff), insert the
  row with the frozen snapshot and fingerprint, `spend.reserve(localise_scope, 3 × llm)` **in
  the same transaction** (the manual path's `_reserve` commits on its own; this one must not),
  `created_refs`, one commit; enqueue `localisation_prompt_job.run_durable(job_id)` on
  `localise` with RQ id `localise-prompt-db-{job_id}`.
* **Worker**: claim, dispatch generation and fencing as in §3.1; the computation runs exactly
  as `localisation_prompt_job.run` does today — inside `_job_translate_timeout()` with
  `translate_budgets()` and the RQ `job_timeout_sec()` (§7 #11) — then one commit of
  `result`/`error` + status `WHERE status='running' AND claim_token=:t`. The result is the same
  `LocalisationPromptResponse` the manual poll returns.
* **Reaper**: as §3.1 (`queued` lost → re-enqueue; `running` stale → `interrupted`, never
  replayed). It runs in the run workers' recovery tick and checks the `localise` queue.
* The flow builds the prompts from the job's **frozen snapshot** (its transcript and segments),
  never from the live project, so a prompt is always consistent with the translation it came
  from; it uses the result only while `fingerprint` equals the project's current one.
* The manual Redis-only path is unchanged.

### 3.3 `recut_at_pauses` (free)

AICP-347's endpoint `db.rollback()`s on an unchanged or infeasible plan, which would also
discard the core's already-flushed effect row (§7 #3). Phase 2 extracts a transaction-neutral
service from it (`speech_recut.recut_project(db, project) -> result`: validates, plans,
`apply_plan`, never commits or rolls back); the endpoint and the handler both call it, the
handler commits once with the effect and its `result` (`applied`, `feasible`, `boundaries`,
`warnings`). Its 409s map to `ActionRefused`. It fences on `project_segments:<pid>` under the
project row lock.

### 3.4 `create_run` — `voice_from_library_clip` (request to the core owner)

The core's `create_run` handler passes the endpoint's parameters explicitly; a voice is an
`UploadFile`. The flow needs `voice_from_library_clip: <clip id>`: the handler checks the clip
exists and is referenced by the same request's stitch plan, and runs `media.make_voice_reference`
on the clip's file into the new run's own voice dir (fresh path, rule (f)). Equivalent to the
journal's "cut the seg-1 dialogue": the probe final IS segment 1's dialogue. The TEMP clip
records the sha256 of the probe final it was copied from (in `created_refs`), so the handler
and the listening approval's artifact hash can be compared.

### 3.5 `create_temp_library_clip` (free)

`library_clips` has no "temporary" flag; the clip is untagged (`language=NULL`) and named
`TEMP autopilot …`. Execute: lock the source run (target), require terminal + `done`, copy its
`result_local_path` into a fresh `library_clip_dir(new id)` (streamed copy), `media.probe` it,
insert the row, `created_refs={"library_clip": id}`, one commit. Reconcile: the row exists →
`launched`, else `not_launched` (an orphan dir at most). Deletion (C2 cleanup) goes through
`delete_library_clip`, which already refuses while an active run's plan references it.

## 4. Requests to the core (found while mapping the flow)

1. **A transcript version key** (prerequisite for phase 2, §7 #6). `project_segments:<pid>` fences segment edits, but nothing
   fences the transcript: a manual `PATCH /transcript` or re-transcription after `breakdown`
   would not reopen `breakdown`/`translate:L`/`prompts:L`. Proposal: a fifth key kind
   `project_transcript:<pid>` = digest of the transcript (`localisation_plan.transcript_digest`),
   fenced by the project row lock, with `PATCH /transcript` and the transcription worker's
   final write taking it first; `breakdown`, `translate:L` and `prompts:L` `depends_on` it.
   A planning-time digest check is NOT a substitute (an edit can land between PLAN and APPLY).
2. **A ctx accessor for non-run keys**: `ctx.version("project_segments:<pid>")` — StepContext
   only has `run_version`.
3. **`voice_from_run` on `create_run`** (§3.4).
4. **Free handlers' results**: `recut_at_pauses` returns data the step needs; the effect's
   `result` must be writable by the handler inside its transaction (like `created_refs`).
5. **Rolls in the segments key**: `project_segments:<pid>` digests (id, index, action, start,
   end); pre/post-roll change the clip windows and the translate fingerprint too and must be in
   it (§7 #9).
6. **A generation accessor**: `ctx` must expose the step's generation / `previous_outputs`, so
   replacement paid launches are marked `regeneration=True` (§7 #2).
7. **`mute_source` frozen into the run** (§7 #10): the pipeline reads `project.mute_source` at
   submit time, after admission. Proposal: a nullable `Run.mute_source` set by the autopilot's
   `create_run` handler (NULL = read the project, today's manual behaviour), read by the submit
   path. Small migration; C1 can own it if the core prefers.
8. §5.7's manual-spend hook must see the durable jobs' reservations as autopilot ones when
   they are made inside a handler's `execute` (they are: the reserve runs there) — and the
   worker's draws are draws, not reserves, so they never trigger it.

## 5. What can strand a language, and the answer

| Situation | Outcome |
|---|---|
| Deploy kills a translation / transcription mid-call | job `interrupted` by the reaper → regeneration approval; never auto-replayed |
| Enqueue lost after the admission commit | reaper re-enqueues the `queued` job (nothing was sent) |
| Probe run fails | regeneration approval to rerun the probe segment |
| Operator edits segments after `segments` | `project_segments` reopens `segments` and every later step; runs already launched keep their `segment_snapshot` |
| Operator reruns the probe by hand | autopilot paused (Q5); after Resume the listening approval is superseded and re-asked |
| TEMP clip deleted by hand before `rest:L` launched | before launching, the step checks the clip still exists and otherwise creates a new one under the next round name (`temp_clip:r2`) |
| Transcript edited by hand | see §4.1 |

## 6. Phase-1 code and tests

`backend/app/autopilot/flows/localisation_plan.py` (no `__init__.py` in `app/autopilot/` or
`flows/` on this branch: namespace packages, so the core's own `__init__` merges without a
conflict), `backend/tests/test_autopilot_localisation_plan.py`,
`backend/tests/fixtures/autopilot_localisation_46cc6443_probe.json` (the refs' probe prompts).
The AICP-347 call is a boundary (`plan_recut`), stubbed while `app.speech_recut` is absent and
exercised against the real module when present.

## 7. §2 design review (GPT-6.1 Terra, `codex exec`, reasoning effort high)

Round 1, 2026-10-31 (session `01a19034-a9b1-7fd3-bb1c-151b4f965be9`), on commit e9cfc72:
6 BLOCKER + 6 MAJOR, all checked against the code and accepted. No round 2 yet: the resolutions
are applied in this document and re-checked by the phase-2 §1 gate on the implementation.

| # | Sev | Finding | Resolution |
|---|---|---|---|
| 1 | BLOCKER | intake acts on the `source_captions` answer without consuming it | named approval, consumed by the `Complete`/`Fail` that acts on it (§2.1) |
| 2 | BLOCKER | a reopened paid step re-buys work under the budget without a regeneration approval; `probe:L1` as a `needs` edge reopens every language | replacement launches are `regeneration=True`; the L1 gate is a soft `Wait`, not `needs` (§2); core request 6 |
| 3 | BLOCKER | the AICP-347 endpoint's rollback discards the core's flushed effect | transaction-neutral recut service, one handler commit (§3.3); verified: the endpoint calls `db.rollback()` on an unchanged plan |
| 4 | BLOCKER | reaper `interrupted` does not revoke the claim; a late worker overwrites | the interrupt revokes `claim_token` atomically; worker writes require `running` + token (§3.1, §3.2) |
| 5 | BLOCKER | manual / after-analysis transcription outside the durable admission races it | one admission for every origin; hand-off suppressed for assistant projects (§3.1); manual 409 on a live job is an operator decision |
| 6 | BLOCKER | a planning-time transcript digest check does not fence | transcript version key is a phase-2 prerequisite on the core (§4.1); prompts built from the job's frozen snapshot (§3.2) |
| 7 | MAJOR | `Complete(consumes=[all answered])` lists already-consumed approvals; retry branches unspecified | consume only the current round's answer; every retry branch has action-bound options and its consumption (§2.7, §2.8) |
| 8 | MAJOR | "no live RQ job, re-enqueue the same id" is not atomic | `dispatch_gen` on the row, generation in the RQ id, claim requires it; reaper rows `FOR UPDATE SKIP LOCKED` (§3.1) |
| 9 | MAJOR | translate fingerprint inputs beyond `project_segments` (transcript, rolls) are not fenced | core requests 1 and 5; prompts from the frozen snapshot |
| 10 | MAJOR | `mute_source` is read at submit and can change after admission | `Run.mute_source` frozen at admission for autopilot runs (§4.7) |
| 11 | MAJOR | the durable worker would lose the job's timeout and translate budgets | reuse `_job_translate_timeout`, `translate_budgets`, `job_timeout_sec` (§3.2) |
| 12 | MAJOR | TEMP clip and voice read the probe's mutable final separately | voice from the TEMP clip (one immutable copy), both launches `expect` the accepted probe version (§2.8, §3.4); `rest_run_spec` changed accordingly |

## 8. Implementation notes (phase 2)

* **Modules.** `app/autopilot/flows/localisation.py` (the flow, registered from
  `flows/__init__.py`), `localisation_handlers.py` (handlers + compute functions),
  `localisation_plan.py` (pure); `app/localisation_jobs.py` + `app/localisation_jobs_models.py`
  (durable jobs); migration `s3b5d7f9a1c2` (chained onto `r1a7c3e5b9d2`, **not applied**):
  `runs.mute_source`, `project_transcriptions`, `localisation_prompt_jobs`.
* **Core requests** 1–4 and 6 were taken by the core (`project_transcript:` key,
  `ctx.version` / `transcript_version` / `previous_outputs`, rolls in the segments digest,
  `voice_from_library_clip`, handler-written `effect.result`). Request 7 (`Run.mute_source`) is
  C1's: `pipeline_v2.effective_mute_source(run, project)` reads the run's frozen value first (the
  run submit and the range-attempt submit); only the flow's `localisation_create_run` (a
  `create_run` subclass) freezes it, to `True`. Manual runs keep NULL = today's behaviour.
* **Rejected listening answers.** The core stops an autopilot whose step keeps `Wait`ing with a
  rejected, unconsumed answer (600 s grace). So a rejection is consumed at once by a free
  `localisation_probe_feedback` launch (`cue:rN`) that records the comment and the corrected
  prompt in its effect result; the `fix:rN` regeneration is asked afterwards.
* **Retry rounds.** Transcription and translation: `retry:rK` (regeneration with options) after
  the K-th launch failed / was interrupted / went stale; approve → `launch:r(K+1)` with
  `regeneration=True`, which raises the budget by the option's cost (Q6).
* **Replacement launches.** Every paid launch of a reopened step (`previous_outputs()` not
  empty) is `regeneration=True`. The segments step does not re-cut on a reopen (the operator's
  own edit is kept).
* **Fences.** `breakdown`, `translate:L`, `prompts:L` complete with `depends_on` the segments
  and transcript keys; every paid launch `expect`s them. `probe:L` depends on its probe run;
  `rest:L` depends on nothing of its own run (C2 regenerates it on purpose, §15 of the core).
* **Prompts** are built from the translate job's frozen snapshot (transcript, clip ceiling) and
  the live segments; `prompts:L` stores only their sha256, `probe:L` / `rest:L` rebuild and
  check it.
* **Transcription for every origin** goes through `localisation_jobs.admit_transcription`:
  `POST /transcribe` (409 while one is live — operator decision 2026-10-31), the
  after-analysis hand-off (skipped while one is live), the autopilot. The old RQ function
  `tasks.run_transcribe_project` stays registered for jobs queued before the deploy.
* **Reaper**: `localisation_jobs.reap()` runs in `tasks.run_reconcile_tick` (180 s).
* **Drain**: `deploy/drain-check.sql` counts live `localisation_prompt_jobs`; transcriptions are
  already counted through `transcript_status`.
* **Tests**: `tests/test_autopilot_localisation_flow.py` (the flow on the real core, SQLite,
  mocked providers: budget first, probe = segment 1 only without voice, nothing of 2..N before
  the listening approval, rest plan = TEMP clip + 2..7 with the clip's audio as voice,
  byte-exact AICP-335 reference prompts, reject → cue → regeneration → second round, rejected
  fix → failed, failed transcription → retry approval, interrupted translation never replayed,
  captioned source → hand-off, second language waits for the first probe),
  `tests/test_localisation_jobs.py` (claims, generations, fencing, reaper, hand-off,
  `effective_mute_source`), `tests/test_autopilot_localisation_plan.py` (pure pieces).

## 9. §1 gate (GPT-6.1 Terra, `codex exec`, reasoning effort high)

Five rounds on `61cf69b..HEAD` (phase 2) plus `localisation_plan.py` in full. Round 5 left no
BLOCKER or MAJOR; its one MINOR was fixed as well.

| Round | Findings | Resolution |
|---|---|---|
| R1 | BLOCKER: replacement launches detected only via `previous_outputs` (a step reopened mid-flight re-bought its work) · MAJOR: legacy transcriptions in the mixed old/new window invisible to the one-live-job check · MAJOR: probe sheet reused after a manual rerun | `_bought_before()` looks for earlier-generation paid effects of the step; sheet keyed by the probe run's version · **rejected**: the drain check requires `transcript_status` pending/running = 0 before a deploy and the old RQ function stays registered; refusing on a legacy `running` without a durable row would bring back the stale lock (Terra agreed in R2) |
| R2 | MAJOR: a consumed rejection of take A applied to a manual take B | the whole listening round (`sheet`/`listen`/`cue`/`fix`) is keyed by the run version; the probe's decisions consume every open answer so stale ones are superseded |
| R3 | MAJOR: stale rejection unconsumed while waiting on B (core grace stop) · MAJOR: `rest:L` completion ignored an answered failure approval | `_clear_stale()`: answers whose bound rows changed are consumed by a no-op launch the core supersedes and discards; `rest:L`'s `Complete` consumes open answers |
| R4 | MAJOR: clearing ran only after the run was terminal · MAJOR: the acknowledgement listed only stale answers (`_check_unconsumed`) · MINOR: pending failure approval reused after a failing manual rerun | clearing runs before any wait; it lists every open answer (a stale one guarantees the discard, so nothing current is consumed); `fail:rN` tagged with the run version |
| R5 | MINOR: a manual transcription after our failed job was ignored | any ready v3 transcript completes `transcribe` |

## 10. The source's own CTA end card (AICP-390, flow v2)

Project ecb9ac03 (2026-11-05) ended with the source's own CTA: an off-screen narrator, "Listen
to anything. Download Listenify now.", over a promo graphic (49.07–54.47 s). Flow v1 sent it to
Seedance with the rest (it re-voiced it wrongly and pasted the woman onto the graphic, 380 cr)
and `tail:L` would have appended the language CTA after it: two CTAs. Flow **v2** proposes the
card to the operator at `segments`; v1 stays registered unchanged for the attempts created under
it (§4.7 of docs/autopilot-design.md), new attempts get v2.

**Detection** (`localisation_plan.detect_end_card`, a candidate, never a verdict): the trailing
lines that are all off screen (`on_screen` false, `face_visible` not true), spoken by voices heard
nowhere before them and not a device. The interviewer's closing "Thank you" is not one (the
interviewer speaks earlier), nor a phone voice. The card starts at the scene cut between the last
line before it and its first line (≤ 0.5 s into that line — transcript times are coarse); without
one at its first line, with a note to check the boundary. Only a card of 1–15 s with a line before
it is proposed; a silent end card is not found. Over the 18 localisation transcripts in prod
(2026-11-05) it fired on ecb9ac03 and 5eb99ba8 (the same source) and on b8d5b3c2, a v2 transcript
whose phone voice has no `kind` (the autopilot re-transcribes v2 to v3 first).

**The question** `endcard:<segments version[:12]>` (kind `handoff`, `about` segments + transcript):
start, lines, frame descriptions, on-screen text, per language what happens (a tail replaces it /
no tail: source footage kept), the segments after the cut. Options `replace` ("Cut it from X s",
pre-authorising `localisation_cut_end_card` with the planned patch) and `generate` ("It is not an
end card: localise it" = v1 behaviour) and `stop`; a reject or `stop` fails the attempt, like
every localisation hand-off (the approval card words its "no" control "End this attempt"). While it is open the operator may
move the boundary in the Segment Editor: the segments version changes, the open question is
retired and a new one is asked about the new state; an answer about segments that changed since is
cleared by `_clear_stale` before anything else.

**The cut** (`end_card_plan` + handler `localisation_cut_end_card`, free): one PATCH /segments
through `api_v2.update_project_segments` (ready / active-run 409s, partition normalisation, its
commit), fenced by `expect` = segments + transcript and consuming the answer. Segments wholly
before the card stay; the one the card starts inside ends at the card (a swap piece under the
model floor joins the swap before it); everything after becomes ONE keep segment (the id of a
segment already starting there is reused); the swap ending at the card loses its post-roll. Not
offered (only `generate`, with the reasons) when a spoken line other than the card's runs past
the cut, a card line starts before it, or the result breaks the partition (a swap over the
14.72 s target). The planned result is written to the effect before the commit; the step then
re-reads the segments (`end_card_segment_ids`) and completes with `end_card: {choice: replace,
start_sec, segment_ids, transcript_version}`. A reopened step reuses an earlier generation's cut
only when the segments and the transcript still show it exactly; anything else is asked again.

**Downstream** nothing new: keep segments get no lines, no prompt and no estimate; the translate
window ends at the last swap segment, so the card's line is not translated; `rest:L` leaves the
card out when the language has a tail (`tail:L` then appends the CTA to the shortened captioned
final at `duration − 0.01` as before) and keeps it as source footage when it has none — its
captions then come from the ASR only (the script fallback has no lines for a keep piece).

**Deploy**: no migration, no new worker. Attempts already past `segments` stay on v1 (ecb9ac03,
paused at `qa:es`, must be finished by hand: its `tail:es` would append a second CTA).

**Tests**: `tests/test_autopilot_localisation_end_card.py` — detection and the plan on ecb9ac03's
real inputs (`tests/fixtures/localisation_ecb9ac03_end_card.json`), the flow on the real core
(cut → the tail language's run has no card, the no-tail language keeps it; generate; edit while
open → asked again; infeasible boundary → generate only; v1 never asks; no card → no question).
