# Rebuild from reference — implementation notes (wave 1)

Companion to `docs/rebuild-from-reference-design.md` (the contract). This file
says what exists after wave 1 (branch `feat/rebuild-core`), where the seams
are, and what waves 2a (storyboard/keyframe jobs + API/UI) and 2b (images-only
run path) must call. Nothing here is deployed; migration `t4d9f2b7c315` is not
applied anywhere but disposable databases.

## 1. What exists

| Area | Where | Notes |
|---|---|---|
| Flag | `settings.REBUILD_ENABLED` (`app/config.py`, default false) | Gates in `api_v2._validate_project_type` (create_project + move-run-to-project: 409 for `rebuild` while off) and `api_v2.create_run` (a rebuild project answers 409; Kling is refused on every non-rebuild project, flag or not, because only `ai_models.V2V_MODELS` pass there). `web_v2` lists `rebuild` in the Create Project / Move forms only while on (`project_types.selectable_project_types`). |
| Project type | `app/project_types.py` | `REBUILD`, `SEG_SHOT_CHUNKS`, `GENERATION_V2V` / `GENERATION_IMAGES_ONLY`, `ProjectTypeSpec.generation_mode`, `REBUILD_DEFAULT_RESOLUTION = "720p"`, `is_rebuild()`. Default model `kling-4-0`. |
| Models registry | `app/ai_models.py` | `kling-4-0` (family `kling`, `supports_v2v=False`), new spec fields `supports_v2v`, `supports_images_only`, `images_only_min_sec/max_sec`, `max_reference_images`, `image_ref_token`; `seedance-2-5` gained images-only 4–15 s / 9 images / `[Image {n}]`. `V2V_MODELS` / `IMAGES_ONLY_MODELS`; every v2v aggregate (segment caps, floors, the New Run form's list, localisation's model check) reads `V2V_MODELS`, so Kling moves no v2v number. Helpers: `images_only_duration`, `validate_images_only_duration`, `image_ref`, `supports_v2v`, `supports_images_only`. |
| Data model | `app/rebuild_models.py` (imported at the end of `models.py`), migration `backend/alembic/versions/t4d9f2b7c315_rebuild_from_reference.py` (down_revision `t4d9a2c6e815` since the integration; wave 1 chained it on `s3c8e1a6d204`) | Tables `rebuild_shots`, `project_storyboards`, `rebuild_keyframes`, `rebuild_chunk_attempts`, `rebuild_llm_calls`; `runs.generation_mode` (CHECK v2v/images_only, server default v2v), `runs.rebuild_snapshot`, `video_projects.rebuild_analysis_token`. Status constants and partial-index WHERE texts are module constants shared byte-for-byte with the migration (a test compares them). |
| Pure helpers | `app/rebuild.py` | Constants (§5.1/§6.4/§6.5), `detect_shots`, `effective_shots`, `build_chunks`, `plan_segments` (with brand-card keeps), `chunk_shot_pieces`, `chunk_keyframe_ids`, `check_chunk_image_count`, `sheet_for_chunk`, `render_chunk_prompt`, `request_sec_for_chunk`, `frame_index` / `chunk_frames` / `fit_plan` / `fit_filter` / `fit_ffmpeg_args` / `keep_cut_ffmpeg_args`, `cut_timing_report`, `trailing_end_card`, `default_stitch_plan_with_end_card`. |
| Analysis | `pipeline_v2.analyze_project` (`SEG_SHOT_CHUNKS` branch) + `app/rebuild_analysis.py` | Detects shots, then under the project row lock and only if the project is still `analyzing`: new shot generation, old-generation project keyframes and sheets deleted, folding written (`folded_into`), chunk SegmentDefs, `ready` — one commit. `rebuild_analysis.rechunk(db, project, keep_brand_cards=...)` re-runs the partition on the current shots (unfold / brand-card keep). |
| Re-analysis / delete guards | `api_v2.reanalyze_project`, `delete_project`, `delete_run` | A rebuild re-analysis is refused (409) while any storyboard / keyframe / LLM call / chunk attempt row is live, and flips the project to `analyzing` in the admission transaction under the project row lock; `analyze_project` accepts an already-`analyzing` rebuild project. `delete_project` 409s on live rebuild rows; `delete_run` 409s while any chunk attempt of the run is live. |
| kie.ai | `app/kie_client.py` | `create_task_once(payload, *, timeout)` and `chat_completion_once(model, messages, *, timeout, response_format=None)` — one POST each, returning a `SendOutcome` (`accepted` / `refused` / `unknown`, `task_id`, `content`, `body`, `file_unavailable`); pure classifiers `classify_create_response` / `classify_chat_response`. Payload builders `build_kling_payload`, `build_seedance_images_payload`, `build_images_only_payload` (dispatch on family). `build_seedance_payload` and every v2v path are untouched (byte-identity test against main's payloads). |
| Spend | `app/spend.py` | `IMAGES_ONLY_CREDITS_PER_SEC`, `images_only_video_credits`, `storyboard_credits(n_shots)`, `keyframes_credits(n)`, scopes `storyboard_scope(storyboard_id)`, `keyframes_scope(batch_id)`, draw key `rebuild_call_key(row_id)`; `reserve_computed` (scope lock -> day locks -> estimate -> reserve); `draw_in_session(..., scope_lock=True)`; `settle_in_session` (strict, in the caller's transaction, scope-locked by default); `estimate_images_only_run_credits`, `check_allowance_credits`, `reserve_images_only_run`. `estimate_run_credits` and `reserve_run` dispatch an images-only run to these, so the existing Retry / rerun endpoints get the locked estimate; v2v behaviour is unchanged. |
| Paid-call protocol | `app/rebuild_paid_call.py` | The §7.2 state machine for all three paid-call tables: `claim`, `begin_send`, `send_window_open`, `record_send_outcome`, `record_poll`, `complete_fetch`, `fail_fetch`, `cancel_unsent`; boot ids (`register_boot`, `sender_identity`); `sender_proof`; step job ids + `enqueue_step` + `run_step`; `reaper_decision` + `reap` (wired into `tasks.run_reconcile_tick`); `release_sender` + CLI; live-row guards `project_live_rows`, `run_live_attempts`, `live_rows_message`. |
| Worker boot | `worker/worker.py` | `rebuild_paid_call.register_boot(redis_conn)` before any work-horse forks. |
| v2v defence in depth | `pipeline_v2.submit_run` (returns None), `_build_submit_state` / `poll_run_once` (raise `ImagesOnlyRunRefused`), `process_run` (returns), `tasks.run_poll_run` (returns), `recovery._repoll_generating_segments` (returns), `pipeline_v2._swap_task_call` (refuses a non-v2v model) | ERROR log, no state change. |
| Storage paths | `app/storage.py` | `rebuild_project_dir`, `rebuild_shots_cache_path`, `rebuild_frame_path`, `rebuild_keyframe_path`, `rebuild_run_chunk_dir`, `rebuild_attempt_dir` (none under a `disk_cleanup.sweep` path; tested). |
| Drain | `deploy/drain-check.sql` | Counts live `project_storyboards` and the three paid-call tables. Needs the migration first. |
| Host attestation | `scripts/rebuild_release_sender.sh <row id>` | `docker ps -a --no-trunc` check, then `docker exec re-skin-api python -m app.rebuild_paid_call release-sender <id> --attest <container>:<absent/exited>`. Not exercised against Docker in tests (the Python side is). |

## 2. Seams for wave 2

### 2.1 Driving a paid-call row (both halves)

1. **Admission** (API, under the project row lock, `project.status == ready`):
   insert the row(s) `queued` with `estimate_credits` set, reserve
   (`spend.reserve` for a keyframe batch / `reserve_computed` for a
   storyboard; the run scope via `reserve_images_only_run`), commit, then
   `rebuild_paid_call.enqueue_step(table, row.id, 1, "claim")`.
2. **Register the step handler**: set
   `rebuild_paid_call.STEP_HANDLERS["<table>"] = "app.<module>.<func>"`
   (a module-level constant edit, NOT a runtime registration — the reaper and
   `run_step` resolve it by dotted path in whatever process runs them). The
   handler is `func(row_id, gen, step, job_id)`. Until it is set the reaper
   only resolves `sending` rows for that table.
3. **claim** step: `token = claim(db, Model, id, gen)`; None -> return. Do the
   free work (frame extraction, uploads; idempotent), commit, enqueue `send`
   with the SAME gen.
4. **send** step: read the row; stop unless `status == claimed` and
   `dispatch_gen == gen`; take `token = row.claim_token`; build the request
   (`kie_client.build_images_only_payload`, `build_image_payload`, or the chat
   messages); `sender = sender_identity(job_id)` (raises `SenderUnavailable`
   in a process without a boot id: do not send); `begin_send(...)` then
   COMMIT; `t0 = time.monotonic()`; if `send_window_open(t0)` call ONCE
   `kie.create_task_once(payload, timeout=SEND_HTTP_TIMEOUT_SEC)` /
   `kie.chat_completion_once(...)`, else `outcome = None`;
   `record_send_outcome(...)`, commit; enqueue `poll` (delay
   `POLL_DELAY_IMAGE_SEC` / `POLL_DELAY_VIDEO_SEC`) when `submitted`.
5. **poll** step: `kie.get_task(task_id)` is fine here (polling is
   idempotent and creates nothing); `record_poll(..., state, credits_consumed)`;
   re-enqueue `poll` while pending, enqueue `fetch` on `fetching`.
6. **fetch** step: download to `.part`, verify, `os.replace`; (chunks: fit);
   `complete_fetch(...)` plus the table's own same-transaction effects
   (auto-select a keyframe; select the attempt + segment `completed` +
   `local_result_path`); on error `fail_fetch(...)` -> re-enqueue `fetch` on
   `"retry"`. Never a new create.

Every step must refresh nothing else it does not own; all transitions are the
module's CAS helpers so the reaper's generation bump fences stale jobs.

### 2.2 Wave 2a (storyboard + keyframes + API/UI)

- `ProjectStoryboard` uses the AICP-365 `_DispatchColumns` (claim/fence like
  `localisation_jobs`); its slices create `RebuildLlmCall` rows (kinds
  `global` / `shot`) driven by §2.1. Interrupting a storyboard must call
  `cancel_unsent(db, RebuildLlmCall, RebuildLlmCall.storyboard_id == sb.id)`
  in the same transaction. The partial indexes `uq_rebuild_llm_shot_live` /
  `uq_rebuild_llm_global_live` refuse duplicate live calls — catch the
  `IntegrityError` as "already live".
- `budget_credits` holds what admission reserved
  (`spend.storyboard_credits(n_shots)` under `storyboard_scope(sb.id)`); stop
  admitting call rows once drawn + next estimate would exceed it.
- `file_unavailable=True` on a refused chat outcome = "unavailable for this
  input": admit the next model as a NEW row.
- Keyframe selection: under the project row lock, `UPDATE ... selected=false`
  for the slot then `selected=true` for the row (indexes
  `uq_rebuild_kf_shot_selected` / `uq_rebuild_kf_sheet_selected`).
- Unfold / brand-card keep / shot edits: lock project, require `ready`,
  refuse while a run is active, then `rebuild_analysis.rechunk(...)`.
- Every rebuild admission must lock the project row and require
  `status == ready` (design §5.1.8); `rebuild_paid_call.project_live_rows`
  gives the live counts.

### 2.3 Wave 2b (images-only runs)

- `api_v2.create_run`: replace the 409 "not available yet" branch with the
  §6.1 launch (snapshot via `rebuild.chunk_shot_pieces` /
  `render_chunk_prompt` / `request_sec_for_chunk` / `sheet_for_chunk`, run-owned
  keyframe copies under `storage.rebuild_run_chunk_dir`, `RunSegment` rows in
  `pending`, `generation_mode="images_only"`, then
  `spend.reserve_images_only_run` last).
- `tasks.run_submit_run`: branch on `run.generation_mode` right after the lock
  (the v2v code already refuses images-only runs as defence in depth).
- Attempt rows: `RebuildChunkAttempt` with `request_sec` / `chunk_frames`
  (`rebuild.chunk_frames(start, end, fps_out)`, `fps_out` =
  `media.get_default_target(info)[2]`) frozen at admission; the live index
  `uq_rebuild_attempt_live` makes a racing duplicate admission fail.
- Fit: `rebuild.fit_plan(raw_frames, chunk_frames, fps_out)` then
  `rebuild.fit_ffmpeg_args(raw, fitted, plan, fps_out)`; a failed plan is
  `fail_fetch(..., final=True, error=plan.reason)`. Keep segments of an
  images-only run: `rebuild.keep_cut_ffmpeg_args(...)` on `F(start)..F(end)`.
- QA: `rebuild.cut_timing_report(planned_cuts(pieces), detected)`; the
  character check is a `RebuildLlmCall(kind="character_check", slot=0|1)`;
  `spend.check_allowance_credits` already counts its undrawn slots.
- Retry Run / segment rerun: reset the failed/selected segment(s) to `pending`
  and call `spend.reserve_run(...)` (it dispatches to
  `reserve_images_only_run`) in the same transaction.

## 3. Deviations from the design (and the doc changes made for them)

1. **Storyboard reservation scope** is per storyboard (`storyboard:<id>`), not
   per project (§5.2 wrote `storyboard_scope(pid)`): the "drawn vs reserved"
   budget check then reads exactly this storyboard's draws.
2. **A skipped send** (send window elapsed) ends `failed` ("not sent"), draw
   settled to 0 — §7.2 wrote `unknown`. Nothing left the process, so `failed`
   is the truthful state and avoids a false "may have billed" warning.
3. **Step job ids** are `rb:<table>:<id>:<gen>:<step>` for claim/send and
   `...:<step>:<seq>` for poll/fetch (a step re-schedules itself; reusing one
   id would delete the running job's own hash). Liveness is "any job with the
   `rb:<table>:<id>:<gen>:` prefix is queued, scheduled or current on a live
   worker". Step jobs keep their hash and result for 7 days
   (`STEP_JOB_TTL_SEC`): the sender proof reads the send job's final status,
   which rq's default 500 s result TTL would lose.
4. **Sender proof** also accepts a send job that `failed` with its own
   traceback (the job function exited by raising, a death-penalty timeout
   included) besides `finished` and rq's work-horse reap record; an
   AbandonedJobError cleanup still proves nothing.
5. **The release CLI** is `python -m app.rebuild_paid_call release-sender`
   (§7.2 named `app.rebuild_jobs`); the attestation must name the row's
   `sender_host` (prefix match) and say `absent` or `exited`.
6. **Extra columns / indexes**: paid-call rows have `sender_note` (proof /
   attestation log) and `fetch_attempts`; storyboards have `build_mode`,
   `global_call_id`, `budget_credits`, `overwrite_edits`, `applied_at`; shots
   have `storyboard_id`; keyframes have `aspect_ratio`; LLM calls have `slot`
   plus `uq_rebuild_llm_check_live` (one live check per attempt) and
   `uq_rebuild_llm_check_slot` (each slot once). `rebuild_llm_calls.shot_id`
   is a soft reference (a re-analysis replaces shots; the call record of a
   possibly billed request stays).
7. **Brand-card keep rule**: a group is kept as original footage when its
   ANCHOR shot `is_brand_card`.
8. **§14 "the late accepted adopting an unknown row"** is not implemented: it
   contradicts §7.2 (an `unknown` row is final because `unknown` is declared
   only once no sender can still answer).
9. **`reserve_computed` does not ping the autopilot** (`note_reservation`):
   rebuild projects have no autopilot (§1 non-goals).
10. **The fit** pads (`tpad`) BEFORE it trims and encodes with
    `-fps_mode cfr -r <fps>`: with ffmpeg 7.1, `trim,tpad` lost the padding and
    the mp4 default (vfr) dropped a cloned tail frame (verified in the image).
11. **`delete_run`** refuses while ANY attempt of the run, or any character
    check of its attempts, is live (the design named `claimed`/`sending` and
    relied on the active-run guard for the rest; Terra round 3 added the checks).
12. **create_run on a rebuild project** answers 409 "not available yet" with
    the flag on, until wave 2b replaces it with the launch.
13. **Re-analysis** deletes the old generation's project keyframes AND
    character sheets (sheets have no shot to cascade through).
14. **The run estimator also counts live attempts** (Terra code gate round 1):
    a `queued`/`claimed` attempt counts its `estimate_credits` + its check
    allowance, a sent one its check allowance only — §7.3 counted only
    `pending` segments without a live attempt, which let a Retry/rerun's
    `release_unused` release an admitted attempt's undrawn money.
15. **A rebuild analysis job runs only from `created` or `analyzing`**
    (Terra round 1): a stale/duplicate job finding `ready`/`failed` does
    nothing, so only an admission (after the live-row guard) can put a
    rebuild project into `analyzing`; each publish needs `analyzing` and
    ends `ready`, so two queued jobs publish at most once per admission.
    Round 2 tightened it: the job reads the status under the project row lock
    and records the shot generation it starts from; it publishes, and on an
    error marks the project `failed`, only while the project is still
    `analyzing` at that same generation (nobody published meanwhile).
    Round 3 added a per-admission fence: `video_projects.rebuild_analysis_token`
    (new nullable column — §4 said "no new column" on `video_projects`) is set by
    the re-analysis admission (or by the first job leaving `created`); a job
    publishes / marks failed only while the token is still the one it started
    under, so a job reaped by recovery cannot touch the next admission's
    analysis. Round 4: the token travels in the job's arguments
    (`enqueue_analyze_project(pid, analysis_token=...)` ->
    `run_analyze_project` -> `analyze_project(..., analysis_token=...)`), so
    an old queued job can never adopt a newer admission's token; a rebuild
    job without a token runs only from `created` (the first analysis).
    Recovery re-checks a rebuild project under the row lock before resetting
    it to `failed` (its liveness snapshot predates its write). Residual
    (Terra round 4, P2, accepted): that snapshot can still predate a
    just-enqueued re-analysis job, which recovery then marks `failed`; the
    job's publish is then refused (no wrong publish, no lost money) and the
    operator re-analyzes — the same exposure every project type has today.
16. **A provider `fail` with no reported `creditsConsumed`** leaves the draw at
    its estimate (Terra round 2); §7.2 settled it "with creditsConsumed".

## 4. Tests

- `tests/test_rebuild_helpers.py` — pure helpers (folding cases from §5.1.3,
  chunking, brand-card keeps, §6.2 pieces, prompts for both models, duration,
  cumulative frames over 94.9 s at several fps, fit decisions and a real
  ffmpeg frame count, cut-timing QA on the pilot's numbers, end card).
- `tests/test_rebuild_core.py` — SQLite: payloads, `create_task_once` /
  `chat_completion_once` outcome matrices (one request each), the structural
  test (any `app/rebuild*.py` / `app/*rebuild*.py` referencing a retrying
  entry point fails — wave 2 modules are covered by their names), v2v
  byte-identity against `tests/fixtures/rebuild_v2v_payloads_main.json`
  (generated from main `a0a182f`), spend, the paid-call machine + reaper,
  API gates and guards, analysis, migration vs ORM, drain check, disk cleanup.
- `tests/test_rebuild_pg.py` — Postgres (`RESKIN_TEST_PG_URL`): full Alembic
  chain + `upgrade head` + ORM-vs-schema compare + downgrade refused/allowed,
  live-attempt and selected-keyframe uniqueness under concurrency, the scope
  lock (with a no-lock control that reproduces the round-9 race), and the
  round-8 `reserve_computed` interleavings.

Still to rehearse before a deploy: `upgrade head` on a copy of the PRODUCTION
schema (wave 1 rehearsed the full chain on a fresh database only), and the
§7.6 rollout sequence + `scripts/rebuild_quarantine.py` (not written yet).

## 5. Code gate record (GPT-6.1 Terra, Codex, reasoning high, read-only, `main...HEAD`)

| Round | Result | Fixed in |
|---|---|---|
| 1 | 2 × P1: the run estimator dropped undrawn live attempts (a Retry's `release_unused` could release them); a stale analysis job could flip a `ready` rebuild project to `analyzing` itself | `588812d` (§3 items 14, 15) |
| 2 | 2 × P1 + P2: the job's `created → analyzing` read was unlocked/stale; a failed task with no reported cost was settled to 0; the error path marked `failed` from a cached status | `652dc91` (generation fence under the lock; §3 item 16) |
| 3 | 2 × P1 + P2: `delete_run` ignored live character checks; the generation cannot tell a retry from a reaped job; recovery wrote `failed` without re-checking | `88ac994` (analysis token, check guard, locked recovery re-check) |
| 4 | 2 × P2: the job adopted the current token instead of its admission's; recovery's liveness snapshot can predate a just-enqueued job | `6afebae` (token in job args); the second accepted as a residual (§3 item 15) |
| 5 | P0: `Queue.enqueue(..., project_id, kwargs=...)` is rejected by rq 1.15 — every analysis enqueue would fail (the API tests mock the enqueue) | `9c6526d` (plain keyword + real-rq tests) |
| 6 | NO FINDINGS | — |

## Wave 2a (branch `feat/rebuild-2a-keyframes`, from `b16a062`)

Storyboard + keyframe jobs, their step handlers, the project-phase API and the
project page. No schema change (migration `t4d9f2b7c315` unchanged). Nothing
deployed; no provider was called (every test fakes kie.ai and counts POSTs).

### What exists (wave 2a)

| File | Content |
|---|---|
| `app/rebuild_jobs.py` | Shared plumbing: `session_factory()` / `kie_client()` (patched in tests); `send_claimed_row` (the §2.1 send step: fenced `begin_send` + COMMIT, ONE `create_task_once` / `chat_completion_once` inside the send window, `record_send_outcome`); `fail_unsent` (a `claimed` row whose free preparation failed → `failed` "not sent: …", cost 0); `store_prepared_request`; `frame_path` / `extract_frame`; `build_llm_proxy` (ladder below); `upload`; `aspect_ratio_for`. |
| `app/rebuild_storyboard.py` | Storyboard admission (`admit_storyboard`: Resume / From scratch, budget, reserve), the advance chain (`run_storyboard_job` → `advance` → `_finish`), the per-call step handler `llm_step` (kinds `global` / `shot`), prompts + tolerant parse + validation, `cancel_storyboard`, the storyboard `reap`, reads (`applied_storyboard`, `latest_storyboard`, `live_storyboard`, `described_shots`, `speech_warning`). |
| `app/rebuild_keyframes.py` | Keyframe/sheet admission (`admit_keyframes`), `select_keyframe`, `_auto_select`, the step handler `step` (claim → send → poll → fetch), prompts, sheet input-frame pick. |
| `app/rebuild_llm_steps.py` | `STEP_HANDLERS["rebuild_llm_calls"]` = `handle`, a dispatcher by `kind` through `KIND_HANDLERS` (dotted paths). Unknown kind → ERROR log, nothing done. |
| `app/api_rebuild.py` | `router` (`/api/v2`) + `web_router` (`/v2/projects/{pid}/rebuild-fragment`), `rebuild_context` (everything the card renders: chunks, preconditions, estimates). |
| `templates/v2/partials/project_rebuild_card.html`, `project_rebuild_attempt.html` | The project page body (§10): chunk strip, storyboard card, sheets, shots grid by chunk, launch form. |

### Shared-file edits

- `app/rebuild_paid_call.py`: `STEP_HANDLERS["rebuild_keyframes"] = "app.rebuild_keyframes.step"`, `["rebuild_llm_calls"] = "app.rebuild_llm_steps.handle"` (`rebuild_chunk_attempts` stays None for 2b); `STEP_JOB_TIMEOUT_OVERRIDES` + `step_job_timeout(table, step)` (LLM send jobs get 420 s; `enqueue_step` uses it).
- `app/tasks.py`: `run_reconcile_tick` also calls `rebuild_storyboard.reap()`.
- `app/main.py`: registers `api_rebuild.router` (`/api/v2`) and `web_router` (`/v2`).
- `templates/v2/partials/project_status_content.html`: a rebuild project's ready branch loads `#rebuild-content` (hx `load`) above the Runs panel, and the v2v New Run `<details>` is rendered `hidden` (kept in the DOM: its scripts run at load).
- `tests/test_rebuild_core.py`: the "no handler deployed" reaper test now sets the keyframe handler to None itself.

### How the storyboard runs

- One `project_storyboards` row; a chain of short **advance** jobs on `process`
  (`rb-sb:<sb id>:<gen>:<seq>`, liveness by that prefix). The first job claims
  `queued → running` (token); each continuation is fenced by (`running`, gen,
  token) and re-schedules itself every 5 s while calls are in flight. An advance
  takes the project row lock, then the storyboard row lock, admits call rows and
  observes them; it never calls a provider.
- Every Gemini call is a `rebuild_llm_calls` row driven by its own step jobs
  (claim = proxy + frame + uploads, free; send = ONE `chat_completion_once`).
  At most 8 shot calls live at once; the global call first.
- Per call and storyboard at most 2 rows: the chain's primary model, then ONE
  more row with the next model after any refusal (`file_unavailable` included)
  or an invalid answer. `unknown` is never re-called automatically ("may have
  been billed"; the next Build calls it again).
- Budget: `budget_credits` = ((0 when Resume reuses the global call, else 1) +
  shots needing a call) × 2 × `llm_credits()`, reserved with `reserve_computed`
  under `storyboard_scope(id)`. A row is admitted only while committed (every
  row that drew or will draw; refused / not-sent / cancelled rows count 0) + one
  call ≤ budget; the rest get "budget for this storyboard used up". When the
  storyboard ends (finished or interrupted) the reservation is shrunk to what
  its calls drew (`release_unused`); a call that ends later re-releases, and
  the storyboard reaper sweeps any ended (or deleted) storyboard still
  over-reserved.
- Finish: `succeeded` (and applied) unless the global call failed or > 10 % of
  the unfolded shots stayed undescribed. A failed storyboard applies nothing;
  Resume reuses its valid answers.
- Interruption: the operator's Cancel (`POST …/storyboard/cancel`) and the
  reaper (advance chain not live for 120 s while `running`) set `interrupted`
  and call `cancel_unsent(RebuildLlmCall, storyboard_id == id)` in the same
  transaction; a `queued` storyboard whose first job is lost is re-enqueued
  with a new `dispatch_gen`.

### Deviations from the design (and the doc changes made for them)

1. **Fallback trigger** is any refusal or invalid answer, not only
   `file_unavailable` (§5.2 named both the file error and "a retry after an
   invalid answer"; one rule covers both). Still one fallback row per call.
2. **Budget** counts the global call only when it is not reused (§5.2 wrote
   `(1 + N) × 2`), and only the shots that still need a call.
3. **A failed storyboard applies nothing** (§5.2 left partial application open).
4. **Frame files** are `rebuild/frames/<shot id>_<ms>.jpg`
   (`rebuild_jobs.frame_path`), not `storage.rebuild_frame_path`'s
   `<shot id>.jpg`: an operator may move `frame_sec` while an older row still
   names the old frame. `input_paths` holds `[{shot_id, t, path}]`.
5. **Manual pick without a column**: a pick stamps `updated_at` after the
   row's `finished_at`; an automatic selection writes both with one instant.
   Auto-select keeps a manual pick and a newer attempt.
6. **Keyframe model pinned** to `gpt-image-2-5-flare-image-to-image` (not
   `settings.KIE_IMAGE_MODEL`): the 2K 9:16/16:9 request depends on 2.5.
7. **LLM timeouts**: HTTP 300 s (global) / 240 s (shot), send job 420 s
   (`STEP_JOB_TIMEOUT_OVERRIDES`), instead of `SEND_HTTP_TIMEOUT_SEC` = 60 s —
   a video call answers in minutes; a timeout would mean `unknown`.
8. **Preparation failure** (ffmpeg/upload in the claim step) ends the row
   `failed` "not sent: …" with cost 0 (nothing was sent or drawn).
9. **Keyframe polling gives up after 1 h** (`POLL_MAX_SEC`): `failed`, draw
   left at the estimate, task id kept.
10. **Flag off**: every mutating route 409s; reads, images and the (read-only)
    fragment stay, so existing rows remain visible (§7.6).
11. **Keyframe admission is refused while a storyboard is live**; re-chunking
    (unfold, brand cards) is refused while a storyboard is live, a run is
    active or a chunk attempt is live.
12. **API shapes**: forms, not JSON bodies (HTMX/fetch friendly).
    `POST …/rebuild/keyframes`: `shots` = `all_missing` | comma ids (one id +
    `prompt` = regenerate), `sheets` = `all_missing` | comma keys, `prompt`,
    `sheet_prompt`, `sheet_inputs` (JSON `{key: [shot ids]}`). Unfold is
    `PATCH /rebuild/shots/{sid}` with `force_anchor=true` (no separate route);
    a re-chunk keeps the brand-card keeps when any keep segment exists.
    Added: `POST …/storyboard/cancel`, `GET /rebuild/keyframes/{kid}/image`,
    `GET /rebuild/shots/{sid}/frame`.
13. **LLM proxy ladder**: (854 px long side, CRF 28, 15 fps, 48k mono AAC) →
    (854, 34, 15, 32k) → (640, 38, 12, 24k) → (480, 42, 10, 16k), stopping at
    ≤ 1 MB (the last rung is kept whatever its size; the fallback model
    accepts bigger files). Upload names are alphanumeric (`sb<hex>.mp4`).
14. **`_lock_project` re-reads the row** (`populate_existing`): found by the
    Postgres admission-vs-re-analysis test — a waiting admission otherwise read
    the `ready` it had loaded before the lock.

### What wave 2b / the integration must know

- **Launch form** (`#rb-launch-form`) posts multipart to the existing
  `POST /api/v2/projects/{pid}/runs`:
  `prompt` = `"Rebuild from reference (prompts are rendered per chunk)"`
  (create_run requires the field; ignore it for images-only), `model`
  (`kling-4-0` default | `seedance-2-5`), `resolution` (720p default; options
  from `IMAGES_ONLY_CREDITS_PER_SEC` ∩ the model's resolutions), `language`
  (code, default `en`), `audio_mode` = `original`, `name` (optional, omitted
  when blank), `stitch_plan` (JSON, ONLY when the operator chose "Keep the
  source end card" — every segment in order — or a library clip — segments
  minus the trailing end card + `{"kind": "library", "clip_id"}`; omitted =
  the server's §6.1 default incl. the automatic language-matched end card),
  `chunk_sheets` (JSON `{segment_def_id: sheet_key}`, ONLY the chunks whose
  sheet the operator changed from the auto pick; omitted when none). On 201
  with `run_id` it navigates to `/v2/runs/{run_id}`; any other answer shows
  `detail` (today: the 409 "not available yet").
- The page computes the §6.1 precondition list in `api_rebuild.rebuild_context`
  (`unmet`); the server must still enforce it. Helpers:
  `rebuild_storyboard.applied_storyboard(db, pid)`,
  `rebuild_keyframes.selected_by_slot(rows)` / `live_slots(rows)` /
  `project_keyframes(db, pid)`, `rebuild_storyboard.live_storyboard`.
- **Character check**: add `"character_check": "app.<module>.<func>"` to
  `rebuild_llm_steps.KIND_HANDLERS`. `rebuild_jobs.send_claimed_row` /
  `fail_unsent` / `store_prepared_request` / `upload` are reusable; a check's
  scope is the run scope (pass `scope_key_fn`).
- Per-table step timeouts: `rebuild_paid_call.STEP_JOB_TIMEOUT_OVERRIDES`
  (a chunk fetch + fit may want more than 1800 s).
- The paid-call reaper now re-drives keyframe and LLM rows (handlers
  registered); `rebuild_chunk_attempts` is still handler-less.
- `rebuild_storyboard._lock_project` re-reads the project after the lock;
  any new admission that first `db.get`s the project must do the same
  (`db.refresh(project, with_for_update=True)` also re-reads).

### Tests (wave 2a)

- `tests/test_rebuild_2a.py` (SQLite, 46 tests): storyboard happy path with one
  POST per row, `file_unavailable` / invalid-answer fallback as a new row,
  `unknown` never re-sent, budget stop, cancel + `cancel_unsent`, Resume / From
  scratch, Build-again waits for `sending`, edited shots kept / overwritten,
  speech warning, validation, the storyboard reaper (requeue, interrupt, live
  chain left alone, reservation sweep incl. a deleted storyboard), single send
  across a reaper re-queue + stale and duplicate send deliveries, `sending`
  waits for a proof, no boot id → no send, unknown LLM kind, handler registry,
  keyframes (missing + sheets, midpoint, 2K 9:16, creditsConsumed settle,
  regenerate, selection rules, folded shot refused, download-only retry,
  provider fail, poll timeout), 429s, `ready` required on every admission, the
  flag gate, non-rebuild 404, re-analysis refused while a storyboard is live,
  shot edit / frame time, unfold + brand cards (refused while a run is active),
  reservation release (finish, cancel, late refusal), the rendered fragment in
  every state (no storyboard / running / done with warnings / keyframes partial
  / all selected / flag off) and the full project page, and the real ffmpeg
  proxy (480×854, ≤ 15 fps, ≤ 1 MB).
- `tests/test_rebuild_pg.py` (+3 on Postgres): concurrent storyboard admissions
  admit exactly one; admission vs re-analysis in both orders (the lock-wait
  order found deviation 14); manual select racing auto-select leaves one.
- Rebuild suites: 211 passed, 1 skipped (Postgres on). Full suite in the image
  (`-n 3`): 23 failed / 5386 passed / 352 skipped — the 23 are main's known
  environment failures (kie_client ×2, knowledge_unit ×2, pipeline_provider ×3,
  range_attempt ×1, pipeline_v2 missing_project ×1, reframe ×1, tasks ×1,
  assistant_* ×4, backup_db_script ×4, gdrive_run_result ×4).

### Code gate record (GPT-6.1 Terra, Codex, reasoning high, read-only, `b16a062...HEAD`; rounds 2+ review the delta only)

| Round | Result | Fixed in |
|---|---|---|
| pre-gate | The Postgres admission-vs-re-analysis test caught a stale `ready` read after the project lock (identity map) | `6f505d2` (deviation 14) |
| 1 | 3 × MAJOR + 1 MINOR: unused storyboard reservations never released (the 2× fallback allowance stayed charged all day); Resume ignored an answered global call an interrupted build never copied; Resume reached past a failed From-scratch build to an older global context; keyframe poll exceptions bypassed the 1 h timeout | `969ff86` (`release_unused` at finish / interrupt, `_finish` cancels its own unsent rows; `storyboard_global`, newest storyboard only; poll errors fall through to the timeout) |
| 2 | MAJOR: a call still `sending` at the interrupt keeps its estimate reserved after it settles lower (a refusal settles to 0) | `4198926` (`_release_if_ended` after the outcome) |
| 3 | MAJOR: that release is a separate transaction — a worker dying between the two commits leaves the estimate charged | `df8ebdf` (`sweep_unreleased` in the storyboard reaper) |
| 4 | MAJOR: the sweep skipped the scope of a storyboard deleted with its project | `47df2f7` (release the orphaned scope directly) |
| 5 | NO FINDINGS | — |

Residual (not a finding): keyframe batches keep `spend.reserve`'s plain
reservation semantics (as transcription does) — a refused or cheaper attempt
leaves its 10 cr reserved until the UTC day ends. Follow-up if it matters:
release a batch scope when its last row is terminal.

## Wave 2b — the images-only run path (branch `feat/rebuild-2b-run`, off `b16a062`)

### What exists

| Area | Where | Notes |
|---|---|---|
| Launch (§6.1) | `app/rebuild_launch.py` (`launch_run`, `LaunchRequest`, `LaunchRefused`), called by `api_v2._create_images_only_run` from `create_run` | Project row lock + `ready`; no live storyboard / keyframe / storyboard-LLM row; storyboard applied to the current shot generation (`status='succeeded'`, `applied_at` set, `shots_generation` = max shot generation); per ACTIVE swap chunk: pieces (`chunk_shot_pieces`), a selected succeeded keyframe per anchor with its file on disk, the chunk's sheet (`sheet_for_chunk`, or the operator's override; when no shot names a group and exactly one sheet is selected, that one), image count (`check_chunk_image_count`) and duration ceiling (`request_sec_for_chunk`) — every miss is listed in a 409 `{"message", "missing": [...]}`. Then the Run (`images_only`, `audio_mode='original'`, no preprocessing / references / voice, `prompt=''`), `segment_snapshot` NOW, run-owned copies `runs/<rid>/rebuild/<chunk idx>/<n>.png` + `sheet.png` (hard link, else copy; sha256 in the snapshot), `rebuild_snapshot` (below), one `pending` RunSegment per active chunk, `queued`, `spend.reserve_images_only_run` LAST. A failed launch removes the run dir. |
| Snapshot | `runs.rebuild_snapshot` | `{version: 1, model, resolution, aspect_ratio, fps_out, storyboard_id, shots_generation, style, end_card_ids, chunks: [{segment_def_id, index, start_sec, end_sec, request_sec, chunk_frames, prompt, shots: [{shot_id, rel_start, rel_end, framing, camera, action, characters, keyframe_id, keyframe_path, sha256}], images: [paths, sheet last], sheet_key, sheet_keyframe_id, sheet_path, sheet_sha256, characters: [{id, name_hint, look, outfit}]}]}`. `fps_out` = `media.get_default_target(probe(source))[2]` at launch (the stitch's grid; fallback `project.fps`). |
| End card (§6.1, Q2) | `rebuild_launch.default_end_card_plan` | With no `stitch_plan` from the form and no `keep_source_end_card`: the trailing all-brand-card keep run is replaced by the newest `LibraryClip` whose `language` equals the run's (`en` when unset; language-agnostic clips are never picked). No such clip → default plan + a warning. Every other source `keep` gets a "keeps its burned text" warning when the storyboard's `on_screen_text` is not empty. |
| Run chain (§6.4) | `tasks.run_submit_run` → `tasks._submit_images_only_run` → `pipeline_rebuild.submit_images_only`; `tasks.run_rebuild_observe` → `pipeline_rebuild.observe_run`; then the existing `run_finalize_run` | The branch sits right after the lock is acquired / refreshed. Submit: `queued → processing` (accepts `processing`), repairs a missing RunSegment, admits one `RebuildChunkAttempt` (`queued`, `attempt_no` = max + 1, `request_sec` / `chunk_frames` from the snapshot, `estimate_credits` = `images_only_video_credits`) per active chunk that is `pending` with no live attempt, inside a SAVEPOINT (the live index refuses a racing duplicate), segment → `uploading`; claim steps enqueued after the commit. The observer job (run id as its first argument, so recovery sees the run live) refreshes the lock, then: all chunks completed → `run_finalize_run` with the lock; a chunk failed and nothing live → `incomplete`, lock released; else re-schedules itself every `RUN_POLL_INTERVAL_SEC`. |
| Attempt steps (§7.2) | `pipeline_rebuild.chunk_attempt_step` = `STEP_HANDLERS["rebuild_chunk_attempts"]` | claim: CAS + sha256 check of the run-owned copies (a missing / altered file → `failed` from `claimed`, nothing sent, no draw). send: free uploads, `build_images_only_payload` (pilot shapes), `sender_identity`, `begin_send` (draw at the estimate in `run:<rid>`), commit, ONE `create_task_once` unless the send window closed, `record_send_outcome`; segment → `submitted` / `failed` in the same transaction. poll: `get_task` (idempotent), `record_poll`; segment → `generating` / `failed`; gives up 6 h after the send (`failed`, draw left at the estimate). fetch: `get_task` for the URL, `download_result` (`.part` + replace) to `raw.mp4`, `fit_attempt` → `fitted.mp4` (frame count verified with ffprobe), then in one transaction: the previous selected attempt unselected, `complete_fetch` (selected, `qa.duration`), segment `completed` + `local_result_path` + `cost_credits`; a too-short take is `fail_fetch(final=True)`; a download error re-schedules the fetch only. |
| Fit (§6.4) | `pipeline_rebuild.fit_attempt` | `raw_frames = floor(duration x fps_out)`, `rebuild.fit_plan`, `rebuild.fit_ffmpeg_args` into a `.part.mp4`, exact frame count checked, `os.replace`. |
| Stitch (§6.6) | `pipeline_v2._run_stitch_and_deliver` (2 edits) | An images-only run's keep segments are cut frame-exact (`pipeline_rebuild.cut_keep_frames`, `rebuild.keep_cut_ffmpeg_args`, `clips/rbkeep_<idx>_<F0>_<F1>.mp4`, video only); under a custom plan they get their source audio muxed like swap clips (`mux_original_audio` copies a silent source unchanged). Everything else — the gate, original-audio stitch, delivery in place, Job ID, Final Creatives (`Pipeline Type` re-skin-v2, `Idea` tag `KL4`) — is the existing code. |
| QA (§6.5) | `pipeline_rebuild.run_attempt_qa` (RQ, `autopilot_compute`, job id `rbqa:<attempt>`), `free_qa`, `qa_flags` | Free part: cut timing (`detect_scenes` at `REBUILD_SHOT_MIN_SEC` on the fitted clip vs `planned_cuts`), burned text (`qa_tools.locate_captions`, flagged when a band is found), duration (written by the fetch), contact sheet `attempts/<aid>/sheet.jpg` (one row per planned shot: midpoint frame | keyframe). Each part independent; errors in `qa.errors`. Then the automatic character check. `qa.flags` is recomputed on every write. |
| Character check (Q9) | `pipeline_rebuild.character_check_step` via `app/rebuild_llm_steps.py: KIND_HANDLERS["character_check"]` | `RebuildLlmCall(kind="character_check", slot 0)`, model = `LOCALISATION_TRANSCRIBE_MODEL`, slot 1 = the first fallback. claim → send (upload the contact sheet, ONE `chat_completion_once`, draw in the run scope). Success → `qa.character_check = {shots: [{shot, present, missing, identity_drift, notes}], flagged, call_id}`; a non-success admits slot 1 once, then `"unavailable"`. Operator's "Run character check" = `POST /runs/{rid}/rebuild/attempts/{aid}/character-check` (slot >= 2, reserves `llm_credits()` with `reserve_computed`). |
| QA reaper | `pipeline_rebuild.reap_qa` in `tasks.run_reconcile_tick` | Succeeded attempts of the last 7 days: re-enqueue a lost free QA (only the free part), admit a never-admitted slot-0 check, fold a check whose last call the paid-call reaper released (`sending → unknown`) — fallback once, else `unavailable`. |
| Repair (§6.6, §7.3) | `api_v2.retry_run` → `_retry_images_only_run`; `rerun_segment` / `rerun_segments` → `_rerun_images_only`; `api_rebuild_run.use_attempt` | Retry from failed/incomplete: `pipeline_rebuild.reset_for_retry` (every active chunk not completed-with-file → `pending`), `queued`, `reserve_run(release_unused)` (→ the images-only estimator under the scope lock), one transaction under the Run-row lock; an orphan in an active status is resumed without a reset or reservation. Re-roll: `reset_for_rerun` (→ `pending`, attempt unselected, `local_result_path` cleared, optional prompt → `prompt_override`, which the next request uses) + `_invalidate_final_and_requeue`. Use attempt N: `pipeline_rebuild.use_attempt` + `_invalidate_final_and_requeue` (free; re-delivers in place). All refuse (409) while any chunk attempt of the run is live; re-roll / Retry from failed-incomplete / manual check refuse while `REBUILD_ENABLED` is off. |
| Refusals (§6.6) | `api_v2._refuse_images_only_run` | 409 for: approve / regenerate preprocessing (both), delete a preprocess ref, `PATCH /runs/{rid}/segments/{rsid}` (whole), take restore, range rerun, segment GPT Image regeneration + apply, Copy run, copy-batch, enabling an automation; segment caption fill is already refused (subtitle_removal only). A re-roll with reference images is a 400. `web_v2._copies_enabled` hides Copy / Automation. |
| Run page (§10) | `app/rebuild_run_view.py`, `templates/v2/partials/run_rebuild_chunks.html` (included in `run_status_content.html` for active AND terminal states; the v2v per-segment panel is hidden), `web_v2._rebuild_run_context` | Per chunk: status, times, request vs chunk length, shots, sheet, error, QA flags of the selected attempt, attempts table (status incl. "waiting for the sending worker" / "may have billed", cost or ~estimate, raw / fitted / contact-sheet links, cut timing, padding, text band, check state, "Use this attempt"), the prompt, the re-roll box with its cost (video + 2 x LLM). Flag off: a banner, no re-roll. |
| Files | `api_rebuild_run.download_attempt_file` | `GET /runs/{rid}/rebuild/attempts/{aid}/{raw|fitted|sheet}` (the row's recorded paths). |

### Launch fields accepted by `POST /api/v2/projects/{pid}/runs` on a rebuild project

Matches the 2a launch form (`#rb-launch-form`): `prompt` (REQUIRED by the form signature,
ignored — the run stores `''`), `model` (`kling-4-0` | `seedance-2-5`), `resolution` (not
sent → `720p`; Kling 720p/1080p, Seedance 480p/720p/1080p), `language` (`en` = English),
`audio_mode` (only `original`), `name`, `gdrive_folder_id`, `stitch_plan` (as for any run;
ABSENT → the §6.1 default with the end-card rule; the form's explicit every-segment plan for
"keep source end card" normalises to the default plan and keeps the card), `chunk_sheets`
(JSON `{segment_def_id: sheet_key}`, only the overridden chunks; `sheet_overrides` is an
alias, not both), `keep_source_end_card` (bool, API convenience; only without a plan).
Refused with 400: reference files/URLs, `segment_prompts`, preprocessing fields, voice
fields. `chunk_sheets` / `keep_source_end_card` on a v2v project: 400. A refusal's `detail`
is always a STRING (the missing preconditions are joined into it), which the form shows as
is. 201 → `{run_id, status, warnings}` (end-card and burned-text advisories).

### For the integration

- `app/rebuild_llm_steps.py` holds only `"character_check": "app.pipeline_rebuild.character_check_step"`
  (signature `(row_id, gen, step, job_id)`; steps `claim` and `send` — an LLM call has no
  poll / fetch); 2a's dispatcher must route `kind == "character_check"` to it. Until it is
  registered, check rows only advance through the steps 2b enqueues itself.
- `rebuild_launch.applied_storyboard` / `_live_project_rows` / `_selected_keyframes`
  duplicate 2a's `rebuild_storyboard.applied_storyboard` / `live_storyboard` and
  `rebuild_keyframes.selected_by_slot` / `live_slots` (same semantics; 2a's generation is
  `rebuild_analysis.current_generation`, 2b's the max shot generation — equal unless a
  storyboard describes a generation with no shots). The integration may swap them.
- `pipeline_rebuild.character_check_step`'s send (upload, `begin_send`, ONE
  `chat_completion_once`, `record_send_outcome`) and `_fail_unsent` duplicate 2a's
  `rebuild_jobs.send_claimed_row` / `fail_unsent` / `upload`; dedupe at integration.
- `tasks.run_reconcile_tick` gains `pipeline_rebuild.reap_qa()` right after
  `rebuild_paid_call.reap()` (2a adds `rebuild_storyboard.reap()` there too).
- `STEP_HANDLERS["rebuild_chunk_attempts"]` is the only `rebuild_paid_call.py` edit; the
  chunk fetch + fit fits the existing 1800 s fetch timeout (no `STEP_JOB_TIMEOUT_OVERRIDES`).
- `main.py`: one router (`api_rebuild_run`). Launch locks with
  `db.refresh(project, with_for_update=True)` before reading `ready` (2a's lesson).

### Deviations (wave 2b)

1. **Observer write**: a non-completed chunk with no live attempt is marked `failed` by the
   observer (its attempt was released by the paid-call reaper, which never touches segments).
2. **Uploads in the send step**, right before the `sending` CAS; the claim step checks the
   run-owned copies' sha256. A deterministic input error found before the send ends the
   attempt `failed` from `claimed` (nothing sent, no draw).
3. **Poll give-up**: 6 h after the send a still-unfinished task is `failed` with the draw at
   the estimate (the design had no bound).
4. **Character-check fallback** also on a primary that ended `unknown` (the allowance the
   estimator reserves covers it); a manual check (slot >= 2) is reserved on top and is not
   in the estimator, so a later `release_unused` may release its undrawn estimate (its draw
   still lands; the books only lose the headroom).
5. **Re-roll of a completed chunk** unselects its attempt and clears `local_result_path`
   (design: the older attempt "may still be selected"); "use attempt N" restores it.
6. **`create_run`'s `resolution`** form default is now `None` (v2v falls back to
   `settings.DEFAULT_RESOLUTION` at call time — same result; images-only to 720p, Q1). The
   autopilot calls `create_run` as a plain function (`app/autopilot/actions.py`), so the
   new fields (`keep_source_end_card`, `sheet_overrides`) and `resolution` are normalised
   at the top: anything that is not a real value reads as "not sent" (found by the suite
   comparison against `b16a062`; regression test `test_create_run_called_directly_like_the_autopilot_still_works`).
7. **Flag off** refuses re-roll / Retry-from-failed-incomplete / manual check; "use attempt N"
   (free) and an orphan Retry stay allowed.
8. **The whole segment PATCH** is refused for images-only runs (the prompt edit travels with
   the re-roll).
9. **Launch's live-row check** counts storyboard LLM calls only (another run's character
   checks do not block a launch).
10. **End-card clip choice**: exact language match, newest first; language-agnostic clips are
    never auto-picked.
11. **Keep clips under a custom plan** get their source audio muxed (the frame-exact keep cut
    is video-only).
12. **Attempt files are named per fetch execution** — `attempts/<aid>/raw-g<gen>-<rand>.mp4`
    and `fitted-g<gen>-<rand>.mp4` (design §4: `raw.mp4` / `fitted.mp4`); the row records the
    winner's paths with the fenced `succeeded` CAS and a fenced-out execution removes its own
    files (Terra round 1). A too-short take keeps its `raw_path` on the failed row.
13. **A character-check answer with no JSON object** counts as a failed check (fallback once,
    then `unavailable`), never as a clean one.

### Tests (wave 2b)

- `tests/test_rebuild_run.py` (SQLite, real ffmpeg on synthetic clips, 31 tests): launch
  (snapshot, copies + sha, segments, reservation, Q1 default, end-card rule incl. a middle
  keep and no matching clip, refusals writing nothing, 429 leaving no files); submit
  idempotence; ONE POST per attempt across a reaper re-drive, a stale generation, a
  duplicate delivery and a recovery re-entry; the request equals the pilot shape through
  the real `KieClient.create_task_once` (respx) for Kling and Seedance; refused / unknown
  terminal, never re-sent; altered run-owned keyframe fails unsent; the fit (exact 120
  frames, 2 padded, too short, 29.97 fps grid); fetch → selected + completed + settled;
  observer incl. a reaper-made `unknown`; `run_submit_run` branch + observer lock handover
  on a real rq queue (fakeredis); finalize with the original soundtrack (default plan 180
  frames, library end card 150 frames, silent source); the v2v keep cut untouched; free QA
  (cut timing, burned text, contact sheet) and flags; character check single send,
  fallback, unavailable; QA reaper (re-enqueue only the free part, fold a released check);
  Retry / re-roll at the budget limit (429 changes nothing) and below it / use-attempt /
  refusals through the endpoints; attempt files; the run page in pending / generating /
  one chunk failed / done with flags / flag off, and no card on a v2v run.
- `tests/test_rebuild_run_pg.py` (Postgres, 3 tests): six concurrent `submit_images_only`
  entries admit one attempt per chunk; the launch + a Retry's reset and re-baselined
  reservation; two racing Retry admissions count once.

### Code gate record — wave 2b (GPT-6.1 Terra, Codex, reasoning high, read-only)

| Round | Range | Result | Fixed in |
|---|---|---|---|
| 1 | `b16a062..457e3a5` (full wave) | MAJOR: two fetch executions (a stale one overlapping a reaper re-drive) wrote the same `raw.mp4` / `fitted.mp4` (and the same `.part`) before the fence, so the stale one could overwrite or corrupt the selected file. MINOR: a non-JSON check answer was stored as a clean check. MINOR: no "Run character check" control. | `35dc835` (per-execution file names, fenced-out execution removes its files; deviations 12–13; the button + endpoint test) |
| 2 | `457e3a5..8947f2b` (delta) | 2 × MINOR: a losing concurrent fold (handler + QA reaper) could write `unavailable` over the fallback; files of executions killed after their download were never cleaned. No BLOCKER/MAJOR. | `ae41596` (a fold that loses the slot race defers to the later slot; the fetch sweeps older generations' files) |
| 3 | `8947f2b..ae41596` (delta) | MINOR: a fenced-out execution killed between its (late) download and its fence check still leaves its file (the sweep ran earlier and the attempt is terminal). No BLOCKER/MAJOR — gate passed. | accepted residual: the files go with the run directory on `delete_run`; disk only, never a wrong result |
| 4 | `e4723af..27d4795` (delta) | The suite comparison against `b16a062` found the autopilot's direct `create_run` call broken by the two new Form fields (deviation 6); fixed, then reviewed (incl. a search for other direct callers of a changed endpoint): NO FINDINGS. | `27d4795` |
| 5 | `9116dae..ae7776e` (delta) | The 2a launch-form contract (`chunk_sheets`, string `detail`, the form's exact fields): NO FINDINGS. | — |

## Integration (branch `feat/rebuild-integration`, from main `730216b`)

Waves 1, 2a and 2b combined on current main, made one feature, proven end to
end. Nothing deployed, nothing pushed, no provider called (every test fakes
kie.ai / Drive and counts POSTs). Operator checklist:
`docs/rebuild-release-packet.md`.

### Merges and conflicts

`feat/rebuild-core` (`c76f259`), then `feat/rebuild-2a-keyframes` (`1d577bb`),
then `feat/rebuild-2b-run` (`b33c742`), merge commits. core and 2a merged
clean; 2b conflicted in four files, resolved so both sides survive:

- `tasks.run_reconcile_tick`: `rebuild_paid_call.reap()`, then
  `rebuild_storyboard.reap()`, then `pipeline_rebuild.reap_qa()`, each in its
  own try/except (the tick re-schedules itself in the last `finally`).
- `rebuild_paid_call.STEP_HANDLERS`: all three set — keyframes →
  `app.rebuild_keyframes.step`, chunk attempts →
  `app.pipeline_rebuild.chunk_attempt_step`, LLM calls →
  `app.rebuild_llm_steps.handle` (2a's dispatcher). `STEP_JOB_TIMEOUT_OVERRIDES`
  keeps 2a's 420 s LLM send job (the character check's 180 s HTTP timeout fits;
  asserted).
- `rebuild_llm_steps.py`: 2a's dispatcher + `KIND_HANDLERS` = `global` / `shot`
  → `app.rebuild_storyboard.llm_step`, `character_check` →
  `app.pipeline_rebuild.character_check_step`. The paid-call reaper re-drives a
  lost check row through it (`test_dispatcher_routes_a_check_and_the_reaper_redrives_a_lost_one`:
  lost claim job, then lost send job → ONE POST).
- `main.py` routers (auto-merged: `api_rebuild` router + web router,
  `api_rebuild_run` router); `project_status_content.html` (2a only) and
  `run_status_content.html` (main's AICP-348 card + 2b's chunk card, auto).
- The notes: 2a's and 2b's sections both kept; design doc header names all
  three branches and the integration.

### Migration re-chain

`t4d9f2b7c315` now revises `t4d9a2c6e815` (main's AICP-348
`run_drive_relinks`, applied on prod); docstring `Revises:`, the
`test_rebuild_core` down_revision assert, the Postgres rehearsal's
upgrade/downgrade targets, wave 1 §1 and the design baseline (`impl:`) follow.
`alembic heads` → `t4d9f2b7c315 (head)` (one head).

**Rehearsed on a copy of the production schema** (read-only `pg_dump
--schema-only` of `re-skin-db`, loaded into a disposable `postgres:16`,
version set to `t4d9a2c6e815`): upgrade head; ORM vs schema — 0 differences
anywhere; labels present; downgrade on the empty feature; re-upgrade;
downgrade refused with one rebuild project; `deploy/drain-check.sql` runs.

### Main since `a0a182f` (AICP-349 / AICP-348)

No schema overlap (`run_drive_relinks` vs the rebuild tables). Code:

- **AICP-349** (range-attempt promotion in place): only reachable through
  `create_range_rerun`, which 2b refuses for images-only runs (409, before the
  Run lock) — no other caller of `create_range_attempt` exists, so an
  images-only run can never get a range attempt or its promotion.
- **AICP-348** (Drive re-link): works on images-only runs unchanged — it
  re-uploads `result_local_path` (the same stitched final), takes the same
  Run-row lock, and requires a `done`/`failed` run with no range attempt /
  audio patch / fill / assistant intent in flight. A re-roll, "use attempt N"
  or Retry admitted first moves the run out of `done`, so the re-link worker
  refuses; one admitted after waits on the Run lock and then re-delivers in
  place behind the NEW id (`_invalidate_final_and_requeue` → delivery
  `files().update`). Images-only delivery is the existing
  `_run_stitch_and_deliver`, so in-place re-delivery (the E2E asserts the
  same Drive id across a re-roll and a Retry) is main's code.

### Dedupe

- **One launch-precondition source**: `rebuild_launch.load_inputs` +
  `preconditions` (`basic_missing` + `plan_chunks`). `create_run` enforces it;
  `api_rebuild.rebuild_context` shows it per images-only model
  (`unmet_by_model`; `unmet` = the default model's; the form shows the
  selected model's list and enables the button from `launch_ready_by_model`).
  A refusal is the same text joined into `detail`
  (`test_project_page_preconditions_are_the_servers` compares them in four
  states). A launch with no plan possible (no storyboard) still refuses with
  the FULL list.
- **Generation**: 2a's `rebuild_storyboard.applied_storyboard`
  (`rebuild_analysis.current_generation`, max over shots AND storyboards) —
  the analysis's definition of "current", and equal to 2b's max-shot
  generation whenever shots exist (generations only grow). 2b's copy removed.
- **Selected keyframes**: 2a's `rebuild_keyframes.selected_by_slot` over
  `project_keyframes`, plus the file-on-disk filter the launch needs for the
  copy (`rebuild_launch.selected_keyframes`). 2b's `_selected_keyframes` removed.
- **Live rows** (`rebuild_launch.live_rows`, design §6.1 "no keyframe or
  storyboard row live"): `queued`/`running` storyboards, live project
  keyframe/sheet rows, live storyboard LLM calls (`global`/`shot`, incl. one
  still `sending` after its storyboard ended). Another run's attempts and
  checks do not block. The page polls while any of them is live.
- **Sheet per chunk** (both sides now): the override, else the key held by
  most shots; a chunk where no shot names a group uses the first group (in
  storyboard order) with a selected sheet — 2b refused when 2+ sheets
  existed while the page called it ready.
- **End card ids**: computed over ANCHOR shots on both sides (the wave 1
  brand-card rule); the page calls `rebuild_launch.trailing_end_card_ids`.
- **Character check**: claim = CAS + free preparation (contact sheet upload,
  request) stored with `rebuild_jobs.store_prepared_request`; a missing input
  → `rebuild_jobs.fail_unsent`; send = `rebuild_jobs.send_claimed_row` (ONE
  `chat_completion_once`, drawn in the run scope). The chunk attempt's
  `_fail_unsent` also uses `rebuild_jobs.fail_unsent` (error prefixed "not
  sent:"); uploads go through `rebuild_jobs.upload(…, client=)`.
- **Upload names are unique per row** (found while deduping): every chunk's
  run-owned copies are named `1.png … sheet.png`, and 2b uploaded them into
  ONE folder per run; the kie.ai upload host OVERWRITES a same-named file
  (the 2026-09-25 captions bug), so two chunks sending at once could be sent
  each other's keyframes. Now `ca<attempt hex>n<i>.<ext>` / `cc<call hex>.jpg`
  (`rebuild_jobs.upload_name`).
- The structural no-retry test lists every module of the feature by name and
  fails if a file named like the feature escapes its glob.

### Other integration changes

- A v2v project page's New Run JS maps (`MODEL_RESOLUTIONS`, `MODEL_LIMITS`)
  list only the form's models, and the dashboard filters list the rebuild type
  / Kling only while `REBUILD_ENABLED`: a v2v project page, run page and the
  dashboard render as on main `730216b` (diffed with fixed rows; the only
  differences are blank lines and the file-mtime `?v=` cache-buster).
- Run page: Copy run / Automation explain why they are unavailable for a
  rebuild run (they showed the localisation text).
- Launch form: resolution options in registry order (the sorted JSON keys put
  1080p first; found by the browser test).

### Tests (integration)

- `tests/test_rebuild_e2e.py` (SQLite, real ffmpeg, POST-counting fake
  kie.ai, Drive stubbed at the client): the whole path — upload → real
  analysis (10 hard-cut shots) → storyboard (1 + 10 calls) → keep brand cards
  → 10 keyframes + 1 sheet → regenerate one + select the older back → launch
  with the page form's exact fields (Kling, then a Seedance run) → attempts →
  fit → free QA + automatic character check → stitch with the original audio
  + the library end card → delivery; re-roll of one chunk (in place, same
  Drive id) → a refused re-roll (incomplete) → Retry (in place). Asserts:
  exactly one provider POST per sent paid row (each row's stored request
  posted once), every draw = its row's cost, every reservation's consumed =
  its draws and charged = max(reserved, consumed), the Kling run's video draw
  = Σ request_sec × 28.5 over succeeded takes, the estimate 0 at the end,
  final frames = source frames (600), audio present, the v2v createTask
  bodies **byte-identical** to main `730216b`'s
  (`tests/fixtures/rebuild_v2v_request_bytes_main730.json`, 15 bodies,
  generator `tests/fixtures/gen_rebuild_v2v_request_bytes.py` run on a clean
  `git archive 730216b`). Plus the state matrix (below), the dispatcher /
  reaper seam, page == server preconditions, and the gate regressions (the
  observer race reproduces `incomplete` on the old read order).
- Rendered-HTML state matrix (`test_project_page_state_matrix`,
  `test_run_page_state_matrix`, `test_v2v_pages_carry_no_rebuild_ui`):
  project — flag off, analysing, no storyboard, storyboard running,
  storyboard done with the speech warning, keyframes partial, all selected
  (launch enabled), launch refused with the unmet list; run — pending,
  generating, one chunk failed, done with flags.
- `tests/test_rebuild_browser.py` (Playwright; skipped without it): the
  launch form navigates to the new run, model switch swaps resolutions /
  estimate / launch state, a refused launch shows `detail` and stays, the
  card's polling pauses while a form field has focus and resumes after, the
  run page done with flags. Run in a scratch image `reskin-rebuild-browser:scratch`
  (`reskin-rebuild-test` + playwright + chromium; host deps are stale).

### Code gate record — integration (GPT-6.1 Terra, Codex, reasoning high, read-only)

Round 1 reviewed the FULL `730216b...7cd8b6e` with the seams between the waves
and with main as the focus; later rounds the deltas.

| Round | Range | Result | Fixed in |
|---|---|---|---|
| 1 | `730216b...7cd8b6e` (full) | MAJOR: `observe_run` read segments then attempts — a fetch committing in between made it fail a completed, paid chunk and move the run to `incomplete`. MAJOR: `admit_check` did not take the Run-row lock `delete_run` takes. MAJOR: the estimator dropped the fallback slot when the primary "succeeded" with an unusable answer (a re-baseline could release its headroom). MINOR: a free QA without a contact sheet left the check un-admitted forever. | `daae4ac` (attempts read first + a regression test that reproduces `incomplete` on the old order; Run lock; verdict-keyed allowance; bounded free-QA re-run then `unavailable`) |
| 2 | `7cd8b6e..daae4ac` | MAJOR: free-QA runs were counted only when a run finished, so a crashing / timing-out QA job was retried forever and kept the allowance reserved. | `2fa96ad` (`free_runs` written at the start; live job left alone; `admit_auto_check` never after a verdict) |
| 3 | `daae4ac..2fa96ad` | MAJOR: the reaper wrote `unavailable` from a stale snapshot, possibly over a QA run that had just written the sheet and admitted a check. | `9663cd8` (`_give_up_check` re-checks under the attempt row lock: sheet still missing, runs used up, no verdict, no check call) |
| 4 | `2fa96ad..9663cd8` | MINOR only: a free-QA job that starts after the reaper's liveness read can lose its automatic check to the give-up. No BLOCKER/MAJOR — gate passed. Own follow-up in the same area: `admit_check` now also locks the attempt and refuses an automatic slot after a verdict, so a give-up and an admission cannot both land. | `5cf6491`; the MINOR is residual R11 |
| 5 | `9663cd8..5cf6491` | NO FINDINGS (incl. a deadlock review of the new attempt lock against fetch / observer / delete / retry / rerun / use-attempt / fold / manual check). | — |

### Test record (integration)

- Rebuild suites at `8ff91f2` in `reskin-rebuild-test:latest` with a disposable
  Postgres (`RESKIN_TEST_PG_URL`): **265 passed, 1 skipped** (the browser
  module, no playwright in that image).
- Browser tests (`tests/test_rebuild_browser.py`) in `reskin-rebuild-browser:scratch`:
  5 passed.
- Full suite, same image + Postgres, `-n 3`, compared by test name:
  main `730216b` — 16 failed / 5447 passed / 121 skipped; integration
  `2fa96ad` — 14 failed / 5722 passed / 122 skipped. **No new failure.** The
  14 are main's environment failures (assistant_ending_acceptance ×2,
  gdrive_run_result ×4, kie_client ×2, knowledge_importer_pg ×2,
  pipeline_range_attempt ×1, range_attempt_pg_concurrency ×2,
  pipeline_v2 missing_project ×1); main's other two
  (`test_spend_pg_concurrency` ×2) passed on a rerun on BOTH trees (load
  flakes). The commits after `2fa96ad` touch only the rebuild QA path, tests
  and docs; the rebuild suites above cover them.
- Migration rehearsal on the production schema copy: see "Migration re-chain".

## Follow-ups after the deploy

### AICP-382 — LLM calls settle to kie.ai's reported cost

kie.ai's chat route returns a top-level `credits_consumed` in every accepted
body (all 130 stored `rebuild_llm_calls.response` rows on prod carry it,
2026-11-05: shot call 0.22-0.32 cr, global call 1.8-2.1 cr, character check
0.34-0.50 cr; storyboard `3fbf6191` = 62 calls, 17.5 cr real vs 1,240 cr
booked). `rebuild_paid_call.record_send_outcome` now writes it to
`cost_credits` and settles the call's draw to it in the same transaction as
`succeeded` (`kie_client.chat_credits_consumed`; anything not a finite
non-negative number keeps the estimate, with a warning). Token `usage` is not
used: a least-squares fit of `usage` against `credits_consumed` on the same
rows misses by up to 60 % (media / reasoning tokens kie.ai does not itemise).

Unchanged on purpose: admission still reserves `KIE_LLM_CALL_EST_CREDITS`
(20 cr) per call incl. the fallback row, and a storyboard's call-admission
budget (`committed_credits`) still counts row estimates — both are worst-case
limiters. The storyboard's `release_unused` at its end then shrinks the
reservation to the real draws, so the day's charged total drops to ~real.
Character checks draw in the run scope, whose reservation (like every run's)
is not released when the run ends. Rows written before this change keep
their 20 cr draws (no backfill).

### AICP-381 — library clip kind; the default end card is an end card

Migration `w2c4e6a8b0d3` (on `w4c8e2a6d1f3` since the 2026-11-06 release chain; additive): `library_clips.kind`
VARCHAR(16) NOT NULL, server default `insert`, CHECK `kind IN ('end_card',
'insert')`; the data step marks `name LIKE 'CTA\_%' AND duration_sec <= 10`
as `end_card` (prod 2026-11-05: CTA_English / CTA_Spanish / CTA_Portuguese /
CTA_Poland / CTA_Poland_2). `CTA_Spanish_Book` / `CTA_Portuguese_Book` (24.6 /
24.8 s, an audiobook scene with the card at the end) stay `insert` on
purpose. The old image ignores the column, so the migration may go first.

- `rebuild_launch.language_end_card_clip` = the newest `end_card` in the run
  language with its file on disk; never an `insert`. None -> the source card
  is kept and the launch warns ("the library has no end card in the run
  language"). An explicit `stitch_plan` from the API is not policed.
- The project page's End card picker lists end cards only (with language and
  length) and shows, under the picker, what the selected choice resolves to
  for the selected language (`auto_end_card` per language, from the same
  function); the launch confirm repeats that line.
- `/v2/library`: Kind column (in-place select, `PATCH {"kind": ...}`) and a
  Kind field on upload (default Insert).
