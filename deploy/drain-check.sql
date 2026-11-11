-- Active-work check before a deploy that stops/recreates containers.
-- Read-only. Run from /root/re-skin:
--   docker exec -i re-skin-db psql -U reskin -d reskin -At -F' ' < deploy/drain-check.sql
-- Each line is `<active|parked> <kind> <count>` (AICP-350, docs/assistant-drain-check-design.md §2):
--   active = local work a restart would kill: EVERY active row must read 0 before a deploy.
--   parked = durable states never retried blind (provider tasks waiting for the operator):
--            listed, not waited on.
-- Keep in step with backend/app/ops_status.py (ACTIVE / PARKED), which serves the same split
-- at GET /api/v2/ops/admission. scripts/deploy.sh waits on this file.
-- Drive re-link and caption-placement scan jobs have no running status: they hold the
-- gate's share lock while they run (app/ops_gate.py job_barrier), so begin.sql waits for
-- them, and they do not start while paused (AICP-350). No row here.
-- AICP-261 assistant turns/jobs (queued/running/cancel_requested) and paid-call intents in
-- submitting/unknown count as active: the supervisor acknowledgement that would let an
-- `unknown` intent count as parked (design §2, §9 item 1) is not implemented; resolve it first.
-- Statuses are the non-terminal labels of each *_status_enum (checked against
-- pg_enum on 2026-10-17, head b8d0f2a4c6e9). `created` rows are not queued
-- yet and are safe; `preprocess_review` / `incomplete` runs are parked.
-- provider_tasks.status is a varchar (state_machine.PROVIDER_TASK_DRAIN_STATUSES;
-- submit_unknown and needs_reconciliation are parked for the operator (the `parked` row); a needs_reconciliation
-- row can park for long and is not a stuck deploy).
-- reframe_jobs.status / qa_status and reframe_renders.status are varchars (AICP-251,
-- app/reframe_models.py).
-- run_segment_fill_jobs.status is a varchar (AICP-358, app/segment_fill_models.py): preparing
-- and compositing are worker phases; admitted waits on its provider task (counted above);
-- composited (batch fills, migration u5e1b7c9d3f4) waits for its batch's promote.
-- run_segment_fill_batches (u5e1b7c9d3f4): active = members at Fal / compositing, promoting =
-- the one promote is running. The table exists from u5e1b7c9d3f4 on (apply it first). Never
-- roll the image back while one is active or a member is composited (the old image promotes
-- members one by one).
-- autopilot_effects (AICP-363): an action effect in `launching` is an admission whose
-- bookkeeping has not landed; run `python -m app.autopilot hold on` first (the gate
-- barrier), then this row reads 0 unless one is unreconciled. Compute effects (free,
-- re-enqueued) and judgements (separate stack) are not counted.
-- localisation_prompt_jobs (AICP-365, app/localisation_jobs_models.py): the autopilot's durable
-- translate jobs (queued/running). A deploy interrupts a running one and it is NOT replayed
-- (the reaper marks it interrupted; the autopilot asks a regeneration), so drain it. Durable
-- transcriptions (project_transcriptions) set transcript_status, counted under 'projects'.
-- The table exists from migration s3b5d7f9a1c2 on (apply it before running this check).
-- video_projects.transcript_status is a varchar: pending | running (active) | ready |
-- empty | failed | NULL (models.py).
-- Rebuild from reference (migration t4d9f2b7c315, docs/rebuild-from-reference-design.md §7.5):
-- project_storyboards (dispatch protocol) and the three paid-call tables (varchar status,
-- app/rebuild_models.py). A row in 'sending' clears only when its sender finished or is
-- proven gone (reaper / scripts/rebuild_release_sender.sh) — wait for it, never force it.
-- The tables exist from migration t4d9f2b7c315 on (apply it before running this check).
-- paid_submit_attempts (AICP-351, migration x3b5d7f9a1c4 — apply it BEFORE running this
-- check): one row per paid kie.ai video/image submit or LLM-job attempt, committed before the
-- call. ACTIVE = a sender may still send: `started`, a `superseded` LLM attempt (lease ran out
-- without proof; counted even after its job ended — design §9 item 2), or a `parked` one whose
-- sender is not released. PARKED = an unknown-outcome submit whose sender is gone: it waits for
-- the operator (Resubmit / Abandon on the run page, `python -m app.paid_attempts`). An
-- unreleased sender whose container was recreated: scripts/paid_attempt_release_sender.sh <id>.
SELECT 'active' AS state, 'projects', count(*) FROM video_projects WHERE status = 'analyzing' OR transcript_status IN ('pending', 'running')
UNION ALL SELECT 'active', 'runs', count(*) FROM runs WHERE status IN ('queued', 'processing', 'stitching', 'delivering', 'preprocessing')
UNION ALL SELECT 'active', 'range_attempts', count(*) FROM run_segment_range_attempts WHERE status IN ('queued', 'generating', 'stitching', 'delivering')
UNION ALL SELECT 'active', 'segment_images', count(*) FROM run_segment_image_jobs WHERE status IN ('queued', 'generating')
UNION ALL SELECT 'active', 'tail_edits', count(*) FROM run_tail_edits WHERE status IN ('queued', 'processing', 'delivering')
UNION ALL SELECT 'active', 'captions', count(*) FROM run_captions WHERE status IN ('queued', 'transcribing', 'burning', 'delivering')
UNION ALL SELECT 'active', 'overlays', count(*) FROM overlay_jobs WHERE status IN ('queued', 'processing', 'delivering')
UNION ALL SELECT 'active', 'insert_merges', count(*) FROM insert_merges WHERE status IN ('queued', 'compositing', 'delivering')
UNION ALL SELECT 'active', 'audio_patches', count(*) FROM run_audio_patches WHERE status IN ('queued', 'processing', 'delivering')
UNION ALL SELECT 'active', 'reframe_jobs', count(*) FROM reframe_jobs WHERE status = 'preparing' OR qa_status = 'running'
UNION ALL SELECT 'active', 'reframe_renders', count(*) FROM reframe_renders WHERE status IN ('queued', 'processing', 'delivering')
UNION ALL SELECT 'active', 'segment_fills', count(*) FROM run_segment_fill_jobs WHERE status IN ('preparing', 'compositing', 'composited')
UNION ALL SELECT 'active', 'segment_fill_batches', count(*) FROM run_segment_fill_batches WHERE status IN ('active', 'promoting')
UNION ALL SELECT 'active', 'provider_tasks', count(*) FROM provider_tasks WHERE status IN ('accepted_unrecorded', 'cancelling', 'closing', 'fetching', 'queued', 'submit_started', 'submitted')
UNION ALL SELECT 'active', 'autopilot_launching', count(*) FROM autopilot_effects WHERE kind = 'action' AND state = 'launching'
UNION ALL SELECT 'active', 'translate_jobs', count(*) FROM localisation_prompt_jobs WHERE status IN ('queued', 'running')
UNION ALL SELECT 'active', 'rebuild_storyboards', count(*) FROM project_storyboards WHERE status IN ('queued', 'running')
UNION ALL SELECT 'active', 'rebuild_keyframes', count(*) FROM rebuild_keyframes WHERE status IN ('queued', 'claimed', 'sending', 'submitted', 'fetching')
UNION ALL SELECT 'active', 'rebuild_chunk_attempts', count(*) FROM rebuild_chunk_attempts WHERE status IN ('queued', 'claimed', 'sending', 'submitted', 'fetching')
UNION ALL SELECT 'active', 'rebuild_llm_calls', count(*) FROM rebuild_llm_calls WHERE status IN ('queued', 'claimed', 'sending', 'submitted', 'fetching')
UNION ALL SELECT 'active', 'assistant_turns', count(*) FROM assistant_turns WHERE status IN ('queued', 'running', 'cancel_requested')
UNION ALL SELECT 'active', 'assistant_jobs', count(*) FROM assistant_jobs WHERE status IN ('queued', 'running', 'cancel_requested')
UNION ALL SELECT 'active', 'assistant_paid_intents', count(*) FROM assistant_paid_call_intents WHERE status IN ('submitting', 'unknown')
UNION ALL SELECT 'active', 'paid_submit_attempts', count(*) FROM paid_submit_attempts WHERE status IN ('started', 'superseded') OR (status = 'parked' AND sender_released_at IS NULL)
UNION ALL SELECT 'parked', 'provider_tasks', count(*) FROM provider_tasks WHERE status IN ('needs_reconciliation', 'submit_unknown')
UNION ALL SELECT 'parked', 'paid_submit_attempts', count(*) FROM paid_submit_attempts WHERE status = 'parked' AND sender_released_at IS NOT NULL;
