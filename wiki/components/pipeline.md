---
title: "Pipeline (v2)"
tags: [pipeline, worker, rq, seedance, v2]
sources: [backend/app/pipeline_v2.py, backend/app/tasks.py, backend/app/recovery.py]
updated: 2026-06-25
---

# Pipeline (v2)

The pipeline is the core orchestration logic — it runs in the RQ worker process, never in the API process.

## Two Pipeline Functions

### `analyze_project(project_id)`

Called once per Project after creation.

1. Download source (if gdrive) or use uploaded file → `data/projects/{pid}/source.mp4`
2. `media.probe()` → width, height, fps, aspect_ratio, duration
3. `face.detect_timeline()` → InsightFace detects faces frame-by-frame → `[(start, end, has_face)]`
4. `face.propose_segments()` → groups intervals, applies lead-in, splits at max 15s → `[SegmentDef]`
5. Persists SegmentDefs; updates Project status: `analyzing → ready`

Key parameters:
- `SEGMENT_MAX_SECONDS` (default 15) — max duration for a swap segment
- `face.filter_small_faces()` — ignores faces smaller than a threshold (avoids false positives on background faces)

### `process_run(run_id)`

Called once per Run after submission.

1. Load Project SegmentDefs + Run config (prompt, refs, model, resolution, audio_mode)
2. **Bounded submit**: For each swap RunSegment, cut clip from source (`media.cut_clip()`) → upload to kie.ai (`kie_client.upload()`) → create Seedance/Gemini task.
   - Fresh submissions run through a bounded `ThreadPoolExecutor` controlled by `SUBMIT_CONCURRENCY` (default 2).
   - Each submit worker opens its own DB session; `process_run` commits newly created `RunSegment` rows before starting submit threads so the independent sessions can see them.
   - In production submit threads also create their own `KieClient` instances, avoiding shared HTTP client state across concurrent uploads/task creates. Injected test clients are preserved for deterministic tests.
   - Existing `seedance_task_id` resume/no-rebill handling stays serial before the fresh-submit pool so an in-progress external task is never duplicated.
3. **Concurrent poll**: Round-robin poll all submitted tasks every `RUN_POLL_INTERVAL_SEC` (default 15s):
   - Done → download result → `local_result_path`
   - Failed → log error, use original clip (segment: `failed → skipped`)
   - Pending >2h (`RUN_SKIP_TIMEOUT_SEC`) → timeout, use original clip (segment: `→ skipped`)
4. **Stitch**: `media.stitch()` — FFmpeg concat of all segment results (swap outputs + keep clips) + audio:
   - `audio_mode=original` → mux full source audio track (clean, but can drift if Seedance changes clip length)
   - `audio_mode=seedance` → per-clip audio (swap clips use Seedance audio, keep clips use cut source audio)
5. **Deliver**: Upload `final.mp4` to Google Drive via `gdrive_client.upload_file()` (chunked resumable)
6. Run status: `processing → stitching → delivering → done`

## Error Handling

- Individual segment failures don't kill the run — failed segments fall back to the original clip
- Skip timeout (2h) prevents one stuck Seedance task from blocking the whole run
- GDrive delivery retries `RUN_DELIVER_ATTEMPTS` times (default 3) with `RUN_DELIVER_BACKOFF_SEC` (default 10s) backoff
- If a worker process dies mid-run, the run stays in `processing` state; **startup reconciliation** (TR5b) auto-detects and re-enqueues on the next worker boot — see [[components/parallel-workers]] → "Startup Orphaned-Run Reconciliation"

## Resume / No-Rebill Behavior (TR5b)

`process_run` is safe to call on an orphaned run already in an active state
(`processing`, `stitching`, `delivering`). The entry-point detects a non-`queued`
run and resets it to `queued` first (via the `processing → queued` state-machine
edge added in TR5b), then re-advances normally.

For **segments stuck in `generating` with an existing `seedance_task_id`**,
`process_run` calls `kie.get_task(task_id)` before resubmitting:

- State `success` → download the result, mark segment `completed`, skip
  resubmission entirely. Avoids a redundant Seedance credit (this is what was
  done manually for run `72325061` seg 2 on 2026-06-24; see [[lessons/production-gotchas]]).
- State still in-progress → re-add to the poll loop without re-submitting.
- State `fail` or network error → fall through to reset + resubmit normally.

Only segments with **no `seedance_task_id`** (i.e., never submitted, or cleared
by a previous retry) are submitted fresh.

## RQ Task Wrappers

All pipeline functions are wrapped in `tasks.py` for RQ:
- `enqueue_analyze_project(project_id)` — enqueues `analyze_project`
- `enqueue_process_run(run_id)` — enqueues `process_run`

Both set long `job_timeout` values (`PROCESS_JOB_TIMEOUT` defaults to 10800s / 3h)
to prevent RQ from killing long-running workers. `run_process_run` also takes a
Redis per-run lock (`reskin:run:lock:{run_id}`) before calling `process_run`; a
duplicate RQ job that cannot acquire the lock exits before any external AI
submission. Lock release is token-checked atomically in Redis.

## Observability

`process_run` logs phase timings without prompts, secret URLs, or reference URL
values:

- reference resolution
- submit phase total and per-segment cut/upload/create timings
- poll total
- stitch total
- delivery total and final file id
- total run time

These Docker logs are the first place to check before changing worker count or
ffmpeg settings. If submit time dominates, tune `SUBMIT_CONCURRENCY` carefully;
if stitch dominates, tune `STITCH_CUT_CONCURRENCY`/ffmpeg settings instead.

## Poll / DB Session Scope

The v2 poll phase can wait on external AI tasks for up to
`RUN_SKIP_TIMEOUT_SEC` (default 2h). It must not hold a SQLAlchemy transaction
or DB connection while sleeping between poll rounds. `process_run` now commits
and closes its setup session before entering the poll wait loop; terminal task
states are recorded through short-lived sessions opened only around the
corresponding `RunSegment` update. This keeps the web UI and retry/recovery
writers from being blocked by a sleeping worker.

## Upload Limits

API uploads are streamed in bounded chunks rather than read into memory. The
limit is `MAX_UPLOAD_SIZE_MB` (default 1024 MiB) and must stay aligned with
nginx `client_max_body_size`. Oversized uploads return HTTP 413 and partial
files are removed.

## Known Production Constraints

- **Memory**: Worker is capped at 8g. 1080p stitch can use 2-4GB depending on clip count. The `FFMPEG_MAX_FPS=30` cap (default) prevents OOM on 60fps sources (reduces frame buffer by 2×).
- **Disk**: Intermediate clips + results accumulate fast at 1080p. No auto-cleanup yet (see [[roadmap]]).
- **Single worker**: Currently only one worker process. Multi-worker is a post-v2 growth item.

See [[architecture]] for the full service topology. See [[models/seedance]] for Seedance-specific behavior.
