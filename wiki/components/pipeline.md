---
title: "Pipeline (v2)"
tags: [pipeline, worker, rq, seedance, v2]
sources: [backend/app/pipeline_v2.py, backend/app/tasks.py]
updated: 2026-06-24
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
2. **Parallel upload**: For each swap RunSegment, cut clip from source (`media.cut_clip()`) → upload to kie.ai (`kie_client.upload()`) → create Seedance task (`kie_client.create_task()`)
   - All uploads and task creates happen concurrently via `asyncio.gather()`
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
- If a worker process dies mid-run, the run stays in `processing` state and can be retried via `/api/v2/runs/{id}/retry` — which resets failed/pending RunSegments and re-enqueues

## RQ Task Wrappers

All pipeline functions are wrapped in `tasks.py` for RQ:
- `enqueue_analyze_project(project_id)` — enqueues `analyze_project`
- `enqueue_process_run(run_id)` — enqueues `process_run`

Both set `job_timeout=7200` (2h) to prevent RQ from killing long-running workers.

## Known Production Constraints

- **Memory**: Worker is capped at 8g. 1080p stitch can use 2-4GB depending on clip count. The `FFMPEG_MAX_FPS=30` cap (default) prevents OOM on 60fps sources (reduces frame buffer by 2×).
- **Disk**: Intermediate clips + results accumulate fast at 1080p. No auto-cleanup yet (see [[roadmap]]).
- **Single worker**: Currently only one worker process. Multi-worker is a post-v2 growth item.

See [[architecture]] for the full service topology. See [[models/seedance]] for Seedance-specific behavior.
