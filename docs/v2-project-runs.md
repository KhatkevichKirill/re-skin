# v2 — Project → Runs (multi-character per video)

Status: **DESIGN AGREED, BUILD DEFERRED** until v1 testing is complete (decided 2026-06-20).

## Problem
v1 couples everything into one `Job`: source video + segmentation + one character (prompt+refs) + result. Testing a different character on the same video forces re-upload + re-marking segments. We want to reuse the video and its segmentation across many character attempts.

## Decisions (locked)
- Approach: **Project → Runs** (proper refactor), not lightweight job-duplicate.
- Character scope: **one character per run**, applied to the whole video.
- **Drop per-segment prompt/reference override** (leftover from v1 hybrid). Segment edits on the project keep only timing + swap/keep + pre/post-roll. Prompt+refs live on the Run.
- Run has an optional **name/label** (e.g. "Redhead woman") to distinguish runs in the list.
- No real data to preserve → rebuild schema cleanly (no complex migration).
- Build on a branch; leave the running v1 stack untouched until v2 is verified.

## Target data model
- **VideoProject**: id, timestamps, source_type(upload|gdrive), source_ref, source_local_path, probe(duration/width/height/fps/aspect_ratio), status(created→analyzing→ready→failed). Has many SegmentDef.
- **SegmentDef**: id, project_id, index, start_sec, end_sec, has_face, action(swap|keep), pre_roll_sec, post_roll_sec. (Reused by every run.)
- **Run**: id, project_id, timestamps, name(optional), prompt, reference_image_urls(JSON), resolution, gdrive_folder_id, status(created→queued→processing→stitching→delivering→done|failed), result_local_path, result_gdrive_file_id, error_message. Has many RunSegment.
- **RunSegment**: id, run_id, segment_def_id(FK), status(pending→uploading→submitted→generating→completed|failed|skipped), kie_upload_url, seedance_task_id, seedance_result_url, local_clip_path, local_result_path, error_message. Resumability is per-run.

## Flow
1. Create project (video upload or GDrive link) → analyze (probe + propose segments) → edit/review segments once → project `ready`.
2. Add a run: enter character (name + prompt + reference images + resolution + GDrive folder) → submit → process_run uses the project's SegmentDefs and the run's character → stitch → deliver. Run `done`.
3. Many runs per project, grouped under the video for comparison.

## What changes vs v1
- Pipeline: `analyze_job → analyze_project(project_id)`; `process_job → process_run(run_id)` (loads project SegmentDefs, processes with the run's character into the run's RunSegments). Reuse media/face/kie/gdrive modules as-is.
- Storage: `data/projects/<pid>/source.mp4`, `data/projects/<pid>/runs/<rid>/clips`, `.../results`. Source downloaded once (for GDrive input) at project analyze.
- API: `POST /api/projects` (video only), `GET /api/projects[/{id}]`, segments review/edit on project, `POST /api/projects/{id}/runs` (create run with character), `GET /api/projects/{id}/runs`, `GET /api/runs/{id}` (status), `GET /api/runs/{id}/result`, retry. Frame/thumbnail endpoint stays on the project source.
- UI: dashboard lists projects; project page = segment editor + "Runs" panel (new-run form with name/prompt/refs/resolution/folder + list of runs with status/result); run view = status polling + result preview/download.
- Carry over the v1 hardening: RQ job_timeouts, retry segment reset, BASE_DIR path resolution, min_segment_sec / Seedance 1.8s floor, stitch normalization + original audio, rebuild whole stack (nginx stale-upstream footgun).

## Proposed task breakdown (sub-agents, I orchestrate)
- TR1: data model (Project/SegmentDef/Run/RunSegment) + state machines + clean Alembic schema.
- TR2: pipeline refactor (analyze_project, process_run, storage layout, tasks enqueue with timeouts) + tests.
- TR3: API refactor (projects + runs endpoints) + tests.
- TR4: frontend refactor (projects dashboard, project page with segment editor + runs panel, run view) + tests.
- TR5: deployed-stack e2e + docs/README update.
