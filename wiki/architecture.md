---
title: "System Architecture"
tags: [architecture, infrastructure, data-model, state-machine, postgres]
sources: [tasks/todo.md, backend/app/models.py, docker-compose.yml]
updated: 2026-06-24
---

# System Architecture

## Service Topology

```
Internet
    │
    ▼
┌──────────────────────────────────────────┐
│ Nginx (Alpine) — :8847 (external)         │
│  • Basic auth (.htpasswd from env vars)   │
│  • client_max_body_size 1024m             │
│  • proxy_read_timeout 300s                │
│  • /public/* routes bypass auth (HMAC)   │
└──────────────────┬───────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────┐
│ FastAPI / uvicorn — :8000 (internal)      │
│  Router includes:                         │
│  • api.py     → /api/jobs/* (v1)          │
│  • api_v2.py  → /api/v2/projects,runs    │
│  • public.py  → /public/* (token-signed) │
│  • web.py     → / (v1 HTML)              │
│  • web_v2.py  → /v2/ (v2 HTML)          │
└──────────┬────────────────────┬──────────┘
           │                    │
           ▼                    ▼
    ┌─────────────┐     ┌──────────────┐
    │ PostgreSQL 16│     │ Redis :6379  │
    │ (internal)  │     │ (RQ queue)   │
    └─────────────┘     └──────┬───────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ RQ Worker ×N (Python) │
                    │  analyze_project      │
                    │  process_run          │
                    │  (InsightFace, FFmpeg │
                    │   kie.ai, GDrive)     │
                    │                       │
                    │  Scale: --scale worker=N │
                    └──────────────────────┘
```

**Shared volumes:**
- `./data/` — bound to api + all worker replicas (media files: clips, results, source videos)
- `postgres-data` — named Docker volume for PostgreSQL data (replaces `./data/app.db`)
- `./secrets/` — GDrive service-account JSON (read-only)
- `insightface-models` — named volume, ~300MB buffalo_l model, shared by all worker replicas

**Resource limits (docker-compose.yml) — sized for N=2 workers on 4 vCPU / 16 GiB:**
- `api`: mem_limit 2g
- `worker` (per replica): mem_limit 6g, memswap_limit 7g, cpus 2.0
- Scale: `docker-compose up -d --scale worker=2`. N=3 would exceed host RAM ceiling.

See [[components/parallel-workers]] for resource math, scaling ops notes, and the poll-hold bottleneck proposal.

## Data Model

### v2 Model (canonical going forward)

```
VideoProject
├── id, name, source_type (upload|gdrive), source_ref, source_local_path
├── probe (duration, width, height, fps, aspect_ratio)
├── status: created → analyzing → ready | failed
├── SegmentDef[] (reused across all Runs)
│   └── start_sec, end_sec, has_face, action (swap|keep), pre_roll_sec, post_roll_sec
└── Run[]
    ├── name, prompt, reference_image_urls (JSON), model (seedance|gemini-omni)
    ├── resolution, audio_mode (original|seedance), gdrive_folder_id
    ├── status: created → queued → processing → stitching → delivering → done | failed
    └── RunSegment[] (one per swap SegmentDef)
        ├── segment_def_id (FK), index
        ├── status: pending → uploading → submitted → generating → completed | failed | skipped
        └── kie_upload_url, seedance_task_id, seedance_result_url, local_clip_path, local_result_path
```

### v1 Model (to be retired after v2 ships)

```
Job (source + prompt + refs + one character)
└── Segment[] (coupled to this job's character)
```

## State Machines

### Project status
`created → analyzing → ready | failed`

### Run status
`created → queued → processing → stitching → delivering → done | failed`

### RunSegment status
`pending → uploading → submitted → generating → completed | failed | skipped`

Transitions are validated by `state_machine.py` — invalid transitions raise exceptions.

## Storage Layout

```
data/
├── app.db
└── projects/
    └── {project_id}/
        ├── source.mp4          # Downloaded once for the whole project
        └── runs/
            └── {run_id}/
                ├── clips/      # Cut segments (pre/post-roll included)
                │   └── seg_0.mp4, seg_1.mp4 ...
                └── results/    # Seedance outputs + keep clips
                    ├── seg_0_result.mp4
                    └── final.mp4
```

## Migrations

8 Alembic revisions track schema history (`backend/alembic/versions/`):
- `4c5436a0d0f1` → initial schema (jobs, segments)
- `81b441a0932d` → v2 tables (video_projects, runs, segment_defs, run_segments)
- `c3f9a2e1b450` → run_segment overrides
- `d4e7f2a1c890` → run audio_mode
- `e5a1c7d2f3b6` → run model (seedance/gemini-omni)
- `f1b2c3d4e5a6` → project name
- `a1b2c3d4e5f7` → add '4k' to run_resolution_enum (Postgres-compatible)

Head: `a1b2c3d4e5f7`. The API runs `create_all()` on startup; migrations are applied manually via `alembic upgrade head`. The deployed stack does not auto-apply migrations on startup (TR5b will fix this).

> **Postgres note:** Two migrations (`d4e7f2a1c890`, `e5a1c7d2f3b6`) were fixed to pre-create Enum types before `add_column` — Postgres requires the type to exist. `CURRENT_TIMESTAMP` server defaults were also fixed (SQLite used `(CURRENT_TIMESTAMP)` with extra parens, rejected by Postgres). See [[decisions/postgres-migration]] and [[lessons/production-gotchas]].

## Key Modules

| Module | Role |
|--------|------|
| `pipeline_v2.py` | Parallel submit + concurrent poll for all swap segments |
| `face.py` | InsightFace timeline detection → segment proposal |
| `media.py` | FFmpeg probe, cut_clip, stitch (normalization + fps cap at 30) |
| `kie_client.py` | kie.ai upload, create_task, poll_task, download_result (tenacity retries) |
| `gdrive_client.py` | Chunked resumable GDrive upload, download by link |
| `storage.py` | Path helpers for project/run dirs |
| `state_machine.py` | Status enums + transition validation |
| `public.py` | Token-signed unauthenticated media URLs (HMAC) |

See [[components/pipeline]] for the full pipeline flow.
