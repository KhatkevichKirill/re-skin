---
title: "Project Overview"
tags: [overview, product, status]
sources: [tasks/todo.md, docs/v2-project-runs.md]
updated: 2026-06-24
---

# re-skin — Project Overview

## What It Is

re-skin is an AI video face-swapping service for operators (small team, internal use). An operator uploads a source video; the system automatically detects face-bearing segments using InsightFace, proposes a swap/keep segmentation, and lets the operator review and tune it. The operator then submits one or more "characters" (prompt + reference images + AI model choice), and the system generates replacement face footage for each swap segment, stitches everything together, and delivers the final video to Google Drive.

The key design insight: **separate the video segmentation (done once per video) from the character generation (done many times per video)**. This is the v2 Project → Runs model — see [[architecture]] for the data model.

## Current State (as of 2026-06-24)

| Layer | Status |
|-------|--------|
| **v1** (Job model) | Deployed to production, stable. Will be retired after v2 is verified. |
| **v2** (Project → Runs model) | ~80% complete on `v2` branch. Core pipeline, API, and UI are done; TR6/TR7/TR8 in-flight. |

### v2 In-Flight Tasks
- **TR6**: Per-segment prompt/reference overrides on RunSegment + single-segment re-run (cheap retry for failed swaps)
- **TR7**: Audio mode per run — `original` (mux source audio, current behavior) vs `seedance` (per-clip audio; avoids drift when Seedance clip lengths shift)
- **TR8**: Linked segment boundary editing — editing one segment's end also moves the next segment's start; backend enforces contiguous partition

After TR6/TR7/TR8, TR5b cleans up v1 and merges v2 → main.

## Who Uses It

Internal operators. Access is gated by Nginx basic auth. Public links (token-signed HMAC URLs) are available for sharing individual videos with external AI tools or reviewers without exposing the full UI.

## Tech Stack Summary

```
Browser → Nginx (basic auth, :8847) → FastAPI/uvicorn (:8000) → RQ Worker
                                                ↓
                                    SQLite (WAL) + Redis + Google Drive
                                    InsightFace + FFmpeg + kie.ai + Gemini Omni
```

See [[architecture]] for the full system diagram.

## Growth Trajectory

The project started as a single-job tool (v1) and is evolving toward a multi-character, multi-run platform (v2). Planned directions after v2 ships: see [[roadmap]].

## Key Design Decisions

- **SQLite + WAL** over Postgres: simple, serverless, works on bind-mounted Docker volumes
- **RQ over Celery**: minimal setup, functions as tasks, easy to test with mocking
- **HTMX over React**: no JS bundle, server controls state, fast iteration
- **Parallel submit + concurrent poll** in v2: all Seedance tasks go out at once; 2h skip timeout prevents one stuck segment from blocking the whole run

See [[decisions/v2-project-runs]] for the full v2 design rationale.
