# Wiki Index

Content catalog for the re-skin LLM wiki. Updated on every ingest or new page.

## Core

| Page | Summary |
|------|---------|
| [overview](overview.md) | What re-skin is, current state (v2 ~80%), who uses it, key design choices |
| [architecture](architecture.md) | System topology (nginx→api→worker), data model (Project/Run/SegmentDef), state machines, storage layout |
| [roadmap](roadmap.md) | Immediate v2 completion (TR6/TR7/TR8/TR5b) + post-v2 growth vision |

## Architecture Decisions

| Page | Summary |
|------|---------|
| [decisions/v2-project-runs](decisions/v2-project-runs.md) | ADR: why Project → Runs replaces the monolithic Job; parallel submit; audio modes |
| [decisions/postgres-migration](decisions/postgres-migration.md) | ADR: SQLite → PostgreSQL migration; driver choice (psycopg2); JSON vs JSONB; enum DDL fixes; ETL approach |

## Components

| Page | Summary |
|------|---------|
| [components/pipeline](components/pipeline.md) | v2 pipeline: analyze_project (InsightFace → SegmentDefs), process_run (parallel submit + concurrent poll + stitch + deliver) |

## AI Models

| Page | Summary |
|------|---------|
| [models/ai-models](models/ai-models.md) | Seedance (kie.ai) and Gemini Omni — constraints, configuration, choosing between them |

## Lessons

| Page | Summary |
|------|---------|
| [lessons/production-gotchas](lessons/production-gotchas.md) | FFmpeg OOM, BASE_DIR fragility, nginx stale-upstream, alembic stamp, Seedance duration floor, audio drift, RQ timeouts |

## Activity

| Page | Summary |
|------|---------|
| [log](log.md) | Chronological record of ingests, queries, and sessions |

---

_Pages to create as the project grows:_
- `components/face-detection.md` — InsightFace timeline logic, propose_segments algorithm
- `components/media-processing.md` — FFmpeg stitch normalization, FPS cap, audio modes
- `models/model-quality.md` — Per-scenario quality observations (fill with production data)
- `lessons/model-quality.md` — What prompts and reference image configs work well
