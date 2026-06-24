---
title: "Production Gotchas & Lessons Learned"
tags: [lessons, production, ops, ffmpeg, docker, postgres, alembic]
sources: [tasks/todo.md, git log, deploy/postgres-cutover.md]
updated: 2026-06-24
---

# Production Gotchas & Lessons Learned

Learnings from building and running re-skin in production. Add entries as they're discovered.

## FFmpeg & Video Processing

### OOM kill on 1080p stitch (rc=-9)

**Problem**: FFmpeg stitch was killed by the kernel (OOM) on 1080p content. 60fps → 3MB per frame in RAM, filling the 8g worker limit fast.

**Fix**: Cap output FPS at 30 (`FFMPEG_MAX_FPS=30` default). Reduces frame buffer by 2× without visible quality loss on face-swap content.

**Lesson**: Always cap FPS at the target resolution when stitching. Source videos at 60fps don't need 60fps output for face-swap use cases.

### Relative path fragility (BASE_DIR)

**Problem**: `DATABASE_URL=sqlite:///./data/app.db` — the `./` resolves relative to CWD. In Docker containers the WORKDIR is `/app`; in local dev it's wherever you ran the command. This caused mismatched paths between API and worker.

**Fix**: `config.py` resolves `BASE_DIR` to an absolute path at startup and rewrites relative paths against it. All path helpers in `storage.py` use absolute paths.

**Lesson**: Never use relative paths in a multi-process service. Resolve everything to absolute at config load time.

## Docker & Deploy

### nginx stale-upstream footgun

**Problem**: After rebuilding the API container, nginx continued sending traffic to the old container's internal IP. Requests hung or returned 502 until nginx was restarted.

**Fix**: Rebuild the entire stack (`docker-compose up -d --build`) rather than rebuilding individual services. nginx resolves upstream IPs at startup.

**Lesson**: In Docker Compose, always rebuild the full stack after any service image change. Don't try to hot-swap individual containers behind nginx.

### Build shipping `data/` → disk full

**Problem**: Docker build context included `./data/` (SQLite DB + all media files), causing builds to hang and disk to fill.

**Fix**: Added `.dockerignore` with `data/`, `secrets/`, `.env`. Build is now fast.

**Lesson**: Always add `.dockerignore` before the first large-data accumulation. Add it at project start, not after.

### `docker-compose` vs `docker compose`

**Environment fact**: The VPS has Docker 26.1.3 with `docker-compose` v1.29.2 (standalone). The `docker compose` v2 plugin is NOT installed. Always use `docker-compose`.

## Database

### Alembic stamp on new deploy

**Problem**: On a fresh VPS migration, the DB is rebuilt from `create_all()` but alembic doesn't know this. Subsequent `alembic upgrade head` fails or makes wrong choices.

**Fix**: After fresh deploy, run `alembic stamp head` to mark the DB as at the latest revision without applying migrations.

**Lesson**: Document the deploy sequence: `create_all()` handles schema creation; `alembic stamp head` tells alembic the DB is current. Don't run `alembic upgrade head` on a fresh DB.

## PostgreSQL Migration

### `op.add_column` with Enum type fails on Postgres

**Problem**: `op.add_column('table', sa.Column(..., sa.Enum('a','b', name='my_enum'), ...))` works on SQLite (which ignores the enum type name and stores as VARCHAR) but fails on Postgres with `ERROR: type "my_enum" does not exist`. Alembic only auto-creates the Enum type when used in `op.create_table`; for `add_column`, you must create the type explicitly first.

**Fix**: In the migration's `upgrade()`, detect Postgres and call `enum.create(bind, checkfirst=True)` before `add_column`:
```python
bind = op.get_bind()
if bind.dialect.name == 'postgresql':
    sa.Enum('a', 'b', name='my_enum').create(bind, checkfirst=True)
op.add_column('table', sa.Column('col', sa.Enum('a', 'b', name='my_enum'), ...))
```
Applied to migrations `d4e7f2a1c890` (audio_mode) and `e5a1c7d2f3b6` (model).

**Lesson**: Any migration that adds a column with a named Enum type must pre-create the type on Postgres. SQLite silently succeeds; Postgres enforces type existence.

### `(CURRENT_TIMESTAMP)` server default rejected by Postgres

**Problem**: SQLite accepts `sa.text('(CURRENT_TIMESTAMP)')` (with parens) as a server default. Postgres does not — it expects `CURRENT_TIMESTAMP` without parens.

**Fix**: Replace all occurrences in migration files with `sa.text('CURRENT_TIMESTAMP')`. Both SQLite and Postgres accept the parenthesis-free form.

**Lesson**: Avoid SQLite-specific SQL dialect idioms in Alembic migration files. Prefer ANSI SQL. Test migrations against both databases when adding new server defaults.

### Alembic `upgrade head` vs `stamp head` for Postgres cutover

When cutting over to a fresh Postgres DB (empty), use `alembic upgrade head` — not `stamp head`. The `stamp head` trick (from the [[production-gotchas#Alembic stamp on new deploy]] section below) is for when the schema already exists (e.g. restored from a pg_dump); `stamp head` would skip applying migrations on an empty DB.

The flow:
1. Fresh Postgres: `alembic upgrade head` (creates schema + records revision)
2. Schema pre-exists (dump restore): `alembic stamp head` (records revision without running migrations)

### psycopg2 rejects Python lists as JSON — use `psycopg2.extras.Json`

**Problem**: When inserting rows with Python `list` values into a Postgres `JSON` column via psycopg2, psycopg2 binds them as Postgres `text[]` arrays, causing `DatatypeMismatch: column is of type json but expression is of type text[]`.

**Fix**: Wrap list/dict values with `psycopg2.extras.Json`:
```python
from psycopg2.extras import Json
conn.execute("INSERT INTO t (col) VALUES (%(col)s)", {"col": Json(my_list)})
```
Applied in `scripts/sqlite_to_postgres.py` for all JSON columns.

**Lesson**: psycopg2 does not auto-detect that a Python list should go into a JSON column; it follows PostgreSQL's type inference. Always wrap JSON-destined Python objects with `psycopg2.extras.Json` in ETL scripts.

### SQLite BOOLEAN stored as INTEGER 0/1

**Problem**: SQLite stores `BOOLEAN` columns as `INTEGER` (0=False, 1=True). When reading with SQLAlchemy and inserting into Postgres (which has a native `BOOLEAN` type), psycopg2 rejects `INTEGER` for a boolean column.

**Fix**: Coerce `has_face` (and any other boolean columns) from `int` to Python `bool` during ETL:
```python
if k in BOOLEAN_COLUMNS and isinstance(v, int):
    v = bool(v)
```

**Lesson**: Always audit type coercions when migrating between SQLite and Postgres. Boolean, JSON, and DateTime columns all need explicit handling.

### Enum schema drift: model has more values than migration

**Problem**: The `run_resolution_enum` was created with `('480p', '720p', '1080p')` in migration `81b441a0932d`. The `Run` model was later updated to include `'4k'` for Gemini Omni (in `e5a1c7d2f3b6`), but no migration was added to extend the Postgres type. SQLite silently accepts any string; Postgres enforces enum membership and would reject `'4k'` inserts.

**Fix**: Added migration `a1b2c3d4e5f7` that runs `ALTER TYPE run_resolution_enum ADD VALUE IF NOT EXISTS '4k'` on Postgres.

**Lesson**: When extending an ORM model's Enum values, always add a Postgres migration to extend the type. SQLite's lack of enforcement masks this gap. Run `alembic upgrade head` against a test Postgres DB as part of every PR that touches ORM models.

See [[decisions/postgres-migration]] for the full ADR.

## Seedance / kie.ai

### Minimum segment duration (~1.8s)

**Problem**: Seedance rejects clips shorter than ~1.8s. Face detection can propose very short segments on fast cuts.

**Fix**: `min_segment_sec` floor enforced in face.py proposal + pre/post-roll padding added to ensure clips meet the floor even after trim.

**Lesson**: External AI APIs often have undocumented minimum input size constraints. Test edge cases (very short segments) early.

### Segment audio drift with `original` audio mode

**Problem**: Seedance sometimes outputs clips that are slightly shorter or longer than the input. When stitching with the original source audio (muxed as a single track), the video/audio fall out of sync after enough segments.

**Fix (TR7)**: `audio_mode=seedance` per run — use per-clip audio instead of muxing the full source audio track. Swap segments use Seedance clip audio; keep segments use source clip audio.

**Lesson**: Never assume AI video models output clips with the exact same duration as the input. Design audio handling to be tolerant of length drift.

## RQ / Worker

### `container_name` prevents `docker-compose --scale`

**Problem**: Setting `container_name: re-skin-worker` in docker-compose.yml prevents running more than one worker replica. Docker names containers based on the `container_name` value; a second instance would collide. `docker-compose up --scale worker=2` silently stops at 1, or errors out.

**Fix**: Remove `container_name` from the worker service. Docker then auto-names replicas as `<project>_worker_1`, `<project>_worker_2`, etc. Services that should stay singletons (api, nginx, redis, db) retain their `container_name`.

**Lesson**: Only use `container_name` on services that must be singletons. Omit it from any service that may need to scale.

### RQ job timeout

**Problem**: RQ's default job timeout is short. Long Seedance tasks (up to 2h total for a run) were killed mid-processing.

**Fix**: Set `job_timeout=7200` (2h) on all RQ job enqueues in `tasks.py`.

**Lesson**: Always set explicit long timeouts for AI-dependent jobs. The default is for fast jobs.

### Worker crash leaves runs orphaned (confirmed 2026-06-24)

**Problem**: When the worker is killed mid-run (OOM, `docker-compose up --build` redeploy), the in-flight RQ job is lost from Redis — there is no `wip`/`failed` registry entry and nothing re-enqueues it. The Run is left in whatever DB state it had:
- stuck in `processing` (poll loop died), or
- stuck in `queued` (job was popped/never started or never recorded).

The RQ queue ends up empty with one idle worker, while the DB shows active runs that will never advance. **No automatic recovery exists** (TR5b still unbuilt).

**Two confirmed recovery gaps**:
1. `POST /api/v2/runs/{id}/retry` only accepts `status == 'failed'` (api_v2.py:877) — a run stuck in `processing` or `queued` is **rejected** by the documented mitigation.
2. A bare `enqueue_process_run()` on a `processing` run throws `InvalidTransition: 'processing' -> 'processing'` because `process_run` blindly calls `transition(run, RunStatus.processing)` and there is no `processing → processing` (or `processing → queued`) edge. The thrown exception is caught and marks the run `failed` — after which retry finally works. Fragile two-step accident, not a real path.

**Manual recovery procedure used 2026-06-24** (runs `72325061`, `3b9eec36`):
- For a `queued` orphan: re-enqueue directly — `enqueue_process_run(rid)` (queued→processing is valid). `process_run` is idempotent: completed segments with a result on disk are skipped, others resubmitted.
- For a `processing` orphan: first reset `run.status = RunStatus.queued` in the DB, then re-enqueue.
- **Before resubmitting, check kie.ai for already-`success` tasks** of segments left in `generating`. The poll loop may have died *after* Seedance finished. Run `72325061` seg 2 had a finished result on kie.ai (`KieClient().get_task(task_id)` → state `success`); we downloaded it and marked the segment `completed` manually, so the resubmit skipped it and saved a Seedance call. The bare resume path would have reset and re-billed it.

**Planned fix (TR5b)**: Startup orphaned-run reconciliation — on worker boot, find runs in `processing`/`queued` with no live RQ job and either re-enqueue them or re-poll their `generating` segments' kie task ids before resubmitting. Becomes more urgent with multiple workers (more frequent restarts). See [[parallel-workers]].
