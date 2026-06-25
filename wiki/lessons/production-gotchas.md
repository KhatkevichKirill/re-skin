---
title: "Production Gotchas & Lessons Learned"
tags: [lessons, production, ops, ffmpeg, docker, postgres, alembic, rq, recovery]
sources: [tasks/todo.md, git log, deploy/postgres-cutover.md, backend/app/recovery.py]
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

**Environment fact**: Compose tooling differs by host. The original VPS notes referenced `docker-compose` v1.29.2 (standalone). The host used for the **2026-06-24 Postgres cutover deploy** has only the **`docker compose` v2 plugin** (`docker-compose` v1 is absent). Check `which docker-compose` / `docker compose version` on the target host before deploying. v2 supports the same `up -d --build --scale worker=2` used in the cutover.

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

### SQLite WAL not captured by `cp app.db` during cutover (confirmed 2026-06-24)

**Problem**: The app runs SQLite in **WAL mode** (`PRAGMA journal_mode=WAL` in `db.py`). Recently-committed rows live in `data/app.db-wal` until a checkpoint folds them into the main `data/app.db` file. The cutover ETL snapshotted the DB with `cp data/app.db ...` — which copies ONLY the main file and **silently drops everything still in the WAL**. The ETL migrated a stale snapshot: two runs that were `done` came across into Postgres as `queued`/`processing`. Worse, TR5b startup reconciliation then saw those (falsely) active runs and re-enqueued them for reprocessing.

**Fix**: Checkpoint the WAL into the main file before copying — `sqlite3 data/app.db "PRAGMA wal_checkpoint(TRUNCATE)"` (or the equivalent python one-liner) — or point the ETL `--source` at the **live** `data/app.db` while its `-wal`/`-shm` sidecars are present (SQLite reads the WAL automatically). Never copy `app.db` detached from its WAL. After the fix, re-running the (idempotent, truncate-then-load) ETL from the now-consistent DB corrected all rows. Runbook `deploy/postgres-cutover.md` Step 4 updated with the checkpoint step.

**Lesson**: In WAL mode the `.db` file alone is NOT a complete database. Any backup/snapshot/ETL must either checkpoint first, copy all three files together (`.db` + `-wal` + `-shm`), or use `sqlite3 .backup` / `VACUUM INTO`. Cross-cuts with [[decisions/postgres-migration]].

See [[decisions/postgres-migration]] for the full ADR.

## Google Drive delivery

### Large 1080p uploads fail with socket read timeout — and the wrong fix breaks them worse (2026-06-25)

**Problem**: 1080p `final.mp4` deliveries (~45-75 MB) repeatedly failed with `TimeoutError: The read operation timed out`. A 5 MB resumable chunk on a slow uplink to Google exceeds httplib2's short default socket timeout, and `next_chunk(num_retries=...)` only retries HttpError 5xx — NOT socket read timeouts — so the whole delivery failed even though generation + stitch succeeded. Multiple runs (`3b9eec36`, `52516f5a`, `915f97ed`, `288e6c17`, `23bd462a`) hit it.

**REGRESSION — do not "fix" by replacing the http transport**: The first fix attempt built the Drive service with a custom `google_auth_httplib2.AuthorizedHttp(creds, http=httplib2.Http(timeout=300))`. This made it WORSE: a raw `httplib2.Http` mishandles the resumable-upload **308 "Resume Incomplete"** response (treats it as a redirect, finds no `Location` header) → every upload failed with `RedirectMissingLocation: Redirected but the response is missing a Location: header`. The google-built default transport handles 308 correctly; a hand-rolled Http does not.

**Correct fix** (`gdrive_client.upload_file`, verified in production on 44-75 MB runs):
- Keep the **default** service build: `build("drive","v3", credentials=creds, cache_discovery=False)`.
- Widen the socket read timeout ONLY around the upload: `socket.setdefaulttimeout(GDRIVE_HTTP_TIMEOUT_SEC)` then restore the previous value in a `finally`. Scoped to the upload call so it never affects the worker's redis/DB sockets (those are used between jobs, not during an upload).
- Wrap `next_chunk` in a manual retry that catches socket read timeouts and re-calls it — a resumable upload resumes from the last confirmed byte, so no work is lost.

**Recovery without re-billing**: a delivery-only failure leaves `final.mp4` on disk. `process_run` is delivery-only-idempotent (`reuse_final` skips re-stitch when nothing was reprocessed), so a plain retry re-delivers without re-generating or re-stitching. Used to recover all the failed runs above.

**Lesson**: For Google resumable uploads, set timeouts via `socket.setdefaulttimeout` (scoped) — never by swapping the http transport. Always verify an upload-path change against a real large file in production before pushing; unit tests mock the transport and cannot catch the 308/redirect regression.

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

**Fix shipped (TR5b — `feat/orphan-reconciliation`)**: Startup orphaned-run reconciliation. On worker boot, before the worker starts consuming jobs, `reconcile_orphaned_runs()` (in `backend/app/recovery.py`) runs:

1. **Safety gate + distributed lock (race-safety with N workers)**: Check that the RQ default queue AND the StartedJobRegistry are both empty. If any job is in-flight on any worker, skip reconciliation entirely — we never re-enqueue a run that might be actively executing elsewhere. This is conservative (orphan not recovered until next idle restart). In practice, crashes happen during redeploys/OOM kills which drain all workers simultaneously, so the queue is always idle at restart.

   ⚠️ **The queue-idle gate ALONE is NOT sufficient under simultaneous N-worker boot — confirmed in production 2026-06-24.** When the Postgres-cutover deploy started `--scale worker=2`, both workers booted within ~300 ms, both passed the idle gate at the same instant (queue genuinely empty), and **both re-enqueued the same two orphaned runs** → duplicate jobs. Fix shipped: a **Redis lock** (`SET reskin:reconcile:lock NX EX 120` in `reconcile_orphaned_runs`) so exactly one worker reconciles per cold-start window; the others log and skip. The lock is released in a `finally`, with the TTL as a deadlock guard if the holder dies. Regression test: `tests/test_recovery.py::TestSafetyGate::test_skip_when_reconcile_lock_held_by_another_worker`.

2. **Orphan detection**: Query DB for runs in `queued`, `processing`, `stitching`, `delivering`. All four active states are covered.

3. **Re-poll before resubmit (no-rebill)**: For each `processing` run, check `generating` segments' existing `seedance_task_id` against kie.ai. If already `success`, download the result and mark the segment `completed` — avoids a redundant Seedance call (exactly the manual step done for run `72325061` seg 2).

4. **Reset and re-enqueue**: Reset each orphaned run from its active state to `queued` (via the new `processing → queued` state-machine edge added in TR5b), then call `enqueue_process_run`.

5. **State-machine fix**: Added `processing → queued` to `RUN_TRANSITIONS` so the reset path is a valid transition. Previously, re-enqueuing a `processing` orphan threw `InvalidTransition: 'processing' → 'processing'`. `process_run` itself also now detects a non-`queued` run at entry and resets to `queued` before advancing, so it handles any active-state input gracefully.

6. **`/retry` endpoint extended**: `POST /api/v2/runs/{id}/retry` now accepts `queued`, `processing`, `stitching`, and `delivering` in addition to `failed`. Includes a doc comment warning operators not to call it on a genuinely active run (no automatic safety guard — use startup reconciliation for automatic safe recovery).

Projects stuck in `analyzing` are also reset to `failed` (operator must re-trigger analysis from the UI).

See `backend/app/recovery.py`, `backend/app/state_machine.py`, `backend/app/pipeline_v2.py`, `backend/app/api_v2.py`, and [[components/parallel-workers]] for implementation details.
