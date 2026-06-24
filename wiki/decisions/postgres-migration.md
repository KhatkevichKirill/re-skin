---
title: "ADR: SQLite → PostgreSQL Migration"
tags: [architecture, database, postgres, migration, parallel-workers]
sources: [deploy/postgres-cutover.md, backend/app/db.py, backend/alembic/versions/]
updated: 2026-06-24
---

# ADR: SQLite → PostgreSQL Migration

## Status

**Accepted** — implemented on `feat/postgres-migration` branch (2026-06-24).

## Context

re-skin stores all application state (projects, runs, segments, jobs) in a SQLite database at `data/app.db`. SQLite has served well for a single-writer setup but has a hard constraint that blocks the next infrastructure milestone:

**SQLite has a single writer at a time.** Even in WAL mode, concurrent writes from multiple processes queue up on a filesystem lock. Once a second RQ worker is added (planned in [[parallel-workers]]), two workers attempting to write to the DB simultaneously will produce `OperationalError: database is locked`. The timeout can be tuned but not eliminated — SQLite is not designed for multi-process write concurrency.

Additional pressure points:

1. **Worker crash recovery (TR5b)** requires a worker to transactionally update run/segment state on startup while the API is serving requests — a write-on-write race.
2. **Future horizontal scaling** (multiple API processes behind nginx) would also create write contention.
3. **Production data safety:** SQLite is a single file; no connection pooling, no built-in point-in-time recovery, no WAL archiving.

## Decision

Migrate the production database to **PostgreSQL 16** (`postgres:16-alpine`), added as a `db` service in `docker-compose.yml`.

SQLite is retained for the **test suite** (in-memory and file-based engines in `backend/tests/*.py`) and as a supported `DATABASE_URL` dialect at runtime. The app remains compatible with SQLite — the migration changes the *default production DB*, not the driver interface.

## Driver Choice: psycopg2

We use `psycopg2-binary==2.9.9` (pinned, compatible with SQLAlchemy 2.0.23).

**Why not asyncpg?** The application uses SQLAlchemy's synchronous session API throughout (FastAPI with sync sessions, RQ workers are blocking). asyncpg requires the async SQLAlchemy API. Switching drivers would require a larger refactor. psycopg2 is the default synchronous Postgres driver for SQLAlchemy 2.x and is the right choice here.

**Why not pg8000?** Pure-Python, slower than psycopg2 (C extension). No benefit for this workload.

## JSON vs JSONB

**Decision: keep `JSON`, do not change to `JSONB`.**

Rationale:
- `JSON` stores the exact text representation; `JSONB` stores a binary representation that enables GIN indexing and containment operators (`@>`, `<@`, `?`).
- re-skin never queries *inside* JSON columns (reference image URLs, segment override payloads). They are always read and written as opaque blobs.
- No `WHERE reference_image_urls @> '["..."]'` queries exist or are planned.
- Changing to `JSONB` would require a new Alembic migration touching 5 columns across 3 tables, with `ALTER COLUMN ... USING column::jsonb`. The risk/reward is negative.
- If JSON querying becomes necessary (e.g. filtering runs by reference image), a future migration can add a `JSONB` copy column or use a generated column.

**Consequence:** JSON column performance on Postgres is equivalent to SQLite for our use case (no change in query patterns).

## Enum Handling

Postgres creates native `ENUM` types (`CREATE TYPE name AS ENUM (...)`) when SQLAlchemy's `sa.Enum(..., name='...')` is used in DDL. SQLite stores the same as `VARCHAR` with no enforcement.

Two migration-layer bugs fixed in this migration:

1. `(CURRENT_TIMESTAMP)` in `sa.text(...)` — the extra parentheses are SQLite-specific. Postgres rejects them. Fixed to `CURRENT_TIMESTAMP` (valid on both).
2. `op.add_column` with a named Enum type fails on Postgres if the type doesn't exist yet — Alembic does not auto-create the type for `add_column` (unlike `create_table`). Fixed by calling `enum.create(bind, checkfirst=True)` before the `add_column` in migrations `d4e7f2a1c890` and `e5a1c7d2f3b6`.
3. `run_resolution_enum` was created with only `('480p', '720p', '1080p')` but the model added `'4k'` for Gemini Omni without a matching migration. Fixed in new migration `a1b2c3d4e5f7` which runs `ALTER TYPE run_resolution_enum ADD VALUE IF NOT EXISTS '4k'` on Postgres.

## Connection Pooling

Added Postgres-specific engine kwargs in `backend/app/db.py` (guarded by `not _is_sqlite`):

```python
pool_pre_ping=True   # detect stale connections after Postgres restart
pool_size=5          # persistent connections per process
max_overflow=10      # burst capacity
```

With 1 API process + 1 worker: peak 30 connections, well within Postgres 16 defaults (max_connections=100). When multiple workers are added ([[parallel-workers]]), each adds up to 15 connections — operator should bump `max_connections` if running >5 workers.

## Data Migration

One-time ETL script: `scripts/sqlite_to_postgres.py`. Copies all 6 tables in FK-safe order. Safe to re-run (truncate-then-load). Verified against live DB copy: 250 rows (7 jobs, 41 segments, 14 projects, 53 segment_defs, 50 runs, 85 run_segments) migrated with all counts matching.

SQLite-to-Postgres type coercions needed:
- `has_face` (BOOLEAN): SQLite stores as INTEGER 0/1 → coerce to Python `bool`
- JSON columns: wrap in `psycopg2.extras.Json` to prevent psycopg2 from treating Python lists as `text[]` Postgres arrays

See [[production-gotchas]] for the full list of gotchas hit during this migration.

## Alternatives Considered

| Alternative | Rejected because |
|-------------|-----------------|
| SQLite WAL + busy_timeout tuning | Not viable for persistent multi-writer concurrency; timeout just delays the lock error |
| SQLite in shared-cache mode | Deprecated, Linux-only, still single-writer |
| CockroachDB / Spanner | Massive operational complexity for a single-VPS deployment |
| MySQL/MariaDB | No advantage over Postgres for this workload; SQLAlchemy Enum DDL has similar quirks |

## Consequences

**Positive:**
- Eliminates `database is locked` errors for multi-worker deployments
- Enables proper connection pooling (`pool_pre_ping` catches Postgres restarts)
- Sets the foundation for [[parallel-workers]]
- Postgres has mature backup tooling (pg_dump, WAL archiving, pgbackrest)

**Negative / Risks:**
- Added operational complexity: Postgres container must be healthy before api/worker start
- ETL script must be run once during cutover — operator must follow [[deploy/postgres-cutover]] runbook
- Tests continue on SQLite (different engine semantics); any future DDL must be tested on both

## Related

- [[parallel-workers]] — the primary motivation for this migration
- [[production-gotchas]] — Alembic DDL gotchas, enum migration, JSON coercion
- [[lessons/production-gotchas]] → "Alembic stamp on new deploy" section
- `deploy/postgres-cutover.md` — step-by-step cutover runbook
