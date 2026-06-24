# PostgreSQL Cutover Runbook

Operator steps to cut production from SQLite to PostgreSQL.

**Branch:** `feat/postgres-migration`  
**Schema head:** `a1b2c3d4e5f7` (add_4k_to_run_resolution_enum)  
**ETL script:** `scripts/sqlite_to_postgres.py`

---

## Pre-flight Checks

Before starting:
- Confirm no active runs are in-flight (`runs` table: no rows with `status IN ('queued','processing','stitching','delivering')`).
- Confirm `docker-compose` v1.29.2 is available (`docker-compose --version`).
- Confirm you are on the `feat/postgres-migration` branch (or it has been merged to `main`).
- Back up the SQLite DB: `cp data/app.db data/app.db.pre-pg-$(date +%Y%m%d-%H%M%S)` (keep it on host — do NOT commit to git).

---

## Step 1: Set environment variables

Edit `.env` (never commit this file):

```bash
# Add / uncomment Postgres variables
POSTGRES_USER=reskin
POSTGRES_PASSWORD=<generate with: openssl rand -hex 24>
POSTGRES_DB=reskin

# Switch the app URL (comment out the old sqlite line)
#DATABASE_URL=sqlite:///./data/app.db
DATABASE_URL=postgresql+psycopg2://${POSTGRES_USER}:${POSTGRES_PASSWORD}@db:5432/${POSTGRES_DB}
```

> **Note:** `docker-compose.yml` uses `${POSTGRES_USER:?}` — the `db` service will fail to start if these are not set.

---

## Step 2: Start the Postgres container

Start only the `db` service (do NOT restart api/worker yet):

```bash
docker-compose up -d db
```

Wait for it to be healthy:

```bash
docker-compose ps db
# Should show: Up (healthy)
# Or poll manually:
until docker-compose exec db pg_isready -U "$POSTGRES_USER" -d "$POSTGRES_DB"; do sleep 2; done
```

---

## Step 3: Apply migrations

Run alembic inside the api container (or on the host if Python env is available):

```bash
# Option A: using the running api container (if still pointing to sqlite, you must
# temporarily pass the PG URL; easier to run on the host):

DATABASE_URL="postgresql+psycopg2://${POSTGRES_USER}:${POSTGRES_PASSWORD}@localhost:55432/${POSTGRES_DB}" \
  python3 -m alembic upgrade head
```

> **IMPORTANT:** This is a fresh Postgres DB — do NOT run `alembic stamp head`.
> `alembic upgrade head` builds the schema from scratch via all migrations.
> The `alembic stamp head` trick from [[production-gotchas]] applies only when
> the schema already exists (e.g. a dump was restored); for a net-new DB, run
> `upgrade head`.

Expected output: 7 lines `Running upgrade ... -> ...`  
Final revision: `a1b2c3d4e5f7`

---

## Step 4: Run the ETL

Copy the live SQLite DB (never open it for writing):

```bash
cp data/app.db /tmp/app_etl_$(date +%Y%m%d).db
```

Run the migration script:

```bash
python3 scripts/sqlite_to_postgres.py \
  --source "sqlite:////tmp/app_etl_$(date +%Y%m%d).db" \
  --target "postgresql+psycopg2://${POSTGRES_USER}:${POSTGRES_PASSWORD}@localhost:55432/${POSTGRES_DB}"
```

Verify output shows all row counts matching. Expected approximate counts:
- `jobs`: 7, `segments`: 41, `video_projects`: 14, `segment_defs`: 53, `runs`: 50, `run_segments`: 85
  (actual counts may be higher if more runs have completed since June 2026).

The ETL is **idempotent** — safe to re-run. It truncates then reloads.

---

## Step 5: Restart api + worker

```bash
docker-compose stop api worker
docker-compose up -d api worker
```

This picks up the new `DATABASE_URL` from `.env` pointing to Postgres.

> **Do not** use `docker-compose up -d --build` unless you also want to rebuild images.
> A plain `up -d` recreates containers with the new env without rebuilding.

---

## Step 6: Verify

```bash
# Check services are all healthy
docker-compose ps

# Tail api logs for DB connection errors
docker-compose logs --tail=50 api

# Confirm alembic version in PG
docker-compose exec db psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" \
  -c "SELECT * FROM alembic_version;"
# Expected: a1b2c3d4e5f7

# Quick smoke test: list projects
curl -s -u "${BASIC_AUTH_USER}:${BASIC_AUTH_PASS}" http://localhost:8847/api/v2/projects | python3 -m json.tool | head -20
```

---

## Rollback

If anything goes wrong:

1. Stop api + worker:
   ```bash
   docker-compose stop api worker
   ```

2. Restore SQLite URL in `.env`:
   ```bash
   # Comment out PG line, uncomment SQLite line
   DATABASE_URL=sqlite:///./data/app.db
   ```

3. Restart with SQLite:
   ```bash
   docker-compose up -d api worker
   ```

4. The `db` container can be left running — it won't interfere with SQLite.
   Stop it if desired: `docker-compose stop db`.

5. SQLite data is unchanged — the ETL script never modifies the source DB.

---

## Notes

- **Port exposure:** The `db` service is internal-only (`expose: 5432`, no `ports:`). To connect from the host during maintenance, temporarily add `ports: ["127.0.0.1:55432:5432"]` to the `db` service and remove it after.
- **Connection pool:** api and worker each open a pool of up to 15 connections (`pool_size=5 + max_overflow=10`). With 1 api + 1 worker, peak is 30 connections — well within Postgres 16 defaults (max_connections=100).
- **SQLite fallback:** The app still supports SQLite (`DATABASE_URL=sqlite:///...`). The test suite always uses SQLite in-memory engines and is unaffected by this change.
- See [[decisions/postgres-migration]] for the ADR, and [[production-gotchas]] for migration gotchas encountered.
