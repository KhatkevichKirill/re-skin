# Wiki Log

Append-only chronological record of wiki activity.

Format: `## [YYYY-MM-DD] type | description`
Types: `ingest`, `query`, `lint`, `session`, `update`

---

## [2026-06-24] session | Initial wiki setup

Set up LLM Wiki for re-skin following Karpathy's LLM Wiki pattern. Seeded from full project exploration.

**Pages created:**
- `wiki/overview.md` — project overview, current v2 status
- `wiki/architecture.md` — system topology, data model, state machines, storage layout
- `wiki/roadmap.md` — immediate v2 completion tasks + post-v2 growth ideas
- `wiki/decisions/v2-project-runs.md` — ADR for the v2 Project → Runs architecture
- `wiki/components/pipeline.md` — v2 pipeline deep dive (analyze_project, process_run)
- `wiki/models/ai-models.md` — Seedance and Gemini Omni characteristics
- `wiki/lessons/production-gotchas.md` — learnings from production (OOM, drift, docker, alembic)
- `wiki/index.md` — content catalog
- `CLAUDE.md` — wiki schema and dev environment facts

**Sources used:** project file exploration, `tasks/todo.md`, `docs/v2-project-runs.md`, git log, `docker-compose.yml`, `backend/app/*.py`

## [2026-06-24] session | Recovered 2 orphaned runs + dispatched Postgres & parallel-worker work

**Incident:** Runs `72325061` ("Latina woman", stuck `processing`) and `3b9eec36` ("Q-00201 midplan 480p", stuck `queued`) were orphaned by a worker redeploy (~17:23/17:37). RQ queue was empty, one idle worker, DB showed active runs going nowhere.

**Diagnosis & recovery** (see [[production-gotchas]] → "Worker crash leaves runs orphaned"):
- Confirmed root cause: in-flight RQ jobs lost on worker restart; no auto-reconciliation.
- `72325061` seg 2 was `generating` but already `success` on kie.ai — downloaded the result, marked the segment `completed`, reset run `processing→queued`, re-enqueued (saved a Seedance call).
- `3b9eec36` re-enqueued directly (queued→processing valid).
- **Outcome:** both runs reached `done` and delivered to Drive (`72325061`→`1EoGQPhcQlaSk3tNKaWYdrtrJgYrxHlpP`, `3b9eec36`→`1fP_7_sqQaxo6yNr64j42XPvE1cMfNt6Y`). No data lost; no extra Seedance billing beyond `3b9eec36` seg 1 which genuinely needed regenerating.
- Found two recovery gaps: retry endpoint only accepts `failed`; bare re-enqueue on a `processing` run throws `InvalidTransition`.

**Dispatched to subagents (sonnet 4.6), documenting back to wiki:**
- PostgreSQL migration (SQLite → Postgres) — see [[decisions/postgres-migration]] (to be authored by agent).
- Parallel workers (scale throughput; analyze + stitch are the slow phases) — see [[parallel-workers]] (to be authored by agent).

## [2026-06-24] update | SQLite → PostgreSQL migration

Implemented on branch `feat/postgres-migration`. See [[decisions/postgres-migration]] for the full ADR.

**What changed:**
- `docker-compose.yml`: added `db` service (`postgres:16-alpine`, `re-skin-db`, internal-only, `postgres-data` named volume, `pg_isready` healthcheck). `api` and `worker` now `depends_on: db: condition: service_healthy`.
- `backend/requirements.txt`: added `psycopg2-binary==2.9.9`.
- `backend/app/db.py`: Postgres engine options (`pool_pre_ping`, `pool_size=5`, `max_overflow=10`) applied when not SQLite.
- `backend/app/config.py`: documented Postgres URL form in comments.
- `.env.example`: added `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB`, commented-out production `DATABASE_URL`.
- `backend/alembic/versions/`: fixed `(CURRENT_TIMESTAMP)` → `CURRENT_TIMESTAMP` (Postgres rejects parens); fixed `add_column` with Enum types to pre-create the type on Postgres; added migration `a1b2c3d4e5f7` for `run_resolution_enum` `'4k'` value.
- `scripts/sqlite_to_postgres.py`: new ETL script (truncate-then-load, idempotent). Tested against copy of live DB: 250 rows migrated (7+41+14+53+50+85), all counts verified.
- `deploy/postgres-cutover.md`: operator runbook with cutover steps and rollback.
- `wiki/decisions/postgres-migration.md`: this ADR.
- `wiki/lessons/production-gotchas.md`: new Postgres gotchas section (Enum DDL, server defaults, psycopg2 JSON, boolean coercion, enum drift).
- `wiki/architecture.md`: updated topology diagram and migrations section.
- `wiki/index.md`: added ADR to the decisions table.

**Tests:** 390/392 passed (2 fail on missing `ffmpeg` binary — pre-existing environment issue unrelated to this change). Pipeline tests (26) also error on `ffmpeg` not found — same pre-existing issue.

## [2026-06-24] update | PostgreSQL migration empirically verified

Full end-to-end verification run on branch `feat/postgres-migration` against a throwaway `postgres:16-alpine` container. All checks passed.

**Verification results:**

1. **Enum double-create check (suspected bug):** Confirmed NOT a bug. Migrations `d4e7f2a1c890` and `e5a1c7d2f3b6` already correctly call `enum.create(bind, checkfirst=True)` before `op.add_column`. The sa.Enum in `op.add_column` does not attempt to re-create the type when it was pre-created in the same migration context. `alembic upgrade head` ran to completion with zero errors.

2. **Fresh Postgres bootstrap:** `alembic upgrade head` against empty Postgres succeeded through all 7 migrations, reaching head `a1b2c3d4e5f7`. All 6 application tables and 13 enum types created correctly. `run_resolution_enum` includes '4k'.

3. **ETL:** `scripts/sqlite_to_postgres.py` against `/tmp/app_verify.db` → throwaway Postgres: 250 rows inserted (7 jobs + 41 segments + 14 video_projects + 53 segment_defs + 50 runs + 85 run_segments). All source/destination counts matched. JSON columns (`default_reference_image_urls`, `reference_image_urls`) round-tripped correctly. Datetimes preserved exactly. Re-run confirmed idempotency (truncate-then-load, same counts, no FK errors). The "250 rows" claim from the previous agent was accurate.

4. **Test suite:** `pytest tests/` → 392 passed, 4 skipped, 2 failed, 26 errors. All failures and errors are pre-existing `ffmpeg not in PATH` issues (ffmpeg only available inside Docker). Zero DB-related failures. SQLite path unbroken.

5. **Smoke test:** `python3 -c "from app.db import engine; print(engine.url.get_backend_name())"` with Postgres DATABASE_URL printed `postgresql`. `SELECT 1` and table queries succeeded.

**Bugs found:** None. The migration was implemented correctly; no fixes required.

**Cross-link:** [[parallel-workers]] (sibling task, see dispatched-agents note above).
