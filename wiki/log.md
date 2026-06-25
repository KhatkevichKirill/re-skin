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

## [2026-06-24] update | TR5b orphaned-run reconciliation on worker startup

Implemented on branch `feat/orphan-reconciliation` (stacked on `feat/parallel-workers` → `feat/postgres-migration`).

**What changed:**
- `backend/app/recovery.py` (new): `reconcile_orphaned_runs()` — startup reconciliation routine. Queue-idle safety gate (no reconciliation if StartedJobRegistry is non-empty), re-poll of generating segments' kie task IDs before resubmission, reset active runs to `queued`, re-enqueue. Also resets orphaned `analyzing` projects to `failed`.
- `backend/app/state_machine.py`: Added `processing → queued` to `RUN_TRANSITIONS` so orphaned-run reset is a valid transition.
- `backend/app/pipeline_v2.py`: `process_run` now detects non-`queued` entry status and resets to `queued` before advancing (handles any orphan state cleanly). Submit phase re-polls existing `seedance_task_id` before resubmitting — marks segment `completed` if task is already `success` on kie.ai (no rebilling).
- `backend/app/api_v2.py`: `/runs/{rid}/retry` now accepts `queued`, `processing`, `stitching`, `delivering` in addition to `failed` (for manual operator intervention on stuck runs).
- `worker/worker.py`: Calls `reconcile_orphaned_runs()` on startup before `worker.work()`.
- `backend/tests/test_recovery.py` (new): 19 passing tests covering safety gate, orphan detection, re-poll/no-rebill, state-machine fix, and process_run orphan resume.
- `backend/tests/test_api_v2.py`: Updated `TestRetryRun` — old test asserting `processing` returns 409 is replaced with test asserting `done` returns 409 (correct non-retryable status); added 2 tests for `processing` and `queued` retry (both should return 200).
- `wiki/lessons/production-gotchas.md`: Updated "Worker crash leaves runs orphaned" from "Planned fix" to "Fix shipped".
- `wiki/roadmap.md`: TR5b marked DONE.
- `wiki/components/parallel-workers.md`: New "Startup Orphaned-Run Reconciliation" section documenting the implementation and race-safety reasoning.
- `wiki/components/pipeline.md`: New "Resume / No-Rebill Behavior" section.

**Tests:** 416 passed, 4 skipped, 2 failed (pre-existing ffmpeg), 29 errors (pre-existing ffmpeg) — 21 more passes than the `feat/parallel-workers` baseline (395 passed).

## [2026-06-24] update | Parallel workers for throughput

Implemented on branch `feat/parallel-workers` (branched from `feat/postgres-migration`, commit on top of `d00ae34`).

**What changed:**
- `docker-compose.yml`: removed `container_name: re-skin-worker` (enables `--scale worker=N`); reduced per-worker limits to `cpus=2.0 / mem_limit=6g / memswap_limit=7g` (fits 2 workers on 4 vCPU / 16 GiB); added `STITCH_CUT_CONCURRENCY=2`; `FFMPEG_THREADS=2`.
- `backend/app/pipeline_v2.py`: replaced serial `cut_clip` loop in stitch assembly with `ThreadPoolExecutor(max_workers=STITCH_CUT_CONCURRENCY)`; added `STITCH_CUT_CONCURRENCY` env constant; futures stored in submission order so `clip_paths` stays correctly ordered.
- `backend/tests/test_pipeline_v2.py`: added `TestStitchCutConcurrency` (3 tests, ffmpeg-gated like all pipeline tests) and `TestParallelStitchOrdering` (3 pure-Python tests, no ffmpeg — all pass on host).
- `wiki/components/parallel-workers.md`: new page (this is the `[[parallel-workers]]` page referenced from several existing wiki pages).
- `wiki/architecture.md`: updated worker topology to "RQ Worker ×N", updated resource limits.
- `wiki/roadmap.md`: updated Multi-worker to DONE, added TR-POLL (poll-decoupling) and TR5b urgency note.
- `wiki/index.md`: added parallel-workers component page.

**Verification:**
- 4 × 2s sleep jobs on 2 workers: all done in 4.63s (vs ~8s serial). Jobs 1+2 completed simultaneously at 2.48s — serialization eliminated.
- `pytest tests/`: 395 passed, 4 skipped, 2 failed, 29 errors — 3 more passes than baseline (new TestParallelStitchOrdering). All failures/errors pre-existing ffmpeg-not-found.

**Ops note:** Branch depends on `feat/postgres-migration` — must be deployed together with Postgres. Poll-decoupling (TR-POLL) proposed but not implemented; see [[components/parallel-workers]].

## [2026-06-24] update | Production cutover to PostgreSQL + 2 workers (deployed & pushed)

Deployed the stacked work (Postgres → parallel workers → TR5b) to the live host and pushed to `origin/main` (commits `1e360a6`..`55142f6`, rebased onto the public-repo head `3166ae8`).

**Cutover sequence:** backup SQLite → set POSTGRES_* in `.env` → `docker compose build` → `up -d db` (healthy) → `alembic upgrade head` (one-off container, head `a1b2c3d4e5f7`) → ETL → switch `DATABASE_URL` → `up -d --scale worker=2`. Final: api on Postgres, `worker-1`/`worker-2`, 47 done + 3 failed runs, 14 projects, HTTP 200 via nginx.

**Two production findings (both fixed, see [[production-gotchas]]):**
1. **SQLite WAL not in `cp app.db`** — first ETL snapshot was stale (two `done` runs came across as active), and TR5b then re-enqueued them. Fixed by re-ETL from the now-checkpointed live DB; runbook + lesson updated.
2. **TR5b reconcile double-enqueue race** — both workers cold-started, both passed the idle gate, both re-enqueued the same runs. Fixed with a Redis lock (`SET NX EX 120`) in `reconcile_orphaned_runs` (commit `55142f6`); regression test added.

**Env note:** deploy host has `docker compose` v2 only (no `docker-compose` v1). After recreating `api`, `nginx` must be restarted to re-resolve the upstream IP (stale-upstream 502 — hit and fixed live).

## [2026-06-24] update | Copy run with a new reference photo

Extended `POST /api/v2/runs/{id}/copy` (was resolution-only) so a run can be copied with a **new reference photo** (`reference_files` / `reference_urls`) — the "project as a template" workflow: tune a run once, re-run it on a new face of the same type. New photo **replaces the photo everywhere** (run-level refs + per-segment reference overrides dropped; per-segment prompt overrides kept). Resolution is now optional (defaults to the source run's). Copy form in `run_detail.html` gained a photo URL + upload field. Tests: 4 added to `TestCopyRun` (11 total pass). Branch `feat/copy-run-new-reference`. See [[decisions/v2-project-runs]] → "Run operations".

## [2026-06-25] update | Fix: Google Drive delivery timeout on 1080p (+ regression)

Large 1080p deliveries (~45-75 MB) failed with socket read timeouts; `next_chunk(num_retries=)` doesn't retry those. Fixed `gdrive_client.upload_file` to widen the socket timeout (scoped `setdefaulttimeout`, `GDRIVE_HTTP_TIMEOUT_SEC=300`) + manual chunk-timeout retry. First attempt regressed (custom `httplib2.Http` broke resumable 308 handling → `RedirectMissingLocation`); corrected to keep the default google transport. Verified in production: runs `3b9eec36`, `52516f5a`, `915f97ed`, `288e6c17`, `23bd462a` all delivered to `done`. Commits `92d5159` (initial), `a2be146` (regression fix). See [[production-gotchas]] → "Google Drive delivery".

## [2026-06-25] update | Reliability-first v2 throughput controls

Implemented bounded v2 fresh-submit concurrency and operational guardrails:
`SUBMIT_CONCURRENCY` overlaps swap-segment cut/upload/create-task work while
keeping per-worker resource pressure bounded; `run_process_run` now uses a
Redis per-run lock to prevent duplicate RQ jobs from double-submitting paid AI
tasks; worker startup masks Redis credentials in logs; API uploads stream to
disk with `MAX_UPLOAD_SIZE_MB` aligned to nginx's 1 GiB body limit. `process_run`
now logs phase timings for reference resolution, submit, poll, stitch, delivery,
and total runtime. See [[components/pipeline]], [[components/parallel-workers]],
and [[production-gotchas]].

## [2026-06-25] update | Analyze/poll/stitch micro-optimizations

Implemented low-risk service-speed fixes without changing AI semantics:
`detect_timeline` now uses OpenCV `grab()` for skipped frames and `retrieve()`
only for sampled frames; the v2 poll loop no longer holds a DB session while
sleeping between external task polls; `stitch(audio_mode="seedance")` reuses
the first `ffprobe` result for no-audio clip silence duration; SQLite
connections now set `synchronous=NORMAL` and `busy_timeout=5000` alongside WAL
and foreign keys. Stream-copy keep/fallback cuts were left as a documented
follow-up because global `-c copy` risks non-frame-accurate boundaries. See
[[components/pipeline]], [[components/parallel-workers]], and
[[production-gotchas]].
