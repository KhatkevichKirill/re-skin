# re-skin — Code Review & Bottleneck Report

> **STATUS: CLOSED — every finding below has shipped.** Archived 2026-08-09 as a
> historical record; do not treat any "current behaviour" claim here as live.
>
> - **B1–B4** → wave 4, merged to `main` via PR #1 (`4f70703`), migration `9230657199c5`.
> - **B5–B13** → wave 5 (`perf/bottlenecks-wave5`), deployed 2026-07-30, migration
>   `08433b4098d9`. Merged into `main` since; verified 2026-08-09.
>
> What was actually done, with commit hashes and prod verification, is in
> `tasks/todo.md` (§ Wave 4 / Wave 5). This file is kept for the *analysis* —
> the severity ranking, the two bottleneck themes in §2, and §4's inventory of
> what was already sound — none of which the task log captures.
>
> Two references in the original text are now stale: the untracked scratch
> script named in §1 and §6 was deleted 2026-08-09, and `origin/perf/bottlenecks`
> is long merged.

**Date:** 2026-07-24
**Reviewer:** Claude (read-only review; no code was modified)
**Scope:** `backend/app/`, `worker/`, `deploy/`, `docker-compose.yml`, `backend/alembic/`
**Branch reviewed:** `main` (commit `7129b0f`)

---

## 1. Git comparison vs the private upstream repository

The local `main` is **identical to `origin/main`** on GitHub:

```
$ git log --oneline origin/main..main   # (empty)
$ git log --oneline main..origin/main   # (empty)
```

The only working-tree divergence is one untracked scratch script:
`scripts/_rerun_seg1_e5c5682f.py` (not committed, not part of the review).

So **the code reviewed below is exactly what is on GitHub.**

### Important context: a `perf/bottlenecks` branch already exists on origin

`origin/perf/bottlenecks` (3 commits ahead of `main`, **not merged**) already
addresses many of the bottlenecks found here. Its commit messages:

- **Wave 1** `54a73ab` — kie upload retry seek fix; fast-intermediate keep cuts
  + reuse; nginx X-Accel-Redirect media offload; gzip; DB indexes on
  `runs.status` and `run_segments.segment_def_id`.
- **Wave 2** `c5a9604` — split `process_run` into staged submit → self-
  re-enqueueing poll → finalize jobs; overlay/analyze/process queue
  separation; pre-warm InsightFace; cached Redis conn/queues; token-checked
  lock TTL refresh.
- **Wave 3** `4f3f60a` — crash-resume clip checkpoints; concurrent poll-round
  downloads; concurrent reference resolution; batch-copy offloaded to
  threadpool; dashboards bounded to newest 100; Sheets export moved to RQ.

Throughout this report, items already covered by that branch are marked
**[perf-branch: done]**; items it does **not** address are marked
**[perf-branch: NOT addressed]** — those are the highest-value next targets.

> Note: the `main` working tree already contains Waves 1–3 (the code I read
> reflects them). So "[perf-branch: done]" means "the fix exists on the
> unmerged branch and is also present in main's current code"; the branch
> itself simply hasn't been merged via a PR.

## 2. Executive summary

The architecture is solid and clearly battle-tested: staged RQ jobs with a
per-run Redis lock, crash-resume checkpoints, per-run liveness reconciliation,
nginx media offload, bounded dashboards, and concurrent submit/poll/download.
Most "obvious" bottlenecks have already been removed.

The remaining bottlenecks cluster around **two themes**:

1. **The HTMX status-fragment poll path (every 2 s) does too much per tick.**
   The single worst item: `run_status_fragment` shells out to `ffprobe` on
   every 2 s poll for any done run being viewed, plus runs 3–4 extra DB
   queries and an unbounded `library_clips` scan each tick. This is the
   highest-impact, lowest-effort fix. **[perf-branch: NOT addressed]**

2. **Unbounded `SELECT *` / full-column reads in a few hot paths.**
   `library_clips`, `overlay_assets`, Sheets dedupe columns, and the
   automation tick's full-sheet + full-launch-set reads grow linearly with
   history and never prune. **[perf-branch: NOT addressed]**

Plus a real **resource leak** (`KieClient` per poll round, never closed) and a
handful of missing indexes / minor issues.

Severity legend: 🔴 High · 🟠 Medium · 🟡 Low

## 3. Bottlenecks (by severity)

### 🔴 B1. `ffprobe` spawned on every 2 s status poll for done runs
**File:** `backend/app/web_v2.py` — `run_status_fragment` (L1109) → `_get_tail_edit_context` (L330) → `media_mod.probe(run.result_local_path)` (L343)

`run_status_content.html` self-polls with `hx-trigger="every 2s"`. Each poll,
for any run in `done` status with a result on disk, calls `media_mod.probe()`,
which **forks an `ffprobe` subprocess** to read the result video's duration.
There is no caching and no `Run.result_duration_sec` column.

Impact: while an operator watches a finished run page, the worker/API spawns
an `ffprobe` process every 2 s indefinitely. With several run pages open (or
several operators) this is a steady subprocess storm on the API container
(mem_limit 2 g). The duration is immutable once the run is done — it should
be probed once at finalize and stored on the `Run` row.

**[perf-branch: NOT addressed]**

Suggested fix (no code changed here): add `result_duration_sec` to `Run`,
populate it once in `finalize_run` after stitch, and have
`_get_tail_edit_context` read the column instead of probing.

---

### 🔴 B2. `run_status_fragment` runs 3–4 extra queries every 2 s poll
**File:** `backend/app/web_v2.py` — `run_status_fragment` (L1109)

Every poll unconditionally calls:
- `_get_run_segments` (1 query, selectinload — fine),
- `_get_insert_merge_context` → `InsertExtraction` lookup by `child_project_id`
  (1 query), and when an extraction exists, a second query loading up to 100
  candidate target projects (L221–231),
- `_get_tail_edit_context` → `_get_tail_edits` (1 query), `_get_library_clips`
  (1 unbounded `SELECT * FROM library_clips ORDER BY created_at`), and
  `_last_segment_start_sec` (1 query) — **all on every 2 s tick**, regardless
  of whether the run is done / has an extraction / has tail edits.

The merge-targets dropdown and the tail-edit form are only meaningful once the
run is `done`; they are being rebuilt every 2 s for `processing`/`stitching`
runs that will never render them.

**[perf-branch: NOT addressed]**

Suggested fix: gate the merge/tail-edit context construction on
`status_val == "done"` (the templates already gate rendering on status); cache
`library_clips` at module level with a short TTL instead of querying each poll.

---

### 🔴 B3. `KieClient` created per poll round and never closed (httpx pool leak)
**File:** `backend/app/pipeline_v2.py` — `poll_run_once` (L1588) `kie = _default_kie()`

`_default_kie()` returns a brand-new `KieClient()`, whose constructor opens an
`httpx.Client(timeout=60)` connection pool. `poll_run_once` is re-enqueued
every `RUN_POLL_INTERVAL_SEC` (default 15 s) for the whole life of a run. Each
round builds a new client and **never calls `.close()`**. A 2 h run with 2 h
skip-timeout ≈ 480 poll rounds → 480 abandoned `httpx.Client` pools per run,
held until GC (and CPython's GC may not promptly collect them because the
underlying `httpx.Client` holds no `__del__` that frees sockets eagerly).

`kie_client.get_shared_client()` already exists (thread-safe singleton) and
is used by the credit-balance path, but `pipeline_v2` ignores it.

Same pattern (less severe, once per phase) in `submit_run` (L1525) and
`process_run` (L1699).

**[perf-branch: NOT addressed]** (Wave 1 fixed the *upload* retry seek bug and
added `close()`/shared-client support, but `pipeline_v2` was not switched to
the shared client.)

Suggested fix: use `get_shared_client()` in `poll_run_once`/`submit_run`, or
`with KieClient() as kie:` so the pool is released each round.

---

### 🟠 B4. Unbounded `SELECT *` on `library_clips` and `overlay_assets`
**Files:** `backend/app/web_v2.py` — `_get_library_clips` (L252), `_get_overlay_library_assets` (L261); `backend/app/api_v2.py` — `list_library_clips` (L814)

All four do `SELECT * FROM library_clips ORDER BY created_at DESC` with **no
LIMIT**. They are called from `project_status_fragment` (when ready),
`run_status_fragment` (via tail-edit context, every 2 s — see B2), the library
page, and the `GET /library-clips` API. As the stitch library grows, every
dashboard poll pays a full-table scan.

**[perf-branch: NOT addressed]** (dashboards were bounded to 100, but the
library queries were not.)

Suggested fix: `.limit(100)` and/or cache the library list with a short TTL.

---

### 🟠 B5. Sheets export dedupe reads an entire column every append
**File:** `backend/app/sheets_client.py` — `column_values` (L129); used in `tasks.py` `run_export_*_to_sheet` (L770, L849, L987) and `_append_caption_demo_row` (L908)

Every export job does `get_header` → `column_values` (reads the **whole Job-ID
column** for the entire sheet) → linear `any(...)` scan → `append_row`. Three
HTTP round-trips plus an O(rows) scan, per export. As the "Final Creatives" /
"Caption Demo" tabs grow over months, each export gets slower and heavier.

These are now correctly run as RQ jobs on the fast queue (Wave 3), so they no
longer block the request — but the unbounded column read itself is still a
latent scaling cliff.

**[perf-branch: NOT addressed]**

Suggested fix: dedupe via the DB flag (`final_creatives_exported_at` /
`caption_demo_status`) which is already the fast path; the sheet-side dedupe is
belt-and-suspenders and could be skipped once the DB flag is set, or bounded to
the last N rows.

---

### 🟠 B6. Automation tick: full-sheet read + full-launch-set scans every 5 min
**File:** `backend/app/automation.py` — `run_tick` (L467) → `_launch_batch` (L359)

Each tick (default `AUTOMATION_TICK_INTERVAL_SEC = 300`), for **every enabled
automation**:
1. `sheets.read_rows` reads the **entire** "Daily Queue" tab `A2:Y` (no
   pagination) — repeated once per automation, not cached across automations
   sharing the same sheet.
2. `used` = `SELECT queue_id FROM automation_launches WHERE automation_id=…`
   — full scan of that automation's launch history.
3. `existing_names` = `SELECT name FROM runs WHERE project_id=…` — full scan
   of all runs in the project.
4. `launched_today` is a separate `SELECT` per automation (and is called again
   in the log line after the launch).

With N automations and a long-lived sheet/launch history, this is O(automations
× sheet_rows + automations × launches) every 5 min. The sheet read in particular
is repeated N times for the same spreadsheet.

**[perf-branch: NOT addressed]**

Suggested fix: read the sheet once per tick (cache across automations in the
same `run_tick` call); add an index or a "today" bound to the
`automation_launches.launched_at` query (the column is already indexed, but the
queue_id scan is unbounded); compute `launched_today` once and reuse.

---

### 🟠 B7. Missing `created_at` indexes on listing tables
**File:** `backend/alembic/versions/e6f7a2b3c4d5_add_missing_indexes.py`

Wave 1 added `ix_runs_status` and `ix_run_segments_segment_def_id`. But every
listing endpoint orders by `created_at DESC`:

| Table | Query site | Indexed? |
|-------|-----------|----------|
| `video_projects.created_at` | projects dashboard, merge-targets | ❌ |
| `runs.created_at` | runs-fragment, list_project_runs | ❌ (filtered by indexed `project_id` first, so OK) |
| `library_clips.created_at` | library page, `_get_library_clips` | ❌ |
| `overlay_assets.created_at` | library page | ❌ |
| `overlay_jobs.created_at` | overlay dashboards | ❌ |
| `insert_merges.created_at` | merges-fragment | ❌ |
| `run_tail_edits.created_at` | tail-edits-fragment | ❌ |

For `video_projects` (the main dashboard) and `library_clips`/`overlay_assets`
(the unbounded library queries), `ORDER BY created_at DESC LIMIT N` currently
does a full-table sort. At current scale this is cheap; as history grows it
becomes a seq scan + sort per dashboard load.

**[perf-branch: NOT addressed]**

Suggested fix: add `created_at` indexes for `video_projects`,
`library_clips`, `overlay_assets`, `overlay_jobs`, `insert_merges`,
`run_tail_edits`.

---

### 🟠 B8. `stitch` is one monolithic ffmpeg `filter_complex` over all clips
**File:** `backend/app/media.py` — `stitch` (L327)

For N clips, `stitch` builds a single `filter_complex` with N `scale/pad/
setsar/fps` chains and one `concat`. In `"seedance"` audio mode it additionally
**probes every clip** (`probe(clip)` in a Python loop, L431) and adds one
`anullsrc` lavfi input per audio-less clip. So a 30-segment stitch = 30 probe
subprocesses + a 30-input ffmpeg with a 30-chain filter graph, all in one job.

Consequences:
- One OOM (rc=-9) kills the **entire** stitch — no partial progress, the whole
  run fails and must retry from the start.
- The `"original"` audio path re-encodes **every** clip (including untouched
  keep segments) with `libx264 -preset fast -crf 18`. For a video that is
  mostly keep segments this is pure CPU waste on the 2-vCPU, 6 g worker.
- `mux_original_audio` (custom-plan path) calls `probe(video)` per swap segment
  for its duration — another N probes inside the stitch phase.

**[perf-branch: NOT addressed]** (Wave 1 added the `-threads` cap and
fast-intermediate keep cuts, which bound memory, but the monolithic graph and
the per-clip re-encode remain.)

Suggested fix: store each result clip's duration in the DB (already probed at
download) and pass it to `stitch` to avoid N probes; consider a concat-demuxer
(`-c copy`) pass for the common case where all clips already share the target
params, falling back to re-encode only when codecs differ.

---

### 🟡 B9. `delete_library_clip` scans every active run's `stitch_plan` in Python
**File:** `backend/app/api_v2.py` — `delete_library_clip` (L840, L853–867)

Loads `(Run.id, Run.stitch_plan)` for **all** runs in `_STITCH_ACTIVE_RUN_STATUSES`
and walks each plan's JSON in Python to test membership of `clip_id`. O(active
runs) Python work on every clip delete. Fine today; becomes a linear scan as the
run count grows. A Postgres JSON-containment predicate (`stitch_plan @> …`) would
push it into the DB. Low frequency (deletes are rare), hence 🟡.

**[perf-branch: NOT addressed]**

---

### 🟡 B10. `_frame_locks` dict grows unbounded
**File:** `backend/app/api_v2.py` — `_frame_locks` (L144)

Per-path `threading.Lock` objects are created on demand for every distinct
frame cache path (`frame_<t_tenths>.jpg`) and **never removed**. Each project
has its own frames dir, so the dict accumulates one entry per distinct
requested timestamp across all projects for the lifetime of the API process.
Minor memory leak (locks are small), but on a long-lived API with many projects
and many scrub positions it is unbounded.

**[perf-branch: NOT addressed]**

Suggested fix: use a `weakref`-based lock map, or prune entries whose cache file
already exists when the lock is uncontended.

---

### 🟡 B11. `get_db` dependency + manual `db.close()` interaction
**File:** `backend/app/api_v2.py` — `get_project_source` (L658), `get_library_clip_file` (L837), `get_run_result` etc.

To release the DB connection before streaming a file, these endpoints call
`db.commit(); db.close()` on the dependency-injected session, then return.
`get_db` wraps `get_session()`, whose context manager on exit runs
`session.commit()` (then `session.close()` in `finally`). After the endpoint
already closed the session, the post-response `session.commit()` runs on a
closed session and raises `InvalidRequestError`/`StatementError`, which
`get_session`'s `except` turns into a `rollback()`+`raise`. The HTTP response
has already been sent (or is an X-Accel-Redirect header response), so the user
sees nothing, but the API logs a traceback **per media request** — noise that
can mask real errors and costs a tiny bit of CPU per request.

**[perf-branch: NOT addressed]**

Suggested fix: give these streaming endpoints their own session lifecycle
(`with get_session() as db:` inside the handler, no `Depends(get_db)`) instead
of closing the dependency session early.

---

### 🟡 B12. `reconcile_orphaned_runs` boot lock TTL (120 s) vs pre-poll cost
**File:** `backend/app/recovery.py` — `reconcile_orphaned_runs` (L308), `_repoll_generating_segments` (L207)

The boot reconciliation lock is held for `_RECONCILE_LOCK_TTL_SEC = 120` s.
For each orphaned `processing` run, `_repoll_generating_segments` calls
`kie.get_task()` **serially** for every `generating` segment. A run with many
segments × many orphaned runs × slow kie.ai latency can exceed 120 s, at which
point the lock expires and a second booting worker can start a parallel
reconcile. The per-run liveness check prevents double-*processing*, so this is
not a correctness bug, but it can cause duplicate `enqueue_process_run` calls
(the run lock in `run_submit_run` dedupes them safely). 🟡.

**[perf-branch: NOT addressed]**

---

### 🟡 B13. `_live_run_ids` does O(jobs) `Job.fetch` per reconcile tick
**File:** `backend/app/recovery.py` — `_live_run_ids` (L150)

Every 180 s tick (and at boot) iterates all pending/scheduled/started job ids
across 4 queues and calls `Job.fetch(jid)` for each — one Redis round-trip per
job. With a large backlog (e.g. many queued overlays or a long scheduled-poll
registry) this is O(backlog) round-trips per tick. Fine at current scale; worth
knowing if the process queue ever grows large.

**[perf-branch: NOT addressed]**

---

## 4. What is already done well (no action needed)

Verified as already implemented in `main` (via the unmerged `perf/bottlenecks`
branch) — **not** bottlenecks:

- **Staged job lifecycle** — `process_run` split into `submit_run` →
  `run_poll_run` (self-re-enqueueing via `enqueue_in`, state in Redis) →
  `run_finalize_run`. Workers never sleep-blocked for hours. (`tasks.py`)
- **Queue priority** `overlay` → `analyze` → `process`. (`worker/worker.py`)
- **InsightFace pre-warm** in worker bootstrap (COW inheritance).
- **Per-run Redis lock** with token-checked release + TTL refresh per poll.
- **Crash-resume checkpoints** (`kie_upload_url` / `local_clip_path` reuse);
  pre-poll recovers already-succeeded kie tasks without rebilling.
- **Concurrent submit / poll-download / reference resolution** via bounded
  `ThreadPoolExecutor`s with per-thread DB sessions.
- **nginx X-Accel-Redirect** media offload for source / library / result.
- **Dashboards bounded** (100 projects, 200 generations; CSV cap 10 000).
- **Sheets export off the request path** (RQ job on the fast queue).
- **Batch-copy off the event loop** (`run_in_threadpool`).
- **Per-path frame-extraction lock** + 0.1 s timestamp quantisation.
- **Credit-balance serve-stale-while-refresh** with stampede lock.
- **Drive resumable upload** with chunked retries + socket-timeout widening.
- **Source normalization at ingest** (≤1080p 8-bit) bounding worker memory.

## 5. Recommended priority (if you act on this report)

1. **B1 + B2** — cache result duration on `Run`; gate merge/tail-edit context
   on `done`; cache `library_clips`. Eliminates the per-2 s-poll `ffprobe` and
   ~4 extra queries.
2. **B3** — switch `pipeline_v2` to `get_shared_client()` (or `with KieClient()`)
   to stop the per-poll httpx pool leak.
3. **B4 + B7** — add LIMIT to library queries and `created_at` indexes to the
   listing tables.
4. **B8** — pass stored clip durations into `stitch`; evaluate a concat-demuxer
   `-c copy` fast path.
5. **B5 + B6** — bound the Sheets dedupe reads; read the automation sheet once
   per tick.
6. **B11** — fix the media-endpoint session lifecycle to stop per-request log
   noise.
7. B9, B10, B12, B13 — minor; address when scale demands.

## 6. Scope & integrity

**Files reviewed (read-only):**
`backend/app/{api_v2,api_overlay,automation,config,db,face,gdrive_client,kie_client,main,media,models,overlay_pipeline,pipeline_common,pipeline_v2,recovery,sheets_client,state_machine,tasks,web_v2}.py`,
`worker/worker.py`, `docker-compose.yml`, `deploy/nginx/nginx.conf`,
`backend/alembic/versions/*.py`, `backend/app/templates/v2/**`.

**No source files were modified.** The only file created is this report,
`BOTTLENECK_REPORT.md`. The pre-existing untracked
`scripts/_rerun_seg1_e5c5682f.py` was left untouched.

**Git comparison:** local `main` == `origin/main` on
the private upstream repository (no divergence). A separate
`origin/perf/bottlenecks` branch (3 commits, unmerged) already contains many
performance fixes; this report flags the items that branch does **not** cover.
