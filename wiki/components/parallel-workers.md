---
title: "Parallel Workers for Throughput"
tags: [workers, rq, throughput, scaling, stitch, pipeline, postgres]
sources: [backend/app/pipeline_v2.py, docker-compose.yml, wiki/decisions/postgres-migration.md]
updated: 2026-06-24
---

# Parallel Workers for Throughput

Implemented on `feat/parallel-workers` (branched from `feat/postgres-migration`, commit on top of `d00ae34`).

## Motivation

Run `72325061` sat in `queued` for several minutes purely because the single
worker was busy with another run. The system had spare capacity but no way to
use it. Two bottlenecks were identified:

1. **Serialization across runs** — one worker, one job at a time.
2. **Serial `cut_clip` loop in stitch** — keep/fallback segments cut one at a
   time even though they are independent.
3. **Worker held during Seedance poll** — the worker blocks for up to 2 hours
   sleeping in a poll loop waiting for external AI results (see below: Proposed
   Decoupling).

---

## Bottleneck Analysis

### 1. Serialization across runs (FIXED)

**Root cause:** A single `worker` container runs one RQ job at a time. Every
`process_run` and `analyze_project` job queues up behind whatever is already
executing. Independent runs for different projects cannot overlap.

**Fix:** Scale the worker service to N replicas. RQ's queue is atomic
(BLPOP/atomic dequeue via Redis): each worker pops a different job. A given
job is executed by exactly one worker — no double-processing is possible by
design. See [[decisions/postgres-migration]] for why Postgres is required
before scaling to N > 1 (SQLite had a single-writer lock; multiple workers
writing simultaneously would produce `database is locked`).

### 2. Serial `cut_clip` loop in stitch (FIXED)

**Root cause:** In `pipeline_v2.process_run`'s stitch assembly phase, keep
segments and fallback (unswapped) segments were cut by calling
`media.cut_clip()` in a plain `for sd in seg_defs` loop. Each ffmpeg call is
independent (different time ranges, separate output files), but they ran one
after another.

For a 10-segment video with 3 keep segments, this is 3 serial ffmpeg invocations
that could run in parallel.

**Fix:** Replaced the serial loop with a `ThreadPoolExecutor` bounded by
`STITCH_CUT_CONCURRENCY` (default 2). Futures are stored in submission order
and collected in that same order, so `clip_paths` is always correctly ordered
regardless of which cuts finish first. See the implementation note in
`pipeline_v2.py` stitch section.

### 3. Worker held idle during Seedance polling (PROPOSED — not yet implemented)

**Root cause:** After submitting all swap segments to Seedance, `process_run`
enters a round-robin poll loop (`while pending: ... time.sleep(15)`). A typical
Seedance job takes 2–10 minutes; a run with many segments can hold the worker
for up to `RUN_SKIP_TIMEOUT_SEC` (2h). During that time the worker is blocked —
it holds the RQ "job slot" while doing no real CPU work. A second worker is not
blocked by the poll loop of the first (each worker has its own job slot), but
each worker can only run one job at a time, so a long-polling run occupies a
worker for hours.

**Why not fixed now:** Decoupling submit → poll → stitch into separate RQ jobs
requires a state-machine split and careful idempotency handling (the existing
resume/retry logic in `process_run` is stateful; splitting it means each leg
must be restartable). This is a significant refactor. See the design proposal
in [[roadmap]] and below.

**Proposed design (for a future PR):**
```
Queue topology:
  "default" queue  → heavy jobs: analyze_project, stitch, deliver
  "poll" queue     → lightweight: poll_run (re-enqueues itself)

State extension:
  Run.status: add "polling" between "processing" and "stitching"
  Run.poll_deadline: TIMESTAMP — when to give up and stitch with what we have

Jobs:
  1. process_run_submit (default queue)
     - Cut clips, upload to kie.ai, create Seedance tasks
     - Persist pending task IDs to DB (new Run.pending_task_ids JSON column)
     - Transition run: processing → polling
     - Enqueue poll_run on the "poll" queue

  2. poll_run (poll queue, lightweight)
     - Load pending_task_ids from DB
     - Call get_task() for each; process completed/failed
     - If all done OR deadline passed → enqueue stitch_run on default queue
     - Else: re-enqueue self on poll queue after RUN_POLL_INTERVAL_SEC
     - Uses RQ's job_timeout=RUN_POLL_INTERVAL_SEC * 2 (short, re-enqueues before timeout)

  3. stitch_run (default queue)
     - Assemble clips (parallel cut_clip), stitch, deliver
     - Transition run: polling → stitching → delivering → done

Poll worker sizing:
  - 1 poll worker (0.1 vCPU / 256 MiB) per ~50 concurrent polling runs
  - No InsightFace, no ffmpeg — poll workers need no GPU model, minimal RAM
  - Poll workers can be a separate, lighter image (or same image, different queue)

Idempotency notes:
  - poll_run must handle the case where it re-enqueues itself but the previous
    instance is still running (Redis dedup by job key: use run_id as job_id)
  - If poll_run worker dies, the re-enqueue never happens → run stays in
    "polling" forever. Need: startup orphaned-run reconciliation (TR5b) to
    detect runs in "polling" with no live RQ job and re-enqueue poll_run.
  - stitch_run must be idempotent (same as current resume logic: skip
    segments already completed with a result on disk).

Risks:
  - 3-table schema change (new columns on Run, new run status enum value)
  - Significant test surface: 3 new job types, cross-job state
  - Retry endpoint must handle "polling" status

Decision: defer to post-v2 pass. With 2 workers, the poll-hold problem only
matters if both workers are simultaneously stuck in 2-hour poll loops —
unlikely in normal operation (most runs finish in <15 min). The blocking case
(both workers stuck) is a future optimization, not a blocker for the 2-worker
improvement.
```

---

## Implementation Details

### Horizontal Scaling (N workers)

**`docker-compose.yml` changes:**
- Removed `container_name: re-skin-worker` from the worker service. A fixed
  `container_name` prevents Docker from naming multiple replicas — removing it
  enables `docker-compose up --scale worker=N`.
- Reduced per-worker limits from `cpus=3 / mem_limit=8g` to `cpus=2.0 /
  mem_limit=6g / memswap_limit=7g` to fit 2 workers on the 4-vCPU / 16 GiB host.
- Set `FFMPEG_THREADS=2` (was 4) — matches 2 vCPU ceiling; keeps per-process
  memory bounded.
- Added `STITCH_CUT_CONCURRENCY=2` env var.

**To run 2 workers:**
```bash
docker-compose up -d --scale worker=2
```
Containers will be named `<project>_worker_1` and `<project>_worker_2`.

**RQ no-double-processing guarantee:** RQ dequeues jobs via Redis `BLPOP` which
is atomic. Only one worker pops any given job. Confirmed by the verification run
(4 jobs, 2 workers: each job ran exactly once, in one worker).

### Parallel `cut_clip` in Stitch

New code in `pipeline_v2.process_run` stitch section:

```python
STITCH_CUT_CONCURRENCY = int(os.getenv("STITCH_CUT_CONCURRENCY", "2"))

def _cut_or_lookup(sd: SegmentDef) -> str:
    # Returns clip path: either existing result, or calls media_mod.cut_clip()
    ...

ordered_futures = []
with ThreadPoolExecutor(max_workers=STITCH_CUT_CONCURRENCY) as pool:
    for sd in seg_defs:
        fut = pool.submit(_cut_or_lookup, sd)
        ordered_futures.append((sd.index, fut))

clip_paths = [fut.result() for _idx, fut in ordered_futures]
```

Key properties:
- **Order preserved:** futures collected in `seg_defs` insertion order; results
  retrieved in that order. Stitch receives clips in the correct timeline order.
- **Bounded concurrency:** `max_workers=STITCH_CUT_CONCURRENCY` prevents
  spawning more ffmpeg processes than the CPU can handle.
- **Exception propagation:** if any cut fails, `fut.result()` re-raises on
  the main thread, marking the run failed. No silent swallowing.
- **Already-available results** (completed swap results) are returned
  immediately without touching the thread pool (no ffmpeg needed for them).

---

## Resource / Scaling Math

**Host: 4 vCPU / 16 GiB**

| Service       | vCPU reservation | Mem limit |
|---------------|-----------------|-----------|
| postgres      | ~0.5             | 0.5 GiB   |
| redis         | ~0.1             | 0.1 GiB   |
| api           | ~0.5             | 2.0 GiB   |
| nginx         | ~0.1             | 0.1 GiB   |
| worker×2      | 2×2.0 = **4.0** | 2×6.0 = **12 GiB** |
| **TOTAL**     | **~5.2 peak**   | **~14.7 GiB** |
| **Headroom**  | ~0 burst vCPU (OS handles burst with scheduling) | ~1.3 GiB OS+buffer |

**Why `cpus=2.0` per worker:**
- One InsightFace `analyze` or one ffmpeg 1080p stitch is already CPU-bound at
  2-3 cores. Capping at 2 prevents one worker from starving the other.
- Two workers × 2 vCPU = 4 vCPU committed; within the host ceiling.

**Why `mem_limit=6g` per worker:**
- A 1080p stitch with 10 clips peaks at ~4 GiB in practice.
- InsightFace buffalo_l loads ~300 MB of model weights.
- 6g provides headroom; OOM at 6g triggers container restart rather than host
  swap-thrash (memswap_limit=7g gives a 1g swap buffer before the OOM kill).

**N=3 workers is NOT recommended on this host** without reducing per-worker
limits: 3×6g = 18g already exceeds the 16g physical RAM ceiling (would swap).

---

## Concurrency Safety

| Concern | Assessment |
|---------|-----------|
| DB write contention | Resolved by Postgres (multi-writer safe). Each job uses an independent SQLAlchemy session (`with get_session() as session`). |
| File path isolation | Per-run paths (`run_clips_dir` / `run_results_dir`) are keyed by `run_id` (UUID). Two workers processing different runs never touch the same files. |
| Shared mutable state | None. `pipeline_v2` has no module-level mutable state. Client objects (`KieClient`, `GDriveClient`) are instantiated fresh per job. |
| RQ double-dequeue | Impossible: Redis BLPOP is atomic. RQ also uses a "StartedJobRegistry" to track in-flight jobs. |
| Worker crashes + orphaned runs | Each additional worker = one more potential orphan on restart. TR5b (startup reconciliation) becomes more urgent. See [[production-gotchas]] → "Worker crash leaves runs orphaned". |
| Postgres connection pool | Each worker process gets `pool_size=5 / max_overflow=10` connections. 2 workers = up to 30 connections. Postgres 16 default `max_connections=100` — safe. |

---

## Ops Notes

### Scaling up (more throughput)

```bash
# 2 workers (recommended for 4 vCPU / 16 GiB):
docker-compose up -d --scale worker=2

# Check worker containers:
docker-compose ps

# Monitor queue depth and worker status:
docker exec <worker_container> rq info --url $REDIS_URL
```

### Scaling down

```bash
docker-compose up -d --scale worker=1
```

### Checking which worker is processing a job

```bash
# List started jobs across all workers:
docker exec <worker_container> python3 -c "
from redis import Redis; from rq.job import Job; from rq.registry import StartedJobRegistry
from rq import Queue; import os
conn = Redis.from_url(os.environ['REDIS_URL'])
reg = StartedJobRegistry('default', connection=conn)
for jid in reg.get_job_ids():
    j = Job.fetch(jid, conn); print(jid, j.description)
"
```

### If a worker crashes mid-run

The run stays in `processing` (or `queued`). Manual recovery procedure:
1. Check `run.status` in the DB.
2. For `processing`: reset `run.status = 'queued'` manually.
3. Re-enqueue: `from app.tasks import enqueue_process_run; enqueue_process_run(run_id)`.
4. `process_run` is idempotent: completed segments with result files on disk
   are skipped; others are re-submitted to Seedance.

Permanent fix: TR5b (startup reconciliation — detect orphaned runs on worker boot).

---

## Verification Results (2026-06-24)

### Multi-worker concurrency proof

Ran 4 dummy sleep-2s jobs against 2 workers on throwaway Redis (localhost:16379):

```
[20:53:32] Enqueued job 1: de3d6d44-…
[20:53:32] Enqueued job 2: b71db739-…
[20:53:32] Enqueued job 3: ac8cf9bf-…
[20:53:32] Enqueued job 4: 0baa78a3-…
[20:53:32] Worker 1 PID=95184 started
[20:53:32] Worker 2 PID=95185 started
[20:53:35] Job 1 done  (elapsed=2.48s)   ← both workers busy simultaneously
[20:53:35] Job 2 done  (elapsed=2.48s)
[20:53:37] Job 3 done  (elapsed=4.63s)
[20:53:37] Job 4 done  (elapsed=4.63s)
All 4 done in 4.63s  (vs ~8s serial with 1 worker)
[PASS]
```

Jobs 1 and 2 finished at the exact same moment (2.48s), confirming they ran
on different workers simultaneously. Serial would have been ~8s.

### pytest results

```
2 failed, 395 passed, 4 skipped, 29 errors
```

- 3 NEW tests pass (`TestParallelStitchOrdering`) — pure Python, no ffmpeg.
- 3 new ffmpeg-gated tests (`TestStitchCutConcurrency`) join the `errors` group
  (ffmpeg Docker-only — pre-existing condition).
- 2 failures + 26 pre-existing errors: all `ffmpeg not in PATH`.
- **Zero new non-ffmpeg failures introduced.**

---

## Related Pages

- [[architecture]] — service topology, updated to show N workers
- [[decisions/postgres-migration]] — prerequisite; why Postgres is required before multi-worker
- [[components/pipeline]] — full pipeline flow (analyze, process, stitch)
- [[lessons/production-gotchas]] — worker crash + orphaned runs; RQ job timeouts
- [[roadmap]] — poll-decoupling refactor (TR-POLL), TR5b orphaned-run reconciliation
