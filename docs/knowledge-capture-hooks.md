# Knowledge capture hooks — implementation addendum (AICP-299)

Status: **revision 10, for the CLAUDE.md §2 design review.** GPT-5.6 Terra
(effort high) reviewed revisions 1-9 and approved none of them:

| Round | Revision |
|---|---|
| 1 | `0ab9e76` |
| 2 | `24dceb4` |
| 3 | `5217950` |
| 4 | `b8ad2eb` |
| 5 | `2ceba94` |
| 6 | `78f3d7b` |
| 7 | `69165de` |
| 8 | `9c4bf47` |
| 9 | `334003d` |

Round 4 confirmed that the atomic-stitch change is sound. Every finding is
resolved below, and each round has a resolution table at the end. No
application code exists yet.

This document turns §6.1, §6.2, §6.6, §7 and §10 of the approved
[knowledge-feedback-design.md](knowledge-feedback-design.md) (revision 8,
line numbers at `e6cd9de`) into concrete hooks. §5 refines that design's §6.5.

Trees read:

| Tree | Branch / commit | How it is cited |
|---|---|---|
| This one | `docs/knowledge-capture-design` (main `5c1080d` + AICP-295 `a7fbac4`) | bare `file:line` |
| Storage | `feat/knowledge-storage` **at or after `94ee5d0`** (the reviewed minimum: `publication_covered` is in `284df2c`, effective-only rule evidence in `898d42b`, SQL-bounded publication lookups in `94ee5d0`; the release packet records the exact gated hash) | S3 (`62070a5`), S16 + the source_record UNIQUE (`3efff8e`), and `file_identity` with C2 identity revalidation (`9c9e3eb`) all live inside the undeployed `k1a7c3e9f205` |
| Importer | `feat/knowledge-importer` **at or after `ba493fc`** (the reviewed minimum), at `/root/re-skin-knowledge-importer` | "importer:"; S9a-S9d, S9f (historical promotion versions), mandatory `file_identity` in reconciliation (`da8d0c5`) are done there; it is not scheduled yet |

"Not verified" marks what I could not check.

Contents:

- §0 Principles and decisions
- §1 Shared mechanics: gate, executor, cases, dispatch identity
- §2 A and B capture sites
- §3 Finals: atomic stitch, two link points, a scanner that orders by live state
- §4 Promotions
- §5 Capture scope (explicit list, recorded as provenance) vs. access scope
- §6 Knowledge worker (slim, budgeted), scheduler, C2, reconciliation, locks
- §7 Failure policy
- §8 Flags, disk, capacity, unpreserved versions
- §9 Recovery, deploy, rollout, pilot gate
- §10 Feedback UI
- §11 Amendments
- §12 Test matrix
- §13 Open risks and operator decisions
- Review resolution tables, rounds 1-4

---

## 0. Principles and decisions

- **P1. No capture between a paid call and its commit.** No capture call runs
  between a paid provider call and the pipeline's durable commit of that call's
  id. A5 records its intent and submission together, after its single commit.
- **P2. Every DB-touching capture call is bounded and never blocks.** It runs
  on the capture executor (§1.3). The caller waits on an `Event` with a timeout
  and then abandons the call. A full queue means the call is skipped. The
  executor supervises itself.
- **P3. No capture DB access or hashing inside pipeline functions or their
  transactions.**
  - Inside a pipeline function, the only capture work is
    `os.stat` + `os.makedirs` + `os.link`, for stitch link (ii) only (§3.2).
  - **Publication links** (stitch link (i) and every promotion) run in the
    **caller** after the pipeline function has returned its committed
    `PublishedFinal` descriptor. That caller is the RQ wrapper or the API
    endpoint (§3.2, §4).
  - Publication links never re-read the DB. A link failure row is written
    afterwards through the executor (≤ 2 s or fire-and-forget), and for
    stitches only after the Redis lock is released.
  - **One timeout policy** for every publication link: absorb RQ's timeout
    and record it.
  - Link (ii) is stitch-only and re-raises.
  - Dispatch identity comes from row locators, never from hashing (§1.5).
- **P4. "Current" is the live state, never link time.**
  - C2 (`drain_run_captures`, storage `9c9e3eb`) decides under the Run lock.
    It compares each queue item's `file_identity` with the live
    `result_local_path` at drain time.
  - A match can become current. A mismatch becomes a **historical** version
    (S16), which never changes the current one (§3.3).
  - The scanner only enqueues; it takes no Run lock.
- **P5. Unpreserved versions get no historical judging.** A version whose
  bytes are not preserved is judged only while it is current and its bytes
  are verified. It is never judged as history (§8.3).
- **P6. Capture scope is an explicit, recorded operator decision.**
  - `KNOWLEDGE_PROJECT_IDS` must be an explicit UUID list. `*` is refused for
    capture.
  - The active list's hash is stored as provenance on every captured row.
  - Reading and feedback keep the assistant-intersected `KnowledgeScope` (§5).

| # | Decision | Evidence |
|---|---|---|
| D1 | `dispatch_key` is built from the **reference locators** the rows already hold, plus the request with its uploaded URLs removed. No content hashing happens in the pipeline. `reference_set_sha256` stays NULL. This is recovered-job matching only. | Round 4 M1. `_resolve_one_ref` re-uploads every time (`pipeline_common.py` L306-323). Revision 4's `note_upload` hashed inside the open submit transaction. |
| D2 | Output capture (B) appends segment versions, locking only the segment case. | `store._locked_case_and_version` L677 |
| D3 | Atomic stitch: write a temp file, then `os.replace`. **Confirmed sound in round 4.** | §3.1 |
| D4 | Two link points per Run final: **(i)** after publication, before the lock release; **(ii)** inside the next stitch, before it replaces the file (kind `superseded`). | Round 4 B2 |
| D5 | Promotions get the same post-commit link. | §4 |
| D6 | The kie.ai payload builders are shared and golden-tested. | `kie_client.py` L460-476, L555-575, L668-676 |
| D7 | Drain before sync. **Done.** | storage `62070a5`, contained in `9c9e3eb` |
| D8 | The pilot gate measures locally observed task ids. Each final counts as captured **or explicitly counted as `link_failed`**. | §9 |
| D9 | **Withdrawn:** `note_upload` and the hash registry. | Round 4 M1 |
| D10 | A slim knowledge worker gets its own entrypoint and an explicit resource budget. | Round 4 B3, §6.1 |
| D11 | **Withdrawn:** revision 3's outbox, savepoint and inline preservation. | Round 3 |
| D12 | The scanner enqueues every publishable link with its inode `file_identity`. C2 revalidates against the live final under the Run lock (storage `bb9b6fb`), not by link time. | Rounds 4 B1, 9 M1 |

---

## 1. Shared mechanics

### 1.1 Modules

New modules:

| Module | Contents |
|---|---|
| `knowledge/capture.py` | `begin_job`, `active_for`, `swap_intent`, `image_intent`, `submitted`, `dispatch_error`, `intent_and_submission_batch`, `terminal`, `output`, `link_publication` (point i), `link_preserve` (point ii), `record_link_failure`, the heartbeat thread |
| `knowledge/capture_exec.py` | the executor |
| `knowledge/finals_scan.py` | the finals scanner |
| `knowledge/service.py` | HTTP-side transactions |
| `worker/knowledge_worker.py` | the slim worker entrypoint |

Changed modules:

- `media.py`: the atomic stitch.
- `kie_client.py`: the payload builders.
- `pipeline_v2.py`, `pipeline_range_attempt.py`, `pipeline_segment_image.py`,
  `pipeline_audio_patch.py`, `assistant_run_result.py`, `recovery.py`: hooks.
- `tasks.py`: `begin_job`, the finalize link, the tick scheduler.
- `docker-compose.yml`: the `knowledge-worker` service.
- One additive Alembic revision with every schema change listed in §11.1.

**`pipeline_common.py` is no longer changed.**

### 1.2 Gate

`capture.in_scope(project_id)` runs four checks, with no I/O:

1. `KNOWLEDGE_CAPTURE_ENABLED` is set.
2. The engine is PostgreSQL.
3. `CaptureScope`, parsed once per process, is a non-empty explicit UUID list
   (§5).
4. The project is a member of that list.

`active_for` adds the deadline check (§1.3). Where only an `rs_id` is known,
the project id comes from one executor `SELECT`.

### 1.3 The capture executor

```python
class CaptureExecutor:                          # one per process; rebuilt when os.getpid() changes
    QUEUE_MAX, THREADS = 8, 2
    def __init__(self):
        self.q = queue.Queue(maxsize=self.QUEUE_MAX)
        self.threads = [None] * self.THREADS
        self.last_ok = None; self.consecutive_failures = 0
        self._ensure_threads()
    def _ensure_threads(self):                  # supervision: called on every call()
        for i, t in enumerate(self.threads):
            if t is None or not t.is_alive():
                if t is not None: count("executor_thread_replaced")
                self.threads[i] = threading.Thread(target=self._loop, name=f"kb-capture-{i}", daemon=True)
                self.threads[i].start()
    def _loop(self):
        session = None
        while True:
            job = self.q.get()
            try:
                if session is None: session = capture_engine.new_session()     # lazy, per thread
                job.result = job.fn(session); session.commit()
                self.last_ok = time.time(); self.consecutive_failures = 0
            except BaseException as exc:
                job.error = exc; self.consecutive_failures += 1
                try: session and session.rollback()
                except Exception: pass
                if isinstance(exc, (OperationalError, InterfaceError)) and session is not None:
                    try: session.close()
                    except Exception: pass
                    session = None                                            # reconnect on next job
            finally:
                job.done.set()
    def call(self, fn, budget_sec):
        self._ensure_threads()
        job = _Job(fn)
        try: self.q.put_nowait(job)
        except queue.Full: count("executor_full"); return None
        if budget_sec == 0: return None                                       # fire-and-forget
        if not job.done.wait(budget_sec): count("budget_exceeded"); return None
        return None if job.error else job.result
```

**Threads.** They are daemon threads. The stdlib `ThreadPoolExecutor` cannot
make daemon threads and joins its threads at exit, so it is not used.
Abandoned work dies when the work-horse calls `os._exit` (RQ 1.15.1
`worker.py` L1302-1313), and PostgreSQL then rolls back the transaction.

**Connection.** Each thread owns one connection, created through
`capture_engine` (`pool_size=2`, `max_overflow=0`, pre-ping) with:

```
connect_args = {connect_timeout: 2, keepalives: 1, keepalives_idle: 5,
                keepalives_interval: 2, keepalives_count: 2, tcp_user_timeout: 5000,
                options: "-c statement_timeout=1000 -c lock_timeout=1000"}
```

**Health (round 5 m1).**
- Every process that owns an executor starts a daemon **heartbeat thread**
  alongside it. That covers RQ work-horses, the API and the knowledge worker.
- Every 15 s the thread sets
  `reskin:knowledge:capture:health:<container>:<pid>` with `ex=60`. The value
  is `{last_attempt, last_ok, consecutive_failures, threads_alive,
  queue_depth}`.
- The thread runs independently of capture calls, so a long provider wait with
  no captures still refreshes the key.
- The thread is supervised the same way as the executor threads: it is
  restarted by `_ensure_threads`.
- **What counts as unhealthy.** A key that *exists* and shows either:
  - `consecutive_failures ≥ 5`, or
  - `last_attempt − last_ok > 600 s`.
- **What does not count.** A missing key. Work-horses exit after each job, so
  keys vanish normally; absence means "no process", not "stale".
- The knowledge tick aggregates all keys into
  `reskin:knowledge-tick:last_ok`.

**Deadline.**
- `begin_job()` is called first in each hosting RQ wrapper and sets
  `_DEADLINE`.
- The budget is `min(site budget, remaining − 90 s)`. Below 1 s the call is
  skipped.
- A non-main thread with no deadline skips.
- A static test asserts that every hosting wrapper calls `begin_job`.

**Site budgets.** A, B and T sites get 3 s. The finals scanner runs on the
knowledge worker under the tick deadline.

No executor job reads or writes a Run's live `final.mp4`.

### 1.4 Cases

Executor sessions follow the storage lock contract
(`docs/knowledge-storage.md`: `runs FOR UPDATE → kb_cases`), round 19 B1.
When a capture needs a Run or segment case that does not exist yet:
- it opens a short transaction of its own, with `SET LOCAL lock_timeout =
  '200ms'`, `SET LOCAL statement_timeout = '250ms'` and `SET LOCAL
  idle_in_transaction_session_timeout = '500ms'` (round 20 M1).

  PostgreSQL enforces these server-side. No single statement can run longer
  than 250 ms. An executor thread that stalls or is abandoned between
  statements, or before `COMMIT`, has its session terminated by the server
  after 500 ms idle, which releases the Run row. The Run lock is therefore
  held for at most about 3 × 250 ms + 500 ms (three statements: lock, case
  insert-or-get, segment case) whatever happens to the client.
- calls `store.lock_run_and_cases(run_id, [])`;
- then `ensure_run_case` and, if needed, `ensure_case(segment)`, and commits.

The Run row is held for one or two indexed inserts (milliseconds). The
existing case path (the common one) takes **no** Run lock: attempts and
observations need only the case id.

**Never waits on the pipeline.** On `lock_timeout` (the Run is held, e.g. by
the submit phase or a promotion), the capture call skips. It is counted as
`capture_skipped_run_locked` in the health key, and the evidence is recovered
later: the importer's sources and draw reconciliation import the attempt and
its cost (as an `ambiguous` attempt when the request is unknown). Capture
never allocates Run versions (R16). The only delay capture can impose on a
pipeline transaction is that millisecond case insert, and only the first time
a Run is captured.

### 1.5 Identity and dispatch basis (D1)

| Site | `attempt_kind` | `origin` | Target | Predecessor | Key |
|---|---|---|---|---|---|
| swap submit | `generation` / `rerun` | `automatic` | `run_segment` | latest submitted (S2) | — |
| resubmit | `resubmit` | `automatic` | `run_segment` | failed task | — |
| range | `range_attempt` | `operator` | `range_attempt` | — | `range_attempt:<id>` |
| preprocess | `preprocess_image` | `automatic` | `run_preprocess` / `<run>:<i>` | — | — |
| segment image | `segment_image` | `operator` | `segment_image_job` | — | — |

**Dispatch basis.** It is built inside the executor body, from rows it reads
in its own session. Nothing is computed in the pipeline. It is `{request
minus volatile URLs, ref_locators, clip_locator, voice_locators}`:

- **`ref_locators`.** Taken from the same source list the pipeline resolved,
  in the same priority order: `rs.generated_reference_urls` →
  `rs.reference_image_urls_override` → `run.preprocessed_reference_urls` (when
  preprocessing is on) → `run.reference_image_urls`. This mirrors
  `_build_submit_state` L1816-1836 and L2104-2135. For range attempts the
  attempt's own override comes first (`pipeline_range_attempt.py` L688-698).
  Each item is normalized:

  | Item | Normalized to |
  |---|---|
  | Drive link | `drive:<file id>` |
  | local path | `local:<path relative to DATA_DIR>` |
  | https URL | the URL itself |

- **`clip_locator`.** `{project_id, clip_start, clip_end, include_audio}`.
- **`voice_locators`.** `run.reference_audio_urls`, normalized the same way.
- **Volatile fields** are removed from the request: every `*_urls` entry and
  `video_list[].url`.

The basis is recorded in `recipe_context["dispatch_basis"]`, and S1 hashes it
into `dispatch_key`.

**Race on the source rows.** Overrides change only through the API, and the
API refuses edits while a run, range attempt or segment-image job is active.
That covers `rerun_segment`'s gates (L4790-4838) and the segment-edit guard
(memory). So the rows are stable for the duration of a dispatch. If they are
not, the basis differs, a new intent is created, and the old one becomes
`intent_without_submission`. That is the weaker, counted outcome.

**Guarantee, stated honestly.**
- The basis matches two dispatches whose *locators* and request are equal.
- It does not prove the *content* behind a locator was the same. A local path
  rewritten in place, or a Drive file replaced, would still match.
- `input_hashes` and `reference_set_sha256` stay NULL at capture.
- The knowledge worker may later hash readable locators (local paths under
  `DATA_DIR`) and append a new observation kind, `input_content` (S13), with
  `{locator: sha256}`. The attempt row stays immutable.
- AICP-301 retrieval treats an attempt's content hashes as known only through
  such observations.

Model and code fields are unchanged:
- swap uses `ai_models.kie_model_id`, and Omni `"gemini-omni-video"`;
- image uses `settings.KIE_IMAGE_MODEL`;
- `RESKIN_CODE_REVISION` is NULL until a build arg sets it (R12).

---

## 2. A and B sites

These are unchanged since revision 3; round 2 verified their P1 placement.
Each site is one executor call with a 3 s budget. The hash registry is gone,
so **no site hashes inputs**.

| # | Site | Intent | After-commit capture |
|---|---|---|---|
| A1 | `_submit_swap_segment_isolated` (pool thread) | L1060-1062 | submission after L1070 |
| A2 | `_resubmit_segment_task` | L1505-1507 | submission + predecessor `failed`, after L1525 |
| A3 | `submit_range_attempt` | before L761 | after L826, success path only |
| A4 | `all_as_refs` | before L2915 | after L2926 |
| A5 | `per_image` loop L2959-2993 | none inside the loop | batch after L2996 |
| A6 | `submit_segment_image_job` | L148-149 | after L157 |

| # | Output site | After commit | Version |
|---|---|---|---|
| B1 | `_download_and_complete_segment` | L1407 (+ settle L1411-1414) | segment |
| B2 | no-rebill resume | L2009 (a commit of the outer session, so no runs-row write is pending) | segment |
| B3 | `recovery._repoll_generating_segments` | L677 | segment |
| B4 | range `_handle_terminal_kie_state` | L1411; hardlink into `attempts/` | at promotion (§4) |
| B5 | `poll_segment_image_job_once` | L288 | none |
| B6 | `poll_preprocess_once` | L3237 | none |

Output hashing (B1-B6) runs in the executor body after the commit, not in the
pipeline thread. The caller waits at most 3 s, then abandons the call. An
abandoned hash finishes, or dies with the horse; the output is still on disk
for the importer.

The T1-T7 terminal captures, the orphan path and S7 dedupe are unchanged.

---

## 3. Finals

### 3.1 Atomic stitch (D3; confirmed in round 4)

`media.stitch` writes its ffmpeg output to `<dst root>.<uuid8>.part.mp4`
instead of `dst` (L1155 and L1249). It then calls `os.replace(tmp, dst)` and
unlinks the temp file in a `finally`.

What this covers:

- **Callers.** All three were checked: the run final (L2682 → L2491
  `final.mp4`), the range candidate (L1081, its own directory) and the tail
  edit (L271, its own output).
- **`enforce_size_limit`** already replaces atomically (L650-680).
- **Readers** see either the old or the new complete file:
  - Drive and nginx read by path or open fd;
  - the assistant result revision includes the inode, and changes on a
    restitch anyway;
  - tail, caption and overlay readers.

Side effect: a truncated `final.mp4` left by a killed ffmpeg can no longer be
reused by a delivery-only retry (L2492-2497).

### 3.2 Two link points (D4; rounds 4 B2, 5 B1-B2-M2)

`capture.link_publication(pf: PublishedFinal)` (point (i); the **only**
linker signature, used by all four publishers) and
`capture.link_preserve(run_id, project_id, path)` (point (ii), stitch-only;
kind `superseded`) share one implementation. Each makes a hardlink into

```
data/knowledge/finals/<project_id>/<run_id>/<time_ns>__<kind>__<pub>__<dev>-<ino>-<mtime_ns>__s<scope8>.mp4
```

`<scope8>` is the capture-scope hash prefix (§5). A link is a directory entry
pointing at an existing inode, so it copies **no data blocks**. That is why
neither link point has a deadline gate or a disk guard (rounds 6 B1, 7 M3).
In order, each:

1. checks `in_scope`, with no I/O;
2. calls `os.stat(path)` **first**, so `(dev, ino, mtime_ns)` is known
   whatever fails next;
3. calls `os.makedirs(dir, exist_ok=True)`, then `os.link(path, target)`.

Space is governed only by `disk-report` / `release-preserved` (§8). The disk
guard stays only on writers that copy bytes: attempt-media copies and
external uploads.

A link can still fail for these reasons:
- `EXDEV`: a nested mount, not verified (§8);
- `EACCES` / `EPERM`;
- `ENOSPC` / `EDQUOT`: the directory needs a new block for the entry, or
  inodes for a new directory are exhausted;
- `EMLINK`;
- `EIO`;
- `os.stat` failing because the file vanished.

**Exception contract (round 5 B2).** From the installed RQ 1.15.1 source
(`rq/timeouts.py` L6-17):

- `BaseTimeoutException(Exception)`
- `JobTimeoutException(BaseTimeoutException)`
- `HorseMonitorTimeoutException(BaseTimeoutException)`

So a bare `except Exception` *would* swallow the job timeout. The two link
points therefore treat RQ's timeout **differently** (round 7 M1):

```python
# link point (ii) — capture.link_preserve(): before the stitch
except BaseTimeoutException:
    raise                                     # propagates unchanged into finalize's normal timeout path
except Exception as exc:
    count(...); log(...); return LinkFailed(exc)

# link point (i) — capture.link_publication(): the last statement before lock release + return
except (BaseTimeoutException, Exception) as exc:   # NOT BaseException: KeyboardInterrupt/SystemExit still propagate
    count(...); log(...); return LinkFailed(exc, stat=st_or_None, failed_at_ns=time.time_ns())
```

**Why (i) may absorb the timeout and (ii) may not.** This is the distinction
round 5 B2 was about.

- **Point (ii) runs before work.** It sits immediately before the stitch
  (minutes of ffmpeg), then the Drive delivery, then the `done` commit.
  Absorbing the alarm there would let the job run on past its timeout with
  the Redis lock held. RQ's monitor would then SIGKILL the horse at
  timeout + 60 s, mid-stitch or mid-upload (RQ `worker.py` L1218-1220),
  leaving the run in `stitching`/`delivering`. So (ii) re-raises.
- **Point (i) runs after all pipeline work.** It follows every committed
  write: the run is `done` and nothing pipeline-wise remains. After it come
  only:
  1. `_release_lock` (one Redis EVAL);
  2. the failure-row submission (executor wait ≤ 2 s, or fire-and-forget);
  3. the function's `return`.

  `signal.alarm` is one-shot (`rq/timeouts.py` L61-77), so absorbing it here
  delays the horse's exit by at most about 2 s, far inside the monitor's
  60 s grace.

  Re-raising instead would turn a `done` run's last step into a failed RQ job
  **with no `kb_link_failures` row**. Recovery ignores `done` runs
  (`recovery.py` L118-125), and the finalize job timeout is 1 h (`tasks.py`
  L133). So a later rerun would leave the delivered final only as
  "delivery unverified" (round 7 M1). Absorbing the timeout lets the row be
  written.
- **The same rule applies to the promotion (i) points (§4)**, which are also
  the last statement of their job or request:
  - range: just before `return "done"`, whose wrapper then returns
    (`tasks.py` L1512-1514);
  - audio: the last statement of `run_process_audio_patch`;
  - assistant: the API process, which has no alarm.

**Failure-row identity.**
- `link_publication` calls `os.stat` first. If the stat completed, the row
  carries `(st_dev, st_ino, st_mtime_ns)`.
- If the stat itself failed or was interrupted, the row has a **NULL
  identity** and carries `run_id`, `path` and `failed_at_ns`.
- **Pairing rule for a NULL-identity row.** It pairs with the **first**
  point (ii) link that satisfies all of:
  - same `run_id`, made for the same `path`;
  - `time_ns > failed_at_ns`;
  - `time_ns` earlier than the next point (i) link of that run, if there is
    one.

  The path match is reliable because (ii) always links `r_dir/final.mp4`
  (L2491). For promotion (i) rows the path is the promoted file, which (ii)
  never links. So a NULL-identity promotion row never pairs and stays a
  counted failure. All links and rows use `time.time_ns()` from the same
  host clock; the containers share the host kernel clock.

**Link point (i): publication. This is the only source of Run-case
versions.**

- **Return contract (round 7 B1).** Today `_run_stitch_and_deliver`
  (L2388-2394) and `finalize_run` (L3348-3352) are both `-> None`. They
  change as follows:

  ```python
  # app/knowledge/publication.py — imported by pipeline modules; no DB, no I/O
  @dataclass(frozen=True)
  class PublishedFinal:            # ONE descriptor for all four publishers (round 9 M3)
      run_id: str
      project_id: str
      path: str
      kind: Literal["stitch", "range", "audio", "assistant"]
      publication_id: str          # "stitch:<run>:<inv>" | "range_attempt:<id>" | "audio_patch:<id>" | "assistant_promotion:<id>"

  def _run_stitch_and_deliver(run_id, *, gdrive, session, did_submit,
                              capture_links: bool = False,
                              publication_id: Optional[str] = None) -> Optional[PublishedFinal]
  def finalize_run(run_id, *, gdrive=None, publication_id: Optional[str] = None) -> Optional[PublishedFinal]
  ```

  `publication_id=None` means `_run_stitch_and_deliver` mints
  `stitch:<run_id>:<uuid4 hex>` at entry. `run_finalize_run` passes
  `stitch:<run_id>:<RQ job id>`.

  Every exit of `_run_stitch_and_deliver`:

  | Exit | Line | Returns |
  |---|---|---|
  | Incomplete gate | L2448-2458 | `None` |
  | No-stitch plan | L2475-2483 | `None` |
  | Success | after `transition(run, RunStatus.done); session.commit()` (L2747-2748) | `PublishedFinal(run_id, run.project_id, final_dst, "stitch", publication_id)`. `final_dst` is `r_dir/final.mp4`, or `run.result_local_path` on the delivery-only reuse path (L2492-2497). The attributes are already loaded; `expire_on_commit=False` (`db.py` L68-73). |
  | Any exception | — | propagates unchanged |

  `finalize_run`:
  - Run not found: raises `ValueError` as today (L3367).
  - Returns what `_run_stitch_and_deliver` returned, with
    `capture_links=True`.
  - The exception path (L3401-3416) is unchanged and re-raises.

  `process_run` (L3423):
  - stays `-> None`;
  - calls with `capture_links=False` (the default) and discards the result.

  The module docstring line at L41 is updated.

  Existing callers keep working:
  - `tasks.run_finalize_run` (L964) is changed as below.
  - `backend/tests/test_pipeline_v2.py` L4948, L5022, L6652 and L6687 call
    `finalize_run(...)` as a statement and ignore the return.
  - No other callers exist (grep over `backend`, `worker`, `scripts`).
  - Tests that monkeypatch `finalize_run` with a function returning `None`
    simply produce no link.
- **Stitch site.** `tasks.run_finalize_run`. The link is the last statement
  inside the `try`, before `finally: _release_lock` (L966):

  ```python
  capture.begin_job()
  link = None
  try:
      published = finalize_run(run_id, gdrive=GDriveClient(),
                               publication_id=f"stitch:{run_id}:{rq.get_current_job().id}")
      if published:
          link = capture.link_publication(published)
  finally:
      _release_lock(conn, lock_key, lock_token)                         # L966: released FIRST (round 6 M1)
  if isinstance(link, LinkFailed):                                     # after release; never raises
      capture.record_link_failure(..., identity=link.stat, error=link.exc)
  ```

- **A timeout at this point is absorbed and recorded** (contract above). The
  run is `done` and committed; the `finally` releases the lock; the job
  returns normally. `_enqueue_finalize` registers no `on_failure` (`tasks.py`
  L1014-1026), and recovery ignores `done` runs (`recovery.py` L118-125).
- **`pipeline_v2.process_run` is excluded from capture (round 6 m1).** It is
  the monolithic compatibility path, and its finalize `try` (L3515-3531)
  catches every `Exception`. A timeout raised by a post-`done` link there would
  write `error_message` on a `done` Run.
  - **Callers.** It is **not** a production path, but it does have a caller.
    `scripts/e2e_v2_acme.py` L58 calls it in a manual, real-Seedance e2e
    script; revision 7 wrongly said there was none.
  - `tasks.run_process_run` delegates to `run_submit_run` (L975-985), not to
    `pipeline_v2.process_run`.
  - It calls `_run_stitch_and_deliver(capture_links=False)`, so neither link
    point fires there.
  - **Consequence.** That e2e script cannot validate final-link behaviour.
    What does validate it:
    - the `test_capture_links.py` unit and PG tests (§12), which drive
      `tasks.run_finalize_run` with a temporary `DATA_DIR`;
    - the AICP-300 rollout step 7 pilot run through the production RQ chain
      (`enqueue_process_run` → `run_submit_run` → poll → `run_finalize_run`).

**Durable failure record for link (i) (round 5 M2).** On `LinkFailed`,
`capture.record_link_failure` inserts one row through the executor:

- **Table:** new, insert-only `kb_link_failures` (S15): `id, publication_id
  unique, run_id, project_id, kind, path, st_dev, st_ino, st_mtime_ns,
  error_class, recorded_at, seq`.
- **Where it runs:** **after** `_release_lock`, outside any pipeline
  transaction. Nothing in the finalize critical section touches the DB for
  capture, which keeps P3 intact.
- **Wait:** up to `min(2 s, remaining − 90 s)` on the executor. When less
  than that remains, the row is submitted **fire-and-forget** (budget 0)
  rather than skipped, so it still lands if the DB is healthy.
- **Why after the release is safe.** A restitch that would replace the
  unlinked `final.mp4` needs a new chain to start: `run_submit_run` takes the
  lock (L388-405), submits, waits at least one poll interval
  (`RUN_POLL_INTERVAL_SEC`, 15 s, `tasks.py` L268), and only then reaches
  finalize, where link (ii) links the old inode before the stitch. The
  failure-row write takes milliseconds to ≤ 2 s. The scanner considers a
  (ii) link only after it is 30 s old (§3.3). In practice the row therefore
  exists before the (ii) link is evaluated, unless its write failed, which is
  the accepted residual.
- **Late pairing.** The scanner checks pairing from both sides each tick
  (§3.3). A proven (ii) link is enqueued with its `file_identity`, and
  C2 makes it historical when its inode is no longer live, or current when it
  still is. Nothing
  proven is discarded (round 7 M2).

**Link point (ii): pre-restitch preservation. Bytes only, never a version
(round 5 B1).**

- **Where.** In `_run_stitch_and_deliver`, immediately before
  `media_mod.stitch(...)` (L2682), inside the chain that holds the per-run
  Redis lock (`tasks.py` L388-405).
- **What.** If `capture_links` is true and `final_dst` exists with `st_nlink == 1`, call
  `link_preserve(kind="superseded", pub=f"superseded:{run_id}:{dev}-{ino}-{mtime_ns}")`.
  If `st_nlink > 1`, link (i) or an earlier (ii) already holds the inode.
- **On failure.** Log and Redis counter only. Point (ii) is a second chance,
  not a publication.
- **Same exception contract as (i), never skipped.** A
  `JobTimeoutException` here propagates into `finalize_run`'s handler, exactly
  as a timeout at the next line (the stitch itself) would. The cooperative
  timeout is never swallowed, so the run fails through its normal timeout
  path and the lock is released by `run_finalize_run`'s `finally`.
- **Why it can never become a version.** It may hold a final that was **never
  delivered**. For example: Drive delivery fails (L2737), the run goes to
  `failed` through the handler (L3401-3416), the operator retries or reruns
  from `failed` (`api_v2.rerun_segment` L4790, L4867-4886), and the next
  stitch's point (ii) links the undelivered file.
- **How (ii) artifacts appear** (§3.3, §8.3, §10):
  - kind `stitched_final` (not `delivered_final`), `locator_kind=preserved_copy`;
  - an artifact check `present=True`;
  - listed in the Run's timeline under **"Preserved pre-restitch files
    (delivery unverified)"**, with the publication id;
  - **no `kb_capture_queue` item, no case version, not judgeable.**
    Feedback requires a case version, and Run-case versions come only from
    point (i), reconciliation or sync (S14).
- **The one exception.** If a `kb_link_failures` row records a failed point
  (i) for the **same** `(dev, ino, mtime_ns)`, the scanner treats the (ii)
  link as that publication's link. That row proves the inode was published
  after a `done` commit. The scanner then enqueues the link with the failed
  record's `publication_id` and reason. Two independent artifacts must agree
  on the identity.

**Residual loss, stated precisely (round 6 B1).** Link (i) is never skipped
and has no disk guard. A delivered final is left with neither preserved bytes
nor a durable DB record only if **all three** of these happen for the same
publication:

1. a genuine link-(i) syscall failure: `EXDEV`, `EACCES`/`EPERM`,
   `ENOSPC`/`EDQUOT` on the directory entry or new directory, `EMLINK`,
   `EIO`, or the file vanishing before `os.stat`;
2. loss of the `kb_link_failures` row (DB unavailable, the executor queue
   full, or the process killed before the fire-and-forget write lands);
3. a restitch that replaces `final.mp4` before reconciliation versions it,
   **and** link (ii) failing as well.

If (ii) succeeds but (2) happened, the bytes are preserved as history only
("delivery unverified"). The failures share causes, chiefly a full disk or
a misconfigured mount, so they are not independent. What remains is a log
line and Redis counters. A worker killed between the `done` commit and the
link is case (1) with (2) — no row is written — and is caught by (ii) or
reconciliation like any other link failure.

### 3.3 The scanner orders by live state (D12; rounds 4 B1, 5 B1/M1)

`finals_scan.scan_run(session, run_id, deadline)` holds the per-run logic.
The knowledge tick calls it once per Run that has pending links, under its own
deadline. The feedback start/sync path calls the **same function**
synchronously (S14).

**Pending.** A link file older than 30 s is *pending* exactly when
`store.publication_covered(session, run_id, publication_id)` is false
(storage `284df2c`). The storage layer decides, so the scanner never keeps its
own notion of coverage. It is covered when a valid version of the Run's
**current** case carries the publication (or `<id>~<n>`), or a queue item for
it still covers it (`_capture_still_covers`). Consumed `duplicate`/`appended`
items stop covering once their result is no longer valid for the current case
(case or version withdrawn, artifact withdrawn). The scanner then calls
`enqueue_final_capture`, which re-queues under the next `attempt_no` key, and
C2 records the publication for the replacement case (`P` or `P~1`), round 13
B1. For unproven `superseded`
links, *pending* means no valid `kb_artifacts` row exists for its locator.

Pending links fall into two classes:

| Class | Which links | Output |
|---|---|---|
| *Publishable* | point (i) kinds (`stitch`, `range`, `audio`, `assistant`); a point (ii) `superseded` link whose `(dev, ino, mtime_ns)` matches a `kb_link_failures` row, or that pairs under the NULL-identity rule (§3.2). A paired link takes that row's publication id and kind. | a queue item; **C2 decides** whether it is current or historical |
| *History only* | every other `superseded` link | an artifact and an artifact check, listed under "Preserved pre-restitch files (delivery unverified)" |

**Phase 1 — read-only, bounded (round 9 M2).** List the Run's link
directory and select **at most `K = 6` candidates** per Run per tick, in this
priority (round 10 M1): (1) the publishable link whose `(dev, ino)` equals the
live `runs.result_local_path` (one read-only `stat`, no lock), so the current
final is never starved behind a backlog; (2) other publishable links, oldest
`time_ns` first; (3) history-only links.
Only the selected links are stat-ed and sha256-hashed (their inodes are
immutable). Unselected links stay pending for the next tick, with no
continuation state. Nothing is written in Phase 1.

**Phase 2 — one short transaction per Run, with no Run lock.** Statement and
lock timeouts are 1 s. For each selected candidate:

- **Publishable:**
  - `get_or_create_artifact(kind="delivered_final",
    locator_kind="preserved_copy", locator=<link>, sha256)`;
  - `enqueue_final_capture(run_id, artifact_id, reason=<kind→reason>,
    publication_id=<pub>, file_identity=f"{dev}:{ino}:{mtime_ns}")`. The
    identity is the link's, and a hardlink shares its inode with the
    published file. This uses the storage API from `bb9b6fb`.
- **History only:** an artifact (`stitched_final`, `preserved_copy`) plus an
  artifact check.

The cost is K artifact writes and at most K queue inserts. `enqueue_final_capture`
takes no Run lock (store: "no Run lock (§6.1 C1)"), so Phase 2 never holds
`runs` and cannot delay a range promotion's
`session.refresh(run, with_for_update=True)` (`pipeline_range_attempt.py`
L1141-1143). The revision-9 Run-lock section and its 5-historical cap are
**withdrawn**.

Idempotency:
- `enqueue_final_capture` is `ON CONFLICT (capture_key) DO NOTHING`.
- Artifacts are get-or-create.
- A crash between Phase 2 and the next tick leaves the link not pending, or
  still pending; either way it is never lost.

**C2 decides current or historical (round 9 M1).** `store.drain_run_captures`
in `bb9b6fb` runs per Run under the Run lock, in `c1_seq` order (importer
tick, 200 ms lock timeout and skip, S9d). It re-reads the live
`runs.result_local_path` identity under that lock (`_live_identity`), so the
check sees the live final at that moment. The outcome:

| Condition | Outcome |
|---|---|
| Same bytes as the latest version, or a version with the same `source_record` already exists | `duplicate` |
| The item's `file_identity` (mandatory, NOT NULL since storage `221b340`; `enqueue_final_capture` refuses an empty one) differs from the live identity, or the live final cannot be stat-ed | appended as a **historical** version (`historical=True`, `published_at=item.created_at`), outcome `historical`, never current |
| Otherwise | the usual `stale` / `appended` rules |

So **"current = live" holds at the moment of the C2 decision**, not at scan
time. Consider a range promotion that commits under `FOR UPDATE`
(`pipeline_range_attempt.py` L1143, commit L1235) between the scan and the
drain. The previously-live item then fails the identity check and becomes
historical; the promotion's own link becomes current when its item drains.

The scanner no longer appends historical versions itself, and needs no
savepoint or `IntegrityError` handling. Duplicate publications are absorbed
by C2's `source_record` check and by the storage UNIQUE.

**Ordering note (R35).** A queue-made historical version's `published_at`
is the queue item's `created_at`, which is the scan time, not the link's
`time_ns`. The timeline therefore shows the link `time_ns` from the locator
name next to it. Only the relative order of historical versions is affected.

**Synchronous use: feedback start and sync (S14, round 5 M1).**

The problem: `store.sync_run_case_version` (L372-397, drain-first in
`62070a5`) creates a `local_path` artifact for the live file. A preserved link
for the same bytes captured later is consumed by `drain_run_captures`
(L1335-1360) as `duplicate` by sha. That never re-points the version and never
attaches the publication id, so an immediate feedback start would pin an
unpreserved version.

S14 changes the store:

- `sync_run_case_version(session, case, *, hasher, invalid, live_link_lookup=None)`.
- **Order.** Under the locks the caller already holds (Run → case), it runs:
  1. **Drain, bounded** (storage `221b340`): at most `SYNC_DRAIN_MAX = 20`
     queued finals under the Run lock. If more remain, sync raises 409
     "this run's captures are still being recorded; retry shortly" without
     syncing, and the card shows a "recording in progress — retry" state
     (round 10 M2). The backlog stays with the knowledge worker's drain.
  2. **Live-link lookup**, before any sha comparison. `live_link_lookup(run)`
     is supplied by `capture`. It lists `finals/<project>/<run>/` (bounded:
     one directory, at most a few dozen entries) and returns the publishable
     link whose `(dev, ino)` equals the live file's, or `None`.
  3. **If a link is found:** the artifact is get-or-created as
     `preserved_copy` at the **link** locator, with the sha computed by the
     hasher on the live path (same inode). If the latest version's sha
     differs, the version is appended with `source_record=<the link's
     publication id>` and the publication's reason; otherwise it is a no-op.
     The scanner later finds the artifact already exists for that locator, so
     the link is no longer pending and cannot produce a second version.
  4. **If no link is found:** current behaviour, a `local_path` artifact with
     `source_record=sync:<run>:<sha>`. P5/S10 treat it as unpreserved (current
     judging only).
- **Locks.** The lookup runs under the Run lock, so it cannot race a
  promotion's re-point. It reads the filesystem only.
- **Callers.** `capture` passes `live_link_lookup` to the start path (§10),
  to feedback sync and to the scanner. The importer's reconciliation
  (`reconcile_finals`) gets the same lookup, as part of S9f, before it queues
  a `reconcile:` publication.
- **Test (§12):** start immediately after finalize (scanner not yet run) →
  the version is `preserved_copy` with the `stitch:` publication id; the
  following scan creates no new artifact and no second version.

**Why this fixes round-4 B1, round-7 M2 and round-9 M1.** A descheduled stitch worker
linking an obsolete final after a newer promotion produces a publishable link
whose inode is not live. It becomes a **historical** version, which can never
be current. A late-proven delivered final likewise becomes a historical
version, so nothing proven is discarded.

---

## 4. Promotions

**Return contracts (round 8 M1).** Every promotion function returns the same
immutable descriptor as the stitch:

```python
PublishedFinal(run_id, project_id, path, kind, publication_id)   # the §3.2 dataclass, unchanged
```

Each caller links with exactly `capture.link_publication(desc)`, the same
signature as the stitch.

Its fields come from in-memory, already-committed objects. Neither the
function nor its caller re-reads the DB to build it.

| Promotion | Pipeline-side change | Caller that links |
|---|---|---|
| **Range** | New `poll_range_attempt_once_published(attempt_id, *, kie=None, gdrive=None) -> tuple[str, Optional[PublishedFinal]]`, holding today's body. Private `_handle_terminal_kie_state` (single caller, L1309) returns `(status, Optional[PublishedFinal])`. It returns a descriptor only on its own successful `_promote` path when `attempt.final_local_path` is set: built right after `_promote` returns (its commit is at L1235) from the in-memory `run.id`, `run.project_id` and `attempt.final_local_path`, with `range_attempt:<attempt.id>`. Every other return (`pending`, `failed`, "already promoted by another session" at L1517) gives `None`. The existing **`poll_range_attempt_once(...) -> str`** becomes `return poll_range_attempt_once_published(...)[0]`. Its 26 test call sites in `test_pipeline_range_attempt*.py` are unchanged. | `tasks.run_process_range_attempt_poll` (L1502-1514) calls the `_published` variant. On a descriptor it calls `capture.link_publication(desc)` as its last statement, after the existing `pending` → re-enqueue branch. Re-enqueue only happens for `pending`, which never carries a descriptor. |
| **Audio patch** | `process_audio_patch(patch_id, *, gdrive=None) -> Optional[PublishedFinal]`. Today it is `-> None` with returns at L511 (already done) and implicit at the end. It returns a descriptor right after `_promote` returns (L632; commit at L469), built from the in-memory `run.id`, `run.project_id`, `patch.output_path` and `audio_patch:<patch.id>`. The already-done path (L511) returns `None`, and exceptions propagate unchanged. The 20 test calls in `test_pipeline_audio_patch.py` ignore the return. | `tasks.run_process_audio_patch` (L1335-1346): `desc = process_audio_patch(...)`, then `capture.link_publication(desc)` as its last statement. **No fresh session read**; revision 8's re-read is withdrawn. |
| **Assistant** | New `promote_run_result_published(...) -> tuple[dict, Optional[PublishedFinal]]`, holding today's body. A descriptor is returned only on the fresh-promotion path that commits at L402, built from `intent.after_json["path"]`, `run.id`, `run.project_id` and `assistant_promotion:<intent.id>`. Every replay or early `_receipt` return gives `None`. The existing **`promote_run_result(...) -> dict`** returns `[0]`. Its callers in the four assistant test modules (11 calls) are unchanged. | `api_assistant.publish_run_result` (L477-489) calls the `_published` variant inside `_call`. After it returns, it calls `capture.link_publication(desc)` before returning the receipt. The API process has no RQ alarm; the same absorb-and-record policy applies anyway. |

**One timeout policy.** Every publication link (stitch link (i) and all three
promotions) is `link_publication`. It absorbs `BaseTimeoutException` and
ordinary exceptions, records a `kb_link_failures` row (≤ 2 s on the executor,
or fire-and-forget), and returns. Each one is the last statement after all
committed work in its job or request (§3.2).

**Promotions have no link (ii).** Their paths are new and never rewritten. A
claim directory is removed only on claim *loss*
(`pipeline_range_attempt.py` L250-256). A later stitch writes `final.mp4`,
which is a different path.

Ordering against stitches is handled by §3.3: the live inode wins. The scanner
appends the range segment version from B4's artifact.

The importer remains the fallback, with **S9f**:

- it attaches publication ids on dedupe;
- it appends no-longer-current promotions as historical versions;
- it does nothing for a publication id that already has a version or queue
  row (the same pending definition as §3.3).

## 5. Capture scope vs. access scope (P6; refines design §6.5; round 4 M2)

| Purpose | Scope | Notes |
|---|---|---|
| **What is recorded**: hooks, links, scanner, importer sources and reconciliation | `CaptureScope`, an **explicit** list of project UUIDs from `KNOWLEDGE_PROJECT_IDS`. `*` is **refused** for capture: logged, capture stays off. Empty or invalid means nothing is captured. | Needs no assistant variables. Those reach only `api` (`deploy/assistant/compose.app.yml` L5-11), while the workers get `env_file: .env` (`docker-compose.yml` L163). |
| **Who may read or write**: API, timeline, feedback, rules, advisor | `KnowledgeScope` = assistant allowlist ∩ `KNOWLEDGE_PROJECT_IDS`, unchanged | API only |

**Provenance (S13).**

- **Scope table.** New insert-only table `kb_capture_scopes(scope_sha256 pk,
  project_ids text[], first_seen_at, recorded_by)`.
  - `scope_sha256` is the canonical sha of the sorted list.
  - The knowledge tick inserts the row with insert-or-get on every run.
  - Every run worker inserts it at boot through the executor. This is best
    effort; the tick guarantees it.
- **Scope hash on each record.** Every capture-made attempt carries
  `recipe_context["capture_scope_sha256"]`. Every link carries `s<scope8>` in
  its name. Every importer batch report records the scope sha.
- **Result.** Any captured row can be traced to the operator decision that
  allowed it. Changing the list creates a new scope row, so the timeline shows
  which scope was active at capture time.

**Operator decision, needed for AICP-300.** Enabling capture for a project
records its production facts and preserves its finals. That is a retention
decision in its own right, separate from assistant visibility. **Operator must
explicitly approve the exact UUID list** at enablement. The approval goes
into the AICP-300 release packet, and the `kb_capture_scopes` row is the
in-database record. Projects outside the assistant scope are recorded but
cannot be viewed until access is granted.

---

## 6. Knowledge worker, scheduler, C2, reconciliation, locks

### 6.1 Slim worker and resource plan (round 4 B3)

**Why it cannot reuse `worker/worker.py`.** That entrypoint pre-warms
InsightFace at L103-125, about 2 GB RSS according to its own comment. It also
runs the boot run-reconciliation and starts the reconcile and automation
ticks.

**New `worker/knowledge_worker.py`.**

- Imports only `redis`, `rq` and `app.tasks.run_knowledge_tick` /
  `enqueue_knowledge_tick`, plus lazily the knowledge modules.
  - `app.tasks`' module-level imports are light (`tasks.py` L30-44:
    `json`, `logging`, `redis`, `rq`, `languages`, `config`).
  - The tick imports `app.models`, `app.knowledge.*` and `media` (`probe`
    shells out to ffprobe).
- **Never** imports `app.face`, `insightface`, `onnxruntime` or `cv2`. A unit
  test imports the entrypoint and the tick path and asserts none of those are
  in `sys.modules`. Whether `app.models` pulls a heavy dependency was not
  verified beyond reading its imports; the test is the guard.
- `Worker([Queue("knowledge")]).work()`, **without** RQ's scheduler (round 15
  B2). Nothing on this queue is delayed: the supervised ticker (§6.2) enqueues
  each tick when it is due, so there is no unsupervised scheduler subprocess
  and no `ScheduledJobRegistry` entry that could be "live" but never run.
- No boot run reconciliation, no reconcile or automation tick, no InsightFace
  pre-warm.
- It does not schedule anything itself; the ticker does (§6.2). The tick
  inserts the capture-scope row (S13).

**`docker-compose.yml` service `knowledge-worker`.**

- Same image, `env_file`, `./data` and `./secrets` mounts, `user`,
  `cap_drop` and `security_opt` as `worker`.
- `command: python -m worker.knowledge_supervisor` (PID 1; it starts the RQ
  worker `worker.knowledge_worker` and the ticker `worker.knowledge_ticker`,
  §6.2), plus `restart: unless-stopped`.
- Resources: `mem_limit: 768m`, `memswap_limit: 768m`, `cpus: "0.5"`.
- One replica. The `--scale worker=2` rule is unchanged.

**Resource plan.** This adds a line to the budget comment at
`docker-compose.yml` L134-147:

| Item | vCPU peak | RAM |
|---|---|---|
| Existing total | ~5.2 | ~14.7 GiB |
| `knowledge-worker` | 0.5 | 0.75 GiB |
| **New total** | **~5.7** | **~15.45 GiB** |
| Headroom on the 4 vCPU / 16 GiB host | — | ~0.55 GiB |

- That headroom is tight. The worker's 768 MiB is a hard cap, enforced by the
  OOM killer inside its own cgroup, so it can never take the run workers'
  memory.
- CPU: hashing ≤ 100 MB per final at 0.5 vCPU takes seconds, and the tick is
  time-bounded.
- If the owner prefers, set `mem_limit: 512m` and measure in AICP-300. Not
  measured here.

**Recovery and liveness treatment** (verified):
- `recovery._live_run_ids` checks only `overlay`, `analyze`, `process` and
  `default` (`recovery.py` L195-199, L238), so jobs on the `knowledge` queue
  never make a run look live.
- `_live_worker_names` (L158-177) lists every heartbeating worker. That only
  matters for started run jobs, which the knowledge worker never takes. So it
  is never counted as running a run.
- The knowledge worker does no run recovery.
- Its own liveness is the tick's `last_ok` key plus the RQ heartbeat.

### 6.2 Scheduler: a supervised ticker with per-slot fencing (round 15)

The revision-14 design chained ticks with `enqueue_in`, a chain key, a
`finally` continuation and a repair watchdog, and every repair path had a
failure mode. It is replaced by a design with **no chain, no RQ scheduler and
no destructive key operation**.

**Processes.** The knowledge-worker container's PID 1 is
`worker/knowledge_supervisor.py`, a loop with no Redis and no threads. With
`subprocess.Popen` it starts:
- the RQ worker (`worker.knowledge_worker`, queue `knowledge`, no scheduler);
- the ticker (`worker.knowledge_ticker`, its own process and its own Redis
  connections; nothing forks from it).

It `os.wait()`s. If either child exits, it terminates the other and exits
non-zero, and compose `restart: unless-stopped` restarts the container. The
RQ parent therefore never holds extra threads (the redis-py fork-lock concern),
and a dead ticker is replaced in seconds even while a tick runs.

**Ticker loop** (every 30 s, and immediately at boot):

```python
INTERVAL = int(os.getenv("KNOWLEDGE_TICK_INTERVAL_SEC", "300"))
TICK_TIMEOUT = 900                                 # importer LEASE_TTL_SEC; TICK_MAX_SEC=600
slot = int(time.time()) // INTERVAL
if not settings.KNOWLEDGE_CAPTURE_ENABLED or not capture_scope().valid: continue
if tick_is_live(): continue                        # never overlap ticks
if redis.set(f"reskin:knowledge-tick:slot:{slot}", ticker_id, nx=True, ex=2 * INTERVAL):
    Queue("knowledge").enqueue("app.tasks.run_knowledge_tick",
                               job_id=f"knowledge-tick-{slot}", job_timeout=TICK_TIMEOUT)
```

`tick_is_live()` uses the liveness rule from `recovery.py` (L22-29, L241-248):
1. `StartedJobRegistry("knowledge").cleanup()`;
2. a `run_knowledge_tick` is live if it is **queued**, or **started on a
   worker whose heartbeat is still alive** (`Worker.find_by_key(job.worker_name)`
   exists and `last_heartbeat` is within the worker TTL).

An orphaned Started entry from a killed work-horse or a redeploy is not live.

**Why this is safe without a repair lock.** The only write that decides
anything is `SET slot-key NX`. It is idempotent and never deleted early:
- Two tickers (overlapping containers during a restart) race on the same
  slot key, and exactly one enqueues.
- A paused ticker that wakes later works on the slot it computes then, and
  that slot is already taken or legitimately free.
- Nothing is ever DELeted, so no lease can be lost mid-action.

The worst case is one skipped slot (a tick was still live, or the ticker was
down); the next slot runs. A wedged tick is killed by its 900 s `job_timeout`,
its worker heartbeat continues, and the next slot sees no live tick.

```python
def run_knowledge_tick():                          # no self-scheduling, no chain
    # Execution-time mutual exclusion (round 16 M1, round 17 M1). The lock value is
    # the holder's RQ job id; it has a long safety TTL (24 h) and is NEVER taken
    # over by expiry. A newcomer takes it only from a holder that is not live
    # by tick_is_live()'s rule: queued, or started on a worker whose heartbeat
    # is alive. That covers an uninterruptible horse whose parent still
    # heartbeats, which keeps the lock.
    me = get_current_job().id
    if not acquire_tick_lock(me):                  # Lua: SET NX, or CAS-replace a holder proven dead
        log.info("knowledge_tick_skipped_running"); return
    capture.begin_job()
    try:
        record_capture_scope()                     # S13
        finals_scan.run(deadline=monotonic() + 240)   # §3.3
        knowledge_import_tick()                    # importer
        _get_redis().set("reskin:knowledge-tick:last_ok",
                         json.dumps({"at": utcnow_iso(), "capture_health": read_health_keys()}), ex=86400)
    except BaseTimeoutException:
        raise                                      # RQ's timeout is never swallowed (marks the job failed)
    except Exception:
        log.exception("run_knowledge_tick")
    finally:
        release_tick_lock(me)                      # Lua: DEL only if the value is still `me`
```

`acquire_tick_lock(me)`:
1. `SET reskin:knowledge-tick:running <me> NX EX 86400`. If it is set, the
   lock is ours.
2. Otherwise read the holder id `h` and apply **positive evidence of a running
   horse**, not registry cleanup (round 18 M1). `h` is live while
   - any worker whose heartbeat key is alive reports `h` as its current job
     (`Worker.get_current_job_id()`, from the worker hash). The work-horse sets
     it in `prepare_job_execution` (`rq/worker.py` L1334, L1405) before the job
     function runs, so it is always set before the tick takes the lock. It is
     cleared either by the horse's own success/failure handling (L1386, L495),
     which runs after the job function, and so after its `finally` released
     the lock, or by the parent's `handle_work_horse_killed` → failure handling
     once it has reaped a killed horse, including its SIGKILL at
     timeout + 60 s. Or
   - `h` is still queued.

   `StartedJobRegistry` expiry and cleanup are **not** consulted, because
   cleanup can fail a job at the timeout + 60 boundary while the horse is still
   alive. Only when `h` is neither queued nor anyone's current job is it
   replaced, by a Lua compare-and-set (`if GET == h then SET <me> EX 86400`).
3. Otherwise skip.

The proof no longer depends on the relationship between `job_timeout` and a
TTL:
- An expired TTL never causes a takeover.
- A takeover requires the holder's worker heartbeat to be gone. RQ only lets
  that happen when the worker process has stopped or been removed, and the
  work-horse dies with its parent: same container, killed on container stop,
  SIGKILLed by the monitor when the parent is alive.

The remaining window is a worker whose heartbeat key lapses while its horse
runs, i.e. a parent stalled past the RQ worker TTL with no container restart,
or a parent killed while its horse survives. That
window is stated as R39. The scanner and importer are idempotent and
lease-guarded, so a duplicate execution there costs load, not correctness.

Admission (the slot key) bounds how many ticks are *queued*; the `running` key
guarantees that at most one *executes* while its holder is live.

**Liveness gate.** `last_ok` older than 2 × INTERVAL fails capture enablement
and the pilot gate. Supervisor restarts log `knowledge_container_restarted`,
and skipped slots `knowledge_tick_slot_skipped`.

### 6.3 C2, reconciliation, locks

- **C2 and reconciliation** come from the importer at or after `ba493fc` (S9a-S9d, S9f, `file_identity`).
  Reconciliation versions an unlinked live final of a done run.
- **Locks.**
  - **The scanner takes no Run lock at all.** Phase 2 only creates
    artifacts and queue items (§3.3).
  - The Run lock is taken by C2 (`drain_run_captures`, importer tick, 200 ms
    lock timeout, S9d), by S14 sync/start and by feedback, always
    `runs` → `kb_cases`.
  - The link points take no locks.
  - Executor sessions take `runs` **only** to create a missing Run or
    segment case (§1.4), with a 200 ms acquisition timeout and server-side
    statement and idle-in-transaction timeouts bounding the hold. All other
    executor writes (attempts, observations, artifacts) take no Run lock.

---

## 7. Failure policy

- **`_guarded` calls.** All exceptions stay on the executor thread. The
  caller's only blocking operation is `Event.wait(budget)`.
  - A `JobTimeoutException` arriving during that wait (main thread only) is
    re-raised.
  - The 90 s margin makes that unreachable.
- **Link points.**
  - Both are syscalls only: stat, makedirs, link. No data blocks are written,
    there is no deadline gate and no disk guard.
  - Point (ii), `link_preserve`, re-raises `BaseTimeoutException` (RQ
    `timeouts.py` L6-17) into finalize's normal timeout path.
  - Point (i), `link_publication`, absorbs it and records a
    `kb_link_failures` row, because it is the last statement after all
    committed work (§3.2).
  - Both catch ordinary `Exception`s.
  - Neither catches `KeyboardInterrupt` or `SystemExit`.
- **Evidence.**
  - Durable evidence: link files, artifacts, artifact checks, observations
    and `kb_link_failures`.
  - Not evidence: logs and counters. They back only the
    `link_failure_unrecorded` residual.
- **Why the pipeline cannot change.** Capture exceptions never cross into the
  pipeline, and:
  - P1: no capture between a paid call and its commit;
  - P2: bounded waits, nothing queued;
  - P3: no DB access and no hashing on the submit, finalize or promotion
    paths, and no capture writes inside pipeline transactions;
  - pipeline code never reads a capture result;
  - D3 is behaviour-preserving, plus the truncated-final fix;
  - D6 is golden-tested.
- **Remaining timing cost.**
  - A, B and T sites: ≤ 3 s each, pilot projects only, all after commits.
    B2 runs after a commit of the outer session.
  - Links: microseconds.
  - Nothing runs inside an open submit transaction. The `note_upload` hook is
    withdrawn.

---

## 8. Flags, disk, capacity, unpreserved versions

| Flag | Gates |
|---|---|
| `KNOWLEDGE_CAPTURE_ENABLED` | everything below |
| `KNOWLEDGE_PROJECT_IDS` (explicit list) | capture scope; also narrows access |
| `KNOWLEDGE_PRESERVE_ATTEMPT_MEDIA` | `attempts/` preservation only |
| `KNOWLEDGE_ENABLED` + `KNOWLEDGE_FEEDBACK_UI` | API and card |

Final links are not behind the preserve flag: they are the record.

**Layout.**

| Path | Contents |
|---|---|
| `finals/<project>/<run>/…` | links (i) and (ii) |
| `attempts/<project>/<attempt>.<ext>` | attempt media |
| `external/<project>/<sha>.<ext>` | external uploads |
| `tmp/` | in-progress writes |

All of these live under the shared `./data` mount. nginx reaches `data/` only
through an internal location (`nginx.conf` L154-159).

**Disk guard.** The guard `free − nbytes ≥ max(20 GB, 10 %)` covers only the
writers that copy bytes: attempt-media preservation, which is skipped and
recorded unpreserved, and external uploads, which get a 507. **Links have no
guard** (§3.2): they write a directory entry, not data. The space they retain
is managed by `disk-report` and `release-preserved`.

**Capacity.** These are derived figures, not measured:
- a replaced final keeps ≤ 100 MB alive;
- an attempt result is 5-60 MB;
- the planning figure is ≤ 1 GB per pilot run and ≤ 30 GB per pilot month.

**Operations.**
- `disk-report` shows usage, plus stale `*.part.mp4` files older than one day
  (R33).
- `release-preserved` unlinks preserved files and appends
  `kb_artifact_checks(present=false)` for them.
- Nothing is deleted automatically.

**Unpreserved versions (P5, S10).** Link-made artifacts are `preserved_copy`
and immutable. Reconciliation, importer and `local_path` segment versions are
not:
- They can be judged only while current, with verified bytes: the Run sync, or
  `expected_artifact_sha` for other cases.
- Historical judging requires `preserved_copy` or `drive_file`.
- The card shows "not preserved" for these versions, with no player and no
  historical controls.

---

## 9. Recovery, deploy, rollout, pilot gate

| Situation | Behaviour |
|---|---|
| Worker dies between intent and call | Open intent. A recovered dispatch is matched through the locator basis. This is matching, not billing evidence. |
| Worker dies inside A5, or between acceptance and commit | No capture. The draw becomes an importer ambiguous attempt. |
| Worker dies after `done`, before link (i) | No row is written. Reconciliation versions the file while it is live, and link (ii) preserves its bytes before a restitch. This is part of the §3.2 residual. |
| A descheduled stitch worker links an old final late | The link becomes a historical version (S16) and never the current one (§3.3). |
| B1 ∥ B3 | S7 dedupe plus the latest-sha check |
| Rerun clicked during finalize | Capture adds only microseconds, so this is unchanged from today. |
| Rolling deploy | Flags are read at import. Abandoned executor work dies with the process. Link files are atomic. |

**Rollout.**

1. Merge `feat/knowledge-storage` **at or after `94ee5d0`**, the importer
   **at or after `ba493fc`**, and AICP-299. The release packet records both
   exact gated hashes; an older importer would fail on the mandatory
   `file_identity`. The single additive AICP-299 revision in
   §11.1 is unconditional.

   **Invariant (round 9 B1).** `k1a7c3e9f205` must **not** be applied to any
   database (production, staging or a restored dump kept for later use) until
   it contains S16, the `source_record` UNIQUE, and `kb_capture_queue.file_identity`.
   That means the storage branch at or after `bb9b6fb`.
   - Alembic identifies revisions only by id. A database migrated with an
     earlier body of `k1a7c3e9f205` could never be fixed by re-running it:
     `upgrade head` would be a no-op.
   - AICP-296 ships exactly that content, and its release packet must record
     the storage commit hash it migrated with.
   - Pre-flight before `upgrade`: `SELECT version_num FROM alembic_version`
     must be `j0e5f2a7b914`, not `k1a7c3e9f205`.
   - Post-flight: `\d kb_case_versions` shows `historical` and
     `published_at`, and `\d kb_capture_queue` shows `file_identity`.
   - If a database already carries an S16-less `k1a7c3e9f205`, which is only
     possible on disposable rehearsal databases, the fix is to drop and
     re-create it. A newer body under the same id never repairs it.
2. Back up the DB, then migrate first. Flags stay off.
3. Deploy with the overlays and `--scale worker=2`. This also creates
   `knowledge-worker` (768m / 0.5 CPU).
   - Restart nginx and verify through :8847.
   - Check boot lines and the `knowledge-worker` memory with `docker stats`.
   - The atomic stitch ships here, flag-independent.
4. Drain: no active run, range attempt, segment-image job or audio patch. Wait
   one reconcile tick.
5. **Operator approves the exact pilot UUID list.** Record it in the AICP-300
   packet (§5).
6. Enable `KNOWLEDGE_CAPTURE_ENABLED=true` and
   `KNOWLEDGE_PROJECT_IDS=<explicit UUIDs>` with `up -d`, then restart nginx.
   Check:
   - worker boot lines show `capture_scope=<n>`, never `*`;
   - the `kb_capture_scopes` row exists;
   - the `knowledge-worker` container runs its supervisor with both children
     (RQ worker and ticker), a `reskin:knowledge-tick:slot:<n>` key appears
     every interval, and `last_ok` is fresh (< 2 × interval);
   - capture health keys exist.
7. Run one pilot run, then confirm:
   - attempts, submissions and outputs exist;
   - the `finals/` link exists;
   - the artifact, queue item and C2 version exist.

   One tick should show 0 unmatched video and image draws.
8. Decide on attempt preservation, then turn on the UI flags.

**Pilot gate.** Over the window, every one of these must hold:

- **(a) Draws.** Every draw of kind `video`, `image`, `range video` or
  `segment image` is matched to an attempt, or is marked ambiguous.
- **(b) Publications.** Every done-run publication (stitch or promotion)
  has one of:
  - a publishable link artifact;
  - a `kb_link_failures` row, which is durable and counted. It is preserved
    if a matching (ii) link exists;
  - a reconciliation or S14 sync version.

  `link_failure_unrecorded` (log + counter only) is reported separately. Each
  occurrence is a residual case the owner accepted (§13 decision 2).
  Pre-restitch (ii) files are reported as preserved history, not as
  publications.
- **(c) Liveness.** `last_ok` was never stale, and no existing health key
  was unhealthy (§1.3) while pilot activity existed.

Provider billing completeness is not claimed (R8).

---

## 10. Feedback UI

Unchanged from revision 3, as verified in round 2:

- the card mounts outside `#run-status-content`;
- the card shows only when the run is `done`;
- three read-only GET states;
- an idempotent `POST …/knowledge/start`: pre-hash, then a 5 s
  `lock_timeout`, then `done` + ensure + drain + sync;
- the write endpoints S5 and S10, with the error mapping;
- a fresh key on 409;
- multipart forms, the upload disk guard, unhashed Drive links, and no
  invented reasons.

Access uses `KnowledgeScope` (§5). Run versions show "preserved" or "not
preserved".

---

## 11. Amendments

| # | Change | Status |
|---|---|---|
| S1 | `dispatch_basis` parameter (locator-based, §1.5) | 299 |
| S2 | `attempt_for_task`, `latest_submitted_attempt` | 299 |
| S3 | drain before sync | **done** `62070a5` (contained in `9c9e3eb`) |
| S4 | `VERSION_REASONS += regeneration, external_repair` | 299, unconditional |
| S5 | `record_external_repair` | 299 |
| S6 | memoised `InvalidSet.load` | 299 |
| S7 | `record_observation(dedupe=)` + per-attempt lock | 299 |
| S8 | sync and Run-case feedback require `done` | 299 |
| S9a-d | importer | **done** `5892b33` (rollout requires `ba493fc`+) |
| S9f | importer: attach publication ids on dedupe; append no-longer-current promotions as **historical versions** (S16), not skips; skip when a live link exists; `live_link_lookup` in reconcile | 297 follow-up |
| S10 | verified bytes for current judging; preserved copies for historical | 299 |
| S12 | `CaptureScope` (explicit list only) in `access.py` | 299 |
| S14 | `sync_run_case_version(..., live_link_lookup=)`: record the live final's publishable link as a `preserved_copy` version with its publication id; also used by start, scanner and importer reconcile (§3.3) | 299 (+297 for reconcile) |
| S13 | `kb_capture_scopes` table; `OBSERVATION_KINDS += input_content` (schema: §11.1) | 299 |
| S15 | `kb_link_failures` table, with NULL-able identity + `failed_at_ns` (schema: §11.1) | 299 |
| S16 | Historical versions (`kb_case_versions.historical`, `published_at`; current/latest/state/sync/drain ignore historical; judging needs `historical=true`; `version_no` max+1; refused for derived/manual/external) **plus** the partial UNIQUE `(case_id, source_record) WHERE source_record IS NOT NULL`, with `source_record` unique by construction in every writer | **Provided by the storage layer inside the undeployed `k1a7c3e9f205`** (`feat/knowledge-storage` `3efff8e`, verified by Terra in round 9), together with `kb_capture_queue.file_identity`, `store.file_identity(path)`, `enqueue_final_capture(file_identity=)` and C2's live-identity revalidation with outcome `historical` (`9c9e3eb`). Not part of AICP-299's migration. |

Withdrawn: S9e and S11.

### 11.1 Schema changes: the single authoritative list (round 6 m2)

Every schema change AICP-299 makes is listed here and nowhere else. Other
sections refer to this table.

**The revision.** It is **one** additive Alembic revision with
`down_revision` = `k1a7c3e9f205`, or the head at implementation time.
`k1a7c3e9f205` (at or after `bb9b6fb`) already contains S16 (historical
versions + `published_at`), the partial UNIQUE on `(case_id, source_record)`
and `kb_capture_queue.file_identity`; those are not repeated
here. This revision contains **only** S4, S13a, S13b and S15. Other sessions add migrations, so re-check with
`alembic heads` first. It has no data backfill and touches no non-`kb_`
table.

| # | Object | Change | Constraints / triggers | Downgrade |
|---|---|---|---|---|
| S4 | `kb_case_versions.reason` CHECK | drop and recreate the CHECK with `VERSION_REASONS += ("regeneration", "external_repair")` (`taxonomy.py` L13-14) | — | recreate the old CHECK; refuses if any row uses a new value |
| S13a | new table `kb_capture_scopes` | `scope_sha256 text pk, project_ids text[] not null, first_seen_at timestamptz, recorded_by text, seq bigint identity` | insert-only: the row and TRUNCATE immutability triggers, as on every `kb_` table | drop; refuses while it has rows |
| S13b | `kb_attempt_observations.kind` CHECK | drop and recreate the CHECK with `OBSERVATION_KINDS += ("input_content",)` (`taxonomy.py` L27-28) | — | recreate the old CHECK; refuses if any row uses it |
| S15 | new table `kb_link_failures` | `id uuid pk, publication_id text unique not null, run_id text not null, project_id, kind, path text not null, st_dev bigint NULL, st_ino bigint NULL, st_mtime_ns bigint NULL` (all three NULL together; CHECK), `failed_at_ns bigint not null, error_class, recorded_at timestamptz, seq bigint identity`; indexes on `(run_id, failed_at_ns)` and `(st_dev, st_ino, st_mtime_ns)` | insert-only triggers; CHECK `kind IN ('stitch','range','audio','assistant')` | drop; refuses while it has rows |

**Requirements on the revision.**
- **Downgrade guard.** The whole downgrade refuses while any `kb_` table has
  a row. This is the same guard as `k1a7c3e9f205`.
- **Models and parity.** `taxonomy.py` and `knowledge/models.py` change in the
  same commit. The schema-parity PG test (`test_knowledge_pg.py`) is extended
  to the two new tables and both CHECKs.
- **Rehearsal.** Upgrade → downgrade → upgrade on a schema-only dump, the
  same procedure as AICP-295.

**Not schema changes**, listed here for completeness:
- link files under `data/knowledge/`;
- Redis keys (`reskin:knowledge-tick:*`, `reskin:knowledge:capture:*`);
- the `knowledge-worker` compose service.

---

## 12. Test matrix

Runs in the `re-skin-api` image. PG tests use `RESKIN_TEST_PG_URL`.

| Area | Proves |
|---|---|
| Atomic stitch | Both branches write temp then replace. A failure leaves the old final intact and no temp file. An open fd reads old bytes. Existing suites pass. |
| Return contract | `_run_stitch_and_deliver` returns `PublishedFinal` on the success path, including delivery-only reuse, and `None` on the incomplete and no-stitch paths. Exceptions propagate unchanged. `finalize_run` returns it, and raises `ValueError` for a missing run. The four existing `test_pipeline_v2.py` callers pass unchanged. A patched `finalize_run` returning `None` produces no link. |
| Link (i) | Present before `_release_lock`. Absent when finalize raised or on the incomplete or no-stitch paths. Out-of-scope projects make no syscall. Never skipped, at any remaining job time, and no disk guard. `os.stat` runs before `os.link`. Failures (`EXDEV`, `EACCES`, `ENOSPC`, vanished file) leave the run `done`, release the lock **before** the `kb_link_failures` write (spy order), and the row lands within 2 s, or fire-and-forget when the deadline is near. `process_run` (`capture_links=False`) never links. |
| Exception contract | Link (ii) re-raises an injected `JobTimeoutException` unchanged → finalize handler path, lock released. Link (i) **absorbs** an injected `JobTimeoutException`, during either `os.stat` or `os.link` → a `kb_link_failures` row (NULL identity if `stat` was interrupted), lock released, job returns normally, run stays `done`. Neither link catches `KeyboardInterrupt`. No link is ever skipped for time. |
| NULL-identity pairing | Row with NULL identity at `t0` → the first (ii) link of the same run and path with `time_ns > t0`, before the next (i) link, pairs. A (ii) link after a later (i) link does not. A promotion row with NULL identity never pairs. |
| Durable link failure | `os.link` raising `EXDEV`/`ENOSPC` at (i) → `_release_lock` first, then a `kb_link_failures` row within 2 s. A row inserted after the (ii) artifact was recorded as history → the next tick appends it as a **historical version** (S16) with the row's publication id. DB unavailable → `link_failure_unrecorded` log and counter, lock still released on time. |
| (ii) never a version | Drive fails → run `failed` → rerun → (ii) links the undelivered final → scanner: history-only `stitched_final` artifact, no queue item, no version. A matching `kb_link_failures` row → the (ii) link becomes a version: current if it is still live, historical otherwise. |
| S14 | `POST start` right after finalize, before any scan → `preserved_copy` version with the `stitch:` publication id. The later scan adds nothing. No link present → `local_path` version (unpreserved). |
| Heartbeat | No capture calls for 10 min during a provider wait → key still fresh. A killed heartbeat thread is restarted. A missing key is not counted as unhealthy. |
| Link (ii) | Restitch with link (i) present (`st_nlink == 2`) creates no extra link. Restitch with link (i) failed creates a `superseded` link holding the old bytes. A rerun with `result_local_path=None` still links `final.mp4` before the stitch. |
| **Round-4 B1 scenario** | Stitch final A delivered; range promotion B (link B); a descheduled stitch worker then creates link A late. The scan enqueues both with their `file_identity`. Drain: A → `historical` (identity ≠ live), B → `appended`. Current = B whatever the enqueue order. |
| **Round-9 M1 scenario** | Link A enqueued while A is live. A range promotion B then commits under `FOR UPDATE` before the drain. Drain: A → `historical`; B (scanned later) → `appended`. Current never equals A after B's commit. |
| S16 | Appending a historical version never changes `case_state` (current version, verdict, accepted artifact). `expected_version_no` uses the latest non-historical version. Feedback on a historical version without `historical=true` is refused, and with it is accepted and does not affect state. `version_no` = max + 1. Refused on derived cases. Owned by the storage layer's tests in `k1a7c3e9f205`; capture tests cover only the scanner's use of S16. |
| Scanner bounded work (round 9 M2) | 20 pending links → at most 6 stat/hash and 6 artifact/queue writes per tick (spy counts). The remainder is processed over later ticks. Phase 1 writes nothing (no-DB spy). Phase 2 never takes `runs` (lock spy), and a concurrent range `_promote` is not delayed by it. |
| Uniqueness | Two scanners on one Run → one queue item (`capture_key` ON CONFLICT) and one version. S14 and the scanner racing on the same live link → one version: C2 `duplicate` by `source_record`. |
| Promotion descriptors (round 8 M1, round 9 M3) | Range: the `_published` variant returns `("done", desc)` only on its own promotion; `poll_range_attempt_once` still returns `str`, and the existing tests pass. Audio: `process_audio_patch` returns `desc` after promotion and `None` on the already-done path; the wrapper links with no DB read (session spy). Assistant: `_published` returns `desc` on a fresh promotion and `None` on replay; the endpoint links after `_call`. An injected `JobTimeoutException` at each promotion link is absorbed and recorded. |
| C2 lock | Run row held by another session → the drain skips within 200 ms (S9d). Rescans are idempotent: a link that is no longer pending is never re-hashed. |
| Migration invariant | Pre-flight refuses `upgrade` when `alembic_version` is already `k1a7c3e9f205` without `kb_capture_queue.file_identity` (packet check). The parity test covers `historical`, `published_at`, `file_identity` and the UNIQUE. |
| Single descriptor | All four publishers return `PublishedFinal` with all five fields; `link_publication` takes only the descriptor (signature test); `finalize_run(publication_id=None)` mints `stitch:<run>:<hex>`. |
| Feedback before scan | Sync drains and syncs against live bytes (existing `62070a5` tests, extended with pending links). |
| Dispatch basis | Recovered dispatch with re-uploaded refs (new URLs, same locators) → same `dispatch_key`, `redispatch`. Changed override → new intent. `reference_set_sha256` is NULL. No `open()` of reference files from any pipeline thread (mock). |
| `input_content` | The worker appends `{locator: sha}` for readable local locators. The attempt row is unchanged. |
| Executor | A blocking body returns within budget + 50 ms. Queue full → skip. A killed thread is replaced on the next call. An `OperationalError` drops the session and the next job reconnects. Health key written. Daemon threads die with `os._exit`. |
| Capture scope | `*` → capture off + log. Explicit list → capture for listed projects only. A worker with no `ASSISTANT_*` env still captures. The API's access scope still intersects. The `kb_capture_scopes` row exists and `recipe_context` carries the hash. |
| First case creation (§1.4) | First capture of a new Run creates its Run and segment cases under `runs → kb_cases`. With the Run held by another session: skips within ~200 ms, counts `capture_skipped_run_locked`, writes nothing. The importer later creates the cases and imports the attempt (ambiguous from the draw when the request is unknown). A stalled executor (simulated sleep between statements) loses its session after 500 ms and the Run row is free again. |
| Knowledge worker | Importing `worker.knowledge_worker` + the tick path never loads `app.face`, `insightface`, `onnxruntime` or `cv2`. It listens only on `knowledge`, **without** an RQ scheduler. No run reaper or automation. Ticker: one enqueue per slot under two racing tickers (fake clock), a skip while a tick is live, an orphaned Started entry not counted as live, the supervisor exiting when either child dies. Tick mutual exclusion: a second tick started while one runs exits immediately; the running lock is released only by its owner token. `last_ok` includes health. `recovery._live_run_ids` is unaffected by knowledge jobs. |
| A, B, T, S7, P5/S10, UI, builders | as revision 4 |

---

## 13. Open risks and operator decisions

**Operator decisions for Operator (AICP-300):**

1. Approve the exact capture UUID list. It is recorded in `kb_capture_scopes`.
2. Accept the precise residual (§3.2): a delivered final with no bytes and no
   DB record when link (i), the `kb_link_failures` insert, and link (ii) or
   reconciliation before the next restitch all fail. They share causes, for
   example a full disk. What is left is a log line and counters.
3. Approve the `knowledge-worker` resource line (768m / 0.5 CPU), which leaves
   about 0.55 GiB of host headroom.
4. Decide on attempt-media preservation.

**Risks:**

| # | Risk | Status / resolution |
|---|---|---|
| R1, R2 | Redispatch matching and input hashes | Now locator-based (D1). Content equality is not guaranteed. `input_content` observations come later. |
| R5/R30 | Lock-hold extension | Gone (microsecond links) |
| R11 | Capture depends on assistant env | Gone (§5) |
| R16 | Cases created without the Run lock | Reviewer to accept |
| R28 | Pilot timing inside the submit transaction | **Gone** (`note_upload` withdrawn) |
| R29 | Link (i) + `kb_link_failures` + (ii)/reconcile all fail, or the file is deleted outside the pipeline | Residual; operator decision 2; not zero |
| R31 | Third container | Slim, budgeted (§6.1); CLAUDE.md update ships with 299 |
| R32 | libpq `tcp_user_timeout` | Not verified. Affects only abandoned threads. |
| R33 | Atomic stitch doubles disk transiently; SIGKILL leaves a temp file | ≤ 100 MB; reported by `disk-report` |
| R34 | Links keep deleted runs' inodes alive | Intended; `release-preserved` |
| R35 | Historical `version_no` is allocation order, not publication order | History is displayed by `published_at` (link `time_ns`). Current = live. |
| R39 | **New.** A worker whose heartbeat lapses (parent stalled past the RQ worker TTL) while its horse still runs a tick | The tick lock is taken over; for that window two ticks can run. Scanner and importer are idempotent and lease-guarded, so the cost is load, not correctness. |
| R36 | **New.** `knowledge-worker` OOM inside 768m (e.g. a large probe) | It dies alone. The supervisor exits and Docker restarts the container within seconds. The restarted worker has a new name and the old worker's heartbeat key expires, so the killed tick is no worker's current job and its `running` lock is taken over by the next slot's tick, by liveness and not by the 24 h TTL. Measure in AICP-300. |
| R37 | **New.** A local locator rewritten in place matches as "same dispatch" | Documented weaker guarantee (§1.5) |
| R38 | A late-proven (ii) link whose inode is no longer live | **Resolved** by S16: it becomes a historical version, judgeable with `historical=true`. |

---

*The round tables below are a record of each round's decision. Where a
later round changed a decision, §0-§13 are authoritative. For example, round
5's "deadline gate on links" was withdrawn in round 6, and round 6's "re-raise
at (i)" was changed in round 7.*

## Review round 1 resolution (`0ab9e76`)

| Finding | Final resolution (rev 5) |
|---|---|
| B1 interruptible capture | P1, P2, P3 |
| B2 undelivered final published | Links only after the `done` / promotion commit; S8; S9b; the scanner requires `done` (§3.3) |
| B3 lost C1 | Two link points; scanner; reconciliation; mandatory tick |
| M4 coverage claim | D8 |
| M5 B dedupe | S7 |
| M6 bootstrap | §10 |
| M7 D1 wording | §1.5 |
| m8 S4, disk, capacity | S4 unconditional; §8 |

## Review round 2 resolution (`24dceb4`)

| Finding | Final resolution (rev 5) |
|---|---|
| B1 A5 / connect timeout | No capture in the A5 loop; executor + per-thread connection timeouts |
| B2 deferred C1 not durable / fair | Links (i)+(ii); slim dedicated worker |
| M3 unpreserved fallback | P5/S10 |
| M4 `note_upload` on non-pilot | `note_upload` **withdrawn** (rev 5) |

## Review round 3 resolution (`5217950`)

| Finding | Final resolution (rev 5) |
|---|---|
| B1 outbox savepoint | Withdrawn; nothing in pipeline transactions |
| M1 promotion ids / order | Live promotion links; scanner by live state; S9f |
| M2 assistant env in workers | Explicit capture scope (§5) |
| M3 rerun delay | Moot (microsecond link) |
| M4 executor lifecycle | §1.3 (now supervised) |

## Review round 4 resolution (`b8ad2eb`)

| # | Sev | Finding | Verified against code | Resolution |
|---|---|---|---|---|
| B1 | BLOCKER | Link order not serialized: a descheduled stitch links an old final after a newer promotion, and C2 makes it current | **Accepted.** Stitch `done` commits at `pipeline_v2.py` L2747-2748 and link (i) comes after; promotions commit at `pipeline_range_attempt.py` L1235, `pipeline_audio_patch.py` L469 and `assistant_run_result.py` L402; `store.drain_run_captures` appends in `c1_seq` order. | §3.3: under the Run lock (200 ms, skip on contention), the link matching the live inode of a `done` run is enqueued **last**. Without a live match, nothing is enqueued and the links are history only. Current = live by construction. The imprecision left is the relative order of two non-current historical finals (R35). |
| B2 | BLOCKER | Link failure is non-durable; a restitch before the scan loses the final | **Accepted.** The revision-4 fallback was a budget-zero executor insert. | Link point (ii) before every Run-final stitch, under the Redis lock: `st_nlink == 1` → link as `superseded` (§3.2). Loss now needs both links to fail on one file. It is counted as `link_failed` (log + Redis counter + best-effort DB artifact), and the gate reports it with no zero-loss claim. |
| B3 | BLOCKER | A third worker breaks the host budget; `worker.py` pre-warms InsightFace | **Accepted.** `worker/worker.py` L103-125 pre-warms InsightFace; the `docker-compose.yml` L134-147 budget is ~14.7 GiB of 16. | `worker/knowledge_worker.py`: no ML imports (tested via `sys.modules`), queue `knowledge` with its own RQ scheduler (required: `rq/scheduler.py` L49, L95-121), `mem_limit: 768m`, `cpus: 0.5`, and a resource-plan line. Recovery treatment verified: `_live_run_ids` ignores `knowledge` (`recovery.py` L195-199); `_live_worker_names` only matters for run jobs (§6.1). |
| M1 | MAJOR | `note_upload` waits inside the open submit transaction | **Accepted.** Refs are resolved at L1839, inside the submit transaction opened by the L1747 transition and the L1783-1800 flushes, and committed later (L2178). | `note_upload` and the hash registry are **withdrawn**. The dispatch basis comes from row locators read in the executor (§1.5), with no hashing in the pipeline. `reference_set_sha256` is NULL. `input_content` observations come later from the worker. The weaker guarantee is stated (R37). |
| M2 | MAJOR | A capture-scope `*` expands privacy and retention | **Accepted.** | Explicit UUID list only (`*` refused). Provenance in `kb_capture_scopes` (S13) + `recipe_context` + link names. Operator approval by Operator at AICP-300 is listed as decision 1. Access is unchanged. |
| m1 | MINOR | Executor self-healing | **Accepted.** | Thread supervision on every call, lazy per-thread session with reconnect after `OperationalError`, and health keys surfaced through `last_ok` and the gate (§1.3). |

## Review round 5 resolution (`2ceba94`)

| # | Sev | Finding | Verified against code | Resolution |
|---|---|---|---|---|
| B1 | BLOCKER | Link (ii) can turn an undelivered failed output into a Run-case version; `superseded` has no `VERSION_REASONS` mapping | **Accepted.** On a Drive failure (`pipeline_v2.py` L2737), the handler (L3401-3416) fails the run. A rerun from `failed` is allowed (`api_v2.py` L4790, L4867-4886), and the next stitch's (ii) links the undelivered file. Revision 5's scanner enqueued every non-live pending link. `taxonomy.py` L13-14 has no reason for it. | Only point (i) produces versions (§3.2). (ii) links become history-only `stitched_final` artifacts listed as "Preserved pre-restitch files (delivery unverified)". They get no queue item, no version and no judging. The one exception is pairing with a durable `kb_link_failures` row for the same `(dev, ino, mtime_ns)`; that row then supplies the publication id and reason. No `superseded` reason is needed. |
| B2 | BLOCKER | Link (ii) swallowed the RQ timeout before the stitch | **Accepted.** `rq/timeouts.py` L6-17: `JobTimeoutException(BaseTimeoutException(Exception))`. | ~~Both link points re-raise `BaseTimeoutException` … a 90 s deadline gate skips the link.~~ **[Superseded: the deadline gate was removed in round 6; link (i) and all promotion links absorb the timeout since round 7/8. Link (ii) still re-raises. See §3.2 and §4.]** (i) propagating is safe: `finalize_run` has returned and its handler is not on the stack; the run is `done`; the `finally` releases the lock; `_enqueue_finalize` registers no `on_failure` (`tasks.py` L1014-1026); recovery ignores `done` runs (`recovery.py` L119-125). (ii) propagating takes the normal timeout path. The audio promotion link moves into its RQ wrapper so its `except` (L639-656) cannot touch a `done` patch. |
| M1 | MAJOR | Immediate feedback start pins an unpreserved `local_path` version, and the later preserved capture becomes a duplicate without its publication id | **Accepted.** `store.sync_run_case_version` (L372-397) creates a `local_path` artifact; `drain_run_captures` (L1335-1360) consumes the same sha as `duplicate`. | S14: `live_link_lookup` under the Run lock. The live final's publishable link is recorded as the `preserved_copy` version with its publication id. That link is then no longer pending for the scanner. Used by start, sync, the scanner and importer reconciliation. A test is added (§12). |
| M2 | MAJOR | Two same-cause link failures leave only non-durable evidence | **Accepted.** | On a link (i) failure, the `kb_link_failures` row (S15) is written through the executor with a wait of at most 2 s, deadline-gated, before the lock release. Only if that insert also fails is the residual left to logs and counters. That residual is stated exactly and placed under operator decision 2. |
| m1 | MINOR | Health key goes stale during provider waits | **Accepted.** | A per-process heartbeat thread refreshes every 15 s (TTL 60 s) independent of capture calls. A missing key means "no process", not "stale"; unhealthy needs an existing key showing failures (§1.3). |

Kept from round 5's verified list: B1's live-last ordering, M1's withdrawal of `note_upload`, the slim worker and the resource plan.

## Review round 6 resolution (`78f3d7b`)

| # | Sev | Finding | Verified against code / reasoning | Resolution |
|---|---|---|---|---|
| B1 | BLOCKER | A deadline-skipped or disk-guard-refused link (i) left only a counter, so link (ii) could not prove the publication and the delivered final became history only | **Accepted.** Revision 6 §3.2 had both a deadline gate and a `statvfs` guard on link (i). A hardlink adds only a directory entry pointing at an existing inode: no data blocks are written. | Link (i) is **never skipped**. It has no deadline gate (round 6 wording: a landing `BaseTimeoutException` propagates; **superseded in round 7**, where link (i) absorbs and records it) and no disk guard; the guard stays only on writers that copy bytes. `os.stat` runs first, so `(dev, ino, mtime_ns)` is always known for the failure row. The residual is restated precisely as three conjoint failures (§3.2). |
| M1 | MAJOR | The ≤ 2 s failure-row wait sat before the Redis lock release | **Accepted.** | The lock is released first (`finally` at L966); `record_link_failure` runs afterwards: ≤ 2 s on the executor, or fire-and-forget near the deadline. Window argument: a restitch needs a fresh chain (lock, submit, ≥ 15 s poll interval, finalize) before link (ii), and the scanner waits 30 s more. So the row normally exists first. Late pairing is checked from both sides each tick; R38 covers a late pair that cannot be inserted into the past. P3 is intact. |
| m1 | MINOR | `process_run`'s handler would put `error_message` on a `done` run | **Accepted.** The `try` at `pipeline_v2.py` L3515-3531 catches every `Exception`. | `pipeline_v2.process_run` is **excluded from capture**. It has no production caller: grep over `backend/app`, `worker` and `scripts` finds only its definition, and `tasks.run_process_run` delegates to `run_submit_run` (L975-985). Link (ii) is suppressed there through `capture_links=False`. |
| m2 | MINOR | Migration lists omitted S15 | **Accepted** (old lines 118, 864, 947). | §11.1 is the single authoritative schema table (S4, S13a, S13b, S15; S16 added in round 7, then **moved into the storage layer's `k1a7c3e9f205` in round 8**): one additive revision. Every other mention points to it. |

Confirmed in round 6 and kept: the history-only classification, the `BaseTimeoutException` re-raise, S14, the scope restriction, the slim worker and the heartbeat.

## Review round 7 resolution (`69165de`)

| # | Sev | Finding | Verified against code | Resolution |
|---|---|---|---|---|
| B1 | BLOCKER | The doc assumed `finalize_run` returns the final path, but it and `_run_stitch_and_deliver` are `-> None`, so link (i) would never run | **Accepted.** `pipeline_v2.py` L2394 and L3352 are `-> None`; the success path ends at L2747-2749 without returning a value. | §3.2 now specifies the exact return contract. `PublishedFinal` is returned from `_run_stitch_and_deliver`: success, including delivery-only reuse, returns it; the incomplete gate (L2448-2458) and the no-stitch path (L2475-2483) return `None`; exceptions propagate. `finalize_run -> Optional[PublishedFinal]`, and its `ValueError` and handler are unchanged. `process_run` stays `-> None`. All callers are listed: `tasks.py` L964 and `test_pipeline_v2.py` L4948, L5022, L6652, L6687. Tests are in §12. |
| M1 | MAJOR | A timeout during link (i) re-raised without a `kb_link_failures` row, so a later rerun leaves the final "delivery unverified" | **Accepted.** Recovery ignores `done` runs (`recovery.py` L118-125); the finalize timeout is 1 h (`tasks.py` L133). | `link_publication` (point i) **absorbs** `BaseTimeoutException` and records the failure: identity if `stat` completed, otherwise a NULL identity with `run_id`, `path` and `failed_at_ns`. It does not catch `KeyboardInterrupt` or `SystemExit`. The NULL-identity pairing rule is defined (§3.2). Why this differs from round 5 B2: (ii) precedes minutes of stitch and delivery, so absorbing there would run past the deadline with the lock held until the monitor SIGKILLs. (i) follows all committed work, and only the lock release, the ≤ 2 s failure-row submission and the return remain; `signal.alarm` is one-shot. (ii) keeps re-raising. |
| M2 | MAJOR | Late pairing permanently discarded a proven delivered final | **Accepted.** | **S16 historical versions**: an immutable `historical` flag and `published_at` on `kb_case_versions`. The latest/current/state/sync/drain logic ignores historical versions, judging them requires `historical=true`, and `version_no` is max + 1. Late-proven (ii) links, non-live (i) links and the importer's non-current promotions become historical versions. Added to §11.1 and §12; R38 resolved. **[Round 8: S16 now ships inside the storage layer's `k1a7c3e9f205`, not AICP-299's migration.]** |
| M3 | MAJOR | Contradictions: §8 still applied the disk guard to links, and a test still required a < 90 s skip | **Accepted.** | §8 disk-guard text now covers byte-copying writers only. The test rows are rewritten. `link_final` is renamed consistently to `link_publication` / `link_preserve`. The failure-policy bullets are rewritten. Earlier round tables carry a note that §0-§13 are authoritative. I grepped for "guard", "90 s", "skip", "deadline gate" and "history only" and fixed the stale statements. |
| m1 | MINOR | The "`process_run` has no caller" claim was wrong | **Accepted.** `scripts/e2e_v2_acme.py` L58 calls `pipeline_v2.process_run`; revision 7's grep pattern missed it. | Corrected. It is not a production path, so exclusion stands (`capture_links=False`). That e2e script cannot validate the links. `test_capture_links.py` (driving `tasks.run_finalize_run`) and the AICP-300 pilot run through the production RQ chain can. |

Confirmed in round 7 and kept: the schema inventory (now with S16) and the lock-release-first claim.

## Review round 8 resolution (`9c4bf47`)

**Lead decision.** S16 (historical versions + `published_at`) and a partial
UNIQUE `(case_id, source_record) WHERE source_record IS NOT NULL` on
`kb_case_versions` are implemented directly in the undeployed
`k1a7c3e9f205` on `feat/knowledge-storage`. Every store writer makes
`source_record` unique by construction; sync records include the new
`version_no`. The commit is to be supplied by the lead.

The AICP-299 plan to add S16 as a successor migration (`ADD COLUMN`) is
removed. §11.1 now lists only S4, S13a, S13b and S15, on top of
`k1a7c3e9f205`.

| # | Sev | Finding | Verified against code / doc | Resolution |
|---|---|---|---|---|
| B1 | BLOCKER | "Pending" meant "no artifact", but phase 1 wrote the artifact before the Run lock. On contention, the link was no longer pending and was lost. | **Accepted** (revision 8 §3.3, phase 1 wrote artifacts). | Phase 1 is now **read-only**: stat, hash, decode. All DB writes happen in **one** Phase-2 transaction under the Run lock: artifact, queue item or historical version. On contention it rolls back with nothing written. *Pending* = no `kb_case_versions` or `kb_capture_queue` row carries the link's publication id, which is idempotent under the storage layer's UNIQUE. History-only links are pending until their artifact exists. |
| M1 | MAJOR | Promotion contract contradicted itself: a DB re-read for audio, `process_audio_patch` returns nothing, conflicting timeout policies | **Accepted.** `process_audio_patch` returns nothing (returns at L511 and at the end; `_promote` at L632); `tasks.run_process_audio_patch` (L1335-1346) has only `patch_id`; the revision-8 §4 text said "re-raise" while §3.2 said "absorb". | Exact descriptor contracts (§4). Range gets `poll_range_attempt_once_published -> (str, Optional[PublishedFinal])`, with the `str` API kept for its 26 test calls. Audio gets `process_audio_patch -> Optional[PublishedFinal]`. Assistant gets `promote_run_result_published -> (dict, Optional[PublishedFinal])`, with the `dict` API kept. Callers link from the descriptor with **no DB read**: `tasks.run_process_range_attempt_poll`, `tasks.run_process_audio_patch` and `api_assistant.publish_run_result`. **One policy:** every publication link absorbs and records. Promotions have no link (ii). P3 is restated. |
| M2 | MAJOR | Unbounded historical appends while holding the Run row | **Accepted.** Range promotion locks the same row (`pipeline_range_attempt.py` L1141-1143). | **[Superseded in round 9: the scanner no longer takes the Run lock; work is bounded by K = 6 candidates per Run per tick, and C2 decides historical.]** `MAX_HISTORICAL_PER_TICK = 5` per Run per tick. The remainder stays pending and continues next tick through the pending definition, with no continuation state. A test measures the hold and checks that a waiting `_promote` completes. |
| M3 | MAJOR | "Idempotent by `source_record`" was not enforced | **Accepted.** No unique constraint on `kb_case_versions.source_record` in `knowledge/models.py`. | Provided by the lead's partial UNIQUE in `k1a7c3e9f205`. **[Round 9: the scanner no longer appends versions; C2's `source_record` duplicate check plus the UNIQUE cover it.]** The contract is stated in §3.3: each historical append runs in a savepoint of the knowledge-worker Phase-2 transaction; `IntegrityError` → roll back the savepoint and treat the link as done. Anything else rolls back the whole transaction. Race tests cover scanner vs scanner and scanner vs S14. |
| m1 | MINOR | A stale "both links re-raise" statement remained in the round-5 table | **Accepted.** | Struck through and marked **[Superseded]** inline, with a pointer to §3.2 and §4. The round-6 wording is annotated likewise. The note above the round tables already declares §0-§13 authoritative. |

## Review round 9 resolution (`334003d`)

Storage now at `feat/knowledge-storage` `9c9e3eb`, which follows `3efff8e`.
It provides:
- `kb_capture_queue.file_identity` (`"dev:ino:mtime_ns"`);
- `store.file_identity(path)`;
- `enqueue_final_capture(..., file_identity=)`;
- C2 revalidation: `drain_run_captures` compares an item's identity with
  `_live_identity` under the Run lock, and a mismatch is appended
  `historical=True` with outcome `historical` (store L1314-1321, L1363-1399;
  `taxonomy.CAPTURE_OUTCOMES` L88).

| # | Sev | Finding | Verified against code / doc | Resolution |
|---|---|---|---|---|
| B1 | BLOCKER | Rollout named storage `62070a5`, but S16 lives in `3efff8e` and later. An S16-less `k1a7c3e9f205` could never be repaired by the same revision id. | **Accepted.** `git -C /root/re-skin-knowledge-storage log`: `9c9e3eb` → `3efff8e` → `62070a5`. | The sources table, D7, §11, §11.1 and rollout step 1 now require storage **at or after `9c9e3eb`**. A stated invariant: `k1a7c3e9f205` must not be applied anywhere until it contains S16, the UNIQUE and `file_identity`. AICP-296 ships exactly that content and records the hash. Pre- and post-flight checks are added, and a database with an S16-less `k1a7c3e9f205` is drop-and-recreate only (§9). |
| M1 | MAJOR | "Current = live" broke between the scan and C2 when a promotion committed in between | **Accepted.** `pipeline_range_attempt.py` L1143 (`FOR UPDATE`) and the commit at L1235 can land between the scan and the drain. | The scanner passes each link's inode identity as `file_identity`. C2 revalidates under the Run lock at drain time: a mismatch becomes `historical`. "Current = live" now holds at the C2 decision point (§3.3, P4, D12). Round-9 M1 scenario test added. |
| M2 | MAJOR | The cap did not bound Phase-2 work, because artifact creation for every pending link came before the cap | **Accepted.** | Phase 1 selects **at most K = 6 candidates per Run per tick** before any write, and hashes only those. Phase 2 writes artifacts and queue items for the candidates only. Phase 2 also **no longer takes the Run lock** (`enqueue_final_capture` needs none), so it cannot delay a range promotion. The revision-9 Run-locked historical appends and their cap are withdrawn, because C2 now decides historical. |
| M3 | MAJOR | `PublishedFinal` contract contradictory | **Accepted.** §3.2 declared three fields; §4 used five. | ONE frozen dataclass, `PublishedFinal(run_id, project_id, path, kind, publication_id)`, in `app/knowledge/publication.py`. ONE linker, `link_publication(pf)`, for all four publishers. `finalize_run` / `_run_stitch_and_deliver` take `publication_id` (minted when `None`), and `run_finalize_run` passes the RQ job id. |
| m1 | MINOR | §6.3 still said scanner artifacts are created in an earlier transaction | **Accepted.** | §6.3 is rewritten: the scanner takes no Run lock; C2, S14 and feedback take `runs` → `kb_cases`. I grepped for "Run lock", "MAX_HISTORICAL", "IntegrityError", "historical=True", "link_final" and the old `link_publication(published, kind, …)` call, and fixed every residue in §0-§13. Round-8 table entries are annotated **[Superseded in round 9]**. |



## Review round 10 resolution

Storage head is now `feat/knowledge-storage` `bb9b6fb`. It contains S16, the
UNIQUE, mandatory `file_identity` with fresh-read revalidation, a bounded sync
drain, and exact publication re-records. Each was reviewed incrementally by
Terra in storage rounds 12–16.

| # | Severity | Finding | Verification | Resolution |
|---|---|---|---|---|
| B1 | BLOCKER | Rollout allowed storage `9c9e3eb`, which lacks the fresh Run read (`af6ce0e`) | **Accepted.** | Every rollout and provenance pin now reads **at or after `bb9b6fb`**. AICP-296 records the exact gated hash. |
| B2 | BLOCKER | `file_identity` optional, and reconciliation omitted it | **Accepted.** | Storage `221b340`: the column is NOT NULL, `enqueue_final_capture` refuses an empty identity, and C2 makes current only an item whose identity equals the live final, read fresh under the Run lock. Anything else, including an unreadable live final, becomes historical. Importer `da8d0c5`: reconciliation passes `store.file_identity(path)`, read under the Run lock. |
| M1 | MAJOR | The K=6 cap could starve the live final behind old links | **Accepted.** | Phase 1 now selects the live-inode publishable link first (§3.3), then the backlog oldest-first. |
| M2 | MAJOR | Feedback sync drained an unbounded backlog under the Run lock | **Accepted.** | Storage `221b340`: `drain_run_captures(max_items=)`; sync drains at most 20 and otherwise returns 409, so the backlog is left to the worker. The card renders a retry state (§3.3 S14 step 1). |
| m1 | MINOR | §6.3 residue | Fixed in revision 10 (`77e9d14`). | — |


## Review round 11 resolution

Storage head `dd0f616`, importer head `46c86f1`.

| # | Severity | Finding | Verification | Resolution |
|---|---|---|---|---|
| B1 | BLOCKER | Importer S9f historical versions are unpreserved `local_path` yet judgeable (see round 12 for the stronger rule) | **Accepted.** | S10 is now enforced **in the store** (`a8b64f7`, `dd0f616`): historical feedback and automatic assessments are refused unless the version's artifact is `preserved_copy`, `drive_file` or `assistant_artifact`. Importer history stays visible but cannot be judged. Operators judge it only once a preserved link exists (the scanner/S14 records those as `preserved_copy`). |
| B2 | BLOCKER | Rollout pinned importer `5892b33` | **Accepted.** | Sources table, §6.3 and rollout step 1 now require importer **at or after `46c86f1`** and storage **at or after `dd0f616`**; the release packet records the exact gated hashes. |
| M1 | MAJOR | Re-records unreachable through the scanner's pending rule | **Accepted.** | Pending now ignores invalidated queue/version evidence and matches `<id>~<n>` via `store.publication_versions` (§3.3). A withdrawn publication's link becomes pending again and is re-recorded as `P~1`. |
| m1 | MINOR | Importer truncated records to 200 while storage rejects >180 | **Accepted.** | Importer no longer truncates (since `5a3572c`). An over-long or `~`-containing id fails loudly in the store (`_check_record`), and the importer reports it as a row failure. |


## Review round 12 resolution

Storage head `1b0145a`, importer head `d2a997f`.

| # | Severity | Finding | Verification | Resolution |
|---|---|---|---|---|
| B1 | BLOCKER | S10 did not protect automatic assessments of superseded (non-historical) mutable versions, or of withdrawn versions | **Accepted.** | Storage `c4d94ba`/`1b0145a`: `record_auto_assessment` refuses withdrawn cases/versions. It takes the Run → case locks and refuses any **non-current** version (historical or superseded) whose artifact is not preserved. Assessments of non-current versions are stored as historical. Historical operator feedback has the same preserved-only rule (`a8b64f7`). |
| B2 | BLOCKER | A case invalidation stranded the publication: its consumed queue item kept the capture key | **Accepted.** | Storage `c4d94ba`: `enqueue_final_capture` re-queues under `capture_key(..., attempt_no)` when the existing item was consumed and its result is no longer valid for the Run's current case. Pending, stale and withdrawn items still cover. The scanner's next pass therefore delivers the publication to the replacement case revision; tested. |
| M1 | MAJOR | A killed tick stops the chain until key expiry, and boot sees a live key | **Accepted.** | §6.2 chain watchdog in the knowledge worker parent: key and job-presence check every 60 s and at boot; a key with no job behind it is restored with `nx`. |
| M2 | MAJOR | The dependency table still allowed storage `bb9b6fb` | **Accepted.** | The sources table and rollout both require storage **at or after `1b0145a`** (the reviewed minimum). |


## Review round 13 resolution

Storage head `284df2c`, importer head `863e958`.

| # | Severity | Finding | Verification | Resolution |
|---|---|---|---|---|
| B1 | BLOCKER | The scanner's pending rule treated consumed `duplicate`/`stale` items as covering, bypassing the re-queue path | **Accepted.** | Pending is now exactly `not store.publication_covered(...)` (storage `284df2c`): coverage is judged against the Run's current case, and consumed items stop covering once their result was withdrawn. Tested in storage (`test_publication_coverage_follows_the_current_case`). |
| M1 | MAJOR | Watchdog thread in RQ's forking parent: redis-py fork-lock deadlock | **Accepted.** | The watchdog is a separate spawned process; the RQ parent gets no extra threads (§6.2). |
| M2 | MAJOR | Orphaned Started entries count as live | **Accepted.** | Liveness follows `recovery.py`: registry cleanup, plus a live worker heartbeat for a started job; orphans are not live (§6.2). |


## Review round 14 resolution

Storage head `898d42b`, importer head `ca9be7a`.

| # | Severity | Finding | Verification | Resolution |
|---|---|---|---|---|
| B1 | BLOCKER | Rollout permitted storage `1b0145a`, which lacks `publication_covered` | **Accepted.** | Sources table and rollout require storage **at or after `898d42b`**; the release packet records the exact gated hash. |
| M1 | MAJOR | The parent restart hook does not run while a tick executes | **Accepted.** (`check_for_suspension` runs only in the outer loop; `monitor_work_horse` blocks.) | PID-1 supervisor (`knowledge_supervisor.py`) owns both processes and exits when either dies; Docker restarts the container (§6.2). |
| m1 | MINOR | DEL followed by SET NX is not an arbiter between two watchdogs | **Accepted.** | Repair runs only under a separate `SET … NX EX 120` repair lock, which is never released early; the holder re-checks liveness before DEL + enqueue (§6.2). |


## Review round 15 resolution

Storage head `94ee5d0`, importer head `ba493fc`.

| # | Severity | Finding | Verification | Resolution |
|---|---|---|---|---|
| B1 | BLOCKER | The compose command started the RQ worker, not the supervisor | **Accepted.** | §6.1: `command: python -m worker.knowledge_supervisor` with `restart: unless-stopped`. |
| B2 | BLOCKER | RQ's scheduler subprocess was unsupervised, and scheduled-but-never-run ticks counted as live | **Accepted.** | §6.2 redesign: no RQ scheduler and no `enqueue_in`; the supervised ticker enqueues due ticks directly. Liveness counts only queued jobs or jobs started on a live worker. |
| M1 | MAJOR | The repair lock had no fencing, so a paused watchdog could create a second chain | **Accepted.** | §6.2: no chain and no DEL. Each slot is fenced by an idempotent `SET slot NX`, so overlapping tickers enqueue at most one tick per slot and nothing is ever deleted. |
| m1 | MINOR | Importer pin too low | **Accepted.** | Importer **at or after `ba493fc`**, storage **at or after `94ee5d0`**; the release packet records the exact gated hashes. |


## Review round 16 resolution

Storage head `c0a603e` (round 24 NO FINDINGS on `94ee5d0`; `c0a603e` is a bounded-read refinement under review), importer head `662fae9`.

| # | Severity | Finding | Verification | Resolution |
|---|---|---|---|---|
| M1 | MAJOR | Slot fencing does not serialize admission across slots, so two ticks can run | **Accepted.** | Execution-time mutual exclusion in `run_knowledge_tick`: `SET running <token> NX EX TICK_TIMEOUT+60`; a duplicate exits immediately. Released by a compare-and-delete on the owner token. `job_timeout` kills a tick before its lock expires (§6.2). |
| m1 | MINOR | Rollout, test matrix and R36 still referenced the chain and scheduler | **Accepted.** | §9 rollout checks supervisor, children, slot keys and `last_ok`. The §12 test row covers ticker slot races, liveness and orphan handling, supervisor exit, and tick mutual exclusion. R36 is rewritten. |


## Review round 17 resolution

| # | Severity | Finding | Verification | Resolution |
|---|---|---|---|---|
| M1 | MAJOR | The lock's expiry could coincide with RQ's force-kill boundary, and `except Exception` swallowed `JobTimeoutException` | **Accepted.** (`timeouts.py` L6; `worker.py` L1217) | §6.2: `BaseTimeoutException` is re-raised. The lock value is the RQ job id with a 24 h safety TTL and **no expiry-based takeover**: a newcomer may replace a holder only when it is not live by the ticker's heartbeat rule (Lua CAS). The release is owner-checked. Residual R39 (a stalled parent past the worker TTL) is stated. Tests: uninterruptible horse with a live parent keeps the lock; a dead-heartbeat holder is replaced once; a queued duplicate skips. |


## Review round 18 resolution

| # | Severity | Finding | Verification | Resolution |
|---|---|---|---|---|
| M1 | MAJOR | `StartedJobRegistry` cleanup at the timeout + 60 boundary can fail a job whose horse still runs, so the CAS handed the lock over | **Accepted.** (`registry.py` L92, L215; `worker.py` L1181, L1217) | §6.2: lock liveness uses positive evidence, i.e. a live-heartbeat worker whose current job is the holder (set before fork, cleared only after the horse is reaped), or the holder still being queued. Registry expiry and cleanup are not consulted. Test: a horse stuck past timeout + 60 with its parent still monitoring keeps the lock until the parent reaps it. |
| m1 | MINOR | R36 described the withdrawn lease expiry | **Accepted.** | R36 rewritten for liveness-based takeover. |


## Review round 19 resolution

| # | Severity | Finding | Verification | Resolution |
|---|---|---|---|---|
| B1 | BLOCKER | §1.4 created Run cases in executor sessions without the Run lock, against the storage contract | **Accepted.** (`store.ensure_run_case` requires the caller's Run lock) | §1.4: executor case creation takes `lock_run_and_cases` in its own short transaction with a 200 ms `lock_timeout`, and skips with a counted miss on contention (reconciliation recovers the evidence). The existing-case path takes no Run lock. |
| m1 | MINOR | The RQ `current_job` lifecycle was misdescribed | **Accepted.** | §6.2 text corrected (the horse sets it in `prepare_job_execution`; it is cleared by the horse after the job function or by the parent after reaping a killed horse). The predicate is unchanged and still holds. |


## Review round 20 resolution

Gate bar adopted by Operator (2026-10-27): the gate passes with no BLOCKER/MAJOR correctness finding. Performance findings need production-scale evidence or become follow-ups. Rounds review the delta.

| # | Severity | Finding | Verification | Resolution |
|---|---|---|---|---|
| M1 | MAJOR | `lock_timeout` bounds only acquisition; the executor's hold of the Run row was unbounded | **Accepted.** | §1.4: server-enforced `statement_timeout = 250ms` and `idle_in_transaction_session_timeout = 500ms` on the case-creation transaction, so the hold is bounded at about 1.25 s regardless of the client. Test-matrix row added. |
| m1 | MINOR | §6.3 said executors never take `runs`; test coverage missing | **Accepted.** | §6.3 corrected; the first-creation, contention-skip, importer-recovery and stalled-executor test row is added. |

## Review round 21 — DESIGN APPROVED (GATE: PASS)

Terra (GPT-5.6, effort high, session `01a17d2c-2185-7453-8b09-b2e51d8fa516`) approved revision 21 (`91a4b7a`) under the adopted gate bar. No BLOCKER or MAJOR finding remains.

Logged follow-up (PERF, not a gate blocker):
- **F1.** §1.4's "about 1.25 s" bound on the Run lock for first-case creation undercounts statements. `ensure_run_case`/`ensure_case` issue several queries, and each is bounded by `statement_timeout`. A cold cache can therefore hold the lock longer than 1.25 s, though always through bounded server-side statements, and only once per Run or segment case.
- **Action.** Measure during the AICP-300 pilot. If it matters, pre-create the Run case at run admission, which is outside capture.
