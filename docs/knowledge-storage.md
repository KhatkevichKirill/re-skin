# Knowledge storage — implementation notes (AICP-295)

Implements the storage half of [knowledge-feedback-design.md](knowledge-feedback-design.md)
(§4 schema, §5 validation, §6.2 lock hierarchy, §6.5 scope, §6.6 read/rule API,
§8.2 holdout refusal at rule-evidence write). Nothing here captures pipeline
events or changes Run behaviour; capture is AICP-299 and the importer is
AICP-297. Everything is behind `KNOWLEDGE_ENABLED` (default off).

## Files

| File | What it holds |
|---|---|
| `backend/app/knowledge/taxonomy.py` | every code list (each is the single source for its CHECK), §5 defect taxonomy and allowed automatic `(check_kind, claim)` pairs, §9.1 canonical applicability + specificity |
| `backend/app/knowledge/models.py` | the 26 `kb_` tables; ORM + database guards against UPDATE/DELETE/TRUNCATE on the 25 insert-only tables |
| `backend/app/knowledge/access.py` | `KnowledgeScope` = assistant `ProjectAllowlist` ∩ `KNOWLEDGE_PROJECT_IDS` (`*` = the whole assistant scope, empty = nothing); rule visibility |
| `backend/app/knowledge/store.py` | typed idempotent writers and invalidation-aware readers (PostgreSQL only) |
| `backend/app/knowledge/api.py` | `/api/v2/knowledge`: `GET runs/{id}/timeline`, `GET/POST rules`, `POST rules/{id}/versions`, `POST rule-versions/{id}/reviews` |
| `backend/alembic/versions/k1a7c3e9f205_knowledge_base_storage.py` | additive migration on `j0e5f2a7b914`; a fixed copy of the models (no app imports) |
| `backend/tests/test_knowledge_unit.py`, `test_knowledge_pg.py` | unit tests (always) and PostgreSQL tests (`RESKIN_TEST_PG_URL`) |

## Store contract (what callers can rely on)

- **One request, one transaction, no commit inside.** Every writer runs in the
  caller's session; the caller commits or rolls back.
- **Lock order**: idempotency-key advisory lock → `runs` row `FOR UPDATE` →
  `kb_cases` rows by ascending id. Attempt capture (`record_attempt_intent`)
  takes only a per-target advisory lock and never a Run lock.
- **Receipts**: `submit_feedback`, `record_correction`, `retract_event`,
  `record_auto_assessment`, `add_rule_version` and `review_rule_version` store a
  `kb_requests` row keyed by `idempotency_key` with the canonical request
  sha256. Same key + same body → the stored receipt (`replayed=True`), no new
  rows; same key + different body → `KnowledgeConflict` (409). Decisions are
  idempotent on their own unique key and `consulted_sha256`.
- **Versions**: `expected_version_no` must equal the case's latest valid
  version, else `KnowledgeConflict` carrying `current_version_no`. A Run case
  first re-hashes the live `runs.result_local_path` under the Run lock and
  appends a version when it changed (§6.2 step 3). A new version always resets
  the verdict to `unreviewed`; `case_state()` reports the previous accepted
  version separately. Derived, manual and external cases have exactly one version.
- **Bindings**: every artifact named in a version, output, correction or event
  must belong to the case's project; defect/assessment time ranges must lie
  inside the judged artifact's duration when the duration is known.
- **Scope**: every id is resolved to its stored project; out of scope raises
  `KnowledgeAccessDenied`, which the API renders as 404 like a missing id. A
  non-`shareable` rule version is hidden unless every scoped and every cited
  project is visible; a `shareable` one shows out-of-scope citations as a
  count only.
- **Unknown stays NULL**: unhashed artifacts are never merged; a reference set
  with any unknown hash has `reference_set_sha256 = NULL`; costs are separate
  `cost_estimate` / `cost_settled` observations, each naming its source.
- **Holdout**: rule evidence citing a case whose project is in a holdout group
  of the latest loaded manifest (or with an explicit holdout membership) is
  refused, and so is any evidence while no manifest is loaded (fail closed). A
  version that cites holdout anyway is never effective. The run timeline shows
  holdout cases without operator judgements (`verdict: hidden_holdout`).
  Retrieval exclusion is AICP-301.

Additions to the §4 column lists, all bookkeeping: `import_batch_id` on
attempts, observations, events, rule versions, rule evidence and eval
membership (so invalidating a batch reaches every row it created even without
a member row); `kb_decisions.project_id` (scope of an outcome);
`kb_case_versions.id` and a `seq` on `kb_requests`; one invalidation per target
(`uq_kb_invalidation_target`, re-invalidating is a no-op).

**Replacement after withdrawal**: `kb_cases` and `kb_artifacts` carry a
`revision` (unique with `case_key`, or with `locator_kind, locator, sha256`).
Lookups use the highest revision that is not invalidated. When every earlier
revision was withdrawn, the next writer creates revision n+1, so a corrected
batch re-imports under the same key. A capture-queue item whose artifact was
withdrawn after it was queued is consumed as `withdrawn` (a fourth outcome
beside `appended`, `duplicate` and `stale`), never appended.

**Historical versions (S16)**: `kb_case_versions.historical` (immutable, set at
insert) plus `published_at`. A historical version takes the next `version_no`
but never becomes current: `latest_version`, `case_state`, the feedback version
check, sync and drain ignore it. It can be judged only with `historical=true`
feedback. Derived, manual and external cases refuse historical versions. Only a version whose
artifact is preserved (`preserved_copy`, `drive_file`, `assistant_artifact`) can
receive historical feedback (S10); a `local_path` version may hold other bytes
now and is refused.
**One publication = one version**: partial UNIQUE `(case_id, source_record)
WHERE source_record IS NOT NULL`. `append_case_version` returns the existing row
for a retried publication, and raises `KnowledgeConflict` if the same publication
arrives with different content. Drain consumes a repeated publication id as
`duplicate`. Sync records are `sync:<run>:<sha>:<version_no>`, so they are unique
by construction.

**C2 live revalidation**: every capture-queue item must carry `file_identity`
(`dev:ino:mtime_ns` of the published file). Under the Run lock, drain compares
it with the live `runs.result_local_path`. On a mismatch the publication is
appended as a historical version (outcome `historical`), never as current, so
a promotion that committed between capture and drain always stays current.
Automatic assessments of a historical version are stored as historical events.
`sync_run_case_version` (feedback/start) drains at most `SYNC_DRAIN_MAX` (20)
queued finals under the Run lock. If more remain, it refuses with 409 ("still
being recorded") and leaves the backlog to the knowledge worker. A publication
whose version was withdrawn can be recorded again as `<record>~<n>`.

**Shareable rules** cross scope, so besides the reviewer's attestation of the
statement and rationale their structure may not name a project or person:
applicability cannot use `project_id`/`subject_id`, and the recommendation
target must be a structured object over generic dimensions (no free-text
target). Outside scope, provenance, evidence gaps, author/reviewer labels and
review notes are omitted.

`kb_import_members.target_kind` vocabulary (what the importer must write):
`case`, `case_version`, `artifact`, `attempt`, `submission`, `observation`,
`event`, `rule_version`, `rule_evidence`, `eval_membership`. Holdout is judged
against the **latest** loaded manifest version only; memberships under
superseded versions are history.

## Importer (AICP-297)

`backend/app/knowledge/importer/` — `python -m app.knowledge.importer
--dry-run|--apply [--source NAME] [--limit N] [--from-start] [--hash]
[--no-seed]` prints the JSON report; `tasks.knowledge_import_tick` is the
bounded RQ job (no-op unless `KNOWLEDGE_CAPTURE_ENABLED`, not scheduled).

- **Dry run** = one transaction, full report, ROLLBACK (the batch row too). It
  takes no Run row locks and no cursor leases, but it does INSERT before
  rolling back, so it needs a writable connection and burns identity values.
- **Apply** = one batch; each source row in a savepoint, committed with its
  cursor advance (Run lock held per row). Failed rows are reported and skipped;
  `--from-start` rescans. Reruns create nothing twice.
- **Commit horizon**: rows (and stamps/settlements) are scanned only once
  older than 2 × `PROCESS_JOB_TIMEOUT`, so a late-committing writer cannot be
  stepped over by the `(created_at, id)` cursor. Knowledge lags by that much.
- **Revisit**: a non-terminal row is imported as it stands and kept in its
  cursor's `pending` list; passes re-process it (at most a quarter of the
  budget and always leaving one row for new rows; alternate passes at
  `limit=1`; rotating) until it is terminal (`done` for runs, since
  failed/incomplete runs can be retried) or gone. Only Runs age out after 7
  days (a run that becomes done later is caught by `reconcile_finals`); other
  rows stay. When the list is full (2000) the source stops before the next
  non-terminal row instead of skipping it.
- Every source row and reconcile row runs with a 200 ms lock timeout in apply
  mode: a Run held by the pipeline is skipped without advancing and retried.
- No file I/O under a Run lock: each source declares its row's files, which
  are probed (exists, stat, sha256) before the row's transaction takes any
  lock; under the lock only a stat re-verifies (size, mtime_ns), and a file
  that changed makes the row retry next pass. An unreadable file during the
  probe imports the row unhashed.
- Fair tick budget: each source gets an equal share first, starting at a
  rotating source, and the leftover goes to sources that still have work.
  Draw coverage runs every pass, bounded by the slowest task-bearing source's
  cursor (a draw is created after the row that names its task).
- A final that fails 3 times is recorded missing once per (path, size,
  mtime) — the last 8 recorded identities per Run are kept, so A→B→A does
  not record A twice; it is retried only when the file changes.
- Export events keep one lock order, key → Run → case: a source row queues
  its `export_observed` in the cursor's `exports` list in the same commit as
  its cursor advance, and the queue is written afterwards (deadline and
  heartbeat before each, 200 ms lock timeout; a failing one is dropped and
  reported after 3 tries). A full queue (2000) stops the source before the
  next row. Store:
  `record_export_observed` locks the Run whenever the case has a `run_id`
  (derived and manual cases included).
- The capture drain takes Runs by their oldest pending item, rotating with a
  cursor, checks the deadline and heartbeats per Run.
- `reconcile_finals` hashes before taking the Run lock, then re-reads the Run
  and re-checks the file's (size, mtime) under the lock before queueing; a
  failure is retried (cursor not advanced) and after 3 attempts recorded as an
  unhashed artifact with its availability check.
- Promotions: a publication whose content is already a version (sync,
  reconciliation, live capture) maps to that version (report `publications`,
  correction note). A superseded promotion is recorded as an artifact (with
  its correction / attempt output), never as an out-of-order version: the
  highest version is current, so appending it would make an old final
  current. A true historical version would need a schema change (e.g. a
  `historical` flag on `kb_case_versions` excluded from "latest").
- Store: `drain_run_captures(..., max_items=N)` bounds one drain (the importer
  passes 20).
- **Dry run never locks a Run**: `export_observed` (whose writer takes the
  Run lock) is counted, not written; no other store call on the dry-run path
  locks a Run. Exports and final reconciliation commit per row with a 200 ms
  lock timeout (a locked Run stops that scan until the next pass) and check
  the deadline before every row/hash.
- A revised settlement (new `mcredits`/`settled_at` on the draw) appends a
  newer `cost_settled`; the latest by `seq` wins.
- **Leases**: before every commit the cursor lease (and the tick's Redis lease)
  is renewed and verified in the same transaction; a lost lease aborts the
  pass (`aborted` batch observation). The tick stops taking rows after
  `min(600 s, ttl/2)`.
- **P3**: only a `done` run's final becomes a Run-case version (runs source and
  `reconcile_finals`, which also queues `reconcile:<run>:<sha>:0` for a done run
  with no case/version). The drain skips a Run locked for more than 200 ms.
- One attempt per provider task: task-bearing sources reuse the attempt that
  already holds the task (live capture's or an orphan's) for the same case; one
  held by another case is reported as a conflict.
- Manual creatives are attributed from ids in their own `details`
  (`run_id`/`parent_run_id`/`source_run_id`, `project_id`) before the overlay
  job; registration does not copy those today, so in practice the overlay is
  used and a dead/upload overlay is still skipped.
- Newly created artifacts whose file was probed get a `kb_artifact_checks` row.
- Sources, in order: `runs` (Run case + version chain rebuilt from recorded
  promotions: audio patch `base_result_path → output_path`, assistant
  `before_json → after_json`, range `final_local_path`; only the current final
  is ever hashed), `run_segments` (current attempt only; cost from the draw,
  else `run_segments.cost_credits` as `run_segment_snapshot`), segment image
  jobs, range attempts, audio patches, assistant promotions, captions, tail
  edits, overlays, manual creatives, `spend_draws` (unmatched generation draws
  → `ambiguous` attempts; LLM draws counted as excluded; runs only once the
  task-bearing sources are caught up). Then `seed` (CLI only) and `reconcile`
  (settlements, stale intents, `export_observed`, changed finals → `reconcile:`
  publications, capture drain).
- Corrections are written only where both sides are recorded (audio patch,
  assistant promotion), with `resolved=unknown`. No verdict is ever written.
- `kb_attempt_submissions` keeps task ids unique across withdrawn rows, so a
  replacement import records the acceptance as a `submitted` status whose
  payload carries `provider_task_id`/`spend_call_key`; the draw matcher reads both.

## Retention

- **Nothing is deleted.** Evidence tables reject UPDATE, DELETE and TRUNCATE in
  PostgreSQL triggers and in ORM events. A wrong import is withdrawn with an
  append-only `kb_invalidations` row (per batch, case, attempt, event, rule
  version or artifact). Every read and effective-state computation skips
  invalidated rows. The timeline still lists events and versions that
  reference them, marked `invalidated`.
- **Soft references**: `run_id`, `project_id`, `segment_def_id` and so on are
  plain strings, so deleting a Run or Project never cascades into knowledge.
  `run_timeline` keeps answering for a deleted Run from its stored case.
- **Media is referenced, not copied.** In this issue `kb_artifacts` only
  points at files and records their hash. `kb_artifact_checks` records whether
  the file is still there and still matches; preserving bytes is AICP-299
  (`KNOWLEDGE_PRESERVE_ATTEMPT_MEDIA`). Existing media retention is unchanged.
- **Rollback**: with data present, rollback is "flags off, keep tables"; the
  downgrade refuses while any `kb_` table has a row.
- **Growth**: rows are small JSON plus ids. At about 525 runs and 929 segments
  so far, a full backfill plus live capture is in the low tens of thousands of
  rows a year. No pruning is planned.

## Verification (2026-10-27)

- `tests/test_knowledge_unit.py` (7) and `tests/test_knowledge_pg.py` (81) pass
  in the `re-skin-api` image against a disposable `postgres:16-alpine`. The PG
  suite covers:
  - Migration/model schema parity on 26 tables (columns, identity, CHECKs,
    uniques, indexes, FKs, triggers).
  - The downgrade refusal.
  - UPDATE, DELETE and TRUNCATE rejection.
  - Replay without duplicates, including 12 concurrent replays of one key.
  - Stale-version 409.
  - The historical judgement path.
  - Acceptance resetting on a new version, and a verdict racing a new
    version.
  - Defect/acceptance contradiction.
  - Time and code bindings.
  - Retraction rules.
  - Automatic-claim rules.
  - Scope 404, the assistant-allowlist cap and rule-text suppression.
  - Gap-free concurrent rule numbering.
  - Intent reuse as a redispatch.
  - Capture-queue duplicate and A→B→A ordering, and concurrent drains.
  - Effective-version supersession and retirement.
  - The O1 operator constraint, and machine hypotheses never approvable.
  - Holdout evidence refusal.
  - Decision pinning.
  - Batch invalidation.
  - Manifest tamper detection.
  - Subject overlap.
  - The HTTP round trip.
  - One regression test per accepted review finding (below).
- Migration rehearsal (on the final revision) on a schema-only dump of
  production restored into a disposable PostgreSQL 16:
  - stamped `j0e5f2a7b914`, then upgrade → downgrade → upgrade;
  - 26 tables, and 50 triggers (one row trigger and one TRUNCATE trigger per
    insert-only table);
  - the non-`kb_` schema is byte-identical before and after, apart from
    pg_dump's random `\restrict` token;
  - with a row present, the downgrade refuses.
  - Not rehearsed on a full-data dump: copying production data into a test
    container was not authorized in this session. The migration only creates
    new objects and reads no existing rows.

## Review record (CLAUDE.md §1 gate)

GPT-5.6 Terra (Codex CLI, `model_reasoning_effort=high`, read-only sandbox),
one session `01a17cf2-c95a-7223-9c09-829cb037ebe9`, ten rounds on
`f8fe093..698858d`. Every finding was checked against the code, confirmed,
fixed, and given a regression test. None were rejected.

| Round | Fix commit | Findings fixed |
|---|---|---|
| 1 | 46c7d64 | rule replay before evidence scope; invalidated rules browsable; holdout from superseded manifests; retracted events kept as rule evidence; invalidated observations in attempt state |
| 2 | f6222c2 | holdout operator labels in the timeline; holdout fail-open without a manifest; withdrawn rules accepted in decisions |
| 3 | c038036 | acceptance over a live serious defect; cross-project capture; drain compared against a withdrawn version |
| 4 | d684c92 | unscoped outcome attempt; retraction reviving a contradicted acceptance; invalidated attempt state |
| 5 | 691b78d | shareable free text across scope; holdout-backed rules in decisions; open import-member kinds |
| 6 | 986febc | withdrawn case/artifact not replaceable (→ `revision`); shareable structure; withdrawn artifact captured |
| 7 | f2e5baa | holdout cases in decisions; children lost on root replacement; capture key without artifact revision; withdrawn case authorizing a deleted Run |
| 8 | 006f909 | withdrawn rows deciding open intents; attempt replay ignoring model provenance; unlocked submission replay |
| 9 | 698858d | withdrawn attempt replayed by key; withdrawn attempts in the timeline |
| 10 | — | **NO FINDINGS** |
| 11–29 | `62070a5` … `edf30b1` | Store contracts required by the importer review and the AICP-299 design review: S16 historical versions plus one version per publication; mandatory `file_identity` with fresh-read C2 revalidation; bounded sync drain; exact `<id>~<n>` re-records; S10 preserved-only historical judgement (feedback and automatic assessments); re-queue of a withdrawn publication; retroactive version-level invalidation; `publication_covered`; effective-only rule evidence; indexed SQL withdrawn predicates with a seq-keyed `InvalidSet` cache (frozen invalidated batches, `BatchWithdrawn`); one-row `latest_version` / `current_case` / artifact reuse; expression indexes for task-id lookups. One finding rebutted (round 19: corrections reference artifacts, they do not judge bytes); Terra accepted the rebuttal. |
| 25, 26, 27, 28, 29 | — | **NO FINDINGS** (the last five rounds) |

**Gated head: `edf30b1`** (round 29, NO FINDINGS). From 2026-10-27 the gate
bar adopted by the owner applies: the gate passes with no BLOCKER/MAJOR
correctness finding; a performance finding needs production-scale evidence or
becomes a follow-up.

**Final migration rehearsal on `edf30b1`.** On the production schema-only
dump in a disposable PG16: upgrade → downgrade → upgrade on `j0e5f2a7b914`;
26 tables, 50 triggers, 95 indexes; the non-`kb_` schema is identical; with
data present the downgrade refuses.

Contract for AICP-299 that came out of round 9: attempt idempotency keys are
unique forever. Replaying the key of a withdrawn attempt raises
`KnowledgeConflict`, and the capture caller records the dispatch under a new
key.

## Release packet (AICP-296) — release-owner action, not performed

Prerequisites:
- this branch, **at the gated head `edf30b1` or a later gated head**, is merged
  to `main` and pushed. `k1a7c3e9f205` must never be applied from an earlier
  revision of this branch, because its content changed several times before
  release (S16, `file_identity`, indexes);
- the deployed tree equals that `main`;
- `alembic heads` is a single `k1a7c3e9f205`. Re-check before running: other
  sessions add migrations, and a new head means rebasing `down_revision`.

1. Drain: no run in an active status, stitching included. The deploy
   recreates both workers.
2. Back up: `scripts/backup_db.sh predeploy`, and confirm the dump is on Drive.
3. Migrate first (CLAUDE.md):
   ```
   docker cp backend/alembic re-skin-api:/app/alembic
   docker cp backend/alembic.ini re-skin-api:/app/alembic.ini
   docker exec -w /app re-skin-api python3 -m alembic upgrade head
   docker exec re-skin-db psql -U reskin -d reskin -c "SELECT version_num FROM alembic_version;"   # k1a7c3e9f205
   docker exec re-skin-db psql -U reskin -d reskin -tAc "SELECT count(*) FROM information_schema.tables WHERE table_name LIKE 'kb\_%'"   # 26
   ```
4. Deploy with the assistant overlays and two workers, then restart nginx:
   ```
   docker compose --env-file .env --env-file /var/lib/reskin-assistant-runtime/assistant.env \
     -f docker-compose.yml -f deploy/assistant/compose.app.yml -f deploy/assistant/compose.asr.yml \
     up -d --build --scale worker=2
   docker compose <same args> restart nginx
   ```
5. Verify:
   - `docker exec re-skin-worker-1 grep -c knowledge_models /app/app/models.py` is ≥ 1, and the same holds on worker-2 and api;
   - `curl -u … http://localhost:8847/api/v2/projects` returns 200;
   - `curl -u … http://localhost:8847/api/v2/knowledge/rules` returns **404**, because the flags are off.
6. Flags: leave every `KNOWLEDGE_*` unset. Nothing captures, imports or
   serves until AICP-297/299/300 are released and explicitly activated.

Rollback:
- Code: redeploy the previous image with the same command. The new tables
  are unused while the flags are off.
- Schema: `alembic downgrade -1` only while every `kb_` table is empty; it
  refuses otherwise. With data present, rollback is "flags off, keep tables".
- No existing table, row or enum is touched in either direction.
