# Run-local segment range regeneration

Status: **BUILT**, branch `codex/run-local-segment-range-regeneration`, unmerged.
Base: `origin/main` @ `3f958ee74b13b2d50c4b154b948c7e7aefb7c000`.

Lets an operator enlarge or shrink a **completed swap segment's** time range,
moving only a boundary it shares with an immediately adjacent `keep` segment,
and regenerate exactly that segment — without disturbing the Run's last
successful result until the replacement has actually succeeded, and without
touching the reusable Project segmentation other Runs share.

See also [[components/run-local-range-regen]] for the synthesis / why-it-fits
write-up, and [[components/segment-image-regen]] for the closest sibling
feature (this one shares its two-phase submit/poll shape and its
append-only-attempt-history data model, but generates a full video instead of
a single still).

## Two timelines, on purpose

- **Project-level `SegmentDef`** rows are the *reusable, shared* partition —
  `PATCH /api/v2/projects/{pid}/segments` edits them, and every Run on the
  Project shares those rows. This feature never writes to `SegmentDef`.
- **`Run.segment_snapshot`** is the *run-local, frozen* copy of that partition
  taken once, at the Run's first submit (`pipeline_v2._build_submit_state`).
  Every downstream consumer of "what time range did THIS run actually use" —
  the run page, finalize, a later Copy, an Automation launch, and now range
  regeneration — reads this snapshot, never live `SegmentDef` rows. A Project
  edit made after a Run's first submit is invisible to that Run and to
  anything derived from it.

Range regeneration edits `Run.segment_snapshot` (after a successful
promotion) — never `SegmentDef`. Other Runs on the same Project, and the
Project's own segmentation, are completely unaffected by a range correction
on one Run.

A Run that predates run-local snapshots (`segment_snapshot is None`) cannot
use this feature — see "Legacy snapshot-less Runs" below.

## Boundary rules

For the selected `swap` entry in the Run's ACTIVE (last promoted, or original)
timeline:

- The **start** may move only when the immediately previous timeline entry is
  `keep`. Moving it earlier enlarges the swap and shortens that keep; moving
  it later shrinks the swap and enlarges that keep. The shared boundary is
  updated atomically: `previous_keep.end_sec == swap.start_sec`.
- The **end** may move only when the immediately next timeline entry is
  `keep`, symmetric to the above: `swap.end_sec == next_keep.start_sec`.
- When both neighbours are `keep`, both boundaries may move in the same
  request.
- When a side is adjacent to another `swap`, **that side is locked** — this is
  what makes `swap -> swap` merging structurally impossible, not a validation
  rule bolted on top. `swap -> swap` regeneration/merging is explicitly out of
  scope for this feature and is not implemented anywhere in this module.
- No implicit `keep` segment is ever synthesized as a workaround for a missing
  `keep` neighbour.
- A move that would consume a `keep` neighbour down to zero or negative
  duration is rejected outright — this is a first-implementation choice: a
  clear server rejection, not a silent removal of a timeline entry or a
  rewritten stitch plan.
- The resulting timeline must stay an ordered, gap-free, overlap-free
  partition covering exactly `[0, source_duration_sec)` — re-validated as a
  whole after every move, which is what catches "boundary consumed a
  neighbour" without hand-deriving a separate inequality per case.
- The candidate swap duration is validated against the selected model's
  effective min/max clip duration (`ai_models.min_clip_sec` /
  `max_clip_sec`) as a **hard rejection**, not the main submit path's
  pad-to-fit pre/post-roll behaviour — the operator asked for an EXACT range,
  and padding would silently pull in footage outside it.
- All of the above validation (`compute_candidate_snapshot`,
  `validate_model_duration`, `validate_stitch_plan_coherence`) is pure —
  no DB write, no media cut, no upload, no external call, no Run state
  transition happens before every check passes. A rejection is always a 400
  before anything is spent.

Worked examples:

```
keep [0.0, 5.0] | swap [5.0, 10.0]
new swap start = 4.6
=> keep [0.0, 4.6] | swap [4.6, 10.0]

swap [0.0, 8.0] | keep [8.0, 20.0]
new swap end = 8.4
=> swap [0.0, 8.4] | keep [8.4, 20.0]

swap [0.0, 8.0] | swap [8.0, 15.0]
=> REJECTED: the shared boundary is locked (both sides are swap)
```

## Pre-roll / post-roll are explicitly not part of this feature

The main pipeline pads a swap's model input by `pre_roll_sec` /
`post_roll_sec` for model context and trims the returned clip back down. This
feature does not read or write those fields, and does not use padding as a
range-editing mechanism — without precise post-generation trimming, reusing
rolls this way can duplicate neighbouring footage, shift the timeline, and
desynchronize audio. The new range editor edits actual run-local coverage,
never hidden context padding.

## Candidate revision, not an eager rewrite

`RunSegmentRangeAttempt` (`backend/app/models.py`, migration `43c4cab61c52`,
purely additive) is one candidate revision of a single swap segment's
run-local range — mirroring `RunSegmentImageJob`'s append-only-attempt-history
shape (added in `8812409e54a4`), but for a full video generation + re-stitch +
re-deliver instead of a single image call.

The Run's active `segment_snapshot` and the targeted `RunSegment`'s fields are
**left completely untouched** until the attempt reaches a successful `done` —
at which point `_promote()` copies the candidate's fields onto them in one
commit. A `failed` attempt never touches the Run.

Every candidate artifact lives under its own directory,
`storage.range_attempt_dir(attempt_id, run_segment_id, run_id, project_id)` —
never the run's shared `run_clips_dir` / `run_results_dir` paths — so a
candidate byte can never collide with, or be mistaken for, an active
artifact. Promotion re-points `Run.result_local_path` /
`RunSegment.local_result_path` at the attempt's own paths; it never copies
bytes over an existing file.

### Two-phase worker shape (worker never sleeps)

```
submit_range_attempt(attempt_id)      creates (or RESUMES) the kie.ai task
poll_range_attempt_once(attempt_id)   one round of status-check; on success
                                       also re-cuts, stitches, delivers, and
                                       PROMOTES in the same call
```

`submit_range_attempt` is resume-aware exactly like
`pipeline_segment_image.submit_segment_image_job`: if
`attempt.seedance_task_id` is already set, the existing task is returned for
polling rather than re-created — a worker restart or a duplicate enqueue can
never rebill kie.ai for the same attempt. `app/tasks.py` re-enqueues the poll
step on its own while an attempt is pending, the same shape the segment-image
and preprocess phases already use.

### What gets cut, what gets reused

- The candidate swap clip is **always** cut from
  `VideoProject.source_local_path` at the exact promoted bounds — never from
  a rendered final, a generated clip, or an already-transcoded keep clip.
- Whichever `keep` neighbour(s) actually moved are **re-cut from the same
  original source** at their new bounds, into the attempt's own directory —
  never reusing (or overwriting) the run's cached `clip_%04d.mp4`.
- Every segment **not** touched by this attempt — an unaffected `keep` (same
  bounds as before) or an unaffected, already-completed `swap` elsewhere in
  the Run — resolves to the run's EXISTING clip/result at candidate-stitch
  time. It is never re-cut, never resubmitted, and never re-billed.

### Failure handling

On failure at generation, download, keep re-cut, stitch, or delivery: the
attempt is marked `failed` with an operator-readable `error_message`, and the
Run/RunSegment are left exactly as they were — same non-destructive contract
at every step, not just generation failure. The operator can retry the
candidate (new range-rerun request) or issue a corrected one; nothing about a
failed attempt is inherited by anything downstream (see "Copy and Automation
inheritance" below).

### Idempotency / worker-restart safety

A worker restart (or a duplicate task-runner invocation) BETWEEN submit and
poll never double-submits or double-bills: `submit_range_attempt` resumes an
existing `seedance_task_id` rather than creating a new task, mirroring the
existing segment-image and preprocess resume behaviour.

That resume check alone does not protect against two workers delivering the
SAME submit or poll job at the SAME time — see "Same-attempt worker races"
and "Production recovery of a crashed finalize-phase worker" below for the
full claim/lease mechanism and recovery state machine, and their documented
residual gaps (an ambiguous window right around kie.ai's paid createTask
call, which this feature cannot make safely auto-retryable because kie.ai
has no idempotency key; and a possible orphaned duplicate Drive file when a
recovered finalize re-delivers after an already-successful-but-uncommitted
upload).

### Delivery in place — same Drive id (AICP-349)

Since AICP-349 a promoted attempt on a run that already has a Drive file
replaces that file's **content** (`files().update`, keeping the id), like a
segment rerun (AICP-342) and an audio patch. The run's `result_gdrive_file_id`,
Creative Link, Job ID, an already-exported Final Creatives row and `Sent to
Final` stay as they are; the bytes behind them are the new final. Drive keeps
the previous bytes as a revision.

- **Where:** inside `_promote`, after the Run-row lock
  (`session.refresh(run, with_for_update=True)`), the claim re-check under a
  row lock on the attempt itself (Run first, then attempt — the range
  attempt's equivalent of finalize's processing-lock ownership: a takeover's
  claim commit waits behind this transaction instead of landing between the
  check and the upload) and every staleness check (`done`, base snapshot).
  The update is the last fallible step before the commit that promotes, so a
  candidate that is refused never touches Drive.
- **What is uploaded:** a private hard link (`.deliver-<uuid>.mp4`) of the
  claim-scoped candidate final, checked to be the same inode right before the
  transfer and removed afterwards (the AICP-342 delivery pin).
- **No fallback:** the deliver step no longer makes a fresh upload for a run
  that has a Drive file. A failed update (including a 404 for a file deleted
  from Drive) raises before the commit, so the attempt is `failed` and the run
  is byte-for-byte untouched, with the existing link still serving the
  previous video. A deleted file needs an explicit re-link first (AICP-348).
- **A run with no Drive file yet** still gets a fresh upload in the deliver
  step, and promotion points the run at it, as before.
- **Audio-patched result:** the promoted candidate is stitched from the segment
  clips, so it drops a promoted audio fix *from the exported link*.
  `range-rerun` therefore 409s on a run whose current result is a promoted
  audio patch unless `discard_audio_patch=true`, the same acknowledgement a
  rerun or a take restore needs; the run page asks once and resends.
- **Ambiguous update failures are settled by content.** When the update
  reports failure, `GDriveClient.file_md5` is compared with the candidate's
  md5: a match means the transfer landed (lost response) and the attempt
  promotes; a mismatch or a 404 fails it with Drive untouched; an unreadable
  checksum fails it with a message saying the link MAY already hold the
  candidate.
- **A commit failure after the update landed** does not fail the attempt
  (`_PromoteUncommitted`): it goes back to `generating` with its claim
  released, the reaper's resume shape, and the next poll re-sends the same
  content to the same id and promotes.
- **Residual risk** (the class audio patch and finalize accept): a worker
  KILLED between Drive acknowledging the update and the commit leaves Drive
  ahead of the row. The attempt still has its task id, so the reaper resumes
  it and the resumed finalize re-sends an equivalent final. Double failures
  are not reconciled automatically and leave a `failed` attempt with Drive
  possibly serving the candidate: the update's response AND the checksum read
  both failing (the message says so); a commit failure followed by a failed
  re-download/re-stitch on the next poll; a timed-out update that Drive
  finishes only after the checksum was read. In each of them a new attempt or
  a segment rerun re-delivers. (Gate round 2 findings, accepted as MINOR.)

## Stitch-plan coherence

Range regeneration never adds, removes, or reorders a `segment_def_id` /
library entry in `Run.stitch_plan` — it only moves the boundary shared by two
EXISTING entries — so a plan that was coherent before the move stays coherent
after it. `validate_stitch_plan_coherence` rejects only an entry shape this
module cannot resolve at all (an unrecognised `kind`), before any credits are
spent, rather than surfacing a confusing failure mid-flight. Covered at
candidate-stitch time:

- default full partition (`stitch_plan is None`);
- no-final-video plan (`stitch_plan == []`) — the candidate result clip IS the
  deliverable, no candidate final video is built;
- custom plans containing the selected swap and an adjacent keep;
- Localisation hook plans (a following Project keep excluded, a
  language-matched library clip appended) — a `library` stitch-plan entry
  resolves to the `LibraryClip`'s own file, untouched by the range move.

## Copy and Automation inheritance

Both manual Copy, batch Copy, copy-of-a-copy, and every Automation launch go
through the same `api_v2._build_copied_run` helper. It now deep-copies the
source Run's **current** `segment_snapshot` (`copy.deepcopy(src.
segment_snapshot)`) instead of leaving it `None` — a `None` snapshot re-freezes
whatever the Project's LIVE segments are at the copy's own first submit, which
is exactly the bug this closes: before this change, a Copy of a
range-corrected Run silently lost the correction and fell back to the shared
Project boundaries.

- `_build_copied_run` reads `Run.segment_snapshot` directly — it never
  re-derives from live `SegmentDef` rows, so a Project edit made after
  promotion has no effect on what a later Copy/Automation inherits.
- Because `_promote()` only ever mutates `run.segment_snapshot` on a
  **successful** attempt, a failed candidate simply never touches the field a
  Copy reads — nothing extra needed on the Copy/Automation side to exclude a
  rejected candidate.
- Copy-of-a-copy inherits correctly for the same reason: each copy's
  `segment_snapshot` is itself a deep copy of whatever its own source had.
- Existing Copy semantics are unchanged and compose with the above: prompt,
  model, audio mode, language, Drive folder, stitch plan, voice references,
  and applicable preprocessing state are copied as before; per-segment prompt
  overrides are always copied; per-segment reference-image overrides are
  copied only when no new run-level photo is supplied (a new photo drops
  image overrides but keeps prompt overrides — unchanged, pre-existing rule).
- Automation is a **live template reference**, not a frozen settings
  snapshot — it re-reads its template Run on every tick, so a range correction
  promoted after Automation was configured is picked up on the next launch
  without the operator re-saving anything. Already-created Automation child
  Runs are immutable and are never retroactively updated.

### In-flight conflict handling

While a range attempt is active (`queued`/`generating`/`stitching`/
`delivering`) on a Run:

- A second `range-rerun` request on the same Run returns **409** — only one
  attempt in flight per Run at a time.
- The legacy prompt/reference-only `.../rerun` endpoint returns **409** on
  that Run — it must not race a pending promotion.
- Manual `Copy` (`POST /runs/{rid}/copy`) returns **409** and enqueues
  nothing — cloning mid-flight risks freezing whichever side of the race the
  copy lands on (the pre-correction or post-correction range), so it is
  refused outright rather than guessed.
- An Automation tick whose template has an in-flight attempt **skips that
  launch** without consuming quota, reserving a sheet row, or marking one
  used — it does not even read the sheet for that template. The next tick
  after the attempt reaches a terminal state (`done` or `failed`) launches
  normally, with whichever range is active at that point.

## API

`POST /api/v2/runs/{rid}/segments/{rsid}/range-rerun` — extends the existing
per-segment rerun contract narrowly:

- `start_sec` / `end_sec` (required form fields, `0.1s`-precision floats) plus
  the existing optional `prompt` / reference-image override fields, submitted
  atomically.
- Returns `201` with `{attempt_id, status, requested_start_sec,
  requested_end_sec}` on success; the attempt is `queued` and the Run/
  RunSegment are untouched.
- `400` for any boundary/duration/coherence rejection (operator-readable
  message from `RangeValidationError`).
- `409` if the Run is not `done`, if it predates run-local snapshots
  (`"predates run-local ..."`), if another attempt is already active on
  this Run, or if the run's current result is a promoted audio patch and the
  optional `discard_audio_patch` form field is not `true` (AICP-349).
- `404` for a missing Run or RunSegment.

`GET /api/v2/runs/{rid}/segments/{rsid}/range-attempts` — attempt history for
one segment, newest first (diagnosis: which ranges were requested, which
succeeded).

The existing `.../rerun` endpoint (prompt/reference override only, no
timing) is completely unchanged — omitted timing fields on that endpoint keep
their historical legacy behaviour exactly. It only gains the same in-flight
409 gate described above.

## Legacy snapshot-less Runs

A Run with `segment_snapshot is None` predates this feature's data model.
Guessing a timeline for it from possibly-since-edited Project segments would
be unsafe, so range editing is simply **disabled** for it —
`create_range_attempt` raises a `RangeValidationError` ("predates run-local
segment snapshots") before anything is written, surfaced as `409` by the API.
Its existing prompt/reference-only `.../rerun` behaviour is fully preserved.

## What this feature does not claim

Precise range control and safe, non-destructive regeneration are what this
feature gives the operator. It does not guarantee seam quality or that the
regenerated segment's motion will visually match its neighbours — the
underlying model's generation is still probabilistic.

## Hardening pass (2026-08-29 follow-up, unmerged)

An independent audit reviewed the branch before merge and found three real
correctness gaps. All three are fixed on the same branch, on top of the
original implementation described above; nothing in this section changes
the boundary rules, the candidate-revision model, or the API contract
already documented — it closes structural gaps in how those guarantees were
actually enforced.

### 1. A completed Run's history depended on mutable, deletable SegmentDefs

`RunSegment.segment_def_id` was a foreign key to `segment_defs.id` with
`ON DELETE CASCADE`. Two supported, unrelated application paths delete live
`SegmentDef` rows: Project re-analysis (`POST /projects/{pid}/reanalyze`,
which replaces the segmentation wholesale) and a bulk segment edit
(`PATCH /projects/{pid}/segments`, whose `deletes` list — or an update that
collapses a segment to zero length — removes individual rows). Either one,
run against a project with a terminal Run, cascade-deleted that Run's own
`RunSegment` rows and, transitively, its `RunSegmentRangeAttempt` history —
even though `Run.segment_snapshot` was designed from the start to be that
Run's own durable, frozen timeline, independent of whatever the Project's
live segmentation currently says. Concretely this meant:

- a `done` run's per-segment breakdown (clips, Kie task ids, the "re-run
  this segment" button) disappeared the moment anyone re-analyzed the
  project, or edited/deleted the specific segment it used;
- its range-attempt correction history disappeared with it;
- a Manual Copy or Automation launch built from that Run could fail
  outright: `_build_copied_run` inserts a fresh `RunSegment` naming the same
  `segment_def_id` the source used, which — with the FK still in place —
  violated the constraint the moment the live row was gone.

**Fix.** `RunSegment.segment_def_id` is no longer a foreign key at all (see
its docstring in `models.py`). It remains a plain, permanently-stable
string column — the identity key every reader already matches against
`Run.segment_snapshot` entries first (`web_v2._launch_window`,
`api_v2._segment_launch_window`, `web_v2._range_edit_bounds`,
`pipeline_v2.run_segment_defs`, `pipeline_range_attempt`,
`api_v2._build_copied_run`) — with a `viewonly` SQLAlchemy relationship
(`primaryjoin` spelled out by hand, since there is no FK for SQLAlchemy to
infer a join from) standing in only for the one case that still needs a
live read: a Run created before `Run.segment_snapshot` existed. Migration
`73df0ff3f344` drops the constraint; no data is rewritten, and the column's
value, its `NOT NULL`, and its plain index are all untouched. `Run`,
`Project` and `VideoProject` deletion still cascade `RunSegment`/
`RunSegmentRangeAttempt` correctly — that was always via their own direct
FKs to `runs.id`/`video_projects.id`, never via `segment_def_id`.

Not solved by (and deliberately avoided): remapping a historical
`segment_def_id` onto a newly re-analyzed project's ids (there is no safe
way to know which new segment "is" an old one), mutating live SegmentDefs to
match a corrected Run (the Project's segmentation is shared by every Run on
it, not owned by one), or blocking Project editing forever once any Run
exists (re-analysis and bulk edits remain exactly as available as before —
only their blast radius on OTHER Runs' history changed).

### 2. The one-active-attempt invariant and promotion itself were not concurrency-safe

`api_v2._active_range_attempt` — the check every mutating endpoint
(`range-rerun`, Copy, the legacy segment rerun/patch) uses to refuse acting
while a candidate is in flight — was a plain non-locking `SELECT`. Two
`range-rerun` requests arriving close enough together could both observe no
active attempt and both create one, both get enqueued, and both spend an
external generation credit — a real double-billing risk, not a theoretical
one. Separately, `_promote()`'s own defense (refusing when
`run.segment_snapshot != attempt.base_segment_snapshot`) compared against
whatever `run` object the current session had loaded, which — under a
genuine race between two *worker* sessions processing attempts — could
itself be stale: two sessions can each load `run` before either commits,
each pass the check against their own copy, and both commit, with the later
commit silently overwriting the earlier one.

**Fix, two layers.** A partial unique index,
`uq_run_segment_range_attempts_active_per_run` (migration `73df0ff3f344`,
`WHERE status IN ('queued','generating','stitching','delivering')`, present
on both PostgreSQL and SQLite — mirrored in
`RunSegmentRangeAttempt.__table_args__` so the ORM-driven test schema
carries the same constraint the migration installs), makes "at most one
active attempt per run" a real database guarantee instead of an
application-level convention. `api_v2.create_range_rerun` catches the
`IntegrityError` this produces on the losing request, rolls back, cleans up
that (and only that) attempt's own reference-file directory, and returns
the same 409 the app-level check already produced for the common case.
Separately, `_promote()` now opens with
`session.refresh(run, with_for_update=True)` — a real row lock on
PostgreSQL (a no-op on SQLite, which has no row-level locking; the
concurrency *guarantee* is proved against PostgreSQL specifically, this
feature's production database, in
`tests/test_pipeline_range_attempt_pg_concurrency.py`), together with
`RunSegmentRangeAttempt.version_id` (migration `df45372e4eb2`) protecting
every earlier intermediate commit. This closes the race the unique index
cannot — the SAME attempt processed by two worker sessions at once — **for
which committed database write wins**: the loser's `refresh`/version check
picks up the winner's already-committed change and correctly refuses.

It does NOT, by itself, stop both sessions from actually PERFORMING the
shared-file and external-call side effects (cutting, uploading, calling
kie.ai's createTask, downloading, re-cutting, stitching, delivering)
concurrently before either commit lands — versioning only arbitrates the
commits *afterwards*. That gap is what "Same-attempt worker races" (§4
below) closes.

### 3. The shared keep-clip cache was not crash-safe

`_promote()` refreshed the run's shared keep-clip cache by `shutil.copy2`ing
the moved neighbour's freshly re-cut file onto the run's existing
`clip_%04d.mp4` path — keyed by INDEX ALONE — before the commit that made
the promotion real. A crash or exception between that copy and the commit
left the cache holding NEW bytes at a path the DB still said belonged to
the OLD, still-active snapshot; the next reader (another range attempt, or
a plain `rerun_segment`) would trust that file's mere existence and reuse
the wrong footage. This was not a matter of moving the copy to the other
side of the commit — the deeper problem is that INDEX ALONE is not a valid
cache key once a promotion can change a `keep` neighbour's bounds without
changing its index; whichever side of the commit the copy lands on, the
path being written was always the one thing a still-active old snapshot
might also read.

**Fix.** `storage.keep_clip_path(run_id, project_id, index, start_sec,
end_sec)` keys the cache by **index and bounds together**, replacing the
bare `clip_{index:04d}.mp4` convention everywhere it was used for this
purpose — `pipeline_v2._cut_source` (shared by `keep` segments and
swap-segment source-fallback) and `pipeline_range_attempt._resolve_keep_clip`
/ `_promote`'s cache warm-up. A promotion now only ever *writes* the file at
the NEW bounds' key; the OLD bounds' file, if it exists, is never touched by
a promotion at all, successful or not. A stale file becomes a harmless
orphan — nothing ever asks for its filename again once the active snapshot
has moved on — never a silent wrong answer; a missing one is always safe to
re-cut, since this cache only ever holds a plain cut of
`VideoProject.source_local_path`, never a rendered or transcoded artifact,
so re-cutting the same bounds twice is always byte-identical.

### 4. Same-attempt worker races over shared files and the one paid external call (2026-08-30 follow-up)

§2's fix makes "which commit wins" deterministic, but a second, independent
delivery of the SAME submit or poll job can still reach ffmpeg, kie.ai, or
Drive concurrently with the first — *before* either one's commit lands.
Concretely, before this pass:

- `submit_range_attempt` cut the candidate into the shared `clip.mp4` path,
  committed, then uploaded and called `createTask` — two workers delivering
  the same queued attempt could both run ffmpeg against that same path, and
  whichever won the DB commit could still be uploading a file the loser's
  own `-y` overwrite had truncated.
- The poll success path downloaded into a shared `result.mp4` (via
  `KieClient.download_result`'s own `result.mp4.part` → `os.replace`) — two
  pollers observing the same "success" state could race that same `.part`
  file, one moving it out from under the other's still-open handle.

**Fix.** Three new nullable columns on `RunSegmentRangeAttempt`
(`claim_owner`, `claim_expires_at`, `submit_billed_at` — migration
`bc2e47aad2a2`, purely additive; pre-migration rows default to "unclaimed",
a safe, immediately-recoverable state). A worker must commit `claim_owner`
(an opaque per-call token) via an optimistic, `version_id`-checked
compare-and-set — no new locking primitive, `version_id` already gives an
atomic CAS — **before** cutting, uploading, calling `createTask`,
downloading, re-cutting, stitching, delivering, or promoting. A second
worker finding a live (non-expired) claim exits immediately, benignly,
without touching any file or external client. Every claimed phase's WORKING
files live in a claim-scoped subdirectory (`range_attempt_dir/.../
claims/<token>/`), never the bare attempt directory, so two live owners
never write the same working path.

**Lease sizing — an independent follow-up audit's finding.** The first cut
of this fix sized `_CLAIM_LEASE_SEC` at 900s against generous assumptions
about how long a finalize round "normally" takes, while the RQ
`job_timeout` both phases actually run under
(`tasks.RANGE_ATTEMPT_POLL_JOB_TIMEOUT`) is 3600s. A finalize whose
stitch+Drive-delivery genuinely took longer than 15 minutes — plausible for
a large source video — would let a duplicate poller (arriving on its normal
15s re-poll cadence) find the claim "expired" and wrongly take over while
the original worker was still very much alive, reopening concurrent
download/stitch/delivery/promotion and risking a duplicate Drive upload.
**Fixed**: `_CLAIM_LEASE_SEC` is now floored at `job_timeout + 300` — the
exact same relationship `tasks._RUN_LOCK_TTL_SEC` already uses for the main
pipeline's Redis lock (`PROCESS_JOB_TIMEOUT + 300`) — so an "expired" claim
is no longer merely *usually* safe to take over, it is **provable**: RQ
enforces `job_timeout` as a hard kill, so a claim that has genuinely expired
proves its owner's job was already terminated. No heartbeat/renewal loop
was needed as a result — the lease is sized correctly from the start rather
than extended mid-flight. `_CLAIM_LEASE_SEC` cannot be configured below this
floor even via its environment override, closing the misconfiguration path
too.

**Explicit ownership re-verification — defense in depth on top of the
lease.** `_assert_still_owner` re-queries `claim_owner` fresh (not the
in-memory copy, which cannot observe another session's commit without an
explicit read) immediately before every remaining external or publish side
effect: downloading, re-cutting a moved `keep` neighbour, publishing into
the shared keep-cache, stitching, calling kie.ai's `createTask`, delivering
to Drive, and promoting. Raises `_ClaimLost`, handled identically to
`StaleDataError` everywhere in this module. This does not replace the
job-timeout-sized lease as the primary guarantee — it guards the residual
risk of clock skew, a manually-edited claim row, or an unknown bug, per "do
not rely only on a normally-finishes-quickly assumption."

**The shared keep-clip cache is the one path outside `claims/<token>/` a
claimed phase writes to — and it is now published atomically, ownership-
checked.** `_resolve_keep_clip` (called from `_stitch_candidate` for every
UNAFFECTED `keep` entry) and `_promote`'s own cache warm-up both target
`storage.keep_clip_path` — deliberately SHARED across every attempt on a
run, since it is a reuse cache, not per-claim state. Before this fix, both
cut/copied directly into that shared path: two overlapping finalizers (only
reachable via the lease-sizing bug above, or the recovery takeover in §5
below) could both run ffmpeg with `-y`, or both `shutil.copy2`, into the
SAME file at once — the module docstring's "every worker writes only
inside `claims/<token>/`" claim did not actually hold for this one path
(found by the same follow-up audit). **Fixed**: a missing/stale entry is
cut or copied into a claim-scoped TEMP path first, ownership is
re-verified (`_assert_still_owner`), and only then is it published into the
shared path via `os.replace` — atomic on the same filesystem, so a
concurrent reader's existence/size check always sees either nothing yet or
the fully-written file, never a partial one, and a superseded worker's
already-cut copy is never published over a newer owner's. The `(index,
bounds)` cache key from §3 is unchanged. A stale worker's cleanup
(`_cleanup_claim`) only ever removes its OWN `claims/<token>/` directory —
the temp files it staged there, never a file another claim already
published to the shared path.

**Submission is the one place this cannot be made fully safe.**
`KieClient.create_task` (kie.ai's `createTask`) has no idempotency key and
no lookup-by-reference endpoint (`get_task` needs a `taskId` this feature
would not yet have) — confirmed by reading `kie_client.py`, not assumed.
So a worker that crashes between calling it (paid) and persisting the
resulting `seedance_task_id` leaves a genuinely ambiguous window: retrying
risks a second billed generation for the same segment; giving up risks
silently losing one that already succeeded. `submit_billed_at`, committed
immediately before that one call, exists to flag exactly this window: a
LATER call finding it set with no `seedance_task_id`, past an expired
claim, marks the attempt `failed` with an explicit "submission ambiguous —
check the kie.ai dashboard, do not auto-retry" message instead of guessing
either way. This is a real, accepted residual limitation, not a gap this
pass closes — see "Failure/recovery test matrix" below for what IS proved
about it (the window is flagged and never silently retried) versus what
provably cannot be (whether kie.ai actually created a task in that window).

The finalize phase (download → recut → stitch → deliver → promote) has no
such ambiguity — nothing in it makes a paid call — so a claim that has
GENUINELY expired (per the job-timeout-sized lease above) is always safe to
take over, and a duplicate/late finalize call correctly reports `"pending"`
(not a fabricated `"done"`/`"failed"`) when it either finds a live claim or
loses the claim-acquisition race itself; see `poll_range_attempt_once`'s
`except (StaleDataError, _ClaimLost)` handling, which now distinguishes
"the other session is already terminal" from "the other session is still
mid-flight" rather than assuming the former.

In production, a genuinely crashed finalize worker is essentially never
recovered by WAITING for this lease to expire on its own — the lease is
deliberately long (§ above), so the periodic reaper (§5 below) almost
always reaches a dead worker's row first, within minutes rather than an
hour. This self-service expiry path is the fallback for when that reaper
does not run.

Claims are per-phase, not per-attempt-lifetime: `submit_range_attempt`
releases its claim the moment the submit phase concludes (successfully or
not — a terminal `failed`/`done` row is never claim-checked again), so the
finalize phase always starts from a genuinely unclaimed row rather than
waiting out a stale lease.

**What this still relies on the database for, explicitly:** the atomicity
of the claim compare-and-set (SQLAlchemy's `version_id_col`, i.e.
PostgreSQL's own row-level serialization of two competing `UPDATE`s to the
same row) and the one-active-attempt partial unique index from §2. Neither
Redis nor any in-process lock is the source of truth for either guarantee —
consistent with keeping optimistic versioning as defense in depth rather
than replacing it.

### 5. Production recovery of a crashed finalize-phase worker (2026-08-30 follow-up)

`recovery._reap_orphaned_derived_creatives` reaps EVERY derived creative
(including `RunSegmentRangeAttempt`) with no live RQ job, aged past its
5-minute grace window, to `failed`. Applied unconditionally to a range
attempt, this DISCARDS an already-paid-for result: an attempt whose
`seedance_task_id` is already persisted (submission fully succeeded —
kie.ai has already generated and billed the video) that then crashes during
download/stitch/delivery makes NO further paid call to finish; failing it
instead of resuming it throws away real, already-spent money for nothing
(found by the same follow-up audit).

**State machine.** The reaper now classifies every orphaned, aged-out range
attempt into exactly one of three outcomes:

| Row state | Outcome | Why |
|---|---|---|
| `submit_billed_at` set, `seedance_task_id` NULL | `failed`, ambiguous-billing message (§4, unchanged) | Cannot tell whether kie.ai's paid call landed — manual-only, never auto-retried |
| `seedance_task_id` set (regardless of `submit_billed_at`) | Claim cleared (`claim_owner`/`claim_expires_at` → NULL), status forced back to `generating`, exactly one recovery poll enqueued | Submission already succeeded — finalizing makes no further paid call, so it is safe to resume |
| Neither set | `failed`, generic "Press Retry" message (unchanged) | Crashed before any billing-adjacent commit — an ordinary orphan |

The claim is cleared **transactionally**, in the SAME batched commit the
reaper already uses for every other row in this table — not merely left to
expire — because the reaper's own 5-minute grace window is far shorter than
the (now job-timeout-sized) claim lease: without clearing it explicitly, a
freshly-enqueued recovery poll would see the dead worker's claim as still
"live" (per `claim_expires_at`) and just report `"pending"` again, stuck
until the FULL lease naturally expires — up to roughly an hour later,
defeating the point of recovering it now. `poll_range_attempt_once` is then
enqueued exactly once per recovered row (`tasks._enqueue_range_attempt_poll`,
`delay_sec=0`), AFTER the reaper's own commit has landed — never before, so
a commit failure never leaves an enqueue pointing at a claim that was never
actually cleared. Status is force-set to `generating` (not via `transition()`
— there is no `stitching`/`delivering` self-loop or backward edge, so this
follows the same table's existing precedent for a forced value, e.g.
`OverlayStatus`'s missing `queued -> failed` edge) regardless of which
active status the crash happened in, because `poll_range_attempt_once`'s
terminal-state handler has no notion of "resume from stitching" — it always
re-runs the WHOLE download → recut → stitch → deliver → promote sequence
from scratch on its next invocation, and its own `transition(attempt,
stitching)` call would otherwise raise `InvalidTransition` for a row still
sitting at `stitching`/`delivering`. On its next run the enqueued poll
re-polls kie.ai (a read, not billed), takes a FRESH claim, and redoes that
whole sequence using new claim-scoped paths — idempotent in outcome (one
final promoted state) even though the underlying local work is redone.

**Racing a genuinely live worker is prevented by the SAME liveness check
every other reaped row already relies on**: a range attempt whose id is in
`live_ids` (a live RQ job exists for it) is never touched, recovered or
otherwise, regardless of its `seedance_task_id` — unchanged from the
existing per-row liveness gate.

**Residual limitation — a resumed finalize can leave an orphaned duplicate
Drive file** (since AICP-349 only for a run that had no Drive file yet — a
delivered run is updated in place, which is idempotent). If the dead worker's crash happened AFTER `_deliver_with_retry`
successfully created a Drive file but BEFORE the commit that records
`attempt.gdrive_file_id`, the resumed finalize (which redoes delivery from
scratch, having no record of that upload) creates a SECOND Drive file; the
first is never referenced by anything and becomes storage waste. This is
explicitly NOT the same class of risk as double-billing kie.ai — Drive
storage is comparatively low-stakes and outside this task's scope to
eliminate — but it is a real, documented residual gap, not a hidden one.
Reconciling truly-orphaned Drive files (if ever needed) is a separate,
future concern.

### Failure/recovery test matrix

| Scenario | Proof |
|---|---|
| Two full `submit_range_attempt` workers race the same queued attempt (real threads, real overlap) | `test_full_duplicate_submission_only_one_worker_cuts_uploads_and_creates_a_task` (PG) |
| Two full `poll_range_attempt_once` workers race the same generating attempt (real threads, real overlap) | `test_full_duplicate_poll_only_one_worker_finalizes_and_promotes` (PG) |
| A full finalizer running LONGER than a deliberately shortened lease is not overtaken by a duplicate poll chain started while it is still alive; exactly one download/recut/stitch/upload/promote happens, verified from a third session | `test_full_duplicate_poll_survives_a_shorter_than_finalize_lease` (PG) |
| Two finalizers never write the same keep-cache path; the published file is complete | `test_full_duplicate_poll_never_double_writes_the_shared_keep_cache` (PG) |
| A live, non-expired claim rejects a duplicate submit before any side effect | `test_submit_live_claim_blocks_a_duplicate_worker_without_touching_anything` |
| An expired finalize claim is safely taken over and completes | `test_poll_takes_over_an_expired_finalize_claim_and_completes` |
| `submit_billed_at` set + no task id + expired claim → failed, NOT retried | `test_submit_ambiguous_billing_window_is_never_retried_and_marked_failed` |
| Expired claim with `submit_billed_at` still NULL → safely retaken from scratch | `test_submit_expired_claim_without_billing_is_safely_retaken` |
| A takeover never touches a dead claim's OWN (never-cleaned) files | `test_takeover_of_a_dead_claim_never_touches_that_claims_own_files` |
| A claim superseded mid-flight cleans up only its own directory, never fails the attempt, never touches the new owner's claim or the Run | `test_claim_superseded_mid_cut_cleans_up_only_its_own_directory`, `test_stale_owner_cannot_deliver_promote_or_fail_the_winner_after_a_takeover` |
| A crashed finalizer with a persisted `seedance_task_id` is recovered (claim cleared, exactly one poll enqueued) without another `createTask` | `test_recovery.py::TestRangeAttemptFinalizeRecovery` |
| The ambiguous submission window (no task id) remains manual-only through the real recovery path too | `test_recovery.py::TestRangeAttemptFinalizeRecovery::test_ambiguous_billing_window_is_not_recovered_automatically` |
| `_CLAIM_LEASE_SEC` can never be configured below the job-timeout floor | `test_claim_lease_floor_matches_the_job_timeout_relationship` |
| Failed stitch/delivery still leaves the previous live Run untouched | `test_failed_stitch_leaves_previous_result_intact`, `test_failed_delivery_leaves_previous_result_intact` (pre-existing, still green) |

None of these claim "no remaining race" more broadly than this table — in
particular, the ambiguous-submission window is flagged and never silently
retried, not eliminated; see the residual-risk note above.

### Verification

- Every fix has a dedicated regression test in
  `backend/tests/test_pipeline_range_attempt.py`,
  `backend/tests/test_api_v2.py` (`TestSegmentDefDeletionSurvival`) and
  `backend/tests/test_automation.py`.
- The genuine concurrency guarantee (two threads, a barrier, real
  simultaneity — not a sequenced approximation) is proved against
  disposable PostgreSQL 15 in
  `backend/tests/test_pipeline_range_attempt_pg_concurrency.py`, skipped by
  default (set `RESKIN_TEST_PG_URL` to opt in) — including the two full
  duplicate-worker tests in §4's table above, which drive the ENTIRE
  submit/finalize call end to end from two independent sessions/threads,
  not just a promotion or an insert.
- Migrations `73df0ff3f344` and `bc2e47aad2a2` were applied, exercised and
  downgraded against a disposable `postgres:15` container (never the
  production database), then re-upgraded cleanly.
- See `tasks/reports/run_local_segment_range_regeneration.md` for the full
  hardening-pass report, including exact test counts and the independent
  code-review evidence.
