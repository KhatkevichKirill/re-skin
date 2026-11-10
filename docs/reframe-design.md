# Horizontal ⇄ vertical reframe — workflow design

AICP-251. Status: **revision 6** (2026-10-28), implemented (see *Implementation*). Revision 5 was approved by the owner on
2026-10-28 (R1–R9 accepted as recommended, the pass-5 residual accepted under the
review-gate rule); revision 6 only aligns it with the Stage 1 code as merged. Planning only:
nothing here authorizes implementation, migration, deployment or paid generation.

Depends on the Fal provider layer of AICP-312 Stage 1, merged to main as `b73e3bf`
(`docs/fal-provider-design.md`: the contract frozen at revision 11, plus its *Stage 1
implementation (AICP-330)* section, which wins where the two disagree). This document is a
**consumer** of that layer and does not change it. §10 records how Stage 1 answered the
requests revision 5 made of it.

Evidence: [wiki/models/horizontal-to-vertical-reframe.md](../wiki/models/horizontal-to-vertical-reframe.md)
(the AICP-252 manual test and the AICP-312 probe) and
[wiki/decisions/fal-second-provider.md](../wiki/decisions/fal-second-provider.md).
Baseline: main `2c1f15d`.

## 1. Goal and non-goals

**Goal.** An operator uploads a landscape (or portrait) video and gets back a delivered
vertical (or horizontal) creative: the original scene kept, the missing canvas outpainted by
MiniMax H3 Max Reframe on Fal, the original audio and timing kept, and an ending that is
trimmed on the spoken phrase, faded on the picture only, and left with black for a CTA. What
the service did, what it cost (as an estimate), and whether the output passed its checks are
visible on the page.

The AICP-252 delivery is the acceptance shape: picture to 25.75 s, last frame held and faded
to black by about 26.40 s, source audio continuing to 26.65 s, a short black tail, no music,
no audio fade.

**Non-goals.**

- The provider layer itself (submit, poll, fetch, cancel, spend, recovery of provider
  tasks): AICP-312 Stage 1.
- Reframing a Run's final, a tail edit or a caption. v1 reframes a project's source only
  (§11, R1 lists the later option).
- Aspect ratios other than 9:16 and 16:9, and 1080p output from Fal (no tariff; the layer
  refuses it).
- Any change to face swap, localisation, or to how a kie.ai run is routed or priced.
- Automatic Final Creatives export in v1 (§8; decision R4).

## 2. Facts this design rests on

From the probe and the manual test unless marked otherwise.

1. Reframe works in both directions. 16:9 → 9:16 at 768p gives 768×1344; 9:16 → 16:9 gives
   1344×768; 480p vertical is 480×832.
2. **Duration is preserved, frame rate is not**: the output is always 24 fps (a 60 fps input
   came back at 24 fps, 5.000 s in, 5.000 s out).
3. **The source audio comes back untouched** (correlation 1.0000 at lag 0, same sample
   count). This design still takes the audio from the project source, not from the output
   (§6), so the ending can run past the generated picture, as AICP-252's did.
4. The source is kept nearly pixel-for-pixel (SSIM 0.94-0.96 in the placement rectangle),
   and the result JSON reports that rectangle as `placement`.
5. Billing is per **output second**: 480p 0.05 USD, 768p 0.08 USD (docs). Long inputs are
   processed in overlapping windows that are also billed. The key cannot read the debit.
6. Input 0.25-300 s (docs). One generation per POST, no idempotency: a duplicate submit is
   a second paid generation.
7. Code facts (main `27dc1fe`, unchanged at `2c1f15d`):
   - `create_project` takes an upload or a Drive link. `analyze_project` downloads Drive
     sources, runs `media.normalize_source`, which caps the short side at 1080 and keeps the
     orientation and frame rate, then probes the result. The only limit is
     `MAX_UPLOAD_SIZE_MB` (1 GiB); there is no duration limit.
   - The project type registry is `app/project_types.py`, mirrored by the Postgres enum
     `project_type_enum`; the Create Project form lists the registry automatically.
   - `media.stitch`, `cut_clip`, `delivery_video_opts` and `enforce_size_limit` cover
     concatenation, the 100 MB delivery cap and the CRF encode. **Nothing in the pipeline
     does a fade or a black tail today** (only the assistant's ending contract does).
   - Tail edits and captions require a parent `Run`; only overlays accept an arbitrary file.
   - `delete_project` (since Stage 1, `b73e3bf`) locks the project row, then its Run rows,
     and 409s while `pipeline_provider.active_task_for_project` finds an active task; it does
     not see tail edits, captions or overlays.
   - `disk_cleanup.sweep` deletes only `tail_edits/<id>/body.mp4` and `runs/<id>/clips/`.
   - `recovery._reap_orphaned_derived_creatives` fails orphaned derived rows after 300 s
     through `_DERIVED_CREATIVE_SPECS`.
8. Provider layer facts (main `b73e3bf`, `app/pipeline_provider.py`):
   - `admit(db, *, endpoint_id, params, inputs, output_sec, resolution, owner_kind, owner_id,
     owner_project_id, destination_path, operation_key, owner_slot="main",
     owner_run_id=None) -> (task, created)` runs in the caller's transaction and never
     commits. It locks the `video_projects` row (then the run row when `owner_run_id` is
     given), inserts the task `queued`, then `spend.reserve`s under the kind
     `fal_<family>` = `fal_reframe`. The caller commits and then calls
     `tasks.enqueue_fal_job("fal_submit", task.id)` only when `created`; a `queued` row
     without a job is resubmitted by the layer's recovery after `FAL_QUEUED_GRACE_SEC`.
   - It raises `AdmissionRefused(status_code, detail)` with nothing written: 403 (Fal
     disabled, or the project is not in `FAL_PROJECT_IDS`), **503 (no live `fal-worker`; the
     container is behind the compose profile `fal`)**, 404 (owner gone), 409 (destination
     exists, or the owner slot is taken), 422 (unknown parameter or aspect, duration outside
     0.25-300 s, an input over `FAL_MAX_INPUT_BYTES`, an input whose probed duration differs
     from `output_sec` by more than `FAL_OUTPUT_DURATION_TOLERANCE_SEC` = 1.5 s, a missing
     input, or a destination directory that does not exist). It raises
     `spend.BudgetExceeded` from the reservation.
   - Destinations are unique per task (`UNIQUE (destination_path)`), must not exist at
     admission, and must lie under the owner project's directory outside `runs/` (no
     `owner_run_id`) and never under `tail_edits/`.
   - The reframe endpoint accepts only `aspect_ratio`, `resolution` and `seed`; the input
     parameter is `video_url`.
   - `done` writes `provider_tasks.result_meta`: the whitelisted, URL-free fields
     `placement`, `seed`, `frame_count`, `frames_per_second`, `timings` and
     `video.{content_type, file_size, width, height, duration, fps}`.
   - Operator actions are Python functions, `adopt(db, task_id, request_id, *, actor, note)`,
     `abandon`, `close` and `resume(db, task_id, *, actor, note)`. Each does its own CAS,
     writes a `provider_task_actions` row, **commits**, and raises
     `ActionRefused(status_code, detail)` when the task is in the wrong state. There is a
     CLI (`python -m app.pipeline_provider list|adopt|abandon|close|resume`) and no HTTP
     endpoint.

## 3. Shape

A new project type, **`reframe`**, whose page has no segments and no runs. It holds two kinds
of child rows:

```
VideoProject (project_type = reframe, source = landscape or portrait video)
 ├─ ReframeJob      one paid generation attempt: window + target aspect + resolution
 │    │              owns at most one provider task (owner_kind "reframe_job")
 │    └─ ReframeRender   one free, repeatable finish: ending + optional CTA → delivered file
 └─ ReframeJob ...  a later attempt (new window, new aspect, or a retry)
```

**Why a project type and not a derived creative.** The input is a source video, not a Run's
result, and everything around the source already exists for projects: upload and Drive
ingest, the 1 GiB cap, normalization, probing, the project list and its filters, and a
deletion path the provider contract already names as a required call site. A derived
creative would need a Run, which a reframe has no use for. The cost is one enum value and a
project page that branches on the type.

**Why generation and finishing are separate rows.** Generation is the only paid step and its
cost scales with output seconds, so the operator picks the window before paying. The ending
(where the picture stops, how long the fade is, where the audio stops, how much black) is a
free ffmpeg render over the stored output, so it can be redone any number of times without a
second paid call. AICP-252 needed exactly this: the first cut point dropped a word, and the
fix was a different ending, not a different generation.

**Why the consumer adds no state machine for generation.** The provider task already has
thirteen statuses, a claim protocol and recovery. A `ReframeJob` does not mirror them: once
it is admitted its generation state **is** its provider task's state, read through the
task row (`ReframeJob.provider_task_id`). The job has only the statuses for the part the
layer does not cover, the local preparation before admission (§5). That avoids a second
copy of the task state and a callback seam between the two, which is the kind of mechanism
that produced new review findings on every revision of the provider contract.

### Modules

| File | Content |
|---|---|
| `app/project_types.py` | `REFRAME = "reframe"`, segmentation `SEG_BLANK` (one full-length keep segment, harmless and never shown), a new `supports_runs=False` field that hides the New Run form and makes `create_run` answer 409 for this type |
| `app/reframe.py` | pure helpers: window and ending validation, target dimensions, estimate, the ffmpeg filter graph for the render, QA checks |
| `app/pipeline_reframe.py` | RQ job bodies: `reframe_prepare(job_id)`, `reframe_qa(job_id)` and `reframe_render(render_id)` |
| `app/api_reframe.py` | `POST /api/v2/projects/{pid}/reframe-jobs`, `POST /api/v2/reframe-jobs/{jid}/renders`, `DELETE` for both, read endpoints |
| `templates/v2/partials/project_reframe_card.html` | the project page body for this type (§9) |
| `models.py` | `ReframeJob`, `ReframeRender` |
| `recovery.py` | a locked CAS reaper for `preparing` jobs, a QA re-enqueue, and a `_DERIVED_CREATIVE_SPECS` entry for renders (§7) |
| `api_v2.py` | the `delete_project` and re-analysis guards (§7), and the four operator-action endpoints for a job's task (§5) |

## 4. Data model

One additive Alembic revision chained on Stage 1's revision `m3c9e5a1b427` (the head of main at `b73e3bf`; re-check the head when the branch is cut):
`ALTER TYPE project_type_enum ADD VALUE IF NOT EXISTS 'reframe'` (the localisation precedent,
`a4c8e2b6d139`), and two tables. **The revision never uses the new label** (no backfill, no
default, no CHECK naming it): PostgreSQL refuses to use an enum value added in the same
transaction, and the whole upgrade runs in one. Rehearse `upgrade head` from Stage 1's head on
a copy of the production schema, including the case where Stage 1's revision and this one run
in one `upgrade` call. Downgrade drops the two tables (refused while any `reframe_jobs` row
has a `provider_task_id`: the link to a possibly billed request lives there) and leaves the
enum label, as every enum migration here does.

**`reframe_jobs`**

| Column | Notes |
|---|---|
| `id` (uuid str), `project_id` (FK `video_projects.id`, `ON DELETE CASCADE`) | |
| `status` | `preparing`, `prepare_failed`, `admitted` (varchar + CHECK) |
| `target_aspect` | `9:16` or `16:9` |
| `resolution` | `480p` or `768p` |
| `window_start_sec`, `window_end_sec` | on the source timeline, as requested; `0 ≤ start < end ≤ source duration`, `end - start` in [1, `REFRAME_MAX_WINDOW_SEC`] |
| `input_start_sec`, `input_frames` | what was actually cut: the presentation time of the first source video frame at or after the window start, and the frame count of `input.mp4`; every later time computation uses these (§6) |
| `input_path`, `input_sha256`, `input_bytes` | the prepared input (§5) |
| `audio_path`, `audio_start_sec`, `audio_duration_sec`, `source_has_audio` | the pinned audio snapshot (§5) |
| `estimated_usd_base`, `estimated_usd_reserved` | display: `output_sec × tariff` and the layer's worst case (`× 2.0` overlap) |
| `provider_task_id` | soft reference to `provider_tasks.id`, set in the admission transaction; NULL before |
| `error_code`, `error_message` | for `prepare_failed` only; generation errors live on the task |
| `prep_claim_owner`, `prep_claim_expires_at`, `prep_attempts` | the prepare claim (§5) and how many times it has been taken |
| `qa_status`, `job_qa` | `pending` / `running` / `passed` / `failed`, plus the J-check values (§6), including `raw_frames`; `qa_claim_owner`, `qa_claim_expires_at` for the QA claim |
| `created_at`, `updated_at` | |

Partial unique index **`uq_reframe_jobs_preparing_per_project`** on `(project_id)` WHERE
`status = 'preparing'`: at most one job is being prepared per project, so two quick clicks
cannot both reach admission. Together with the admission check in §5 this gives **one
generation in flight per project** (R5).

**`reframe_renders`**

| Column | Notes |
|---|---|
| `id`, `job_id` (FK `reframe_jobs.id`, `ON DELETE CASCADE`) | |
| `status` | `queued`, `processing`, `delivering`, `done`, `failed` (the tail-edit set) |
| `picture_end_sec`, `fade_sec`, `audio_end_sec`, `black_tail_sec` | the ending as requested (§6), source timeline |
| `library_clip_id`, `cta_path` | the optional CTA clip's id (soft reference, as tail edits store it) and the render-owned snapshot of its file (§6) |
| `language` | the creative's language (default `en`), only for a later Job ID (§8) |
| `result_local_path`, `result_bytes`, `result_sha256`, `result_duration_sec`, `result_width`, `result_height` | |
| `qa` | JSON: the R-checks of §6 with their values and pass/fail |
| `gdrive_file_id`, `gdrive_link` | |
| `error_message`, `created_at`, `updated_at` | |

Files, all under the project directory and outside every path `disk_cleanup` deletes:

```
projects/<pid>/reframe/<job id>/input.mp4      prepared input (kept: its hash is frozen on the task)
projects/<pid>/reframe/<job id>/audio.flac     pinned source audio for the renders
projects/<pid>/reframe/<job id>/raw.mp4        the provider task's destination_path
projects/<pid>/reframe/<job id>/sheet.jpg      contact sheet for visual QA
projects/<pid>/reframe/<job id>/renders/<render id>/cta.mp4     hard link (or copy) of the CTA clip
projects/<pid>/reframe/<job id>/renders/<render id>/final.mp4
```

`input.mp4` stays until the job is deleted: the provider layer reads it once at dispatch and
fails the task `input_changed` / `input_missing` if it is gone or different, and a recovery
re-dispatch of a `queued` task needs it too.

**Nothing after preparation reads the project source.** The paid input and the audio the
renders use are both cut from the source in the same prepare job, so a later change of the
source (a re-analysis that re-downloads a Drive file, §7) cannot make the delivered picture
and sound come from different bytes.

## 5. Generation: create → prepare → admit

**Lock order, everywhere in this design**: the `video_projects` row, then `reframe_jobs` rows,
then `reframe_renders` rows, then whatever the provider layer takes inside admission (its task
row, then the spend day row, which `spend.reserve` takes last and holds to the commit). No
path takes a project or job lock after a spend lock.

**Create** (`POST /api/v2/projects/{pid}/reframe-jobs`, body: window, aspect, resolution).
In one transaction:

1. `SELECT ... FROM video_projects WHERE id = :pid FOR UPDATE` (the consumer-contract lock
   object; §6 of the provider contract).
2. Refuse with 409 unless: the project type is `reframe`; the project is `ready`; no job of
   the project is `preparing`; no provider task with `owner_project_id = pid` and
   `owner_kind = 'reframe_job'` is **active** (`pipeline_provider.active_task_for_project`;
   only reframe tasks can belong to a `reframe` project, so the helper's project-wide scope
   is the same thing).
   Refuse with 400 on an invalid window, aspect or resolution, or when the target aspect
   equals the source's (a 16:9 source asked for 16:9).
3. Refuse with 403 unless `FAL_ENABLED` and `FAL_API_KEY` are set and the project passes the
   layer's data-handling gate (`pipeline_provider.allowed_project_ids()`, i.e.
   `FAL_PROJECT_IDS`; R2), and with 503 while no `fal-worker` is live. Both are the checks
   admission repeats; asking first saves an encode that admission would throw away. The
   worker check is the layer's private `_fal_worker_alive()`; the implementation exposes it
   as a public one-line wrapper in `pipeline_provider` rather than importing a private name.
4. `api_v2._budget_precheck(db, "reframe", needed=estimated_usd_reserved /
   settings.KIE_USD_PER_CREDIT)`: the existing helper takes credits, and this is the layer's
   conversion (§5 of the contract). It is advisory, so the common "over the daily cap" case
   is a synchronous 429; the real reservation is at admission.
5. Insert the job `preparing`, commit, enqueue `reframe_prepare(job_id)` on `QUEUE_PROCESS`
   (job id first, so `_live_run_ids` sees it).

**Prepare** (`reframe_prepare`, free, local, the range attempt's claim protocol):

- **Claim first.** `UPDATE reframe_jobs SET prep_claim_owner=:token,
  prep_claim_expires_at=now+REFRAME_PREP_LEASE_SEC WHERE id=:jid AND status='preparing' AND
  (prep_claim_owner IS NULL OR prep_claim_expires_at < now) AND prep_attempts < 3`, also
  incrementing `prep_attempts`, must change one row, else the job exits benignly (another execution owns it, or the job is gone). The lease is
  `REFRAME_PREP_JOB_TIMEOUT` (the RQ timeout of this job, 1800 s) plus 300 s, validated at
  start-up as the provider layer does, so a live worker never outlives its lease.
- **All writes go to claim-scoped staging**: `reframe/<job id>/prep-<token>/`. The worker
  never creates the job directory itself after it has been removed (no `makedirs` of a
  parent), so a deleted job cannot be resurrected on disk.
- **One demux, two outputs.** One ffmpeg invocation reads `VideoProject.source_local_path`
  (not an assumed `source.mp4`: a compliant upload keeps its own name and extension through
  `normalize_source`) with an accurate seek to `input_start`, and writes both the video input
  and the audio snapshot, so both are cut at the same container instant and the source's own
  audio/video offset is preserved: `aresample=async=1:first_pts=0` pads a late-starting audio
  stream with leading silence instead of shifting it. `input_start` is the presentation time
  of the first video frame at or after `window_start`, read from the stream's packets
  (`ffprobe -read_intervals`), not computed from `r_frame_rate`, so variable-frame-rate
  sources are handled; the video output is forced to constant frame rate (`fps=` at the
  source's average rate, capped at 30; the output is 24 fps anyway).
- **Video input** `input.mp4`: duration `window_end - input_start`; short side at most 1080,
  orientation kept; libx264 CRF 18 with VBV `-maxrate R -bufsize R`, where `R = min(12 Mb/s,
  0.85 × FAL_MAX_INPUT_BYTES × 8 / duration − 128 kb/s)`; AAC 128k; faststart. With VBV the
  file is bounded by about `R × duration / 8 + one buffer`, under the cap by construction
  (about 5 Mb/s at the 120 s default maximum, ample for a 768p output). **Check, then one
  fallback:** if the probed size is still over `FAL_MAX_INPUT_BYTES`, re-encode once at
  `0.6 R` with the short side at 768; if it is still over, the job ends
  `prepare_failed(input_too_large)` with both sizes in the message. `input_frames` is read
  from the result; its duration must equal the requested length within one source frame,
  else `prepare_failed(input_duration)`.
- **Audio snapshot** `audio.flac`: from `input_start` for `min(source duration, window_end +
  REFRAME_AUDIO_RUNON_MAX_SEC) - input_start` (`REFRAME_AUDIO_RUNON_MAX_SEC` = 3 s, enough
  for AICP-252's 0.9 s), lossless, at the source's sample rate. A source without an audio
  stream stores `source_has_audio = false` and no file.
- **Publish under the lock.** The admission transaction (below) re-checks the claim
  (`prep_claim_owner = :token AND prep_claim_expires_at > now`) inside its CAS, and only
  after that CAS succeeds `os.replace`s the staged files onto `input.mp4` and `audio.flac`
  and computes nothing more (the hashes were computed from the staged bytes). A worker that
  lost its claim deletes its own staging directory and exits; no other execution ever
  writes the published paths.

This replaces the contract's "an ineligible source is offered a smaller re-encode": every
source is re-encoded toward an eligible input, and the rare one that still does not fit ends
with a clear local failure before anything is reserved or sent. The cost is one generation
of encoding loss on the preserved scene; at these bitrates it is below what the model's own
output encode does.

**Admit**, still in `reframe_prepare`, one transaction, in the lock order above:

1. Lock the project row (`FOR UPDATE`).
2. **CAS the job first**: `UPDATE reframe_jobs SET status='admitted', input_*=..., audio_*=...,
   prep_claim_owner=NULL, updated_at=now WHERE id=:jid AND status='preparing' AND
   prep_claim_owner=:token AND prep_claim_expires_at > now`, which must change one row, then
   flush, then publish the staged files (above). Zero rows means a `DELETE`, the reaper or a
   newer claim ended this execution's ownership (§7): roll back, delete this execution's
   staging directory, stop. Nothing was reserved.
3. Call `pipeline_provider.admit` **in the same transaction**:
   `endpoint_id="minimax/h3-max/reframe-video"`,
   `params={"aspect_ratio": job.aspect, "resolution": job.resolution}`,
   `inputs=[TaskInput("video_url", <published input.mp4>, "video/mp4")]`,
   `output_sec` = the probed duration of the published `input.mp4` (so the layer's
   duration check holds by construction), `resolution=job.resolution`,
   `owner_kind="reframe_job"`, `owner_id=job.id`, `owner_slot="main"`,
   `owner_project_id=pid`, no `owner_run_id`,
   `operation_key="reframe_job:<job id>:main:1"`,
   `destination_path=os.path.join(storage.data_dir_root(), "projects", pid, "reframe",
   job.id, "raw.mp4")`, an **absolute** path (the layer resolves a relative one against the
   worker's working directory, not `DATA_DIR`, and would refuse it with 422); its directory
   exists, because the publish created it, and the file does not. The input path is built
   the same way. (Paths elsewhere in this document are written relative to `DATA_DIR` for
   brevity.) The layer locks the project row again (already
   held by this transaction, so no new wait), hashes the input, inserts its task row, then
   reserves under `fal_reframe` (the spend day lock is the last lock taken).
4. Set `job.provider_task_id` on the already-locked job row (no new lock: the row was
   locked by step 2's UPDATE), commit, and then, **only when `admit` returned
   `created=True`**, call `tasks.enqueue_fal_job("fal_submit", task.id)`. If that enqueue
   fails after the commit, nothing is lost: the task is `queued` and the layer's recovery
   resubmits it after `FAL_QUEUED_GRACE_SEC`. (`created=False` cannot happen here, because
   the job CAS admits each job once, but the code still follows the rule.)

`BudgetExceeded` at admission rolls the whole transaction back, the job's CAS included, so
the job is `preparing` again with this worker's claim still on it. File moves are not
transactional, so the published `input.mp4` and `audio.flac` stay on disk; that is harmless,
because only the claim holder ever writes those paths and a `preparing` job has no task that
could read them. A second short transaction (project lock, then the same claim-checked CAS
from `preparing`) sets it `prepare_failed(budget)`, and the worker then deletes the published
files. A crash between the publish and the commit leaves the same state, which the reaper
ends when the claim expires (§7); the files go with the job's deletion.
`AdmissionRefused` is handled the same way, with the layer's `detail` kept in
`error_message` and an `error_code` by status: 403 `fal_disabled`, 503 `fal_worker_down`,
422 `admission_invalid`, 404 and 409 `admission_conflict` (neither should happen under this
job's lock and a fresh destination; they are failures to investigate, not retried). None is
retried by the worker: the operator starts a new job once the cause is fixed, and create's
step 3 makes the 403 and 503 cases rare. Nothing was sent or reserved in any of these
cases.

**One attempt per job.** A retry is a new `ReframeJob`, never `attempt_no + 1` on the same
owner: each attempt keeps its own window, input, output and render history, and the
operation key is simply unique per job. The one-generation-per-project rule (create step 2)
is what stops a retry while an earlier task is still active, including a held `refused` or a
parked `submit_unknown` task, which keeps its slot until the operator settles it.

**Displayed state** of an admitted job is the task's status, mapped for the operator:

| Task status | Card shows |
|---|---|
| `queued`, `submit_started`, `accepted_unrecorded` | Sending to Fal |
| `submitted` | Generating (elapsed time; request id) |
| `fetching` | Downloading |
| `cancelling`, `needs_reconciliation`, `closing` | Waiting for Fal to confirm (`needs_reconciliation`: the Resume and Close actions below) |
| `submit_unknown` | **Unknown — may have been billed.** The Adopt / Abandon / Close actions below |
| `done` | Ready — then the QA state below |
| `refused` | Refused by Fal (category; `moderation` for a policy refusal). While the task still holds its reservation (`reservation_held`), it keeps the project's generation slot: the Abandon and Close actions below settle it |
| `failed` | Failed (`error_code`; `result_gone`, `owner_gone`, `input_changed`, ...) |
| `abandoned` | Abandoned |

**Operator actions on a job's task** (owned here: Stage 1 ships the functions and a CLI, no
HTTP endpoints). `POST /api/v2/reframe-jobs/{jid}/task-actions/{action}`, `action` one of
`adopt`, `abandon`, `close`, `resume`; form body `note` (required, non-empty) and, for
`adopt`, `request_id`.

1. Load the job (404 if absent) and require `job.provider_task_id` (409 otherwise). The task
   id always comes from the job row, never from the request, so the endpoint cannot act on
   another consumer's task.
2. Call the matching function with `actor="dashboard"` and the note. Map `ActionRefused` to
   an HTTP error with its own `status_code` and `detail`; the function is the authority on
   which state allows which action, and the buttons (shown only for the states in the table
   above) are only a convenience.
3. Answer with the refreshed card.

No reframe lock is held around the call: each function does its own CAS and commits, so a
lock taken before it would be released mid-call anyway, and none of them touches a reframe
row. Deletion stays safe because the job and project deletes read the task's status under
their own locks (§7): they see it active and 409, or settled and proceed. `adopt` makes one
status GET to Fal inside the request, bounded by the client's timeout. The CSRF check
applies to these POSTs like any other.

**Job QA is a claimed, durable step that gates renders.** `qa_status` starts `pending`. The
page, when it sees the task `done` and `qa_status = pending`, enqueues `reframe_qa(job_id)`;
the reconcile tick does the same for a `pending` row with no live job, and for a `running`
row whose claim has expired (QA is free and idempotent, so re-running it is safe). The job
claims it in a short transaction that **locks the project row first** and then runs the
job-row CAS (`pending` → `running`, or an expired `running`, with a token and a lease of its
RQ timeout plus 300 s), so a claim and a deletion's QA check are serialized by the same lock;
it runs J1-J4 (§6) on `raw.mp4`, builds the contact sheet
as `sheet-<token>.jpg`, and commits `passed` or `failed` with the values in `job_qa` by a
CAS on its own token and an unexpired lease; only after that CAS succeeds does it
`os.replace` its sheet onto `sheet.jpg`. A worker that lost its claim, or finds the job
directory gone, removes its own temporary file and exits; it never creates directories. Job
and project deletion, under the project lock, 409 while QA is `running` with a live claim;
a deletion that wins the lock first removes the row, and the QA claim then changes zero rows. **Render creation requires
`qa_status = passed`**, read under the project and job locks; a page in any other QA state
shows why the render form is disabled.

## 6. Finishing: the render

**Create** (`POST /api/v2/reframe-jobs/{jid}/renders`, body: `picture_end_sec`, `fade_sec`,
`audio_end_sec`, `black_tail_sec`, optional `library_clip_id`, `language`). Lock the project
row, then the job row; then require the job `admitted`, its task `done`, `qa_status =
passed` and `raw.mp4` present; and validate the ending:

- `1 ≤ NP ≤ job_qa.raw_frames`, with `NP` computed as in the timeline below from the
  **probed** raw frame count, not from `window_end` (J2 allows the raw output to be a frame
  short, so the window end may not exist as a frame);
- `0 ≤ fade_sec ≤ 2`;
- `picture_end − 1/24 ≤ audio_end ≤ audio_start + audio_duration` (one frame of tolerance,
  because the raw output may be a frame longer than the input, which J2 allows; the pinned
  snapshot bounds the end; the audio can
  run on up to 3 s past the window, which is how AICP-252 kept "download below"); on a
  source without audio `audio_end` is ignored;
- `0 ≤ black_tail_sec ≤ 5`;
- an optional CTA clip must exist and have the target's aspect (width/height within 1 %),
  else 400. **The clip is snapshotted** into `renders/<render id>/cta.mp4` in this request
  (a hard link, falling back to a copy to a temporary name and an `os.replace`; library
  clips are a few MB), so deleting the library clip later cannot touch a queued render, and
  the render reads only `cta_path`. A clip deleted between the check and the link (library
  deletion takes no lock this request could share) is a clean 409 "clip no longer
  available": the partial file is removed and the transaction rolled back.

Insert `queued`, commit, enqueue `reframe_render(render_id)`.

**Timeline.** Everything is computed in **output frames at 24 fps** and **audio samples at
48 kHz**, from the job's `input_start` (not the requested window start), so that the
picture and sound are cut on the same grid:

```
s  = input_start                         (the instant output frame 0 shows)
NP = round((P - s) × 24)                 frames of generated picture used, 1 ≤ NP ≤ raw_frames
NF = round(F × 24)                       fade frames (0 = cut to black, no fade filter)
NA = round((A - s) × 48000)              audio samples from the snapshot (0 without audio)
NL = max(NP + NF, ceil(NA / 2000)) + round(T × 24)    total body frames (2000 samples/frame)
```

```
frame 0 ........ NP ........ NP+NF ........ ⌈NA/2000⌉ .... NL
|  generated picture | hold last frame, fade to black | black .......... |
|  snapshot audio, samples 0 .. NA ................. | silence ........ |
                                                         then the optional CTA clip
```

**Video**: `raw.mp4` → `trim=end_frame=NP,setpts=PTS-STARTPTS` → `tpad=stop_mode=clone:
stop=NL-NP` → when `NF ≥ 1`, `fade=t=out:start_frame=NP:nb_frames=NF:color=black` (frames
after the fade stay black); when `NF = 0` (a zero or sub-half-frame fade) no fade filter, and
the frames from `NP` on are drawn black (`drawbox=...:color=black:t=fill:enable='gte(n,NP)'`),
since FFmpeg's `nb_frames` minimum is 1 → `fps=24`, native output size, exactly `NL` frames (`-frames:v NL`). **Audio**:
`audio.flac` → `aresample=48000` → `atrim=end_sample=NA,asetpts=PTS-STARTPTS` (the snapshot
already starts at `s`, so no start offset is needed and none can be forgotten) → a 10 ms
ramp ending at `NA` only to avoid a click (not an audible fade) → `apad=whole_len=NL × 2000`;
a source without audio uses `anullsrc=r=48000` of `NL × 2000` samples (the silent-source
stitch lesson). One encode with `delivery_video_opts`, AAC 48 kHz, faststart, then
`enforce_size_limit`. With a CTA clip the body is passed with `cta.mp4` through
`media.stitch([body, cta], audio_mode="seedance")`, as a tail edit appends one, at the body's
size and 24 fps.

Defaults the form pre-fills, chosen to satisfy the validation above by construction:
`P = s + min(raw_frames, round((window_end − s) × 24)) / 24` (the last generated frame that
exists and is inside the window), `A = min(max(P, window_end), audio_start + audio_duration)`, `F = 0.65`, `T = 0.25`;
the operator moves `A` past `P` when the last word is cut. The page plays the snapshot's last
two seconds around `P` so the operator can hear where the phrase ends (no waveform in v1).

**Checks** (J on the job, gating renders; R on each render, gating delivery):

| Id | Check | Outcome |
|---|---|---|
| J1 | raw size is the expected size for aspect × resolution (768×1344, 1344×768, 480×832, 832×480) | warn |
| J2 | raw frame count = `input_frames` converted to 24 fps (`round(input duration × 24)`) ± 1 | **fail**: `qa_status = failed`, no renders, since audio would drift |
| J3 | `placement` rectangle (`result_meta.placement`, Q1) vs the input: SSIM ≥ 0.85 at three timestamps | warn; shows the measured SSIM. Skipped when Fal's result carried no `placement` |
| J4 | contact sheet (3×3 frames) built | informational |
| R1 | body: exactly `NL` frames at 24 fps and the expected size; final: `NL` + the CTA's frames (± 1 frame for the CTA's resample) | fail |
| R2 | body audio: duration `NL × 2000` samples ± 1 frame; with source audio, the RMS envelope (50 ms windows) of the body's first `NA` samples matches the snapshot's within 1.5 dB on every window louder than −50 dBFS in the snapshot, and stays below −45 dBFS where the snapshot is quieter than −50 dBFS (a silence-aware comparison: a quiet opening is checked as silence, not by an undefined correlation), with the envelopes' best alignment at lag 0 ± 1 window; after `NA`, and throughout when the source has no audio, the body is silent (peak below −60 dBFS) | fail |
| R3 | final audio: duration = final video duration ± 1 frame (body plus the CTA's own audio or synthesized silence) | fail |
| R4 | file ≤ 100 MB after `enforce_size_limit` | fail |

R1-R3 run on the body before the CTA stitch and on the final after it, as stated. A failed
R-check ends the render `failed` with the check in `error_message`; nothing is delivered.

**Delivery**: `gdrive.upload_file(final.mp4, default folder, "reframe_<job8>_<render8>.mp4")`
with the tail edit's `_deliver_with_retry`; `gdrive_file_id` and `gdrive_link` are stored.
The render is `done` only after the Drive upload succeeded.

## 7. Deletion, cleanup and recovery

**Deletion** (the provider contract's consumer obligations (b) and (c), plus our own rows).

- `delete_project` already locks the project row, then its runs, and 409s on an active
  provider task (Stage 1). It gains, right after the active-task check: 409 while the
  project has a `preparing` job or a render in `queued`, `processing` or `delivering`.
  Only then the existing checks, `db.delete` and `rmtree`. Lock order is unchanged: project
  row first, then runs ascending.
- `DELETE /api/v2/reframe-jobs/{jid}`: under the project lock, 409 while the job is
  `preparing`, its task is active, or one of its renders is active; otherwise delete the job
  (cascading renders) and its directory. The provider task row stays: it is the only record
  of a possibly billed request (soft reference, no FK).
- `DELETE /api/v2/reframe-renders/{rid}`: hard delete like tail edits; 409 while active. The
  Drive file is left in place (as today for tail edits).

- **Re-analysis** (`POST /projects/{pid}/analyze`) of a `reframe` project **always answers
  409**; a reframe project whose first analysis failed (a Drive download error, a broken
  file) is deleted and created again. Its segmentation is a placeholder (`SEG_BLANK`), so
  re-analysis has nothing to offer, and refusing it removes the only path that rewrites the
  source after the project is `ready`. The first analysis cannot overlap a prepare: create
  requires `ready`, which only that analysis sets, and once it is `ready` nothing writes the
  source again. (A first analysis that recovery marked `failed` from a stale RQ snapshot and
  that then finishes and sets `ready` anyway is harmless here: it has stopped writing by
  then.) This leaves the existing analysis recovery untouched. Prepare is the only reader of
  the source (§4); an admitted job and its renders read only their own snapshots.
- **Library clip deletion** needs no new guard: a render reads its own `cta.mp4` snapshot.

**Disk cleanup.** No change to what it deletes; a test pins that `sweep` never touches
`projects/<pid>/reframe/`, and its docstring lists the path among those it never touches.

**Recovery.**

- `reframe_jobs` in `preparing`: **not** a `_DERIVED_CREATIVE_SPECS` entry. That reaper reads
  rows and commits plain ORM updates without a lock or a status condition, so it could
  overwrite a job that admission had just moved to `admitted` with a paid task behind it.
  Instead a dedicated step in `reconcile_orphaned_runs`. Prepare is free and its claim makes
  duplicate executions harmless, so the reaper's default answer is to **re-enqueue, never
  to fail**: a `preparing` row with no live claim (none, or expired) and no live RQ job
  older than the 300 s grace gets another `reframe_prepare`. A backed-up queue therefore
  never costs a job its place, and a lost enqueue is repaired. The only failure is bounded
  retries: the claim itself refuses a fourth attempt (`prep_attempts < 3` is part of its
  condition), and when `prep_attempts` has reached 3 and the claim has expired, a short
  transaction locks the project row and runs `UPDATE reframe_jobs SET
  status='prepare_failed', error_code='prepare_crashed' WHERE id=:jid AND
  status='preparing' AND prep_attempts >= 3 AND prep_claim_expires_at < now`, so the
  decision is re-made against the row under the lock and a live claim always survives.
  Admission runs its CAS under the same project lock (§5, admit step 2) and requires its
  own unexpired claim, so exactly one of them wins in either commit order. A prepare worker that wakes after
  losing changes zero rows and stops before reserving anything; its files never left its
  staging directory.
- `reframe_jobs` QA: a `pending` row whose task is `done` and has no live `reframe_qa` job,
  or a `running` row with an expired QA claim, is re-enqueued (§5). Never failed by a reaper:
  it costs nothing to run again.
- `reframe_renders` in `queued`, `processing`, `delivering`: the tail-edit entry pattern. A
  render reaped in `delivering` may already have uploaded a Drive file; Retry uploads a new
  one, as tail edits do today (a known, free duplicate).
- **Provider tasks are never reaped here**, as the provider contract requires. An admitted
  job has no status to reap; its task is recovered by the layer.
- Deploy drain: `deploy/drain-check.sql` gains a `reframe_jobs` line (`preparing`, and QA
  `running`) and a
  `reframe_renders` line (the three active statuses). Provider tasks already have theirs.

## 8. Cost evidence and the Final Creatives sheet

**Cost on the page**, per job: tariff, output seconds, the base estimate (`× tariff`), the
worst case the layer reserved (`× 2.0` overlap), the Fal request id, submit and finish times,
and the sentence "Estimate only: Fal does not report the debit to this key; the account
dashboard does." The daily spend readout shows the `fal_reframe` kind with its provider
label (Stage 1's `provider` column). The project card totals base and reserved estimates
over all of the project's jobs.

**Sheet export is not in v1** (R4). A reframe creative's Job ID and Pipeline Type
(`re-skin-reframe`, `<render8>-<lang>`) are only useful once the downstream renamer in
ai-fb-audit accepts them, which has not been checked. v1 delivers to Drive; the operator
can register a render as a manual creative through the existing `manual_creatives` CLI if
it must reach the sheet sooner, which needs a small `register_reframe_render` path next to
`register_overlay` (in scope only if R4 says so).

## 9. UI

The project page for a `reframe` project replaces the segment editor and the runs list with
one card:

- **Source**: size, duration, orientation, fps; the target aspect it implies (landscape →
  9:16, portrait → 16:9; a square source lets the operator pick).
- **New generation**: window start/end (defaults 0 and `min(duration, REFRAME_MAX_WINDOW_SEC)`),
  resolution (480p / 768p; 1080p is shown disabled, "no Fal price"), and the live estimate.
  The button text names the money: "Generate on Fal — about 2.08 USD (up to 4.16 reserved)".
  A confirm dialog repeats the estimate and that the video is sent to Fal.
- **Attempts**, newest first: status (the table in §5), request id, elapsed, estimate, QA
  badges, the raw output player and contact sheet when `done`, the operator actions (§5)
  when the task needs one, and Delete.
- **Renders** under each finished attempt: the ending form with its defaults and a timeline
  sketch, then each render's player, QA badges, Drive link and Delete.

The Create Project form gets the type from the registry automatically; its hint says the
video will be sent to Fal, a third party.

## 10. What this needed from Stage 1, and the answers

Revision 5 asked four things of Stage 1. The owner's decisions (2026-10-28) and what the
merged code (`b73e3bf`) does:

| | Request | Answer | Consequence here |
|---|---|---|---|
| Q1 | Persist the result's small fields on the task | **In Stage 1**: `provider_tasks.result_meta` (§2, fact 8) | J3 reads `result_meta.placement`; the QA report also shows `seed` and `frame_count` |
| Q2 | One helper for "active task for this project" | **Already there**: `active_task_for_project(db, pid)` on `PROVIDER_TASK_ACTIVE_WHERE` | create step 2, the job delete and `delete_project` use it |
| Q3 | Operator actions as HTTP endpoints | **Not in Stage 1**: functions plus a CLI; the endpoints are AICP-251's | §5, *Operator actions on a job's task* |
| Q4 | Admission in the caller's transaction | **Already the case**: `admit(db, ...)` never commits and returns `(task, created)` | §5, admit steps 3-4; the caller enqueues `fal_submit` after its commit |

## 11. Decisions for the owner

All nine accepted as recommended by the owner on 2026-10-28.

| | Question | Recommendation (accepted) |
|---|---|---|
| R1 | Project type `reframe`, or reframe as an action on a Run's final / tail edit? | Project type in v1. A later "reframe this deliverable" can create a reframe project from a final, the way overlays take a run source |
| R2 | Data handling gate. D3 is closed (main `2c1f15d`: Fal held to the kie.ai standard, no separate terms review), but the contract's `FAL_PROJECT_IDS` env allowlist still needs an `up -d` per project | Keep the allowlist for the canary; then let the project type itself be the gate (creating a `reframe` project is the choice to send that video to Fal, stated in the type's hint), which needs a one-line contract amendment for `owner_kind = reframe_job` |
| R3 | Output size: native (768×1344) or upscaled to 1080×1920 in the render? | Native in v1 (the accepted AICP-252 delivery was native); an upscale option later if ad platforms or CTA clips need it |
| R4 | Final Creatives export with Pipeline Type `re-skin-reframe` and Job ID `<render8>-<lang>` | Phase 2, after checking that ai-fb-audit accepts them |
| R5 | One generation in flight per project? | Yes: a second click cannot double-bill, and Fal's account concurrency is 2 anyway |
| R6 | `REFRAME_MAX_WINDOW_SEC` (model limit 300 s) | 120 s: the worst case is 19.20 USD reserved at 768p, and a 300 s job would run close to the layer's one-hour poll deadline |
| R7 | Ending defaults | Hold + 0.65 s video fade, audio to the window end, 0.25 s black (the AICP-252 shape, measured from its final) |
| R8 | Directions | 16:9 → 9:16 and 9:16 → 16:9 only; the others the model offers stay off until someone needs them |
| R9 | Where job QA lives | On `reframe_jobs` (`qa_status` + `job_qa`), a claimed step that must pass before any render (§5) |

## 12. Verification plan

**Unit** (`tests/test_reframe.py`): window and ending validation edges (P at the window end,
A at the source end, a zero fade, a zero tail, the aspect equal to the source's); the
estimate (base and reserved, 480p and 768p); the input bitrate budget at 1, 120 and 300 s;
the render filter graph on synthetic media (a 2 s colour-bars clip with a tone): output
duration `L`, a black final frame, audio equal to the source's for `A - s`, silence after.

**Flow, with the layer's transport mocked** (`tests/test_reframe_flow.py`, PostgreSQL): create
→ prepare → admit inserts exactly one task with the right owner, slot, key, destination and
spend kind; a second create while preparing is 409 and while the task is active is 409, for
every active task status including `submit_unknown` and a held `refused`; a reaped
`preparing` job does not admit when its worker wakes; `BudgetExceeded` leaves no task and no
reservation; each `AdmissionRefused` status (403, 503, 422, 409) ends the job
`prepare_failed` with its `error_code` and leaves no task and no reservation; `fal_submit` is
enqueued once, after the commit, and not at all when the commit fails; create answers 403
and 503 before any encode; the four operator-action endpoints call the right function with
the job's own task id, map `ActionRefused` to its status, 404 an unknown job and 409 a job
without a task; `delete_project` and job delete 409 for each active case and succeed after; the
render refuses before `done`, on a changed source, and on a failed J2; a CTA clip of the
wrong aspect is a 400; the recovery entries reap `preparing` and active renders and never
touch a provider task; `disk_cleanup.sweep` leaves `reframe/` alone. Races, each in both
commit orders: the `preparing` reaper against admission (including a reaper acting on a
stale snapshot, which must change zero rows); `DELETE` of a job against admission;
re-analysis against prepare (409); two QA jobs for one row (one claim wins); a render
create before QA passed and after QA failed (409). The input encoder's fallback and its
`prepare_failed(input_too_large)` end, with a synthetic high-entropy clip and a lowered
`FAL_MAX_INPUT_BYTES`. A CTA library clip deleted after its render was queued (the render
still succeeds from its snapshot). Frame math: a non-zero, non-frame-aligned window start
(the body's first audio sample is the snapshot's first sample, and `NL`, `NP`, `NA` match
the encoded file); a source without audio; a body plus CTA (R1-R3 on both). Migration:
`upgrade head` from Stage 1's head in one call on a copy of the production schema, and
downgrade refused while a job has a `provider_task_id`.

**Real media, free**: run the render on the probe's `out_t1_reframe_9x16.mp4` + `in_h_5s.mp4`
and on the AICP-252 source with a window ending at 25.75 s and `A = 26.65`, and compare with
the accepted final (duration 26.667 s, 768×1344, 24 fps; the audio must match the source
through 26.65 s).

**Paid canary** (separately authorized, owner reading the Fal dashboard before and after):
one 5 s 480p job (0.25 USD base, 0.50 reserved) on an allowlisted project, then one render,
then the same at 768p. Record request ids, wall time, QA values and the dashboard delta in
the wiki note.

## 13. Rollout

Behind the layer's `FAL_ENABLED=false`; with it off the `reframe` type can be selected but the
generate button is disabled with the reason. Migration before deploy, per CLAUDE.md. Once
Fal is enabled, every deploy passes `--profile fal` (CLAUDE.md, *Fal worker*), or the
`fal-worker` keeps its old image and, if stopped, admission answers 503. Release
follows the provider layer's (it cannot ship before Stage 1). An operator runbook section is
added to the wiki note: how to read a job's state, what to do on `submit_unknown`, how to
pick the ending, and where the cost evidence is.

## Implementation (AICP-251, branch `feat/reframe`)

Built 2026-10-28 on main `2842bad` (Stage 1 `b73e3bf`); migration `n4d0f6b2c538` on
`m3c9e5a1b427`. The sections above stay the approved revision 6; this section records
where the code departs from them. **Where the two disagree, this section and the code
win.**

- **R2 gate relaxed (owner, 2026-10-29).** A `reframe` project no longer needs its id in
  `FAL_PROJECT_IDS`: `pipeline_provider.project_allowed()` admits the allowlist OR any
  project of type `reframe` (the type exists only to send its source to Fal; its hint
  says so, and each generation is confirmed with its cost estimate). `FAL_ENABLED` +
  `FAL_API_KEY` stay the master switch, and the daily spend cap, one-in-flight rule and
  503-without-worker are unchanged. Both `admit()` and the page's `fal_gate()` use it.

| Item | Code | Why |
|---|---|---|
| Render reaping (§7) | Not a `_DERIVED_CREATIVE_SPECS` entry: `recovery._reconcile_reframe_jobs` fails an orphaned render with `pipeline_reframe.reap_orphaned_render`, a CAS on the observed `status`, `claim_owner` and `updated_at` | Terra gate passes 2-3: the shared reaper writes `failed` unconditionally from a stale read, so it could overwrite a render that finished (or was retried) in between |
| Render ownership (§6) | New column `reframe_renders.claim_owner`; every status change after `queued` is a CAS on (status, claim); Retry clears it | Terra gate pass 1: a reaped-then-retried render could otherwise have two workers writing one directory |
| Render files (§4) | `renders/<id>/final-<token>.mp4`, one per execution, the winner's path recorded on the row; no shared `final.mp4` | Terra gate pass 2: a superseded worker could overwrite a shared final after its CAS. A superseded worker that already reached `delivering` may still upload a duplicate Drive file (the tail-edit precedent) |
| R2 on the final (§6) | R2 also runs on the delivered file; with a CTA it judges only `[0, NL)` | Terra gate pass 1: the CTA stitch and the size cap re-encode the body audio |
| `input_start` (§5) | Read on the container timeline (the format's `start_time`), which is what an input `-ss` seeks on | Terra gate pass 1: a source whose video starts after its audio was cut at the wrong instant |
| CTA aspect (§6) | Within 1 % of the nominal 9:16 / 16:9 **or** of the model's native ratio (768×1344 is 1.6 % off 9:16) | Terra gate pass 1: a standard 1080×1920 clip was refused |
| Job QA trigger (§5) | The project page fragment enqueues `reframe_qa` at most once a minute per job (a Redis `SET NX EX 60`); the reconcile tick covers a page nobody opens | The claim makes duplicates harmless; the key only keeps the queue short |
| Extra column | `reframe_jobs.input_duration_sec` (the probed input duration; admission's `output_sec` and J2 use it) | |
| Endpoints | `POST /reframe-renders/{rid}/retry`; file reads `/reframe-jobs/{jid}/raw`, `/sheet`, `/audio`, `/reframe-renders/{rid}/result` (paths checked under `DATA_DIR`) | §7's Retry and §9's players |
| Tests (§12) | `tests/test_reframe.py` (pure helpers, real synthetic media, the AICP-312 probe's T1 input + output when `/probe` is mounted, schema), `tests/test_reframe_flow.py` (the flow on SQLite with real ffmpeg and the real `pipeline_provider.admit`), `tests/test_reframe_pg.py` (opt-in `RESKIN_TEST_PG_URL`: the reaper's CAS against admission in both commit orders, QA claim against deletion, two creates at once) | The flow is on SQLite rather than PostgreSQL; the races SQLite cannot show are in the PG file |

**Verification.** 73 reframe tests plus `test_recovery.py`, `test_project_types.py` and `test_api_v2.py` green in the `re-skin-api`
image; the 4 PostgreSQL proofs green on a disposable `postgres:16`; the migration
`upgrade → downgrade → upgrade` rehearsed on a copy of the production schema
(`pg_dump -s`, head `m3c9e5a1b427`), the enum gaining `reframe` and the partial unique
index refusing a second `preparing` job. The probe's T1 output passes J2, J3 (SSIM 0.94
on the centred band) and R1-R4 through the real render. Code gate: GPT-6.1 Terra (Codex,
effort high), one session: pass 1 on the whole diff 3 MAJOR + 2 MINOR, pass 2 2 MAJOR,
pass 3 1 MAJOR, pass 4 NO FINDINGS; every finding was real and is fixed above.

**Not yet exercised:** a paid canary (§12, separately authorized), the real `placement`
shape of Fal's result (J3 parses the usual rectangle shapes and is skipped otherwise),
and a browser pass of the page.

## Known residuals

- **A stale analysis reaper can mark a just-analyzed reframe project `failed`.** The existing
  orphaned-analysis recovery (`recovery.py`, project pass) acts on an RQ snapshot taken
  earlier in the same pass and updates without a status condition, so it can overwrite a
  `ready` that the analyze worker committed in between. This is pre-existing and affects every
  project type. For a reframe project, which refuses re-analysis (§7), the consequence is that
  the operator deletes and re-creates the project: no paid call, no reservation and no
  delivered file is involved, because jobs require `ready`. Follow-up, not fixed here: a
  conditional (`WHERE status = 'analyzing'`) update and a fresh liveness check under the
  project lock in that reaper, for all types. Accepted by the owner on 2026-10-28.

## Review record

| Pass | Reviewer | Result | Changes in the next revision |
|---|---|---|---|
| 1 (rev 1) | GPT-6.1 Terra, Codex CLI, effort high | 1 BLOCKER, 7 MAJOR, 3 MINOR; all verified against the code and accepted | rev 2: the `preparing` reaper became a locked CAS instead of a `_DERIVED_CREATIVE_SPECS` entry; one lock order (project → job → render → layer task → spend day) with the job's CAS flushed before admission reserves; job QA became a claimed step that must pass before a render; R-checks split into body and final, with an explicit silent-source check; the render timeline is computed in 24 fps frames and 48 kHz samples from a frame-snapped `input_start`, with zero-based timestamps; prepare pins an audio snapshot so nothing after it reads the project source, and re-analysis 409s while preparing; the CTA clip is snapshotted per render; the input encode uses VBV plus one bounded fallback and a terminal `input_too_large`; the budget precheck is `api_v2._budget_precheck` in credits; the source is read from `source_local_path`; the migration never uses the new enum label in its own transaction |
| 2 (rev 2, delta) | GPT-6.1 Terra, Codex CLI (`codex exec resume` of pass 1's session, `model: gpt-6.1-terra`, `reasoning effort: high` in the header) | Pass-1 findings closed except re-analysis; 8 MAJOR, 2 MINOR on the new mechanisms. Rejected: the MINOR saying no CLI review ran (the `.out` header shows it did). Accepted: the rest | rev 3: prepare has a durable claim with a lease above its RQ timeout, claim-scoped staging and publish-after-CAS; the reaper decides by the claim, re-enqueues unclaimed rows first; re-analysis of a `reframe` project is refused unless `failed`; QA writes claim-scoped files, never creates directories, and blocks deletion while live; `NP` is validated against the probed raw frame count, and the default picture end is the last real frame; `NF = 0` has its own branch; `input_start` comes from real frame timestamps, video and audio are cut in one demux with a late-audio pad; R2 became a silence-aware envelope comparison; a CTA clip lost mid-request is a clean 409 |
| 3 (rev 3, delta from rev 2) | GPT-6.1 Terra, Codex CLI (resume, header verified) | 4 MAJOR, all accepted; the CLI attribution correction accepted by the reviewer | rev 4: a reframe re-analysis marks the project `analyzing` under the project lock before enqueueing (superseded in rev 5); the QA claim is taken under the project lock; the `preparing` reaper only re-enqueues, and fails a job only after 3 claimed attempts with an expired claim; the default picture and audio ends satisfy the validation by construction, with one frame of tolerance for a raw output a frame longer than the input |
| 4 (rev 4, delta) | GPT-6.1 Terra, Codex CLI (resume, header verified) | 1 MAJOR (the pre-enqueue `analyzing` mark relied on analysis recovery that can fail a live analysis from a stale snapshot), 1 MINOR (a fourth prepare claim was possible); both accepted | rev 5: re-analysis of a `reframe` project is always refused, so the existing analysis recovery stays untouched and nothing rewrites the source after `ready`; the prepare claim refuses a fourth attempt |
| 5 (rev 5, delta) | GPT-6.1 Terra, Codex CLI (resume, header verified) | 1 MAJOR: the existing analysis reaper can overwrite a fresh `ready` with `failed`, and a reframe project cannot then be re-analyzed | Not fixed in this design: pre-existing in `recovery.py` for every project type, and for reframe its cost is re-creating a project, with no billing or delivered-data impact. Recorded under *Known residuals* as a follow-up; the owner accepted it on 2026-10-28 and counted the gate as passed |
| 6 (rev 6, delta from rev 5) | GPT-6.1 Terra, Codex CLI, effort high (new session; header verified), then one resumed pass on the fixes | 2 MAJOR (a relative `destination_path` the layer would resolve outside `DATA_DIR`; no dashboard control for a `refused` task still holding its reservation), 1 MINOR (create's pre-check omitted `FAL_API_KEY`); all accepted. Fix pass: NO FINDINGS | Fixed in rev 6 itself: absolute destination and input paths, Abandon/Close for a held `refused` task, the pre-check matches admission's gate |

