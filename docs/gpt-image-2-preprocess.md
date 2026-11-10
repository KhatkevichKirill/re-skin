# GPT Image 2 — Reference image preprocessing

Status: **DESIGN AGREED, BUILD DEFERRED** (decided 2026-07-29). Implementation tracked separately.

## Problem

When creating a Run, the operator can attach several reference images used by Seedance/Gemini Omni. For better generation quality the operator sometimes pre-processes those images with **GPT Image 2 (image-to-image)** before attaching them to the Run:

1. Take a screenshot (or several) from the source video.
2. Write a prompt.
3. Send screenshots + the desired reference photos to GPT Image 2 (via kie.ai).
4. Attach the generated images to the Run.

This is a manual, out-of-band step today. We want to integrate it into the New Run flow as an **optional first phase of the Run**, so the operator does it without leaving re-skin.

## Decisions (locked)

- **Flow**: inline option inside the New Run form — a toggle "Preprocess with GPT Image 2" plus, for each reference image, its own image-to-image prompt. Preprocessing runs automatically as the first phase of the Run, then the preprocessed images flow into Seedance/Gemini. No separate manual approval step, no standalone tool.
- **Prompts**: **per-image** — each reference image gets its own GPT Image 2 prompt (max flexibility). Refs without a prompt are passed through as-is.
- **Storage**: a new `Run.preprocessed_reference_urls` field holds the results; originals stay in `Run.reference_image_urls` for audit. When preprocessing is enabled, Seedance/Gemini receive the preprocessed set; otherwise the originals.
- **Aspect ratio / resolution**: always `auto` + `1K` (simplest, least UI). No per-image override.
- **Visibility**: a new `RunStatus.preprocessing` value, so the operator sees "preprocessing 3/5 images" as a distinct phase before `processing`.
- **Resume**: kie.ai `taskId`s are persisted on the Run (`preprocess_task_ids`). On worker restart, already-succeeded tasks are downloaded without rebilling; in-flight ones are polled — mirroring the existing Seedance resume/no-rebill logic in `pipeline_v2.py`.
- **API**: GPT Image 2 uses the **same** kie.ai `createTask` / `recordInfo` protocol as Seedance and Gemini Omni, so the existing `KieClient` retry/backoff/download machinery is reused. Only a thin `create_image_task` method is added.

## Target data model

New `Run` columns (all nullable / default-off so existing runs are unaffected):

| Column | Type | Notes |
|--------|------|-------|
| `preprocess_enabled` | `Boolean` | `NOT NULL DEFAULT FALSE` |
| `preprocess_image_prompts` | `JSON` | list parallel to `reference_image_urls`; `prompts[i]` applies to `reference_image_urls[i]`; empty string = pass-through |
| `preprocessed_reference_urls` | `JSON` | list of local paths to the generated images, parallel to `reference_image_urls` |
| `preprocess_task_ids` | `JSON` | list of kie.ai `taskId`s (one per ref with a non-empty prompt), for resume |

New enum value:

```
ALTER TYPE run_status_enum ADD VALUE IF NOT EXISTS 'preprocessing';
```

State machine addition (`state_machine.py`):

```
created → preprocessing → queued → processing → stitching → delivering → done
                ↘ failed
```

`created → preprocessing` happens in `create_run` when `preprocess_enabled` is true. `preprocessing → queued` happens once all preprocessed images are downloaded. If `preprocess_enabled` is false, the existing `created → queued` path is unchanged.

Storage layout (reuses `storage.py` helpers):

```
data/projects/<pid>/runs/<rid>/preprocessed/ref_<i:04d>.png
```

## KieClient extension (`backend/app/kie_client.py`)

New method, modeled on `create_omni_task`:

```python
def create_image_task(
    self, *,
    prompt: str,
    input_urls: list[str],
    aspect_ratio: str = "auto",
    resolution: str = "1K",
) -> str:
    url = f"{self._jobs_base}/api/v1/jobs/createTask"
    payload = {
        "model": "gpt-image-2-image-to-image",
        "input": {
            "prompt": prompt,
            "input_urls": input_urls,
            "aspect_ratio": aspect_ratio,
            "resolution": resolution,
        },
    }
    data = self._create_task_with_retry(url, payload)
    task_id = _extract_task_id(data)
    if not task_id:
        raise KieTaskError(_no_task_id_msg(data))
    return task_id
```

Polling and download are **reused** as-is: `get_task()`, `_parse_result_url()`, `download_result()`. The result is a single image URL in `resultJson.resultUrls[0]`.

`input_urls` must be public URLs — produced by the existing `resolve_reference_urls()` helper (`pipeline_common.py`), which already handles local paths, Google Drive links, and plain https URLs via `kie.upload_file()`.

## Pipeline (`backend/app/pipeline_v2.py`)

New function `preprocess_reference_images(run_id, *, kie, gdrive) -> dict[str, int]`:

1. Load run; if `not run.preprocess_enabled` → return `{}` (caller skips to submit).
2. Validate `len(preprocess_image_prompts) == len(reference_image_urls)`.
3. `resolve_reference_urls(reference_image_urls, kie, gdrive)` → public `input_urls`.
4. For each ref with a non-empty prompt: `kie.create_image_task(prompt=..., input_urls=[url])`; collect `pending = {task_id: index}`.
5. Persist `run.preprocess_task_ids`; commit.
6. Return `pending` (empty if all refs are pass-through — caller proceeds straight to submit).

The existing `submit_run` (`pipeline_v2.py:939-948`) changes its ref source:

```python
if run.preprocess_enabled and run.preprocessed_reference_urls:
    run_ref_urls = resolve_reference_urls(
        list(run.preprocessed_reference_urls), kie, gdrive=gdrive
    )
else:
    run_ref_urls = resolve_reference_urls(
        list(run.reference_image_urls or []), kie, gdrive=gdrive
    )
```

Per-segment `reference_image_urls_override` resolution stays unchanged.

## Source frames from the video (`backend/app/api_v2.py`)

The operator's manual "take a screenshot from the video" step is replaced with an in-app frame picker. The existing `GET /api/v2/projects/{pid}/frame?t=<seconds>` endpoint (`api_v2.py:544-623`) already extracts a single JPEG via ffmpeg `-ss` seek and caches it under `data/projects/<pid>/frames/frame_<t_tenths>.jpg` with a per-path striped lock (Fix #11).

**Decision (locked 2026-07-29):** frames are **just another ref source**, on equal footing with uploaded files and URL links. No new model field, no separate "frame role". A frame becomes a normal entry in `Run.reference_image_urls` and flows through the existing `resolve_reference_urls` → `kie.upload_file` → `input_urls` path unchanged.

### UI

In the New Run form, the Reference images block gains a third input alongside `reference_files` and `reference_urls`:

- A numeric field "frame timestamp (s)" + an "Add frame" button (HTMX). On click, the server:
  1. Calls the existing frame-extraction logic (or reuses the cached JPEG) for `t` on the project source.
  2. **Copies** the JPEG into `runs/<rid>/references/frame_<t_tenths>.jpg` (stable path that survives project-frame-cache eviction; keeps the original next to its preprocessed counterpart for audit).
  3. Appends that path to the in-progress ref list and returns a thumbnail fragment.
- Multiple timestamps can be added; each appears as a thumbnail with its own GPT Image 2 prompt textarea (same per-image prompt UI as uploaded refs).

### Why copy rather than reference the cache

`data/projects/<pid>/frames/` is a shared project-level cache that may be evicted; `runs/<rid>/references/` is the run's own durable storage. Copying (cheap — single JPEG) guarantees `reference_image_urls` keeps pointing at a live file for the run's lifetime and for any re-process/retry.

### No pipeline change

Because frames become ordinary local-path refs, `preprocess_reference_images`, `resolve_reference_urls`, and `submit_run` need no frame-specific logic. The only new code is the UI button + a small server handler that extracts/copies the frame and returns the thumbnail.

## RQ chain (`backend/app/tasks.py`)

Current: `enqueue_process_run → run_submit_run → run_poll_run → run_finalize_run`.

With preprocessing enabled:

```
enqueue_process_run
  → run_preprocess_refs        # createTask per ref, persist task_ids
  → run_preprocess_poll        # recordInfo loop, self-reschedule like run_poll_run
  → run_submit_run             # existing, but reads preprocessed_reference_urls
  → run_poll_run
  → run_finalize_run
```

- `run_preprocess_refs` mirrors `run_submit_run`: opens its own session, calls `preprocess_reference_images`, persists `preprocess_task_ids`, transitions `preprocessing → queued` only if `pending == {}` (all pass-through), otherwise enqueues `run_preprocess_poll`.
- `run_preprocess_poll` mirrors `run_poll_run`: round-robins `recordInfo` for each task in `preprocess_task_ids`; on success downloads the image to `preprocessed/ref_<i:04d>.png` and appends to `run.preprocessed_reference_urls`; on `fail` falls back to the original ref for that index (log + continue, do not fail the run); on all-done transitions `preprocessing → queued` and enqueues `run_submit_run`. Per-run Redis lock is reused.

When `preprocess_enabled` is false, `enqueue_process_run` goes straight to `run_submit_run` (no change to existing runs).

## Resume / no-rebill

On worker restart mid-preprocessing, `run_preprocess_refs`/`run_preprocess_poll` inspect `preprocess_task_ids`:

- For each `taskId`, call `kie.get_task(taskId)`.
- `state == "success"` → download result (no rebill), append to `preprocessed_reference_urls`.
- `state` in `waiting`/`running` → keep in `pending`, continue polling.
- `state == "fail"` → fall back to the original ref for that index.

This mirrors the existing Seedance resume path in `submit_run` (`pipeline_v2.py:975-1018`).

### Download failures and what Retry can do about them

Added 2026-08-27 after run `ce9b5bfd`, where one truncated download cost a whole
run and left the operator with a Retry button that could never work.

`download_result` streams to a sibling `.part` file, verifies the body against
`Content-Length`, and only then moves it into place — so a truncated response
never lands at the destination path where a later phase would mistake it for a
finished download. Transient failures (the result CDN closing a connection
mid-body, reported as `httpx.RemoteProtocolError`) are retried by
`_transient_retry` before the phase ever sees an error.

When a download still fails, what happens to the task id decides what a retry
can do:

- **Transient** (connection dropped, 5xx): the id is KEPT. The generated image
  is still on kie.ai, so re-entering the phase re-downloads it for free.
- **Gone** (`KieResultGone`, a 4xx — the CDN's temp URL has expired): the id is
  DROPPED. Re-polling it could only fail forever, so the next entry into the
  phase generates a fresh image. That is one GPT Image 2 rebill for that one
  ref, and it is the only exit.

`POST /runs/{rid}/retry` reads this state rather than assuming a failed run
wants Seedance. When the run has a **hole** — a slot `preprocess_hole_index`
says the phase owed an image and did not deliver one — and no `RunSegment` has
a `seedance_task_id` (nothing was billed), the run goes `failed → preprocessing`
and re-enters the phase. Every other failure still retries through `queued`.

Without that routing, `failed → queued` is the only edge out of `failed` and it
submits: `resolve_reference_urls` cannot upload a `None`, so the run re-uploads
its surviving refs, dies, and lands back in `failed` — forever.

`preprocess_hole_index` is deliberately mode-aware and keyed off the PROMPTS,
never off list length or `preprocess_task_ids`:

- `all_as_refs` holds a ONE-element result list on success, and drops its task
  id when a URL is gone — so both a length check and a task-id check would
  misread a healthy or a holed run;
- a trim with no prompt is a pass-through and owes nothing;
- an EMPTY result list is the documented fallback to the originals, not a hole.

## API (`backend/app/api_v2.py`)

`POST /api/v2/projects/{pid}/runs` gains form fields:

| Field | Type | Notes |
|-------|------|-------|
| `preprocess_enabled` | bool form flag | optional, default false |
| `preprocess_image_prompts` | JSON string | optional; list parallel to `reference_files` + `reference_urls` order |

Validation:
- If `preprocess_enabled`: `len(preprocess_image_prompts)` must equal `len(reference_files) + len(reference_urls)`. Empty prompts allowed (pass-through).
- Persisted onto the new `Run` row.

`create_run` transitions:
- `preprocess_enabled` true → `created → preprocessing`, enqueue `run_preprocess_refs`.
- false → `created → queued`, enqueue `run_submit_run` (unchanged).

`RunResponse` / `RunCreateResponse` schemas (`schemas_v2.py`) gain the new fields.

## UI (`backend/app/templates/v2/partials/project_status_content.html`)

The Reference images block in the New Run form gains:

- A checkbox `preprocess_enabled` ("Preprocess with GPT Image 2").
- When checked (HTMX toggle): for each reference image (uploaded file, URL link, **or extracted frame**), render a thumbnail + a textarea `preprocess_image_prompts[i]`.
- A "frame timestamp (s)" field + "Add frame" button (HTMX): extracts a JPEG from the project source via the existing `GET /api/v2/projects/{pid}/frame?t=`, copies it into `runs/<rid>/references/frame_<t_tenths>.jpg`, and returns a thumbnail fragment. Multiple timestamps allowed.
- The `applyRunFormExtras` JS hook (`project_status_content.html:859-875`) serializes `preprocess_image_prompts` as JSON parallel to the ref order (files + urls + frames) and adds `preprocess_enabled`.

Run status view shows a `preprocessing` badge with progress `done/total` (derived from `len(preprocessed_reference_urls)` vs `len(reference_image_urls)`).


## Migration

Alembic migration (manual, applied before deploy per `CLAUDE.md`):

```python
def upgrade():
    op.add_column("runs", sa.Column("preprocess_enabled", sa.Boolean(), nullable=False, server_default=sa.false()))
    op.add_column("runs", sa.Column("preprocess_image_prompts", sa.JSON(), nullable=True))
    op.add_column("runs", sa.Column("preprocessed_reference_urls", sa.JSON(), nullable=True))
    op.add_column("runs", sa.Column("preprocess_task_ids", sa.JSON(), nullable=True))
    op.execute("ALTER TYPE run_status_enum ADD VALUE IF NOT EXISTS 'preprocessing'")

def downgrade():
    op.drop_column("runs", "preprocess_enabled")
    op.drop_column("runs", "preprocess_image_prompts")
    op.drop_column("runs", "preprocessed_reference_urls")
    op.drop_column("runs", "preprocess_task_ids")
    # enum value cannot be removed — downgrade is a no-op for the type
```

## Task breakdown (sub-agents)

- **GP1**: model fields + `RunStatus.preprocessing` + Alembic migration + schemas.
- **GP2**: `KieClient.create_image_task` + unit tests (mock `_create_task_with_retry`).
- **GP3**: `preprocess_reference_images` pipeline function + `submit_run` ref-source switch + tests.
- **GP4**: RQ tasks `run_preprocess_refs` / `run_preprocess_poll` + resume/no-rebill + tests.
- **GP5**: API form fields + validation + `create_run` transition branch + tests.
- **GP6**: UI toggle + per-image prompt fields + status badge + manual e2e.
- **GP7**: "Add frame from video" UI (timestamp field + HTMX button) + server handler that extracts/copies a frame into `runs/<rid>/references/` and returns a thumbnail fragment.

## Per-segment regeneration (added 2026-08-29)

Status: **BUILT**, `feat/segment-image-regen`, unmerged. Prerequisite: Task 0 /
Task 0b (`Run.preprocess_enabled` honest on copied runs) — merged to `main` at
`99c8835`.

### What it does

In the Per-Segment Overrides block, an operator who doesn't like a swap
segment's result can now regenerate *that segment's* GPT Image 2 reference
image before re-running it — two deliberately separate clicks: **Regenerate
image**, look at the result, then **Re-run this segment**.

### Why this is not a slot in the existing per-segment override

`RunSegment.reference_image_urls_override` (used by `PATCH
/runs/{rid}/segments/{rsid}`) is a **full replacement** of the run-level
reference set (`pipeline_v2.submit_run` ~L1955), not a per-image slot. Writing
one generated still into it would hand that segment a DIFFERENT reference
recipe from its siblings (one ref instead of N) and would be destroyed by the
existing write path the moment the operator saved an empty override (`_apply_
segment_override` nulls the override on zero refs). So the generated image
gets **its own column** — `RunSegment.generated_reference_urls` — and
`submit_run` learns a three-way priority instead of two:

1. `rs.generated_reference_urls` — a per-segment GPT Image 2 rescue.
2. `rs.reference_image_urls_override` — the hand override (unchanged).
3. the run-level set (preprocessed if `preprocess_enabled` and a result
   exists, else the originals — unchanged).

**The generated list is exactly two entries: `[character photo, the
downloaded GPT Image 2 output]`** — the honest analogue of one `per_image`
task's result. If the run had three trims, this segment sends 2 references
while its siblings send 4; the UI says so. The generated still is never
spliced into the run-level set — it was generated from a different input
(one specific extracted frame) and corresponds to no run-level trim.

The character photo in that pair is decoupled from
`reference_image_urls_override` on purpose: it defaults to
`run.reference_image_urls[0]` but can be overridden per-attempt (an upload or
a URL), and reading it from the hand override would let one control silently
destroy the other.

### Data model

`run_segment_image_jobs` — the append-only attempt history (one row per
"Regenerate image" click, never reused as a slot):

| Column | Notes |
|---|---|
| `id` | uuid PK |
| `run_id` | denormalized from `run_segment_id`'s parent Run (cheap cost rollup + provenance) |
| `run_segment_id` | FK, **CASCADE** — the attempt dies with its segment |
| `prompt` | the GPT Image 2 prompt used for this attempt |
| `input_paths` | JSON — `[character photo, extracted frame]`; the API writes index 0 at create time, the worker appends index 1 once it extracts the frame |
| `frame_sec` | the source timestamp the frame was cut at (label + the value the worker clamps/quantises) |
| `kie_task_id` | persisted before polling, so a resumed job never rebills |
| `status` | `SegmentImageStatus`: `queued → generating → done \| failed` (terminal — a retry is a NEW row, not a state reset) |
| `output_path` | the downloaded image |
| `cost_credits` | this attempt's kie.ai credits (survives a Re-run, unlike `RunSegment.cost_credits`, which the next video task overwrites) |
| `error_message` | bounded, same convention as `RunTailEdit` |

Two new nullable columns on `run_segments`: `generated_reference_urls` (JSON)
and `generated_image_job_id` (the currently-applied attempt, for the UI
badge) — composed when a job **succeeds** (auto-apply) or when the operator
explicitly re-Applies an older `done` attempt.

Migration `8812409e54a4` (additive: one table + two nullable columns, no
backfill).

Treated as a **derived INPUT job**, not a derived creative: no Drive upload,
no sheet row, no public token — its only downstream consumer is a later
Seedance/Gemini submission.

### Worker (`app/pipeline_segment_image.py`)

Two-phase like the run-level preprocess chain, so the worker never sleeps:

- `submit_segment_image_job(job_id)` — resume-aware exactly like
  `preprocess_reference_images`: if `job.kie_task_id` is already set (a prior
  attempt got this far before its worker died, or a duplicate invocation),
  the existing task is resumed rather than recreated. Otherwise it extracts
  the frame (`pipeline_common.extract_project_frame`, shared with the
  frame-preview/stage-frame API endpoints) and **copies it into the job's own
  directory** (`storage.segment_image_job_dir`) so a retry never depends on
  the shared project frame cache surviving.
- `poll_segment_image_job_once(job_id)` — one round, no sleep; downloads the
  result on success and **auto-applies** it to the segment.

Both are enqueued on `QUEUE_PROCESS` (`tasks.enqueue_process_segment_image` /
`tasks._enqueue_segment_image_poll`), never `QUEUE_OVERLAY` — this blocks on a
kie.ai poll chain exactly like run-level preprocessing, and `overlay` is the
fast interactive queue workers must never have starved by a blocking wait.

**`args[0]` is the job id, never the run id.** `recovery._live_run_ids`
treats every queued RQ job's first positional arg as an entity id across all
queues; passing the run id would make a `done` run generating a still look
"live" and could make the reconcile tick skip a genuinely orphaned job.

### Why the run's Redis lock is never taken here

Serialisation is **per segment, in the API** (a 409 on a second
`regenerate-image` while one is already `queued`/`generating` for that
segment) — not the run-wide `tasks.run_lock_key`. Two segments of the same
run may regenerate in parallel; only a second attempt on the SAME segment is
refused. Taking the run lock instead would be worse than the race it
prevents: `run_submit_run` finds a held lock and **silently skips**
(`tasks.py` ~370), parking the run in `queued` with no worker — see
`wiki/lessons/production-gotchas.md`.

The same 409 gate, from the other direction: `patch_run_segment` and
`rerun_segment` both refuse while this segment has an active image job.
`rerun_segment` resets the segment and re-enqueues submit, which reads refs
**once** — an image job finishing afterward would write a path that attempt
never sees, and the operator pays for video generated from the old still.

### Recovery

`RunSegmentImageJob` is in `recovery._DERIVED_CREATIVE_SPECS` with the same
300s grace window as captions/tail edits/overlays/merges. A worker that dies
after `create_image_task` has already billed kie.ai; without reaping, the row
is stuck `generating` forever and — because of the 409 gate above — the
operator can never re-run the segment. Reaping just marks it `failed`
(derived creatives are never auto-resumed, same reasoning as everywhere else
in `recovery.py`); a fresh "Regenerate image" click starts a clean new
attempt.

### Copy policy

`generated_reference_urls` / `generated_image_job_id` are **not** cloned by
`_build_copied_run`, and no `RunSegmentImageJob` rows are cloned. A generated
still is a rescue of one specific take cut from the SOURCE run's frozen
snapshot; carrying it onto a copy — especially one with a new photo — would
pin the OLD face into the reference priority chain ahead of the run-level
set, the same class of bug the copy-preprocess-refs fix (Task 0) closed at
the run level. `prompt_override` keeps cloning as it always has.

### UI

A third block in the segment panel, under "Reference images override",
visible only when `Run.preprocess_enabled` and the segment is
`completed`/`failed`. Self-polls via `GET
/runs/{rid}/segment-image-jobs-fragment` while any of the run's image jobs is
active — an `hx-swap-oob` payload per segment (mirroring
`partials/translate_gate.html`), since the content lives scattered across
several per-segment panels rather than one consolidated list.

Fixed in the same change: the hand-override reference box
(`reference_image_urls_override`) used to prefill its URL text field with
whatever the list held, including local WORKER-DISK paths from an uploaded
file — so a plain "Save override" round-tripped a path as visible, editable
text. It now shows only `http(s)` entries as text; a local entry renders as a
thumbnail (`GET /runs/{rid}/segments/{rsid}/override-reference`, new) plus a
hidden input, so a plain Save still carries it forward without ever exposing
or letting the operator hand-edit the raw path.

## Open questions (deferred to implementation)

- Credit cost tracking: should `preprocess_task_ids` results carry `cost_credits` like `RunSegment`? GPT Image 2 returns `costTime` but the docs don't show a credits field — verify against a real response.
- Rate limiting: GPT Image 2 may have different rate limits than Seedance; confirm before enabling concurrent per-ref submit.
- `input_urls` max count per task: docs say "Multiple Files: Yes" but don't state a cap. We submit one image per task (per-image prompt decision), so this is moot — but confirm a single-image task is supported.

---

## Addendum — GPT Image 2.5 (2026-09-12)

kie.ai shipped two GPT Image 2.5 image-to-image variants on 2026-09-08
(`gpt-image-2-5-flare-image-to-image`, `gpt-image-2-5-sunburst-image-to-image`).
This preprocessing phase — and the per-segment regeneration that shares
`KieClient.create_image_task` — now runs **flare** by default.

### Why the switch is nearly free

The three model ids take a **byte-identical request** and return a
byte-identical response: same five `input` fields (`prompt`, `input_urls`,
`aspect_ratio`, `resolution`, `background`), no new params, none newly
required, same `{code,msg,data:{taskId}}` envelope, same `recordInfo` fields.
`input_urls` is **not** renamed. So nothing in `pipeline_v2.py`,
`pipeline_segment_image.py`, `tasks.py` or the poll/download path varies with
the choice, and the model id moved out of the payload literal into
`settings.KIE_IMAGE_MODEL` (`.env`, read per call). A swap or a rollback needs
no rebuild and no migration — edit `.env`, then
`docker compose up -d --scale worker=2` (a **recreate**; `docker compose
restart` does not re-read `env_file` — verified 2026-09-12), then
`docker compose restart nginx`.

Two schema deltas that do not affect us, both read off the docs' OpenAPI:

- `aspect_ratio` shrank from 16 values to 13 — `5:4 4:5 2:1 1:2 3:1 1:3 9:21`
  were dropped, `27:16 16:27 9:8 8:9` added. Every caller sends `auto`, which
  all three accept and all three still default to.
- `background` lost its "1K only" restriction. We never send it.

### Measured, not documented (2026-09-12, 5 live tasks on real prod refs)

kie.ai publishes no rate card for these models and the `creditsConsumed: 3` in
every doc page's callback example is boilerplate — the same number appears on
the gpt-image-2 page, whose real cost is 6. Measured instead:

| model | resolution | credits | costTime | output on a 9:16 input |
|---|---|---|---|---|
| `gpt-image-2` | 1K | 6.0 | ~100 s (prod, n=2) | 941×1672 |
| 2.5 flare | 1K | 6.0 | 88 s, 80 s | 941×1672 |
| 2.5 sunburst | 1K | 6.0 | 73 s, 71 s | 941×1672 |
| 2.5 flare | 2K | 10.0 | 39 s | 1536×2736 |

So 2.5 at 1K costs **exactly what gpt-image-2 costs** and is no slower. This
also answers two of the "Open questions" above: `recordInfo` **does** carry
`creditsConsumed` (the run-level preprocess path still does not persist it —
only `RunSegmentImageJob.cost_credits` does), and `input_urls` caps at
**16 images**, 30 MB each.

### `aspect_ratio: auto` + 2K is legal on 2.5, and was not on gpt-image-2

gpt-image-2's docs state that `auto` "will only be converted to 1K images;
otherwise, the task will fail to create". 2.5 dropped that sentence, and a live
task confirmed it: `auto` + `2K` created and returned 1536×2736.

This matters beyond the pixel count. At 1K a 9:16 frame comes back **941 px
wide — below the 1080 px source frame it was cut from**, so today's
preprocessing quietly hands the video model a downscaled reference while
Seedance 2.5 has done 1080p since 2026-08-18. `settings.KIE_IMAGE_RESOLUTION`
exposes the tier; it ships at `1K` (unchanged behaviour) because 2K is +67%
credits per image on every run, which is an operator's cost decision, not a
default. Do not raise it while rolled back to gpt-image-2 — there it is a
guaranteed create-failure.

### Flare vs sunburst

kie.ai's two doc pages are otherwise word-for-word identical; the distinction
comes from OpenAI: flare is "our fastest model for high-quality, everyday
image generation" and the default variant, sunburst "our most capable model
for image generation and editing", positioned for **precision across
multi-turn edits** at the cost of speed. No public benchmark separates them.

Flare is the default here because sunburst's documented edge is the one axis
this pipeline cannot consume: we fire one task per frame and never chain edits
on the same image. Two caveats worth recording: in both measured pairs
sunburst was *faster* than flare, not slower, contradicting the vendor's own
positioning at n=2; and at equal price the choice is one `.env` line, so it is
worth re-testing on real footage rather than treating it as settled.

Identity preservation from a reference photo is a **2.5-family** improvement,
not a flare-vs-sunburst one — both inherit it. Content moderation is
unchanged: nothing suggests 2.5 relaxes the likeness refusals that block
recognizable real people, so this is not a fix for that.
