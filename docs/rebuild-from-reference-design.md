# Rebuild from reference — project type design

Status: **revision 10** (2026-11-02, full-video pilot folded in; Terra design gate PASS, round 10);
decisions Q1–Q9 taken 2026-11-03 (§15). Wave 1 (data model, registry, helpers, paid-call
primitives) is implemented on `feat/rebuild-core`, wave 2a (storyboard, keyframes, project
page) on `feat/rebuild-2a-keyframes`, wave 2b (the images-only run path) on
`feat/rebuild-2b-run`, and the three are combined on `feat/rebuild-integration` — see
`docs/rebuild-implementation-notes.md`,
whose §3 lists where the code deviates from this text (the affected sentences below are
annotated "impl:"). Nothing here authorizes migration, deployment or paid generation.

Generalises Linear AICP-258 (images-only rebuild of ONE talking-head chunk, Backlog). Evidence:
the manual rebuilds of AICP-243 (run `3b1ea5c7-es`) and AICP-256 (run `0abca08e`), the
Kling 4.0 evaluation (`wiki/models/kling-4-0.md`), and the 2026-11-02 pilot on project
`81d0afc0` (§2.2, §13). Baseline: main `3ec0d16`, alembic head `s3c8e1a6d204` (impl: the integration rebased the revision
onto main `730216b`'s head `t4d9a2c6e815`, AICP-348 `run_drive_relinks`).

## 1. Goal and non-goals

**Goal.** An operator uploads a creative (any number of shots, a voiceover or a song as the
soundtrack, burned captions, an end card) and gets back a delivered creative that tells the
same story with the same characters, shot by shot, **generated from scratch** from still
keyframes instead of edited from the source video. The original soundtrack is kept, so the
picture must change at the same moments the source picture changed. Burned captions are
gone by construction (the keyframes have none), so the subtitle-removal pass is no longer
needed. A later phase swaps the characters at the keyframe stage, which replaces the
face-swap pass too (§11).

Why: on project `81d0afc0` (94.9 s 3D cartoon, 54 shots) the current route is a Gemini
Omni subtitle pass (1848 cr) + VOID fills + a later Seedance face swap (~6500 cr), about
8300 cr ≈ 42 USD, and v2v cannot fix some defects at all (the source's hands survive a
gender swap, AICP-243). The rebuild with Kling 4.0 at 720p measured ≈ 3,100 cr ≈ 15.5 USD
for the same video in the full manual pilot (§13).

**Non-goals (phase 1).**

- Changing the characters (phase 2, §11). Phase 1 rebuilds the SAME characters.
- Model-written speech (the AICP-258 talking-head case: script + voice reference +
  `generate_audio`). Phase 3. Phase 1 always keeps the source soundtrack.
- Lip sync. Phase 1 targets voiceover/song soundtracks; a shot where a character speaks on
  camera gets generic mouth movement (the storyboard flags it, §5.2, and the UI warns).
- Automation, sheet batches, Copy run with a new photo, autopilot ("by Assistant") flows.
- Kling 4.0 for any v2v work (it lost both v2v evaluations: rewrites burned captions,
  paraphrases speech; `reskin-kling-4-eval`).
- Any change to how v2v runs are built, submitted, priced or stitched. Every new branch is
  keyed on `runs.generation_mode = 'images_only'` and v2v payloads stay byte-identical
  (tests prove it, §14).

## 2. Facts this design rests on

### 2.1 Code (main `3ec0d16`)

1. Project types live in `app/project_types.py` (`ProjectTypeSpec`, `segmentation`
   strategies `detect_faces | scene_detect | hook_split | blank`), mirrored by the Postgres
   enum `project_type_enum`. `supports_runs=False` exists (reframe).
2. `analyze_project` dispatches on `segmentation`. `scene_mod.detect_scenes` chunks
   `min_scene_len_sec`; analysis passes `SCENE_MIN_LEN_SECONDS` = 4 s, which MERGES short
   shots (the 81d0afc0 source has 54 cuts at the detector default, mean shot 1.75 s).
   `whole_video_ranges` (AICP-340) is the pure greedy merge of candidate ranges into
   segments ≤ target with a short-tail rebalance; `scene.write_scene_cache` records cuts.
3. A run's swap segments are `SegmentDef(action='swap')`; `RunSegment` rows are created
   lazily in `_build_submit_state`; `segment_snapshot` freezes the defs at submit.
   `RunSegment` already has `prompt_override`, `reference_image_urls_override` and
   `generated_reference_urls` (per-segment refs, priority generated → override → run).
4. `_submit_swap_segment_isolated` always cuts + uploads a source clip, pads it to the
   model's `min_clip_sec`, and calls `_create_swap_task(**recipe)`; the recipe is kept for
   in-poll resubmits. The task id is persisted **after** `create_task` returns: a worker
   death in between can double-bill (known gap, recorded by AICP-293; range attempts guard
   it with `submit_billed_at`, the Fal layer with `submit_started → submit_unknown`).
5. `KieClient.create_task` / `build_seedance_payload` always send `reference_video_urls`;
   `ai_models.validate_duration` forces `duration=-1` + `aspect_ratio=adaptive` for 2.5
   (`follows_input_geometry`). There is no images-only path and no Kling model in
   `ai_models.AI_MODELS` (`run_model_enum` has no Kling label).
6. `media.stitch` concatenates clips **whole** (normalised to the project's
   width×height@fps). With the default plan (`stitch_plan` NULL) and `audio_mode='original'`
   it muxes the source's continuous audio with `-shortest`; with a custom plan it muxes the
   source audio of each swap segment's OWN time range onto that clip
   (`media.mux_original_audio(clip, source, sd.start_sec, …)`, `pipeline_v2.py` ~3164) and
   concatenates per-clip audio. Nothing trims a generated clip to its segment length. v2v
   ASSUMES the model returns roughly the input's length; `_download_and_complete_segment`
   accepts a short result with a warning (`_result_too_short`) and stitches it whole, so
   the assumption has known violations (Seedance 2.5's 14.72 s truncation).
7. Every kie `createTask` goes through `_create_task_with_retry` (tenacity, 4 attempts on
   transport errors and 5xx): a request kie accepted whose response was lost is SENT AGAIN
   inside the client, before any caller-side marker could help.
8. Run recovery (`recovery._recover_orphaned_run`) claims a stale per-run Redis lock,
   resets an active run to `queued`, re-polls `generating` segments that hold a
   `seedance_task_id` (`_repoll_generating_segments`) and re-enqueues `process_run`, whose
   `_build_submit_state` resumes segments by `seedance_task_id` and RESUBMITS when the
   resume check or the download fails. In-poll resubmits (`_resubmit_segment_task`) create
   a new task while the old id is still stored.
9. Spend (`app/spend.py`): `reserve` at admission, `record_draw` / `settle_draw` per paid
   call. Video rates are credits per second of the **submitted clip** (in+out);
   `IMAGE_CREDITS` 10 cr at 2K; `llm_credits()` for Gemini.
10. Durable side jobs (`localisation_jobs.py`, AICP-365): DB row is authoritative, RQ is the
   doorbell; `dispatch_gen` in the RQ job id, CAS claim with `claim_token`, a reaper
   re-enqueues `queued` rows whose job is gone and marks `running` rows `interrupted`
   (never resubmitted: the call may have been billed).
11. Gemini on kie: OpenAI-compatible chat route, video by URL in the `image_url` envelope,
   JSON fenced and not schema-enforced (tolerant parse), errors as HTTP 200 with a `code`,
   upload ceiling ~100 MB → `media.make_llm_proxy`. Fallback chain
   `LOCALISATION_TRANSCRIBE_MODEL` + `..._FALLBACK_MODELS`.

### 2.2 Measured (2026-11-02 pilot, project 81d0afc0, 0–15.2 s, 8 shots)

| | Kling 4.0 `reference-to-video` 720p | Seedance 2.5 images-only 720p |
|---|---|---|
| Request | 9 image refs (8 keyframes + character sheet), `duration` 16, 9:16, no video | same refs, `reference_video_urls: []`, `duration` 16, 9:16, `generate_audio: false` |
| Credits | 456 (= 28.5 cr/s, list price) | 1008 (= 63 cr/s) |
| Wall clock | 526 s | 396 s |
| Output | 720×1280, 24 fps, 16.04 s, has an audio track | 720×1280, 24 fps, 16.04 s |
| Shots | all 8, in order | all 8, in order |
| Cuts vs requested 1.7/3.1/4.1/5.4/7.3/9.4/13.4 | 1.79/3.17/4.21/5.54/7.42/9.58/13.5 (**+0.07…+0.18 s**) | 2.62/3.62/5.29/7.04/8.75/11.42/14.21 (**up to +0.9 s**) |
| Burned text | none | none |
| Characters across shots | consistent | consistent |

- Storyboard: `gemini-3.1-pro` on a 0.7 MB 480p/15 fps proxy answered correctly
  (`voiceover`, 3 characters, 8 shots). A 14.5 MB CRF-16 upload named `src_0_15.2.mp4`
  was refused (`400 Failed to get the file information`) → always send the proxy with a
  plain name. Its shot descriptions were terse (every camera "Static").
- Keyframes: GPT Image 2.5 flare 2K 9:16, prompt "recreate this exact frame as a clean
  render, remove ALL captions, keep everything else" → 8/8 clean on the first try, plus a
  two-character sheet from 5 frames; 10 cr each.
- The prompt that worked: style paragraph, one line per character, then one line per shot
  `Shot k (a–b s), starts from @image_k: <action>`, then "no on-screen text, no captions".
  Kling addresses images as `@image_N`; Seedance got `[Image N]`.

Pilot scripts and outputs: `/root/rebuild-pilot/` (host-side, outside `data/`).

## 3. Shape

**Terminology.** A *shot* is one camera shot of the source (between two cuts). A *chunk*
is a group of consecutive shots generated by ONE video task; it is an ordinary swap
`SegmentDef`. ("Take" is avoided on purpose: `run_segment_takes` already means the kept
previous results of a segment rerun, AICP-342.)

```
VideoProject (project_type = rebuild)
 ├─ SegmentDef …            CHUNKS: swap = generated, keep = original footage (end card)
 ├─ RebuildShot …           detected shots on the source timeline + storyboard fields
 ├─ ProjectStoryboard …     durable Gemini jobs; the latest succeeded one seeds the shots
 ├─ RebuildKeyframe …       durable GPT Image jobs: one row per attempt, ONE selected per
 │                          shot, plus the character sheet (shot_id NULL)
 └─ Run (generation_mode = images_only, model kling-4-0 | seedance-2-5)
      ├─ rebuild_snapshot   frozen at launch: per chunk the shot list, the prompt and
      │                     RUN-OWNED copies of the keyframes
      └─ RunSegment …       one per swap chunk; raw + fitted result
```

**Why a project type plus ordinary Runs (not a reframe-style job table).** Everything after
generation is run-shaped and already exists for runs only: stitch with keep segments and
library tails, Drive delivery, Job ID, Final Creatives, captions, tail edits, overlays, the
run page, retries and segment reruns. A chunk is exactly a swap segment whose input is
images instead of a clip. The parts that are NOT run-shaped (storyboard, shot list,
keyframes) depend only on the source and are reviewed before any video credit is spent, so
they live on the project, like the localisation transcript does.

**Why keyframes are project-level in phase 1.** They are a property of the source (the same
frame redrawn without captions). Phase 2's persona keyframes depend on the run's character
and become run-level (§11); the table already carries an owner for that.

### Modules

| File | Content |
|---|---|
| `app/project_types.py` | `REBUILD = "rebuild"`, segmentation `SEG_SHOT_CHUNKS`, defaults (§9) |
| `app/rebuild.py` | pure helpers: chunk building, prompt rendering, duration request, fit/trim maths, cut-timing QA, estimates |
| `app/rebuild_models.py` | `RebuildShot`, `ProjectStoryboard`, `RebuildKeyframe` (imported at the end of `models.py`) |
| `app/rebuild_jobs.py` | storyboard + keyframe admission, RQ bodies, reaper (the AICP-365 dispatch protocol) |
| `app/api_rebuild.py` | storyboard / shots / keyframes endpoints (§8) |
| `app/ai_models.py` | `kling-4-0` spec, `supports_v2v` / `supports_images_only` flags, images-only duration rule |
| `app/kie_client.py` | `build_kling_payload`, `build_seedance_images_payload`, `build_images_only_payload`, `create_task_once`, `chat_completion_once` |
| `app/rebuild_paid_call.py` | the §7.2 state machine, sender proof, step jobs, reaper, release CLI (impl: wave 1) |
| `app/rebuild_analysis.py` | publishing shots + chunk SegmentDefs; re-chunking (impl: wave 1) |
| `app/pipeline_v2.py` | images-only branch in analysis, submit, download (fit), QA hook |
| `app/spend.py` | `IMAGES_ONLY_CREDITS_PER_SEC`, estimates for storyboard / keyframes / runs |
| `app/recovery.py` | reaper for the two job tables; images-only submit-ambiguity rule (§7.3) |
| `templates/v2/partials/project_rebuild_*.html` | project page body for this type (§10) |

## 4. Data model

One Alembic revision on `s3c8e1a6d204` (re-check the head when the branch is cut; impl: on
`t4d9a2c6e815` since the integration):

- `ALTER TYPE project_type_enum ADD VALUE IF NOT EXISTS 'rebuild'`
- `ALTER TYPE run_model_enum ADD VALUE IF NOT EXISTS 'kling-4-0'`
- The revision **never uses either label** (no default, no backfill, no CHECK naming
  them): Postgres refuses a label added in the same transaction. Enum labels are permanent;
  the names above are final.
- Downgrade is **refused** while any `video_projects.project_type = 'rebuild'`, any
  `runs.generation_mode = 'images_only'` or any row in the new tables exists (it would
  destroy the only record of how those runs were made and of billed provider tasks); on
  an empty feature it drops the new tables/columns and leaves the labels.
- Rehearse `upgrade head` on a copy of the production schema; rehearse the rollout
  sequence of §7.6 with representative rows (a rebuild project, an images-only run with an
  `unknown` attempt) on a disposable database.

**`video_projects`**: no new column. **`runs` (new columns)**

| Column | Notes |
|---|---|
| `generation_mode` | `varchar(16) NOT NULL DEFAULT 'v2v'`, CHECK in (`v2v`, `images_only`). The server default backfills every existing run as `v2v`. Set at creation, never changed. |
| `rebuild_snapshot` | JSON, NULL for v2v. Frozen at launch (§6.1). |

**`run_segments`**: no new column. An images-only segment never writes
`seedance_task_id` / `kie_upload_url`; its provider work lives in `rebuild_chunk_attempts`
and `local_result_path` points at the selected attempt's fitted file (§6.4).

**`rebuild_shots`**

| Column | Notes |
|---|---|
| `id`, `project_id` (FK, `ON DELETE CASCADE`), `index` | unique (`project_id`, `index`) |
| `start_sec`, `end_sec` | source timeline; contiguous, cover `[0, duration]` |
| `frame_sec` | where the keyframe's source frame is taken (default the shot midpoint; operator-editable within the shot) |
| `location`, `framing`, `camera`, `characters` (JSON list of ids), `action`, `speaking_on_camera` (char id or NULL), `is_brand_card` (bool) | seeded from the storyboard, operator-editable |
| `folded_into` | NULL for an anchor, else the anchor of its effective-shot group (§5.1.3) |
| `force_anchor` | bool, operator "unfold" (§5.1.3) |
| `sheet_key` | which character sheet the shot belongs to (outfit/setting group, from the storyboard; editable) |
| `edited_at` | set when the operator edits any storyboard field; a later storyboard re-run does not overwrite an edited shot unless asked (§5.2) |
| `generation` | int, the analysis generation that created the row (§7.4) |

**`project_storyboards`** — the AICP-365 `_DispatchColumns` mixin (`status`, `dispatch_gen`,
`claim_token`, `attempts`, `rq_job_id`, `not_live_since`, `error`, timestamps) plus
`project_id`, `model` (the model that answered), `shots_generation` (the shot list it
described), `result` (JSON: `style`, `characters`, `audio`, `on_screen_text`, `shots`).
Partial unique index: one live (`queued`/`running`) row per project.

**Paid-call columns** (a mixin shared by the next two tables; the protocol is §7.2):
`status` (varchar + CHECK: `queued`, `claimed`, `sending`, `submitted`, `fetching`,
`succeeded`, `failed`, `unknown`), `dispatch_gen`, `claim_token`, `rq_job_id`,
`not_live_since`, `send_deadline`, `sender_host`, `sender_boot`, `sender_job`, `task_id`,
`request` (JSON: the exact createTask body
sent, URLs included), `error`, `estimate_credits`, `cost_credits`, `created_at`,
`updated_at`, `finished_at`.

**`rebuild_keyframes`** — one row per GPT Image attempt.

| Column | Notes |
|---|---|
| `id`, `project_id` (FK CASCADE), `batch_id` (indexed; groups one admission), `shot_id` (FK `rebuild_shots.id` CASCADE, NULL = character sheet), `owner_run_id` (NULL in phase 1; phase-2 persona keyframes) | |
| paid-call columns | §7.2 |
| `sheet_key` | sheet rows only: the outfit/setting group (§5.3) |
| `prompt`, `input_paths` (JSON: the exact files fed to GPT Image), `resolution` (`2K`) | |
| `image_path`, `image_sha256`, `width`, `height` | result, under `projects/<pid>/rebuild/keyframes/<id>.png` |
| `selected` | bool. Two partial unique indexes (Postgres treats NULLs as distinct, so one index over a nullable column would not hold): `uq_rebuild_kf_shot_selected (shot_id) WHERE selected AND owner_run_id IS NULL AND shot_id IS NOT NULL` and `uq_rebuild_kf_sheet_selected (project_id, sheet_key) WHERE selected AND owner_run_id IS NULL AND shot_id IS NULL` (one selected sheet per outfit/setting group; `sheet_key` NOT NULL on sheet rows, NULL on shot rows). Phase 2 adds the run-owned pair. Selecting is `UPDATE … SET selected=false` for the slot then `SET selected=true` for the row, in one transaction under the project lock. |

**`rebuild_chunk_attempts`** — one row per paid video task for a chunk.

| Column | Notes |
|---|---|
| `id`, `run_id` (FK `runs.id` CASCADE), `run_segment_id` (FK `run_segments.id` CASCADE), `attempt_no` | unique (`run_segment_id`, `attempt_no`) |
| paid-call columns | §7.2 |
| `model`, `resolution`, `request_sec`, `chunk_frames` | frozen from the snapshot at admission |
| `raw_path`, `fitted_path` | `runs/<rid>/rebuild/attempts/<attempt id>/raw.mp4` and `fitted.mp4`; immutable once written |
| `qa` | JSON, §6.5 |
| `selected` | bool; partial unique index `(run_segment_id) WHERE selected` — the attempt whose `fitted_path` is the segment's `local_result_path` |

**`rebuild_llm_calls`** — one row per Gemini call (storyboard global / shot calls, the
per-attempt character check): the paid-call columns (§7.2; `task_id` unused, the response
body in `response`), `project_id` (FK CASCADE), `storyboard_id` / `attempt_id` (FK
CASCADE, one of them set), `kind`, `shot_id`, `model`, `global_call_id` (shot calls: the
global call they were made under). Live rows count for deletion / re-analysis exactly
like the other two tables.

Partial unique index `uq_rebuild_attempt_live (run_segment_id) WHERE status IN ('queued',
'claimed','sending','submitted','fetching')`: at most one live paid attempt per chunk.

Files:

```
projects/<pid>/rebuild/shots.json                       detector output (cache, rebuildable)
projects/<pid>/rebuild/frames/<shot id>.jpg             extracted source frame per shot
projects/<pid>/rebuild/keyframes/<kf id>.png            GPT Image results (all attempts)
projects/<pid>/runs/<rid>/rebuild/<chunk idx>/<n>.png   run-owned keyframe copies (§6.1)
projects/<pid>/runs/<rid>/rebuild/attempts/<aid>/raw.mp4, fitted.mp4, sheet.jpg
```

`disk_cleanup.sweep` deletes only the paths it names (`tail_edits/<id>/body.mp4`,
`runs/<id>/clips/`, expired `segment_takes/`); none of the paths above is under them. A test
pins that, because `raw.mp4` is the only copy of a paid output.

## 5. Project phase: analyze → storyboard → keyframes

### 5.1 Analysis (`SEG_SHOT_CHUNKS`, free)

1. `normalize_source` and probe as for every type.
2. Shot detection: `scene_mod.detect_scenes(video, fps, min_scene_len_sec=REBUILD_SHOT_MIN_SEC)`
   with `REBUILD_SHOT_MIN_SEC` = 0.4 (the pilot's 54-cut list came from the detector
   default). Write `rebuild/shots.json` and replace `rebuild_shots` (new `generation`).
3. Short shots → **effective shots**. `rebuild.effective_shots(shots, fold_sec=0.7)` is a
   single deterministic left-to-right pass that partitions the detected shots into
   contiguous groups, each containing exactly one *anchor*:
   - a shot ≥ `REBUILD_SHOT_FOLD_SEC` (0.7 s) is an anchor and opens a group;
   - a short shot joins the group of the **next** anchor (pending shorts are buffered
     until it arrives);
   - shorts left after the last anchor join the **last** group if the result stays ≤ the
     cap; otherwise they form a group of their own (anchor = their longest shot),
     itself closed and restarted at the cap like any buffer;
   - if the whole video has no anchor, its shots form one group whose anchor is the
     longest shot;
   - **length cap**: no group may exceed `REBUILD_CHUNK_MAX_SEC`. When adding a buffered
     short would push the pending buffer past the cap, the buffer is closed as a group of
     its own (anchor = its longest shot) and a new buffer starts; when an anchor plus its
     buffered shorts would exceed the cap, the shorts become their own group (same rule)
     and the anchor stands alone; an anchor alone longer than the cap is split into equal
     pieces ≤ the cap, each piece its own group sharing the anchor's keyframe, with the
     prompt line of each piece covering only its own interval (§6.2).
   The groups cover `[0, duration]` exactly, every group is ≤ the cap, no group is empty, and no folding can form a
   cycle (folding only ever points at the group's anchor). A folded shot keeps its row with
   `folded_into` = its anchor's id, gets no keyframe and no prompt line, and its cut leaves
   the QA cut list; the anchor's prompt line spans the whole group. The pilot's only cut
   errors above 0.3 s were 0.67 s shots the model merged anyway (§13). "Unfold" on a shot
   marks it `force_anchor` and re-runs the same pass (free; refused while a run is active,
   like any segmentation change). Tests: consecutive shorts, shorts at both ends, an
   all-short 20 s video, a 16 s run of shorts before an anchor, an overlong anchor, a
   14.5 s anchor followed by one and by several trailing shorts, unfold then re-chunk.
4. Chunks = `rebuild.build_chunks(groups, target_sec=REBUILD_CHUNK_MAX_SEC, max_groups=REBUILD_CHUNK_MAX_SHOTS)`
   over the effective-shot groups (chunking never splits a group): greedy left-to-right
   merge of whole groups while the chunk stays ≤ 14.72 s **and** ≤ 8 groups (every group
   is already ≤ the cap, step 3). There is no short-tail rebalance: a chunk shorter than
   the model's minimum output (Seedance 2.5 4 s, Kling 3 s) is simply requested at that
   minimum and trimmed back by the fit (§6.4). Constants:
   - 14.72 s: Seedance 2.5 intermittently truncates to exactly 14.72 s, so a chunk ≤ 14.72 s
     requested as `ceil` = 15 s still covers the chunk after a truncation. Kling accepts up
     to 30 s; the full pilot used chunks up to 15.2 s / 9 shots (Q5).
   - 8 groups: Seedance accepts 9 image refs (8 keyframes + sheet); Kling accepts 10.
5. Every chunk becomes a `SegmentDef(action='swap')`. Shots flagged `is_brand_card` by a
   LATER storyboard are not re-segmented automatically; the storyboard card offers
   "Keep brand-card shots as original footage" which re-cuts the chunks so those shots form
   `keep` segments (a free re-segmentation through the same helper, refused while a run is
   active, as every segment edit is).
6. The Segment Editor stays available: an operator may split/merge chunks or turn a chunk
   into `keep`. Chunk boundaries need not fall on shot cuts (§6.2 handles a shot cut by a
   boundary), but the editor **warns** when one does not: the pilot's seams were invisible
   because every boundary sat on a hard cut (§13), and a boundary inside a shot has no
   continuity reference in phase 1.
7. Re-analysis (new source or new detector setting) deletes shots, storyboards and
   keyframes of the old generation (CASCADE through `rebuild_shots` for keyframes; the
   storyboard rows are kept for history but no longer match `shots_generation`). Refused
   (409) while a storyboard or keyframe row is live (non-terminal) or any run is active —
   the existing active-run guard plus the two new live-row checks, under the project row
   lock, in the same transaction that sets the project to `analyzing`.
8. **Every rebuild admission** (storyboard, keyframe batch, keyframe selection, shot edit,
   brand-card re-cut, run launch) takes the project row lock and requires
   `project.status == ready`; re-analysis flips the status to `analyzing` under the same
   lock. So no admission can be made against a shot generation an analysis is about to
   replace, and no job of the old generation is live when analysis publishes (it is
   refused above). The analysis job publishes shots + SegmentDefs in one transaction under
   the project lock and only if the project is still `analyzing` (a second analysis
   admission is already refused today while one is running).

### 5.2 Storyboard (Gemini, LLM credits)

`POST /api/v2/projects/{pid}/rebuild/storyboard` admits a `project_storyboards` row
(reserve `llm_credits()` under `spend.storyboard_scope(<storyboard id>)` — impl: per storyboard, not per project; 409 while one is live; 429 on
budget) and enqueues it on the `process` queue, where transcription jobs already run (the job
encodes an LLM proxy with ffmpeg, which the 512m `localise` worker is not sized for).

The storyboard is built from **per-shot calls**, which are aligned with the detected
boundaries by construction (whole-video and per-chunk calls drifted by ±1 shot in the
pilot, §13):

1. **Global call** (one): a small proxy of the whole source (`make_llm_proxy`, then
   re-encoded further until it is under 1 MB) → `style`, `characters[{id, name_hint, look,
   outfit, role}]`, `sheet_groups[{key, description, characters}]` (one group per
   outfit/setting, e.g. "street", "wedding"), `audio{kind: voiceover|song|
   on_camera_dialogue|mixed, lines[…]}`, `on_screen_text`.
2. **Per-shot calls** (one per unfolded shot): the shot's own sub-clip (≤ 1 MB proxy) and
   its frame at `frame_sec`, plus the global result as context → `{location, framing,
   camera, characters, action, speaking_on_camera, is_brand_card, sheet_key}`. The prompt
   demands camera MOVEMENT words (the first pilot answer said "Static" everywhere).

Model chain: `LOCALISATION_TRANSCRIBE_MODEL` (gemini-3.1-pro) then the configured
fallbacks; a `400 Failed to get the file information` answer is classified as
"unavailable for this input" and moves to the next model (3-pro accepted the 4 MB proxy
3.1-pro refused, §13). (impl: wave 2a moves to the next model after ANY refusal or an
invalid answer — still one fallback row per call; `unknown` is never re-called
automatically.)

**Every LLM call is a paid-call row** (`rebuild_llm_calls`, the §7.2 mixin; a synchronous
call has no task id, so `accepted` stores the response body and goes straight to
`succeeded`). It is sent with a new `KieClient.chat_completion_once(model, messages, *,
timeout)` — one POST, no `_chat_with_retry` — classified like `create_task_once`: a
complete response with content → accepted; a complete response that is a definite error
(4xx, or 200 with a body `code` ≠ 200 such as the `400 Failed to get the file
information`) → refused; transport error, timeout, 5xx, unparseable body → unknown. A
structural test asserts no rebuild path reaches `_chat_with_retry` or
`_create_task_with_retry`. Columns besides the mixin: `project_id`, `storyboard_id` or `attempt_id`,
`kind` (`global`, `shot`, `character_check`), `shot_id`, `model`, `response`. So the
§7.2 rules apply unchanged: the draw is recorded with the `sending` marker, a call whose
sender died mid-call becomes `unknown` only after the sender is proven terminated, and
nothing ever sends a second request for the same row. A fallback to the next model, or a
retry after an invalid answer, is a **new row**, and the job may admit one only while the
storyboard's reservation still covers it.

**Reservation**: storyboard admission reserves `(1 + N) × 2 × llm_credits()` (every call
may fall back once). (impl: the global call is not counted when Resume reuses it, and N is
the shots that still need a call.) The job stops admitting call rows when the drawn total would exceed
the reservation, and leaves the remaining shots undescribed with "budget for this
storyboard used up; build again to continue".

**Execution and resume**: one storyboard row, processed in slices — the global call or up
to 8 shot calls per RQ job, each through its own call row — re-enqueueing itself for the
next slice under the same claim. (impl: the storyboard's own jobs only admit and observe —
at most 8 shot calls live at once; each call row is sent by its own paid-call step jobs.) A missing continuation is "not live" for the reaper →
the storyboard becomes `interrupted`, and **in the same transaction** every call row of
that storyboard still `queued` or `claimed` is cancelled (CAS → `failed`, "cancelled
before send"; nothing was sent, and the cancel revokes the fenced `claimed → sending`
CAS). Rows already `sending` keep their §7.2 life.

Invariant: at most one non-terminal call per (global context, shot) —
`uq_rebuild_llm_shot_live (global_call_id, shot_id) WHERE kind = 'shot' AND status IN
(<non-terminal>)` — and at most one non-terminal global call per project
(`uq_rebuild_llm_global_live (project_id) WHERE kind = 'global' AND status IN
(<non-terminal>)`). Admission of a call row relies on these indexes, so a duplicate is
refused by the database, not by a read-then-write check.

**Build again** (refused while the old storyboard has any call row `sending`: "waiting for
the previous worker to finish"; it unblocks by the §7.2 sender proof) offers:
- **Resume** (default), defined by the old storyboard's global call:
  - `succeeded` → the new row copies its **global result and `global_call_id`** and
    reuses every shot result made under that id for the same `shots_generation`; it admits
    shot calls only for shots with no `succeeded` call under that id. A shot whose last
    call is `unknown` is shown as "may have been billed" and is called again by this
    explicit Build.
  - `unknown` or `failed` (or none) → there is no global result to reuse, so Resume makes
    a new global call (the UI says the previous one may have been billed when it is
    `unknown`) and reuses no shot result — the same as From scratch.
- **From scratch**: a new global call; no shot result of an older global call is reused.
Each shot result stores the `global_call_id` it was made under, and the storyboard applies
only results whose `global_call_id` equals its own. Tests: Resume against an old
storyboard whose shot calls are `queued`, `claimed`, `sending`, `unknown`, `succeeded`,
and whose global call is `sending`, `unknown`, `succeeded`.

**Validation** (per call): tolerant JSON parse; character ids and `sheet_key` exist in the
storyboard's global result. A shot that still fails after one fallback row is left
undescribed with a readable error; the storyboard ends `failed` only if the global call or
more than 10 % of shots failed. If the shot generation changed while the job ran (it
cannot under §5.1.8, but defensively), the result is stored and not applied.

On success the job fills the shot rows' storyboard fields **except shots with
`edited_at`** (impl: a failed storyboard applies nothing; Resume reuses its valid answers); the card offers "Overwrite my edits" explicitly. `audio.kind` drives a
warning: `on_camera_dialogue` / `mixed` → "this video has on-camera speech; phase 1 keeps
the original audio and does not lip-sync" (operator may still proceed).

### 5.3 Keyframes (GPT Image, 10 cr each)

`POST /api/v2/projects/{pid}/rebuild/keyframes` with `{shots: [ids] | "all_missing",
sheet: bool, prompt?: str}` (impl: form fields `shots`, `sheets` = keys | `all_missing`,
`prompt`, `sheet_prompt`, `sheet_inputs`; refused while a storyboard is live) admits one `rebuild_keyframes` row per requested item in one
transaction under the project row lock (reserve `N × image_credits('2K')` under
`spend.keyframes_scope(batch id)`), then enqueues each row. Requires a storyboard applied
to the current shot generation (the sheet prompt needs the character descriptions).

- Shot keyframe input: the source frame at `frame_sec` (extracted by the job into
  `rebuild/frames/<shot id>.jpg`; impl: `<shot id>_<ms>.jpg`, so a moved `frame_sec` never
  changes a frame an older row was fed). Default prompt (operator-editable per batch and per
  regeneration):
  *"Recreate this exact frame of a <style> film as a clean, sharp, high-quality render.
  Remove ALL burned-in captions, subtitles and overlaid text completely; reconstruct what
  is behind them. Keep everything else exactly the same: the same characters with the same
  faces, hair, skin tone, body and outfits, the same pose, expression, framing, camera
  angle, background, lighting and colours. Do not add any text, logos or watermarks."*
- Character sheets: **one per `sheet_groups` key** (outfit/setting; the pilot needed a
  second "wedding" sheet, §13). Input: up to 5 frames of that group's shots where each of
  its characters is largest (storyboard `characters` + `framing`), operator-editable.
  Prompt generated from the group's character lines: "clean character reference sheet …
  side by side, full body, front view, plain light background, exact same faces/outfits,
  no text".
- 2K, `aspect_ratio` = the project's 9:16 / 16:9 (pilot: 1536×2736).
- The newest successful attempt for a shot becomes `selected` automatically unless the
  operator picked another attempt (selecting is a free PATCH). (impl: no column records the
  pick — a manual pick stamps `updated_at` after the row's `finished_at`.)
- Regenerate one: the same endpoint with one shot id and an optional prompt. Each attempt
  is a new row; older ones stay selectable.

Each keyframe row is driven by the paid-call protocol of §7.2 (short step jobs on the
`process` queue: claim → frame extraction + upload → send → poll → fetch). A download
failure retries the download only (the `reskin-preprocess-retry-deadend` lesson), never the
create. `batch_id` only groups rows for the UI and the reservation.

## 6. Run phase: launch → submit → fit → stitch

### 6.1 Launch (`create_run` on a rebuild project)

Form fields: model (`kling-4-0` default, `seedance-2-5`), resolution (`720p` default;
`1080p` offered), language (Job ID suffix only), stitch plan (default = all chunks and keep
segments in order; library tails allowed as usual). Forced: `generation_mode='images_only'`,
`audio_mode='original'`, `preprocess_enabled=False`, no reference images, no voice refs.
The API rejects (400/409) any other combination, and rejects `kling-4-0` on every
non-rebuild project (`supports_v2v=False`).

Preconditions (409 with the list of what is missing): storyboard applied to the current
shot generation; every unfolded shot that intersects an active swap chunk has a
`selected`, `succeeded` keyframe; each active chunk's sheet (below) exists; no keyframe or
storyboard row live.

**Sheet per chunk**: the chunk uses the selected sheet of the `sheet_key` held by most of
its unfolded shots (ties → the earliest shot's key); the New Run form shows it per chunk
and lets the operator pick another group's sheet (stored in the snapshot).

**End card / kept footage with burned text**: every `keep` segment kept from the source
still shows the source's burned captions (pilot defect 1, §13). When the storyboard's
`on_screen_text` is not empty, the New Run form warns on every source `keep` segment.
Only the **trailing end card** gets a default replacement: the maximal run of `keep`
segments at the END of the timeline whose shots are all `is_brand_card`. If such a run
exists **and** a language-matched library end card exists (as localisation picks tails;
impl: the newest library clip whose `language` equals the run's, `en` when unset — a
language-agnostic clip is never picked; the form field `keep_source_end_card` keeps the
source card),
the default stitch plan is every segment in its original order up to the end card, then
the library clip instead of the end-card segments. Every other `keep` segment (a middle
scene the operator kept) stays in place. Without a matching library clip, the default
plan is unchanged and only the warning shows. The operator may keep the source end card
deliberately (Q2). This uses the existing custom-plan stitch (per-clip audio: the library
clip brings its own sound). Test: a timeline with a middle keep and an end card.

In the creating transaction, under the project row lock:

1. `segment_snapshot` is written **now** (v2v writes it at submit) so the chunks, the
   snapshot and the prompts all describe one segmentation, and one `RunSegment` row per
   active swap chunk is created in `pending` (v2v creates them lazily at submit; for
   images-only they must exist before the reservation below, so the estimator sees them).
2. For each active swap chunk: the shots it intersects, clipped to the chunk, with times
   relative to the chunk start (§6.2); the rendered prompt (§6.3); the requested duration
   (§6.4); the keyframe list. Keyframe files are **copied** (hard link, else copy) into
   `runs/<rid>/rebuild/<chunk idx>/` and their sha256 recorded, so regenerating or deleting
   project keyframes never changes a launched run (the voice-reference rule).
3. All of it goes into `runs.rebuild_snapshot = {version: 1, model, resolution,
   storyboard_id, shots_generation, chunks: [{segment_def_id, index, start_sec, end_sec,
   request_sec, prompt, shots: [{shot_id, rel_start, rel_end, keyframe_path, sha256}],
   sheet_key, sheet_path}]}`.
4. `spend.reserve_run` with the images-only estimate (§7.3, §12).

File copies happen before the commit; a failed commit leaves orphan files in a fresh run
dir that no row points to (cleaned by `delete_run`'s directory removal if the id is reused,
otherwise harmless; listed as a residual).

### 6.2 Shots inside a chunk

A shot intersecting a chunk contributes `[max(shot.start, chunk.start), min(shot.end,
chunk.end)]`. Folded shots (§5.1.3) contribute their time to the shot they are folded into. Pieces
shorter than `REBUILD_SHOT_FOLD_SEC` created by an operator boundary are merged into their
neighbour inside the chunk the same way. A long
shot split across two chunks appears in both, each with the same keyframe. The chunk's image
list is its shots' keyframes in order (deduplicated), then the sheet: ≤ 9 images by
construction of §5.1 (an operator-merged chunk with more shots is refused at launch with
"split this chunk": the model limit is `max_reference_images - 1` keyframes).

### 6.3 Prompt

`rebuild.render_chunk_prompt(snapshot_chunk, storyboard, model)`, deterministic:

```
A <request_sec>-second vertical 9:16 <style>. It consists of <k> shots joined by hard
cuts, in exactly this order and timing. Each shot starts from the composition of its
keyframe image. The characters must look exactly the same in every shot, as in the
character sheet <ref(sheet)> and the keyframes: <one clause per character present>.
Shot 1 (0.0-1.7 s), starts from <ref(1)>: <framing>, <camera>: <action>.
…
No on-screen text, no captions, no subtitles, no logos. No dialogue is needed; ambient
sound only.
```

`ref(n)` is `@image_n` for Kling and `[Image n]` for Seedance (`AIModelSpec.image_ref_token`).
The last shot's end is stretched to `request_sec` (the model fills the extra tail, which
§6.4 trims). The rendered prompt is stored in the snapshot and shown (read-only) on the run
page; a per-chunk prompt edit is a segment rerun with `prompt_override` (§6.6).

### 6.4 Submit, download, fit

An images-only run never enters the v2v submit/poll code. The real entry is the RQ wrapper
`tasks.run_submit_run` (what `enqueue_process_run` enqueues, for a launch, a Retry and a
recovery hand-off alike); right after it acquires or refreshes the per-run lock it
branches on `run.generation_mode`. For `images_only` it calls
`pipeline_rebuild.submit_images_only(run_id)` and then schedules
`tasks.run_rebuild_observe(run_id, lock_key=…, lock_token=…)` with `enqueue_in`, handing
the SAME lock over exactly as `run_poll_run` receives it. The v2v code below the branch is
untouched.

**`submit_images_only` (first entry, also after every recovery):**
1. If the run is `queued`, transition `queued → processing` (recovery always resets an
   orphaned active run to `queued` first, so this is the only transition it makes); if it
   is already `processing`, accept it; anything else → log and stop.
2. Create any missing `RunSegment` row for an active swap chunk (normally all exist since
   launch; this only repairs a run created before a crash mid-launch, and never happens
   without a reservation because launch commits the rows and the reservation together).
3. For each active chunk segment in `pending` with no live attempt: admit a
   `rebuild_chunk_attempts` row (`queued`, `attempt_no` = max + 1, request frozen from the
   snapshot) and enqueue its `claim` step. The live-attempt unique index makes a duplicate
   admission (two entries racing) fail harmlessly.

**`run_rebuild_observe` (the run's driver while attempts are in flight):** refreshes the
lock first and stops if it no longer owns it (as every chain job does). Then, read-only
on attempts:
- every active chunk segment `completed` → enqueue the existing `run_finalize_run` with
  the lock handed over (stitch + deliver);
- some segment `failed` and no attempt live → run → `incomplete`, release the lock
  (existing semantics: the operator reruns the chunk);
- otherwise re-schedule itself (`RUN_POLL_INTERVAL_SEC`), lock handed over.
It never transitions `queued → processing` and never admits attempts, so repeated
invocations are pure observation. (impl: one write besides `incomplete` — a chunk whose
attempt the paid-call reaper released to `unknown` never ran a step, so a non-completed
segment with no live attempt is marked `failed` from its last attempt by the observer.) Segment states are written by the attempt steps
(below), not by the observer: `pending → uploading` (attempt `queued`/`claimed`/`sending`),
`→ submitted → generating` (`submitted`/`fetching`), `→ completed` (attempt `succeeded`
and selected), `→ failed` (attempt `failed`/`unknown`), each in the same transaction as the
attempt transition.

Neither function creates a provider task, resubmits, or reads `seedance_task_id`.
Defence in depth: `pipeline_v2.submit_run`, `_build_submit_state`,
`_repoll_generating_segments`, `_resubmit_segment_task`, `poll_run_once_internal` and
`run_poll_run` refuse an images-only run (ERROR log, no state change), with tests that go
through the real `enqueue_process_run` path for a launch and for an orphan recovery.

The attempt's request (built by the attempt's send step from the frozen snapshot):

- Images: the chunk's run-owned keyframes, uploaded at the `claimed` step (impl: the claim
  step verifies the copies against their sha256; the free uploads happen in the send step
  right before the `sending` CAS, so a URL is never older than the request)
  (`resolve_reference_urls` on the paths), then the sheet. No clip, no `min_clip_sec`
  padding, no source-fallback branch (a chunk is ≤ 14.72 s by construction; a longer
  operator-made chunk is refused at launch, never silently replaced by source footage).
- Duration: `request_sec = clamp(ceil(chunk_len - 0.02), spec.images_only_min_sec,
  spec.images_only_max_sec)`; Kling 3–30 integer, Seedance 2.5 4–15 here.
  `validate_duration` is not called (its `follows_input_geometry` rule is about an input
  video).
- Payloads: `build_kling_payload` → `{"model": "kling-4.0/reference-to-video", "input":
  {prompt, duration, resolution, aspect_ratio, image_urls}}`; `build_seedance_images_payload`
  → the Seedance shape with `reference_video_urls: []`, an integer `duration`, a concrete
  `aspect_ratio`, `generate_audio: false`. New pure functions; `build_seedance_payload` is
  untouched.
- Sent with `KieClient.create_task_once(payload)` (§7.2), never `_create_task_with_retry`.

Fetch (the attempt's `fetching` step; the ONLY completion path, used by normal polling and
by every recovery): download to `raw.mp4.part`, verify it decodes, `os.replace` →
`raw.mp4`; probe; then **fit**:

- The output frame grid is the stitch's: `fps_out` = the fps `_run_stitch_and_deliver`
  passes to `media.stitch` for this project. Chunk boundaries are quantised
  **cumulatively** on that grid, `F(t) = round(t × fps_out)`, and the chunk gets
  `chunk_frames = F(end) − F(start)` frames (frozen on the attempt), so rounding never
  accumulates across chunks.
- `fitted.mp4` = the first `chunk_frames` frames of `raw` at `fps_out`
  (`fps=fps_out,trim=end_frame=chunk_frames`), re-encoded CRF 18.
- If `raw` holds fewer than `chunk_frames − round(0.1 × fps_out)` frames: the attempt
  ends `failed` with `result too short (X s < chunk Y s)`. **No automatic resubmit** (a
  Kling chunk is ~400 cr): the run goes `incomplete` and the operator reruns the chunk.
  Up to that tolerance, the last frame is held to fill, recorded in `qa.padded_frames`.
- Then, in one transaction: attempt `succeeded` + `selected` (unselecting the previous
  one), segment `completed`, `local_result_path = fitted_path`. A crash before that commit
  repeats the fetch (idempotent: the files are rewritten via `.part` + replace).

**Every** source-timeline segment of an images-only run is allocated frames by the same
cumulative rule: at finalize, keep segments are cut by `rebuild.cut_keep_frames(source,
F(start), F(end), fps_out)` (frame-exact select after the fps conversion), through an
images-only branch in `_cut_or_lookup`; v2v keeps its existing cut. So the stitched picture
has exactly `F(duration)` frames and every segment starts on its own boundary frame,
however many swap/keep alternations there are.

### 6.5 QA (flags only, never blocks delivery)

Run as a separate short job on the `autopilot_compute` queue (served by the run workers)
right after an attempt succeeds, written to `rebuild_chunk_attempts.qa`, shown on the run
page. QA never changes run or segment state (impl: `pipeline_rebuild.reap_qa` in
`run_reconcile_tick`, a 7-day window); a lost QA job is simply re-enqueued by the
reaper for succeeded attempts whose `qa` is NULL:

- **Cut timing**: `detect_scenes` on the fitted clip vs the planned internal cut times.
  Report every planned cut's |Δ| and missing/extra cuts. Flag when any |Δ| > 0.5 s or the
  cut count differs. (Pilot: Kling max 0.18 s, Seedance 0.9 s.)
- **Duration**: raw duration, requested, padded.
- **Burned text**: `qa_tools.locate_captions` on the fitted chunk (the subtitle locator);
  flag any band found.
- **Contact sheet**: one frame per planned shot (midpoint) next to its keyframe, saved as
  the attempt's `sheet.jpg` for the operator's eye check.
- **Character check** (pilot defects 2–3, §13): one Gemini call per succeeded attempt with
  the contact sheet and the chunk's character list → per shot `{expected characters
  present, missing, identity_drift (a character looks like another one), notes}`. Flags:
  a missing expected character (shot 15 in the pilot) and identity drift (the hero
  turning into the friend at ~84 s). It is the one paid QA step and runs as a
  `rebuild_llm_calls` row (`kind = character_check`, at most two per attempt: the primary
  model and one fallback — impl: the fallback is admitted when the primary ends `failed`
  OR `unknown`; the QA reaper folds a call the paid-call reaper released), so the §7.2 rules hold: drawn at `sending`, never re-sent, an
  interrupted call becomes `unknown` only after the sender proof. The free QA values
  (cut timing, duration, text band, contact sheet) are written separately from the check,
  and the reaper re-enqueues **only the free part** when it is missing; a check that ended
  `failed`/`unknown` leaves `qa.character_check = "unavailable"` with a "Run character
  check" button (a new, reserved call). Never blocks delivery.
- **Repair unit**: a flagged chunk is fixed by re-rolling that chunk (segment rerun, a new
  attempt from the same snapshot, ~285–430 cr at 720p), then "use attempt N" if the
  re-roll is worse. The run page shows the flags per shot with a one-click rerun.

### 6.6 Stitch, delivery and afterwards

The existing finalize path: swap chunks use their fitted results, keep segments are cut
from the source, `audio_mode='original'`. With the default plan the source soundtrack is
muxed continuously; with a custom plan each swap clip gets its own range of source audio
(fact 2.1.6), so an omitted or reordered chunk still carries its matching sound. Because
every fitted chunk is exactly its chunk's length on the output frame grid, the continuous
soundtrack stays aligned. Library tails are allowed as for any run (they bring their own
audio under a custom plan). Drive delivery, Job ID (`Pipeline Type` stays `re-skin-v2`;
Q8), Final Creatives (`Idea` model tag `KL4` for Kling, `SD25` for Seedance), captions, tail
edits, overlays, audio patches and Move-to-project work on the final as for any run.

**Allowed on an images-only run**: Retry Run (failed/incomplete), segment rerun (admits a
new attempt from the frozen snapshot; optional `prompt_override` stored on the RunSegment
and used by the next attempt's request), "use attempt N" (re-select an older succeeded
attempt of a chunk and re-stitch; replaces AICP-342 take restore for this mode), and
everything that works on the delivered final.

**Refused server-side (409 "not supported for images-only runs")** (impl: the whole
`PATCH /runs/{rid}/segments/{rsid}` is refused — a prompt edit goes with the re-roll; while
`REBUILD_ENABLED` is off a re-roll / Retry from failed-incomplete / manual check is refused,
"use attempt N" and an orphan Retry are not), each with a direct-API
test: Copy run and copy-batch, enabling an automation, segment reference-image override
(`reference_image_urls_override` PATCH), per-segment GPT Image regeneration
(`run_segment_image_jobs`), run-level preprocessing / preprocess regen, range attempts
(AICP-349), rerun in place's take archive and take restore (AICP-342) — superseded by
"use attempt N" —, VOID caption fill / source fill, Gemini Omni reruns, and any model change
on an existing run. The New Run form hides the matching controls, but the server is the
guard.

## 7. Concurrency, recovery and billing safety

### 7.1 Lock order

`video_projects` row → `runs` rows → `run_segments` → `rebuild_shots` →
`rebuild_keyframes` / `project_storyboards` / `rebuild_chunk_attempts` → spend rows
(reserve last, as everywhere). Every admission locks the project row first (§5.1.8). The
per-run Redis lock is orthogonal (it serialises run jobs, as today) and is never taken by
attempt or keyframe step jobs.

### 7.2 Paid-call protocol (keyframes and chunk attempts)

Every paid provider call in this feature is one row and at most one provider request.

**Single-attempt create.** `KieClient.create_task_once(payload, *, timeout)` POSTs once,
with no tenacity retry, and classifies the outcome:
- HTTP 200 with a `taskId` → **accepted** (task id returned);
- a complete HTTP response that is a definite refusal (4xx, or 200 with a body `code` ≠
  200 and no `taskId`) → **refused** (nothing billed);
- anything else (connect error after the request may have left, read timeout, 5xx,
  unparseable body) → **unknown**.
Existing callers keep `_create_task_with_retry`; v2v is unchanged.

**States.** `queued → claimed → sending → submitted → fetching → succeeded`, with
`failed` (refused, provider `fail`, too short) and `unknown` (send outcome unknown) as the
other terminals.

**Step jobs.** One short RQ job per step, id `rb:<table>:<row id>:<dispatch_gen>:<step>`
(impl: poll/fetch add `:<seq>`; liveness is the `rb:<table>:<id>:<gen>:` prefix),
on the `process` queue: each does one transition and enqueues the next step (polls via
`enqueue_in`, 10 s for images, 20 s for video). No step holds a worker for more than one
provider call, one download or one ffmpeg fit.

- `claim` (`queued`): CAS `queued → claimed` setting a fresh `claim_token`; prepare
  inputs (frame extraction, uploads) — free and idempotent.
- `send` (`claimed`): one transaction: CAS `claimed → sending` **fenced by
  `claim_token`**, record the **sender identity** on the row (`sender_host` = container
  hostname, `sender_boot` = the worker process's boot id, `sender_job` = this RQ job id),
  set `send_deadline = now + SEND_WINDOW_SEC` (30 s), and record the spend draw at the
  estimate, keyed `rebuild:<row id>` (`spend.draw_in_session`, committed with the marker).
  After the commit the worker skips the call if more than `SEND_WINDOW_SEC − 5` s have
  passed since the commit (cheap insurance, **not** a proof of anything); otherwise it
  calls `create_task_once(timeout=SEND_HTTP_TIMEOUT_SEC)` (60 s).
  - accepted → CAS `sending → submitted` with the task id. This CAS cannot lose to the
    reaper, because the reaper never moves a `sending` row whose sender may still be
    alive (below).
  - refused → CAS `sending → failed`, draw settled to 0.
  - unknown → CAS `sending → unknown`, draw stays at the estimate. (impl: a call
    skipped by the send window → `failed` "not sent", draw settled to 0.)
- `poll` (`submitted`; impl: chunk attempts give up 6 h after the send — `failed`, draw
  left at the estimate): `recordInfo`; `success` → `fetching`; `fail` → `failed`, draw
  settled with `creditsConsumed`; otherwise re-schedule.
- `fetch` (`fetching`): download + verify (+ fit for chunks, §6.4) → `succeeded`, settle
  the draw. A failed download re-schedules the fetch (bounded attempts, then `failed` with
  "result could not be downloaded; task <id>" — the task id stays on the row for manual
  recovery). Never a new create.

**Reaper** (in `run_reconcile_tick`, next to `localisation_jobs.reap`). For each
non-terminal row whose current step job (by its deterministic id) is not queued, scheduled
or started, for longer than `NOT_LIVE_GRACE_SEC`:
- `queued` / `claimed` → bump `dispatch_gen`, new `claim_token`, back to `queued`,
  enqueue `claim` (nothing was sent: the send step only runs from `claimed` after a fenced
  CAS to `sending`, which the bump revokes);
- `sending` → **only when the sender is proven terminated**: either the boot id
  currently registered for `sender_host` differs from `sender_boot` (the container
  restarted, so every process of the old boot is gone), or `sender_job` was ended BY ITS OWN
  WORKER: status `finished`, or `failed` with the work-horse reap record that rq 1.15's
  `Worker.monitor_work_horse` writes after `waitpid` ("Work-horse terminated unexpectedly;
  waitpid returned …"). A job moved to the failed registry by
  `StartedJobRegistry.cleanup` ("due to AbandonedJobError") proves nothing — that is what
  a frozen container looks like — and is not accepted. Then → `unknown`. While neither holds
  (a live but stalled sender, a frozen container) the row stays `sending` however long it
  takes; the UI shows "waiting for the sending worker", and deletion, re-analysis and a
  replacement attempt stay refused because the row is live.
  Boot ids: each worker process writes `rebuild:boot:<hostname> = <uuid>` (no TTL) at
  startup, so a frozen container keeps its old value and is never mistaken for dead.
  `hostname` is the container id, which survives `docker restart` / `restart:
  unless-stopped` (the automatic case: a crashed or OOM-killed container comes back under
  the same hostname with a new boot id) but NOT a recreation (deploy, `up -d`, scale), which
  removes the old container and starts one with a new id. That case cannot be proven from
  inside a container (no Docker socket), so it gets a **host-level attestation**:
  `scripts/rebuild_release_sender.sh <row id>` runs on the VPS host, reads the row's
  `sender_host`, checks with `docker ps -a --no-trunc` that no container with that id is
  running (absent, or `exited`), and only then calls `docker exec re-skin-api python -m
  app.rebuild_paid_call release-sender <row id> --attest <container id>:<state>` (impl: module name), which moves
  the row `sending → unknown` (CAS) and logs the attestation on the row. Deploys rarely
  need it: the drain check (§7.5) waits for `sending` rows to clear, so a recreation only
  meets a `sending` row after a crash during the send itself. Tests: restart (same
  hostname, new boot → automatic), recreation (old host absent → stays `sending` until the
  attested release), frozen container (same boot → stays `sending`), horse reaped by its
  own worker (automatic), abandoned-job cleanup only (stays `sending`).
- `submitted` / `fetching` → bump `dispatch_gen`, enqueue `poll` / `fetch` (polling and
  fetching are idempotent; the task id is known, so there is nothing to re-create).
Every transition the reaper makes is a CAS on the row's current (`status`, `dispatch_gen`),
so a slow step job finds its fence gone and stops. Because `unknown` is declared only once
no process that could still send exists, there is no late response to reconcile: an
`unknown` row is final, and the replacement the operator admits is the only live attempt.
(kie.ai offers no idempotency key; if it ever does, it replaces the sender proof.)

**Accounting.** The draw is recorded with the `sending` transition (keyed by the row,
idempotent), so a paid call can never be missing from the books, and is settled at the
terminal state: `creditsConsumed` on success/fail, 0 on refused, the estimate on unknown
(conservative). Admission reserves the same estimate (`spend.reserve`, scope per batch or
per run), so the cap sees every attempt before it is sent.

**No automatic re-buy.** Nothing in this feature creates a second provider request for the
same row. `failed` and `unknown` are terminal; the operator's regenerate / segment rerun /
Retry Run admits a NEW row with its own reservation. `unknown` is shown as "the provider
may have billed this attempt (task id unknown); regenerating bills again".

### 7.3 Images-only runs and the existing run machinery

- `process_run` dispatch (§6.4) is the only entry; the per-run lock, `_recover_orphaned_run`
  (claim stale lock → reset to `queued` → re-enqueue `process_run`) and the drain rules
  apply unchanged. Re-running `process_images_only_run` is idempotent: it admits an attempt
  only for a `pending` segment with no live attempt (the live-attempt unique index makes a
  duplicate admission fail), and otherwise only observes.
- `_repoll_generating_segments` and the v2v resume/resubmit paths return early for
  images-only runs (§6.4), so recovery can never create a provider task for them; attempt
  progress is the reaper's job (§7.2), run progress the run chain's.
- **Retry Run** on an images-only run (failed/incomplete): the existing endpoint only moves
  the run to `queued`; the images-only branch additionally resets every active chunk
  segment that is `failed` (its attempt `failed` or `unknown`) to `pending`, under the
  run row lock and in the same transaction as the estimate and `spend.reserve_run`.
  **Segment rerun** resets the one segment the same way. The estimator branches on
  `generation_mode`: images-only cost = `IMAGES_ONLY_CREDITS_PER_SEC[model][res]
  × request_sec + 2 × llm_credits()` (the video task plus the character check with one
  fallback) for every active chunk segment in `pending` (a fresh launch, or a segment
  the Retry/rerun endpoint has just reset — whether or not an older attempt is still
  selected) that has no live attempt. The rerun endpoints reset the segment and reserve
  in the same transaction (as today), so the reservation always covers the attempt the
  next `submit_images_only` admits. The estimator **also** counts the *undrawn* remainder
  of every succeeded attempt's character-check allowance: `llm_credits()` for each of its
  at most two call slots (primary, fallback) that has no call row yet or whose row is
  still `queued`/`claimed`; a slot whose row reached `sending` has already drawn its
  estimate (§7.2) and counts 0; once a slot is terminal and no further call is allowed,
  the check counts 0. So `release_unused` on a Retry or rerun keeps exactly the
  still-undrawn QA allowance reserved, never double-counts an in-flight call, and never
  releases it.

  **Estimate under the lock.** Today `reserve_run` computes `estimate_run_credits` and
  only then calls `reserve`, which takes the day lock — so an estimate can go stale
  before the lock. Images-only runs therefore use a new
  `spend.reserve_computed(db, scope_key, estimate_fn, *, kind, release_unused, run_id)`:
  it flushes the caller's changes (as `reserve` does), takes every day lock it will touch
  in ascending date order (today, plus the earlier days `release_unused` re-baselines for
  this scope), and **then** calls `estimate_fn(db)`, which reads attempts, call rows and
  segments with plain reads (no row locks, so the lock order "paid-call row update →
  day lock" used by `claimed → sending` + `draw_in_session` cannot deadlock against it),
  and applies the result under the locks it already holds. Interleavings: a `sending`
  transition that committed before the lock is visible (row `sending`, draw in consumed →
  counts 0 in the estimate); one waiting on the day lock is still `claimed` in our read
  (counted as undrawn) and its draw lands after our commit inside an already-reserved
  scope (`charged = max(reserved, consumed)` absorbs it without a second charge). Tests
  for both interleavings, including a scope whose reservation spans two UTC days.

  **Scope lock.** `_apply_draw` picks its target day (`_draw_target_day`) BEFORE it takes
  that day's lock, so a draw could pick yesterday, wait, and land on yesterday's
  allowance after `reserve_computed` re-baselined the scope onto today. For the scopes of
  this feature (`run:<id>` of an images-only run, the storyboard and keyframe-batch
  scopes) both sides first take a **scope lock** — `pg_advisory_xact_lock` on a stable
  64-bit hash of `scope_key` (on SQLite the database-wide write lock already serialises
  them) — and only then choose / re-baseline days and take day locks:
  - draw (`claimed → sending` + `draw_in_session`, settle): paid-call row update →
    scope lock → `_draw_target_day` (now computed under the scope lock) → day lock;
  - `reserve_computed`: flush → scope lock → day locks (ascending) → `estimate_fn` →
    re-baseline + reserve.
  Neither path takes a paid-call row lock after the scope lock (the estimator reads
  rows without locks), so there is no lock cycle. A draw serialised after a re-baseline
  therefore chooses its day from the re-baselined reservations. The scope lock is added
  through a `scope_lock=True` argument on `draw_in_session` / `_apply_draw` used only by
  this feature; `reserve_run` and every v2v caller stay unchanged. Test: a sender that
  would have chosen the old day blocks on the scope lock while the two-day reservation
  transaction moves the allowance, then commits onto today's reservation. The
  same estimator serves launch, Retry and rerun; tests at the cap: a Retry while a
  check's primary call is `sending`, and after the primary ended `failed` but before the
  fallback is admitted, each racing a segment rerun;
  tests through the real endpoints: a fresh multi-chunk launch, a Retry after an attempt
  ended `failed` and one ended `unknown`, and a rerun of a completed chunk, each at the
  budget limit (expects 429) and below it (expects the attempt to be admitted).
- Cancel / delete of a run with a live attempt: refused (409) while any attempt is
  `claimed`/`sending`; for `submitted`/`fetching` the existing delete guard (active run)
  already refuses.

### 7.4 What re-analysis and deletion do to runs

Launched runs never read shots, storyboards or project keyframes after launch: everything
is in `rebuild_snapshot`, `segment_snapshot` and the run-owned files. So a re-analysis
(allowed only when no run is active and no rebuild row is live) cannot change an existing
run or its reruns. `delete_project` additionally 409s while a storyboard, keyframe or
attempt row is live; the cascade deletes the rest. `delete_run` removes the run dir
(keyframe copies, attempts' raw and fitted files) and 409s while an attempt is live.

### 7.5 Deploy and drain

`deploy/drain-check.sql` counts live `project_storyboards`, `rebuild_keyframes`,
`rebuild_chunk_attempts` and `rebuild_llm_calls` (non-terminal statuses). A deploy that kills a step job is
recovered by the reaper per §7.2; a row caught in `sending` becomes `unknown` only after
its deadline, so the drain waits for `sending` rows to clear.

### 7.6 Rollout, feature flag and rollback

- `REBUILD_ENABLED` (`.env`, default `false`, needs `up -d`): while false, `create_project`
  refuses `project_type=rebuild`, `create_run` refuses `kling-4-0` and
  `generation_mode=images_only`, and the type is hidden in the Create Project form. So
  neither new enum label can appear in a row until the flag is flipped.
- Order: DB backup → migration → image (every container that maps `VideoProject`/`Run`:
  api, both workers, localise, autopilot, knowledge, fal if running, AND the assistant
  supervisor, which `up -d --build` does not recreate) → verify every container runs the
  new image → nginx restart + curl → `REBUILD_ENABLED=true` + `up -d` **with the same
  compose arguments and overlays as the deploy** (`--env-file` pair, `compose.app.yml`,
  `compose.asr.yml`, `--scale worker=2`, `--profile fal` if Fal is enabled) → **nginx
  restart again** (the env change recreates the api container with a new IP) → curl
  through the proxy. Turning the flag off follows the same sequence.
- Rollback before the flag was ever on: ordinary image rollback (no row carries the new
  labels). After a rebuild project exists, an OLD image would fail to load those rows
  (SQLAlchemy enum lookup on `project_type` / `model`), breaking project and run lists, so
  **image rollback is not supported**; rollback = `REBUILD_ENABLED=false` on the new image
  (new admissions refused, existing rows still readable and recoverable). A forced
  rollback to an older image first requires `scripts/rebuild_quarantine.py` (delete or
  re-type rebuild projects and images-only runs after exporting them), shipped with the
  feature and rehearsed on the disposable database.

## 8. API

| Method + path | Notes |
|---|---|
| `POST /api/v2/projects/{pid}/rebuild/storyboard` | admit (202); 409 live job / wrong type; 429 budget |
| `GET  /api/v2/projects/{pid}/rebuild/storyboard` | latest rows + applied state |
| `GET  /api/v2/projects/{pid}/rebuild/shots` | shots with storyboard fields and selected keyframe |
| `PATCH /api/v2/rebuild/shots/{sid}` | edit storyboard fields / `frame_sec` (sets `edited_at`) |
| `POST /api/v2/projects/{pid}/rebuild/keyframes` | admit a batch (202) |
| `PATCH /api/v2/rebuild/keyframes/{kid}` | `{selected: true}` |
| `POST /api/v2/projects/{pid}/rebuild/keep-brand-cards` | re-cut chunks so brand-card shots are `keep` (§5.1.5) |
| `POST /api/v2/projects/{pid}/runs` | existing; images-only rules of §6.1 |

All mutating routes go through the existing CSRF and auth layers and lock the project row.
(impl wave 2a: `app/api_rebuild.py`, form fields; added `POST …/rebuild/storyboard/cancel`,
`GET /rebuild/keyframes/{kid}/image`, `GET /rebuild/shots/{sid}/frame`; unfold is the shot
PATCH with `force_anchor`; with `REBUILD_ENABLED` off every mutation 409s and reads stay —
see the implementation notes, Wave 2a.)

## 9. Registry entries

- `ProjectTypeSpec(key="rebuild", label="Rebuild from reference", segmentation=SEG_SHOT_CHUNKS,
  default_model="kling-4-0", default_prompt="" (prompts are generated per chunk),
  default_audio_mode="original", default_mute_source=False, requires_audio_model=False,
  uses_references=False, hint="Recreate the video shot by shot from clean keyframes; original
  soundtrack kept.")` plus a new field `generation_mode="images_only"` (default `"v2v"` for
  every other type) that the New Run form and `create_run` read.
- `AIModelSpec` gains `supports_v2v: bool = True`, `supports_images_only: bool = False`,
  `images_only_min_sec`, `images_only_max_sec`, `max_reference_images`,
  `image_ref_token`. `kling-4-0`: family `kling`, label "Kling 4.0", short `KL4`,
  resolutions 720p/1080p, `supports_v2v=False`, `supports_images_only=True`, 3–30 s,
  10 images, `@image_{n}`, `produces_audio=True` (its track is discarded in phase 1),
  `supports_generate_audio=False`. `seedance-2-5`: `supports_images_only=True`, 4–15 s
  (our ceiling), 9 images, `[Image {n}]`.
- The New Run form on a rebuild project lists only `supports_images_only` models; on every
  other project only `supports_v2v` models (Kling never appears there).

## 10. UI

**Create Project**: the type appears automatically from the registry, with its hint.

**Project page (rebuild)**, top to bottom:

1. Source + analysis status, "N shots, M chunks" and a chunk strip (chunks coloured, shot ticks).
2. **Storyboard card**: Build / Rebuild button with the LLM estimate, job status, then the
   style, characters and audio kind; the on-camera-speech warning; "Keep brand-card shots as
   original footage".
3. **Shots grid**, grouped by chunk: per shot the source frame, the selected keyframe (or a
   placeholder), the editable storyboard fields, the attempts strip with select, and
   "Regenerate" (optional prompt). Batch bar: "Generate missing keyframes (N × 10 cr)" and
   the character sheet slot with its input-frame picker.
4. **New Run** form (model, resolution, language, stitch plan) with the estimate (§12) and
   the precondition list; disabled until met.

**Run page (images-only)**: per chunk — status, request vs chunk length, links to raw and
fitted clips, the contact sheet, QA flags (cut timing table, text band, padding); the
rendered prompt; the Segment rerun (with optional prompt override). Captions / tail edits /
overlays / Send to Final as for any run.

**Segment Editor**: works as today on the chunks; shows shot ticks so the operator can see
when a boundary splits a shot.

## 11. Later phases

- **Phase 2 — new characters.** A run-level keyframe set: GPT Image edits each selected
  project keyframe with the persona photo ("replace character C1 with the person in image 2,
  in the same 3D style; keep pose, framing, scene") plus a new sheet. Rows in
  `rebuild_keyframes` with `owner_run_id`; review on the run page before submit (a new run
  state or the existing `preprocess_review` — decided in the phase-2 design). Copy run with a
  new photo re-runs only this step. Gender agreement and hands are handled by the image
  model (the AICP-243 failure disappears because there is no source video).
- **Phase 3 — model speaks (AICP-258).** `audio_mode='seedance'`, Seedance 2.5 with
  `generate_audio` + voice reference + the script (source transcript or translation) per
  chunk; localisation reuses the translate job. Needs speech-aware chunk boundaries.
- **Phase 4.** Automation/sheet batches; autopilot flow; per-chunk model choice; 1080p
  default if the pilot says so.

## 12. Costs and estimates

Rates for admission (`IMAGES_ONLY_CREDITS_PER_SEC`, credits per **requested output
second**; erring high as the existing table does):

| model | 480p | 720p | 1080p | source |
|---|---|---|---|---|
| `kling-4-0` | — | 28.5 | 42.5 | list price, matched exactly in 7/7 tasks (pilot + AICP-256 eval) |
| `seedance-2-5` | 40 | 64 | 160 | 63/s pilot 720p; 158/s AICP-256 1080p; ~40/s AICP-243 480p |

Estimate for 81d0afc0 (94.9 s; ~87 s of chunks, end card kept; ~7 chunks, each requested as
`ceil`): Kling 720p ≈ 7 × 15 s × 28.5 ≈ 3000 cr; Kling 1080p ≈ 4460 cr; Seedance 720p ≈ 6700
cr. Keyframes: 54 shots + 1 sheet ≈ 550 cr. A Kling 720p rebuild of the whole video was
estimated at ≈ 3550 cr; the full manual pilot **measured ≈ 3,100 cr ≈ 15.5 USD** (§13:
45 keyframes after shot folding + 2 sheets + 1 regen = 470 cr; 7 chunks = 2,622 cr;
Gemini tens of cr), against ≈ 8300 cr for subtitle removal + face swap. Add ~285–430 cr
per re-rolled chunk.

`settle_draw` replaces each estimate with `creditsConsumed`; the run page shows estimate
and actual.

## 13. Full-video pilot results

Manual pilot, 2026-11-02, project `81d0afc0` (94.9 s, 30 fps), Kling 4.0 720p, original
audio. Files: `/root/rebuild-pilot/full/` (`81d0afc0_rebuild_kling_full.mp4`, `_sbs.mp4`,
`seams_strip.jpg`, `gen/cut_timing.json`, scripts).

- **Shape**: 0–89.0 s rebuilt as 7 chunks of 9.3–15.2 s, ≤ 9 shots each (53 shots); the
  end card (89.0 s → end) copied from the source as a keep segment. The final has 2845
  frames, the same as the source.
- **Cost ≈ 3,100 cr ≈ 15.5 USD**: GPT Image 470 cr (45 keyframes + a second "wedding"
  character sheet + 1 regeneration), Kling chunks 2–7 2,166 cr (all first try), chunk 1
  456 cr (reused from the 15 s pilot), Gemini a few tens of credits.
- **Cut timing** (max / mean |Δ| per chunk): 0.22/0.12, 0.10/0.06, 0.53/0.14, 0.60/0.15,
  0.28/0.14, 0.08/0.04, 0.02/0.01 s. The two errors above 0.3 s are 0.67 s source shots
  that Kling merged into a neighbour; every other cut is within 0.3 s, with no extra cuts.
- **Seams**: all 6 chunk boundaries sit on source hard cuts; no visible jump in look,
  outfit or lighting across them. Chunk-to-chunk continuity references (last frame of the
  previous chunk) are therefore **not needed while chunk boundaries are on hard cuts**.
- **Burned text** in the generated part: none; app screens legible.
- **Defects**:
  1. The copied end card still carries the source's burned captions.
  2. Identity blending: at ~84 s (5.9 s aisle walk, shot 51) the hero morphs into the
     friend; some close-ups give the hero the friend's nose.
  3. Shot 15 came out empty: the hero is missing although he is in the keyframe.
  4. Shots shorter than ~0.7 s can be merged into a neighbour by the model (the two cut
     errors above).
  5. Storyboard alignment: a whole-video Gemini call and per-chunk calls both drifted by
     ±1 shot against the real boundaries; per-shot calls (the shot's own sub-clip + its
     mid frame) are aligned by construction.
  6. `gemini-3.1-pro` answered `400 Failed to get the file information` for a 4 MB / 89 s
     proxy; `gemini-3-pro` accepted it; 3.1-pro worked on uploads under 1 MB.
- **What worked**: one character sheet per outfit/setting, chosen per chunk (a "wedding"
  sheet for the wedding chunks).
- **Verdict**: shippable after one review pass (a clean end card + a re-roll of the bad
  chunk). A re-roll of one chunk costs ~285–430 cr at 720p.

**Design changes from this pilot** (revision 5): per-shot storyboard calls (§5.2, defect 5)
with the 3.1-pro → 3-pro fallback on the file error (defect 6); short shots folded at
chunking (§5.1, defect 4); per-outfit/setting character sheets chosen per chunk (§5.3,
§6.1); the end card problem surfaced at launch with a clean library end card as the
default fix (§6.1, Q2, defect 1); a per-shot character check in QA and "re-roll this chunk"
as the repair unit (§6.5, defects 2–3); chunk boundaries off a hard cut warned in the
Segment Editor (seams).

## 14. Verification plan

- Unit: `build_chunks` (cuts, long-shot split, 8-shot cap, tail rebalance, brand-card keep),
  shot clipping inside a chunk (§6.2), prompt rendering per model, duration request,
  fit/pad/too-short decisions, payload builders, estimates.
- **v2v byte-identity**: every existing payload test unchanged; a new test that a v2v run
  on every model still builds exactly today's payloads with the new code present.
- Paid-call protocol (§7.2), each on Postgres: crash/stall at every step boundary
  (before the `sending` commit, after it and before the POST, after the POST and before
  recording the id, during fetch/fit), a reaper tick at each of those points, the late
  `accepted` adopting an `unknown` row, the stalled sender giving up by its own clock, a
  `create_task_once` outcome matrix (200, 4xx, 200+code, 5xx, read timeout), and a
  structural test that no rebuild code path reaches `_create_task_with_retry`.
- Run machinery: `process_images_only_run` idempotence under repeated recovery,
  `_repoll_generating_segments` / `_build_submit_state` / `_resubmit_segment_task` early
  returns, Retry/rerun estimates, every refused endpoint of §6.6 via direct API calls.
- Launch preconditions, `status == ready` admissions vs re-analysis in both orders,
  selection uniqueness under concurrent auto-select + manual select on Postgres.
- Fit: cumulative frame quantisation over many fractional chunks (no drift at the end of a
  94.9 s timeline), capped/fractional fps, short-result tolerance.
- Migration + rollout: upgrade from the head on a schema copy; the labels unused in the
  revision; downgrade refused with rows present; the §7.6 sequence and the quarantine
  script on a disposable database with representative rows.
- Rendered HTML across the project-page state matrix (the localisation lesson: HTMX
  lifecycle bugs are invisible to unit tests).
- Pilot through the app: 81d0afc0 end to end, compared with the manual full-video pilot.

## 15. Open questions for Operator

**Decided by Operator 2026-11-03 — every proposal accepted:** Q1 720p default · Q2 clean
language-matched library end card · Q3 one character sheet per outfit/setting group,
auto-picked per chunk · Q4 keyframe = the shot's midpoint · Q5 chunks ≤ 14.72 s / ≤ 8 shots
for both models · Q6 launching the run approves the keyframes · Q7 on-camera speech: warn and
allow · Q8 `Pipeline Type` stays `re-skin-v2` for now · Q9 the character check runs
automatically. The questions below are kept for the record.


- **Q1. Default resolution**: 720p (≈ 3000 cr for 81d0afc0) or 1080p (≈ 4460 cr)?
- **Q2. End card**: the pilot showed the source end card keeps its burned captions. Default
  to a clean language-matched library end card (proposed, §6.1), or run the end card through
  subtitle removal, or keep the source footage?
- **Q3. Character sheets**: one per outfit/setting group, auto-picked per chunk (proposed,
  worked in the pilot) — or one sheet per project, or none?
- **Q4. Keyframe frame**: the shot's midpoint (pilot) or its first frame? Kling "starts from"
  the keyframe; the first frame may hold the composition at the cut better.
- **Q5. Chunk length**: 14.72 s / 8 shots for both models (Seedance-safe; proposed), or
  up to 15.2 s / 9 shots for Kling-only runs as in the full pilot (7 chunks, invisible
  seams), or longer (untested beyond 16 s)?
- **Q6. Approval**: is launching the run the approval of the keyframes (proposed), or do you
  want an explicit "Approve keyframes" step?
- **Q7. On-camera speech** in phase 1: warn and allow (proposed), or refuse?
- **Q8. Final Creatives `Pipeline Type`**: keep `re-skin-v2` for rebuild runs (no downstream
  change; the model tag `KL4` in Idea already distinguishes Kling), or a new value such as
  `re-skin-rebuild` (needs ai-fb-audit to accept it)?
- **Q9. Character check** (§6.5): run the per-attempt Gemini check automatically (one LLM
  call per chunk, proposed), or only on demand?

## Review record

**Round 1 — GPT-6.1 Terra (Codex, reasoning high), revision 1 @ `25a9e95`: FAIL** (2 BLOCKER,
11 MAJOR, 1 MINOR, 1 PERF). Resolved in revision 2:

| # | Finding | Resolution |
|---|---|---|
| 1 | BLOCKER: `_create_task_with_retry` re-sends inside the client, the marker cannot stop it | `create_task_once` with an accepted / refused / unknown classification (§7.2); test that no rebuild path reaches the retrying client |
| 2 | BLOCKER: in-poll resubmit keeps the old task id, the marker test misses the new one | no resubmits at all in images-only mode; one row per provider request (`rebuild_chunk_attempts`) (§6.4, §7.2) |
| 3 | MAJOR: v2v resume path bypasses fit and re-buys on a failed download | images-only runs never enter v2v submit/resume; the attempt's `fetch` step is the only completion path and never creates (§6.4, §7.3) |
| 4 | MAJOR: claim fencing does not fence the external call | fenced CAS into `sending` + `send_deadline`; the sender gives up by its own clock; the reaper declares `unknown` only after deadline + HTTP timeout; a late id adopts the row (§7.2) |
| 5 | MAJOR: batch liveness/dispatch underspecified | batch dispatch dropped; per-row step jobs with deterministic ids, per-row `dispatch_gen`, reaper per state (§7.2) |
| 6 | MAJOR: selected-keyframe unique index broken by NULL `owner_run_id` | two partial unique indexes (§4) |
| 7 | MAJOR: old images cannot read new enum labels; downgrade destroys records | `REBUILD_ENABLED` flag, rollout order incl. the assistant supervisor, no image rollback once rows exist, quarantine script, downgrade refused with rows (§4, §7.6) |
| 8 | MAJOR: custom stitch plans break source-audio alignment | **Rejected**: under a custom plan `_run_stitch_and_deliver` muxes each swap clip's OWN source range (`media.mux_original_audio(clip, source, sd.start_sec, …)`, `pipeline_v2.py` ~3164) and concatenates per-clip audio, so an omitted or reordered chunk keeps its matching sound. Fact 2.1.6 now states this. |
| 9 | MAJOR: per-chunk frame rounding accumulates | cumulative quantisation on the stitch's frame grid, `chunk_frames = F(end) − F(start)` (§6.4) |
| 10 | MAJOR: unsupported run mutations still reachable | enumerated server-side refusals with direct-API tests (§6.6) |
| 11 | MAJOR: draws can be missed; estimates use v2v rates; resubmits unreserved | draw committed with the `sending` marker, keyed by row; one images-only estimator for launch/Retry/rerun; no automatic re-buy (§7.2, §7.3) |
| 12 | MAJOR: re-analysis can race admissions | every rebuild admission requires `status == ready` under the project lock; re-analysis refused while any rebuild row is live (§5.1.7–8) |
| 13 | MAJOR: take archive moving the raw file is unsafe | attempts are immutable files; AICP-342 take archive/restore refused for images-only runs, replaced by "use attempt N" (§4, §6.6) |
| 14 | MINOR: v2v length claim wrong | fact 2.1.6 corrected |
| 15 | PERF: serial batch + QA on the two run workers | short per-row steps; QA as a separate short job on `autopilot_compute`. Scale: today a handful of rebuild projects per day (≤ ~55 keyframes + ~7 chunks each); each step is one HTTP call, one download or one ≤ 15 s ffmpeg fit, so two concurrent projects add ≤ ~110 short jobs interleaved with run jobs. Load test kept in §14's pilot; not a gate blocker. |

**Round 2 — GPT-6.1 Terra (Codex resume, reasoning high), revision 2 @ `5efe045`: FAIL**
(1 BLOCKER, 6 MAJOR). Terra withdrew round-1 #8 after checking the cited code. Resolved in
revision 3:

| # | Finding | Resolution |
|---|---|---|
| R2-1 | BLOCKER: a sender paused after its deadline check can send after the reaper declared `unknown` | elapsed time is no longer a proof: a `sending` row becomes `unknown` only when the sender is proven terminated (container boot id changed, or RQ recorded the sending job as ended); otherwise it stays live (§7.2) |
| R2-2 | MAJOR: a late acceptance conflicts with a replacement / deleted rows | follows from R2-1: `unknown` exists only when no sender can still answer; `sending` rows count as live for deletion, re-analysis and the live-attempt index (§7.2) |
| R2-3 | MAJOR: dispatch was placed in `process_run`, not on the real `run_submit_run` path | branch in `tasks.run_submit_run` after the lock; observer chain with the lock handed over; tests through `enqueue_process_run` for launch and recovery (§6.4) |
| R2-4 | MAJOR: the observer repeated `queued → processing` | first entry (`submit_images_only`) vs pure observation (`run_rebuild_observe`) split (§6.4) |
| R2-5 | MAJOR: a rerun of a completed chunk estimated at zero | the estimator counts every `pending` chunk segment regardless of older selected attempts; reset + reserve in one transaction; test at the budget limit (§7.3) |
| R2-6 | MAJOR: keep segments not quantised cumulatively | images-only finalize cuts keep segments on the same frame grid (`rebuild.cut_keep_frames`) (§6.4) |
| R2-7 | MAJOR: flag flip without the nginx restart → 502 | flag change uses the full deploy arguments and restarts nginx afterwards; same for turning it off (§7.6) |

**Round 3 — GPT-6.1 Terra (Codex resume, reasoning high), revision 3 @ `b61ff7b`: FAIL**
(3 MAJOR). Terra confirmed the rq 1.15 grounding of the sender proof. Resolved in revision 4:

| # | Finding | Resolution |
|---|---|---|
| R3-1 | MAJOR: a recreated container gets a new hostname, so the boot-id proof never fires and the row stays `sending` | restart vs recreation defined; recreation is released by a host-level attestation script that checks `docker ps -a` for the old container id; tests for restart, recreation, freeze, horse reap, abandoned cleanup (§7.2) |
| R3-2 | MAJOR: a fresh launch estimates zero because RunSegments are created later | images-only launch creates the pending RunSegments in the reservation transaction (§6.1, §6.4) |
| R3-3 | MAJOR: Retry Run does not reset failed segments | the images-only Retry branch resets failed chunk segments to `pending` in the reservation transaction; endpoint tests (§7.3) |


**Round 4 — GPT-6.1 Terra (Codex resume, reasoning high), revision 4 @ `6984c8f`: PASS —
NO FINDINGS.** Design gate passed under the review-gate bar (no BLOCKER/MAJOR correctness
finding). Later changes (e.g. folding in §13) are reviewed as a delta from `6984c8f`.

**Round 5 — GPT-6.1 Terra (Codex resume, reasoning high), revision 5 @ `6b09552` (delta from
the approved `873161e`): FAIL** (6 MAJOR, all in the pilot-driven additions). Resolved in
revision 6:

| # | Finding | Resolution |
|---|---|---|
| R5-1 | MAJOR: recovery could re-send a billed character check | every LLM call is a `rebuild_llm_calls` row under the §7.2 protocol; only the free QA part is re-enqueued (§6.5) |
| R5-2 | MAJOR: LLM fallbacks and the QA call not covered by the reservation | storyboard reserves `(1 + N) × 2` calls and stops at the reservation; the run estimator adds `2 × llm_credits()` per chunk (§5.2, §7.3) |
| R5-3 | MAJOR: a revoked storyboard sender could still bill a shot the new Build calls again | per-call rows with the sender proof; a shot whose call is `sending` waits, `unknown` is re-called only on explicit Build (§5.2) |
| R5-4 | MAJOR: reused shot results not tied to the global context | Resume copies the global result + `global_call_id`; only results made under that id are reused or applied; "From scratch" reuses nothing (§5.2) |
| R5-5 | MAJOR: folding could cycle and leave gaps | deterministic anchor-group partition covering the timeline, chunking never splits a group, tests (§5.1.3) |
| R5-6 | MAJOR: the end-card default dropped middle keep segments | only the trailing all-brand-card keep run is replaced, and only when a language-matched library clip exists (§6.1) |

**Round 6 — GPT-6.1 Terra (Codex resume, reasoning high), revision 6 @ `4024f32`: FAIL**
(5 MAJOR). Resolved in revision 7:

| # | Finding | Resolution |
|---|---|---|
| R6-1 | MAJOR: Gemini calls go through `_chat_with_retry` | `chat_completion_once` with accepted/refused/unknown; structural test (§5.2) |
| R6-2 | MAJOR: Resume could duplicate a queued/claimed shot call | interrupting a storyboard cancels its `queued`/`claimed` call rows atomically; DB partial unique indexes for live shot/global calls; Build again refused while any call is `sending` (§5.2) |
| R6-3 | MAJOR: Retry/rerun releases the reservation of a delayed character check | the estimator counts outstanding checks of succeeded attempts, so `release_unused` keeps them reserved; race test (§7.3) |
| R6-4 | MAJOR: an overlong effective-shot group conflicts with the chunk cap | groups are capped during folding (buffer split, anchor split into pieces) (§5.1.3–4) |
| R6-5 | MAJOR: Resume undefined when the global call was interrupted | Resume defined per global-call state: `sending` waits, `unknown`/`failed` → new global call, `succeeded` → copy (§5.2) |

**Round 7 — GPT-6.1 Terra (Codex resume, reasoning high), revision 7 @ `62f6a91`: FAIL**
(2 MAJOR). Resolved in revision 8:

| # | Finding | Resolution |
|---|---|---|
| R7-1 | MAJOR: the QA estimate double-counts a check already drawn at `sending` | per-slot undrawn remainder (primary, fallback), computed under the reservation's day lock; tests at the cap (§7.3) |
| R7-2 | MAJOR: trailing shorts after the last anchor could exceed the cap | trailing shorts join the last group only within the cap, else form their own capped group; tests (§5.1.3) |

**Round 8 — GPT-6.1 Terra (Codex resume, reasoning high), revision 8 @ `14efae4`: FAIL**
(1 MAJOR). Resolved in revision 9:

| # | Finding | Resolution |
|---|---|---|
| R8-1 | MAJOR: `reserve_run` estimates before `reserve` takes the day lock, so the QA estimate can go stale | `spend.reserve_computed` takes the day locks first (ascending, incl. re-baselined days) and computes the estimate inside them with lock-free reads; both interleavings + a two-day scope tested (§7.3) |

**Round 9 — GPT-6.1 Terra (Codex resume, reasoning high), revision 9 @ `f8b6084`: FAIL**
(1 MAJOR). Resolved in revision 10:

| # | Finding | Resolution |
|---|---|---|
| R9-1 | MAJOR: a draw picks its day before the day lock and can land on an allowance a concurrent re-baseline moved | per-scope advisory lock taken before day selection by both the draw and `reserve_computed` for this feature's scopes; v2v untouched; interleaving test (§7.3) |


**Round 10 — GPT-6.1 Terra (Codex resume, reasoning high), revision 10 @ `02ba442`: PASS —
NO FINDINGS.** Design gate passed again after the full-video pilot changes (rounds 5–10
reviewed only the delta from the round-4 approval `873161e`).
