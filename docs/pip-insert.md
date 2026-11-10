# PIP Insert Extraction & Reinsertion — Design Spec

**Status**: agreed 2026-07-14 (discussion with operator), not yet implemented
**Scope**: Seedance only. Static rectangle. One insert per video. Gemini Omni explicitly out of scope for v1.

## Problem

UGC-style source videos (TikTok/Reels ads) often contain a small picture-in-picture
insert (a "reaction" clip) overlaid on the main frame — sometimes square-cornered,
sometimes rounded with a visible border. We need to re-skin (face-swap) **only the
insert's content** via Seedance without touching the rest of the video, then get a
final video where the processed insert sits back at the exact same coordinates and
time window, with the original audio intact.

## Core decision: insert = child project + merge, not a segment layer

The extracted insert becomes a **normal child `VideoProject`** that flows through the
existing analyze → segment review → Run (Seedance submit/poll/stitch) machinery
completely unchanged. Two new operations bracket that flow:

1. **Extract** — crop + trim the insert region out of a parent project's source into
   a new child project, recording provenance + placement.
2. **Merge** — composite a chosen child Run's `final.mp4` back onto a target video at
   given coordinates/time window, applying a geometric corner mask, keeping the
   target's original audio.

Why this shape (vs. an "insert layer" inside a run):

- **Maximum reuse.** Chunking to Seedance's 4–15s duration caps, min-duration floor,
  concurrent submit/poll, retry/skip/resume, stitching, per-run character attempts,
  reference handling — all already exist and apply verbatim to the child project.
  `stitch()` already conforms the result back to the child source's WxH/fps
  (`get_default_target`), which *is* the "match fps/resolution" anti-drift step.
- **Timeline invariant untouched.** `SegmentDef`s tile the timeline; a PIP insert
  overlaps keep/swap segments in time. Keeping it out of the segment model means
  `analyze_project` and `stitch` need zero changes.
- **Reuse across videos.** The same processed insert (child run result) can be merged
  into *different* target videos/projects — placement lives on the merge, not on the
  insert. The extraction's placement is just the default.
- **Operator review.** The child run is reviewed as a normal run before anyone merges
  it — no blind compositing.

## Data model (additive migration)

```
insert_extractions
  id                 uuid PK
  parent_project_id  FK video_projects  (index)
  child_project_id   FK video_projects  (unique — one extraction per child)
  x, y, width, height   int   -- crop rect in parent source pixels
  start_sec, end_sec    float -- insert's time window in the parent
  corner_radius_px      int   default 0
  created_at

insert_merges
  id                 uuid PK
  target_project_id  FK video_projects  (the video being merged INTO; defaults to
                                         the extraction's parent, but can be any project)
  child_run_id       FK runs            (must be status=done with final.mp4)
  x, y, width, height, start_sec, end_sec, corner_radius_px
                                        (prefilled from the extraction, editable —
                                         a different target video may place it elsewhere)
  status             enum: queued | compositing | delivering | done | failed
  output_path        text  -- data/projects/<target>/merges/<merge_id>/merged.mp4
  gdrive_url         text
  error_message      text
  created_at, updated_at
```

`VideoProject` itself is untouched; a child project is a regular project whose
`source_local_path` points at the extracted crop clip. The extraction row is the
back-link (UI shows "extracted from <parent>" via it).

## Pipeline

### Extract (fast, runs in API worker or a short RQ job)

```
ffmpeg -ss START -i parent_source.mp4 -to (END-START)
       -vf "crop=W:H:X:Y" -c:v libx264 -preset fast -crf 18 -c:a aac ...
```

- **Crop rect = inner content, inside any border.** The original border/frame pixels
  stay in the base video and never round-trip through Seedance — no shimmer, no mask
  complexity for the border. Operator draws the rect inside the border in the UI.
- **W/H rounded down to even** (libx264/yuv420p requirement). X/Y can be odd.
- Keep audio in the child clip (Seedance clips keep audio; irrelevant at merge time
  since merge uses the target's audio).
- Then create the child `VideoProject` (source_type `upload`, source_local_path =
  crop path), write the `insert_extractions` row, enqueue `analyze_project`.
- Aspect ratio: the crop is typically ~1:1 or 4:5; `_map_aspect` in
  `pipeline_common.py` already returns `"adaptive"` for anything outside the Seedance
  set — no changes needed.
- Upscale/downscale asymmetry works in our favor: a small crop (e.g. 424×522) goes up
  to a Seedance resolution tier for generation and is downscaled back to insert size
  at merge, which hides generation artifacts.

### Child project processing — existing flow, zero changes

analyze → operator reviews segments → create Run (Seedance, prompt + refs) →
process_run → child `final.mp4` at child source WxH/fps.

Note: `analyze_project` caps segments at 10s (the shared Seedance/Omni cap), so an
insert visible for 30–60s becomes several sequential swap segments. Identity/lighting
may jump at chunk boundaries inside the insert — same character-consistency behavior
as the main pipeline, just more visible in continuous PIP footage. Accepted for v1.

### Merge (RQ job, `process_merge(merge_id)`)

Single ffmpeg pass over the target video:

```
[1:v] scale=W:H, setsar=1, fps=TARGET_FPS, trim/tpad to exact (END-START) [ins]
[ins][2:v] alphamerge [insa]                     # 2 = rounded-rect mask PNG (skip if radius=0)
[0:v][insa] overlay=X:Y:enable='between(t,START,END)' [v]
-map [v] -map 0:a -c:a copy -c:v libx264 -preset fast -crf 18 -threads $FFMPEG_THREADS
```

- **Conform before overlay** (scale + `setsar=1` + `fps=` + exact-duration trim, or
  `tpad=stop_mode=clone` if the result runs a few frames short) — this is what
  prevents timing drift and size mismatch against the base.
- **Mask is geometric, not extracted from pixels**: a rounded-rect alpha PNG rendered
  from `(W, H, corner_radius_px)` with cv2/numpy (already available via InsightFace
  deps), stored under the merge dir. Deterministic, operator-tunable. radius=0 →
  plain overlay, no alphamerge.
- **Audio: `-map 0:a -c:a copy`** — the target's audio is never decoded, generation
  doesn't touch sound.
- Output → `merges/<merge_id>/merged.mp4` → deliver to GDrive with the existing
  `_deliver_with_retry` pattern → status `done`.
- Memory: one 1080p decode + one small overlay input; cheaper than stitch. Respect
  `FFMPEG_THREADS` and `FFMPEG_MAX_FPS` conventions from `media.py`.

## media.py additions

- `crop_clip(src, start, end, x, y, w, h, dst)` — trim + crop, even-dim rounding.
- `make_rounded_mask(w, h, radius, dst_png)` — cv2/numpy alpha mask.
- `overlay_insert(base, insert, dst, *, x, y, start, end, w, h, fps, mask_png=None)` —
  the conform + alphamerge + overlay + audio-copy command above.

All follow the existing style: shell out, `MediaError` with stderr tail, logged.

## API / UI (v2, HTMX)

- **Parent project page → "Extract insert"**: frame preview (frames already extracted
  to `data/projects/<id>/frames/`) with a draggable/resizable rectangle; inputs for
  start/end sec and corner radius; submit → child project created + analyze enqueued;
  redirect to child project page with an "extracted from <parent>" banner.
- **Child run page (status=done) → "Merge back"**: target project selector (default:
  extraction parent), placement fields prefilled from the extraction, editable for a
  different target; submit → enqueue `process_merge`; merge status block on the
  target project page (HTMX poll, like runs).
- Endpoints: `POST /v2/projects/{id}/extract-insert`,
  `POST /v2/runs/{run_id}/merge` (+ merge status/list partials).

## Out of scope v1 (explicit)

- **Gemini Omni** for insert runs (its 16:9/9:16-only aspect constraint needs a
  pad-or-overcrop strategy; revisit if needed).
- **Animated/moving inserts** — static rect only.
- **Multiple inserts per video** — schema doesn't forbid it, UI/pipeline assume one.
- **Auto-detection** of the rect — ship manual-first. Planned v1.5 assist: average an
  edge map over sampled frames (cv2) so the persistent rectangle border pops out of
  moving content, contour-detect the axis-aligned rect, prefill the UI; timing found
  by checking which frames match the border signature.

## Suggested milestones

1. **M1** `media.py`: crop_clip / make_rounded_mask / overlay_insert + unit tests
   (synthetic clips, like existing media tests).
2. **M2** Models + Alembic migration (`insert_extractions`, `insert_merges`,
   merge-status enum).
3. **M3** Extract endpoint + UI (rect-drawing over frame preview) + child project
   creation.
4. **M4** `process_merge` RQ job + merge UI + GDrive delivery.
5. **M5** (optional) auto-detect assist.

E2E acceptance: extract the PIP from a real UGC source (see example project
`a172185c…`-style content), run Seedance on the child, merge back, verify no drift
(frame-step the boundary at start/end), corners/border intact, original audio bit-identical.
