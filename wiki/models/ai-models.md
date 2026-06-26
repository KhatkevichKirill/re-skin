---
title: "AI Models: Seedance & Gemini Omni"
tags: [ai-models, seedance, gemini, kie-ai]
sources: [backend/app/kie_client.py, backend/app/pipeline_v2.py, tasks/todo.md]
updated: 2026-06-24
---

# AI Models

re-skin supports two AI face-swap models, selectable per Run.

## Seedance (via kie.ai)

**Primary model.** Accessed via the kie.ai API wrapper.

### How It Works

1. Upload source video clip to kie.ai (`KieClient.upload()`) → returns `kie_upload_url`
2. Create Seedance task (`KieClient.create_task(kie_upload_url, prompt, reference_image_urls, resolution)`) → returns `seedance_task_id`
3. Poll task status (`KieClient.get_task(task_id)`) every 15s until done or failed
4. Download result (`KieClient.download_result(result_url)`) → local MP4

### Known Constraints

| Constraint | Value | Notes |
|-----------|-------|-------|
| **Minimum segment duration** | ~1.8s | Seedance rejects clips shorter than this; pipeline enforces `min_segment_sec` floor + pre/post-roll padding |
| **Maximum segment duration** | ~15s | Configurable via `SEGMENT_MAX_SECONDS`; face.py splits longer face intervals |
| **Clip audio** | Optional | Seedance can generate with or without original audio; `audio_mode` controls this |
| **Resolutions** | 480p, 720p, 1080p | Set at Run creation; 1080p requires more worker memory |
| **Typical generation time** | 2–10 min per segment | Varies by resolution and server load |
| **Task timeout** | 2h (configurable) | Segments exceeding `RUN_SKIP_TIMEOUT_SEC` are skipped with original clip |

### API Key

`KIE_API_KEY` in `.env`. Accessed via `KieClient` (initialized in pipeline, injected into worker).

### Retry Logic

`KieClient` uses `tenacity` for exponential backoff on HTTP errors. Separate from the run-level retry (`/retry` endpoint).

## Gemini Omni (Google)

**Alternative model.** Per-run selectable via `model=gemini-omni` in the Run creation payload (accessed through the same kie.ai jobs API, `model=gemini-omni-video`).

### How It Works

Unlike Seedance, Gemini takes its reference clip via `video_list` (`{url, start, ends}`, trim ≤10s) plus reference images via `image_urls`, and returns a fixed-length output (one of 4/6/8/10s). Submitted by `KieClient.create_omni_task`.

### Known Constraints

| Constraint | Value | Notes |
|-----------|-------|-------|
| **Max clip / output duration** | **10s** (hard) | `OMNI_MAX_CLIP_SECONDS` in `pipeline.py`. Trim range and output duration must both be ≤10s. |
| **Output durations** | 4, 6, 8, 10s | Fixed enum; clip length is snapped to the nearest via `_snap_omni_duration`. |
| **Audio** | **None** — must upload video-only | Gemini **fails when its reference clip carries an audio track**. Clips are cut with `include_audio=False` (`-an`); the original audio is re-applied at stitch. |
| **Resolutions** | 720p, 1080p, 4k | No 480p (unlike Seedance). |
| **Aspect ratios** | 16:9 or 9:16 only | `_map_omni_aspect` maps by orientation. |

### Audio handling (important)

Gemini produces **no audio**. The pipeline therefore:
1. Cuts each swap clip **video-only** before upload (`media.cut_clip(..., include_audio=False)`).
2. **Forces `audio_mode="original"`** for Gemini runs (in `process_run` and at Run creation) so the continuous source soundtrack is muxed over the final stitch. A requested `audio_mode="seedance"` is ignored for Gemini.

### 10s limit enforcement

The model is chosen **per-Run**, but a Project's segmentation is shared across all runs, so `analyze_project` caps segments at `min(SEGMENT_MAX_SECONDS, OMNI_MAX_CLIP_SECONDS)` = **10s** — every swap segment fits both Seedance (≤15s) and Gemini (≤10s) regardless of which model a run picks. At submit time:
- Pre/post-roll that pushes a clip past 10s → the clip is **trimmed back to 10s**.
- A *segment* (without rolls) longer than 10s — only possible from a stale segmentation built with a larger cap — is **skipped** (marked failed → the original un-swapped clip is used) rather than swapping only its first 10s and desyncing the timeline.

### Configuration

Set `model=gemini-omni` when creating a Run via the API or UI. The UI rebuilds the resolution dropdown and locks the audio control to "original" when Gemini is selected.

_Add quality learnings to [[lessons/model-quality]]._

## Choosing Between Models

_Fill in as production experience develops._

| Scenario | Recommended Model | Reason |
|----------|------------------|--------|
| Short clips (<5s) | Seedance | More consistent at short durations |
| Long clips (>10s) | Seedance | Gemini Omni's hard 10s cap means longer segments would be skipped (original clip used) |
| High-res (1080p) | TBD | Memory/quality tradeoffs unclear |

## Future Models

kie.ai exposes multiple models; Seedance is the current primary. See [[roadmap]] for the model expansion plan.
