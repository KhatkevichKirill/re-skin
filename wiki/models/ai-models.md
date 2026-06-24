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

**Alternative model.** Per-run selectable via `model=gemini-omni` in the Run creation payload.

### Status

Integrated and working. Some differences vs Seedance:
- Different resolution handling (aspect-ratio mapping differs)
- Different audio handling
- Different generation time characteristics

### Configuration

Set `model=gemini-omni` when creating a Run via the API or UI.

_This section needs expansion as production experience accumulates. Add learnings to [[lessons/model-quality]]._

## Choosing Between Models

_Fill in as production experience develops._

| Scenario | Recommended Model | Reason |
|----------|------------------|--------|
| Short clips (<5s) | Seedance | More consistent at short durations |
| Long clips (>10s) | TBD | Need more data |
| High-res (1080p) | TBD | Memory/quality tradeoffs unclear |

## Future Models

kie.ai exposes multiple models; Seedance is the current primary. See [[roadmap]] for the model expansion plan.
