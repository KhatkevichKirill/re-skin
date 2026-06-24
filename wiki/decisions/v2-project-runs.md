---
title: "ADR: v2 Project → Runs Model"
tags: [architecture, decision, v2, data-model]
sources: [docs/v2-project-runs.md, tasks/todo.md]
updated: 2026-06-24
---

# ADR: v2 Project → Runs Model

## Context

v1 couples everything into one `Job`: source video + segmentation + one character (prompt + refs) + result. Testing a different character on the same video forces re-upload + re-analysis + re-segmentation. For real usage, operators want to try 3–5 different characters on the same video efficiently.

## Decision

Decompose the Job into:
- **VideoProject** — holds the source video and the segmentation (done once)
- **Run** — one character attempt on a Project (prompt + refs + model → one result)

Multiple Runs share the same Project segmentation. The video is downloaded/uploaded once.

## Key Choices Made

| Question | Decision | Rationale |
|----------|----------|-----------|
| One character per run? | Yes | Simpler mental model; easy to compare character A vs B |
| Per-segment prompt/ref overrides on RunSegment? | **Yes (TR6, reversed)** | Initially dropped for simplicity; re-added because some segments need tuning independently |
| Run label? | Optional `name` field | e.g. "Redhead woman" — helps distinguish runs in the list |
| Audio mode? | Per-run: `original` or `seedance` | `original` = mux full source audio (clean); `seedance` = use Seedance clip audio (avoids drift) |
| Process strategy? | **Parallel submit + concurrent poll** | All swap segments uploaded and submitted at once; round-robin poll at 15s intervals; 2h skip timeout per segment |
| Branch strategy? | v2 branch, additive alongside v1 | v1 stays deployed and green; v2 merged to main only after full e2e verified |
| Migration? | Clean schema (no complex migration) | No real data to preserve at decision time |
| Startup alembic? | Deferred to TR5b | Not needed for v2 branch; added during v1 cleanup |

## Implementation Notes

- **Parallel submit**: All swap segments for a Run are uploaded to kie.ai and submitted as tasks simultaneously. Unlike v1 which processed segments sequentially.
- **Skip timeout**: If a Seedance task exceeds 2h, the segment is skipped (uses original clip) rather than blocking the whole run. Controlled by `RUN_SKIP_TIMEOUT_SEC`.
- **Audio modes**: `original` muxes the full source audio track (ignores segment audio); `seedance` uses clip audio from Seedance and source audio for keep segments. The `seedance` mode avoids the drift issue where Seedance outputs are sometimes slightly shorter/longer than the original clip, causing desync when muxed with the source audio.
- **SegmentDef reuse**: SegmentDefs are created once during `analyze_project` and shared across all Runs. RunSegments are created at Run creation time (one per swap SegmentDef).

## What Didn't Change

All v1 modules (media.py, face.py, kie_client.py, gdrive_client.py, storage.py) were reused as-is. Only the pipeline orchestration, API routes, and HTML templates changed.

## Outcome

Full e2e verified (TR5a): 3 swap segments submitted in parallel, all completed, stitched at 1080p, drive delivery confirmed. Parallel processing confirmed working.

See [[architecture]] for the data model diagram.
