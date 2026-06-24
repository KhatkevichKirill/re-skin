---
title: "Roadmap & Growth Plans"
tags: [roadmap, growth, features, vision]
sources: [tasks/todo.md]
updated: 2026-06-24
---

# Roadmap & Growth Plans

## Immediate: Complete v2 (on `v2` branch)

Three tasks remain before v2 ships:

| Task | Description | Why it matters |
|------|-------------|----------------|
| **TR6** | Per-segment prompt/reference overrides + single-segment re-run | Cheap recovery when one segment fails to swap |
| **TR7** | Audio mode per run: `original` vs `seedance` | Source audio sometimes desyncs due to clip-length drift from Seedance |
| **TR8** | Linked segment boundary editing | Editing one segment's end must move the next segment's start to prevent gaps/overlaps |
| **TR5b** | Remove v1, merge v2 → main, startup alembic, orphaned-run reconciliation | Clean deployment |

See `tasks/todo.md` for task-level detail.

## Post-v2 Vision

_This section is for growth ideas — add to it as plans develop._

### Scaling

- **Multi-worker (DONE on `feat/parallel-workers`)**: Worker service now supports `docker-compose up --scale worker=N`. N=2 is the recommended maximum on the current 4 vCPU / 16 GiB host (see [[components/parallel-workers]] for resource math). Requires [[decisions/postgres-migration]] to be deployed first.
- **Poll-decoupling refactor (TR-POLL)**: Each RQ worker holds its job slot during the entire Seedance poll loop (up to 2h). Decoupling submit → poll (self-re-enqueueing, lightweight queue) → stitch would free CPU workers during the wait. Deferred: significant state-machine work. Concrete design in [[components/parallel-workers]] → "Proposed Decoupling". Do after v2 ships.
- **TR5b — Orphaned-run reconciliation**: More workers = more container restarts = more orphaned runs. Detect runs in `processing`/`queued`/`polling` with no live RQ job on worker startup and re-enqueue them. See [[lessons/production-gotchas]] → "Worker crash leaves runs orphaned". Priority increases with N>1 workers.
- **Horizontal scaling**: If load grows beyond a single VPS, multi-node deployment. Postgres + Redis are already external to the app; the main work would be a shared NFS/object-storage mount for `./data/` (currently a local bind mount).

### UX / Workflow

- **Batch project creation**: Upload a folder of videos → create N projects at once. Useful for processing many similar videos.
- **Segment editor improvements**: Visual timeline scrubber, thumbnail previews at segment boundaries, keyboard shortcuts.
- **Result comparison view**: Side-by-side diff between two Runs on the same Project (useful when tuning prompt/refs).
- **Run templates**: Save a prompt + reference set as a named template, reuse across projects.
- **Webhook notifications**: Notify Slack/Discord when a run completes (currently operators poll the UI).

### AI Model Expansion

- **Additional models**: kie.ai supports multiple models beyond Seedance. Gemini Omni is already integrated. Room to add others as they mature.
- **Model auto-selection**: Automatically pick the best model per segment based on segment duration, face size, resolution. Seedance has a ~1.8s floor; Gemini Omni handles longer clips differently.
- **Quality scoring**: After generation, run a lightweight face-detection pass to score how well the swap worked. Surface score in the UI; auto-flag failed swaps for re-run.

### Production Hardening

- **Startup alembic**: Run `alembic upgrade head` automatically on api startup (TR5b). Currently requires manual run on deploy.
- **Orphaned run reconciliation**: Detect runs stuck in `processing` from a previous worker crash and reset them to `queued` on startup.
- **Monitoring**: Grafana/Prometheus integration for queue depth, processing time per segment, error rates.
- **Disk management**: Auto-delete intermediate clips (keep only final.mp4 + source) after successful delivery. Disk fills fast at 1080p.

### Business / Product

_Fill in as plans develop._

---

## Ideas Parking Lot

_Unvetted ideas — capture here before deciding whether to promote to a real plan._

- Per-project reference image library (reuse refs across runs without re-uploading)
- Mobile upload (allow phone video upload directly to re-skin)
- Custom segment duration cap per project (currently global 15s; some content needs longer)
- Preview generation: low-res preview of a swap segment before processing the whole run
