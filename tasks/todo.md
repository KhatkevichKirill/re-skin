# re-skin — Orchestration Tracker

Repo: github.com/KhatkevichKirill/re-skin (private, SSH push)
Stack: FastAPI + RQ/Redis + HTMX/Jinja + SQLite + Docker Compose + Nginx
Orchestrator: Opus (review each PR). Owner/tester: user.

## Environment facts (verified 2026-06-19/20)
- git 2.34.1, SSH auth OK as KhatkevichKirill; remote repo exists.
- `gh` NOT installed → feature branches + local `git diff main..branch` review + merge (no PR CLI).
- Docker 26.1.3 daemon running; only `docker-compose` v1.29.2 (no `docker compose` v2 plugin).
- Python 3.10.12. FFmpeg 4.4.2 present. InsightFace (buffalo_l) installed & working.
- Secrets present locally and MUST stay gitignored: `.env`, `secrets/gdrive-sa.json`.

## Dependency graph
T1 → T2 → (T3, T4, T5, T6 parallel) → T7 → T8 → T9 → T10
Checkpoints for user acceptance: after T1, after T7 (full run, no UI), after T9 (full browser cycle).

## Tasks
- [x] **T1** Repo scaffold (Haiku) — **DONE** (adbcbf5, on main)
- [x] **T2** Data model + migrations + state machine (Sonnet) — **DONE** (merged 3729b3e)
- [x] **T3** Media module: ffprobe, accurate cut, stitch (normalize + original audio) (Sonnet) — **DONE** (merged 12c790b)
- [x] **T4** Face module: timeline, small-face filter, grouping, grow-back start, ≤15s split, pre/post-roll (Sonnet) — **DONE** (merged 8df2734)
- [x] **T5** kie.ai client: upload + Seedance create/poll/download + retries (Sonnet) — **DONE** (merged 8c27ae4)
- [x] **T6** Google Drive client: download by link, upload by folder id (service account) (Sonnet) — **DONE** (merged 6781c1f)
- [ ] **T7** Orchestrator/worker: pipeline over state machine, RQ queue, resumable, sequential (Sonnet) — **NEXT**
- [ ] **T8** REST API: create/proposal/edit/submit/status/result (Sonnet)
- [ ] **T9** HTMX frontend: ingest, segment review+edit, status, result preview/download (Sonnet)
- [ ] **T10** Deploy hardening: upload limits, long-job timeouts, logs, e2e on Erewhon (Haiku)

Full test suite on main: 138 passing.

## Review log
- **T1** (Haiku, adbcbf5) — APPROVED. Secrets untracked (only .env.example); scaffold matches spec; smoke 401(no auth)/200(auth) on :8847. Direct to main (empty repo).
- **T2** (Sonnet, 3729b3e) — APPROVED after fix. SQLAlchemy 2.0 + Alembic, Job/Segment, state_machine. Orchestrator caught Segment missing created_at/updated_at though transition() writes updated_at — fixed (+migration +test). 49 tests.
- **T3** (Sonnet, 12c790b) — APPROVED. media.py probe/cut_clip/cut_segments/stitch/get_default_target. Key stitch normalization test (mixed res/fps → target) passes. 68 tests.
- **T4** (Sonnet, 8df2734) — APPROVED. face.py: detect_timeline + pure logic (filter_small_faces, group_intervals, apply_lead_in grow-back, split_max_duration, propose_segments full partition). 107 tests.
- **T5** (Sonnet, 8c27ae4) — APPROVED. kie_client.py: upload/create_task/get_task/poll_task/download_result + tenacity retries. API key not hardcoded/leaked. 119 tests.
- **T6** (Sonnet, 6781c1f) — APPROVED. gdrive_client.py: extract_file_id (all link formats), download_file, upload_file; lazy SA build (import needs no creds); service injectable for tests. SA json not committed. 138 tests.
