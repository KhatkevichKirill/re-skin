# re-skin — Orchestration Tracker

Repo: github.com/KhatkevichKirill/re-skin (private, SSH push)
Stack: FastAPI + RQ/Redis + HTMX/Jinja + SQLite + Docker Compose + Nginx
Orchestrator: Opus (review each PR). Owner/tester: user.

## Environment facts (verified 2026-06-19)
- git 2.34.1, SSH auth OK as KhatkevichKirill; remote repo exists & empty.
- `gh` NOT installed → use git+SSH (`git@github.com:KhatkevichKirill/re-skin.git`).
- Docker 26.1.3 daemon running; only `docker-compose` v1.29.2 (no `docker compose` v2 plugin).
- Python 3.10.12. FFmpeg 4.4.2 present. InsightFace (buffalo_l) installed & working.
- Secrets already present locally and MUST stay gitignored: `.env`, `secrets/gdrive-sa.json`.

## Dependency graph
T1 → T2 → (T3, T4, T5, T6 parallel) → T7 → T8 → T9 → T10
Checkpoints for user acceptance: after T1, after T7 (full run, no UI), after T9 (full browser cycle).

## Tasks
- [ ] **T1** Repo scaffold (Haiku) — docker-compose, Nginx basic auth :8847, FastAPI /health, README, .gitignore, dir structure. **IN PROGRESS**
- [ ] **T2** Data model + migrations + job/segment state machine (Sonnet)
- [ ] **T3** Media module: ffprobe, accurate cut, stitch (normalize + original audio) (Sonnet)
- [ ] **T4** Face module: timeline, small-face filter, grouping, grow-back start, ≤15s split, pre/post-roll (Sonnet)
- [ ] **T5** kie.ai client: upload + Seedance create/poll/download + retries (Sonnet)
- [ ] **T6** Google Drive client: download by link, upload by folder id (service account) (Sonnet)
- [ ] **T7** Orchestrator/worker: pipeline over state machine, RQ queue, resumable, sequential (Sonnet)
- [ ] **T8** REST API: create/proposal/edit/submit/status/result (Sonnet)
- [ ] **T9** HTMX frontend: ingest, segment review+edit, status, result preview/download (Sonnet)
- [ ] **T10** Deploy hardening: upload limits, long-job timeouts, logs, e2e on Erewhon (Haiku)

## Review log
- (entries added as PRs land)
