# re-skin — CLAUDE.md

## Project Context

**re-skin** is a production AI video face-swapping service. Operators upload source videos; the system uses InsightFace to detect faces, proposes segments (swap/keep), lets the operator review and edit them, then submits swap segments to Seedance (via kie.ai) or Gemini Omni. Results are stitched and delivered to Google Drive.

**Current state**: v2 (Project → Runs model) is ~80% complete on the `v2` branch (TR6/TR7/TR8 in-flight). v1 is deployed to production and stable.

**Stack**: FastAPI + RQ/Redis + SQLite/SQLAlchemy + Jinja2/HTMX + FFmpeg + InsightFace + Docker Compose + Nginx

## Dev Environment

- Python 3.10.12, FFmpeg 4.4.2, InsightFace (buffalo_l), Docker 26.1.3
- Use `docker-compose` (v1.29.2) — NOT `docker compose` v2 plugin
- `gh` CLI is NOT installed — use `git` directly, push via SSH as KhatkevichKirill
- Tests: `cd backend && pytest tests/` (~350 tests)
- Deploy: `docker-compose up -d --build`
- **Never commit**: `.env`, `secrets/gdrive-sa.json`, `data/`
- Task tracker: `tasks/todo.md` — sprint-level task tracking lives there, not in the wiki

## LLM Wiki

`wiki/` is a persistent, LLM-maintained knowledge base. It follows Karpathy's "LLM Wiki" pattern: a compounding artifact that Claude writes and maintains while the user curates sources and steers the direction. The key property: knowledge is compiled once and kept current — not re-derived from scratch each session.

### What the wiki covers (vs other files)

| Layer | File | Purpose |
|-------|------|---------|
| Sprint tasks | `tasks/todo.md` | In-progress tasks, PR log, acceptance notes |
| Design docs | `docs/` | Formal design specs (immutable reference) |
| **Wiki** | `wiki/` | Architecture synthesis, growth plans, AI model learnings, lessons, roadmap vision |
| Raw sources | `raw/` | Ingested documents (articles, notes, transcripts) — never modified |

The wiki captures the WHY and WHERE-ARE-WE-GOING; the code captures the WHAT; `tasks/todo.md` captures the HOW-RIGHT-NOW.

### Directory Layout

```
wiki/
├── index.md            # Content catalog — update on every ingest or new page
├── log.md              # Append-only chronological log
├── overview.md         # Project overview, current state, trajectory
├── architecture.md     # System architecture, data model, deploy topology
├── roadmap.md          # Growth vision, feature ideas, post-v2 direction
├── decisions/          # Architecture Decision Records (one file per decision)
│   └── *.md
├── components/         # Per-component deep dives
│   └── *.md
├── models/             # AI model (Seedance, Gemini Omni) evaluation & characteristics
│   └── *.md
└── lessons/            # Production learnings, gotchas, post-mortems
    └── *.md

raw/                    # Immutable source documents — read but never written by LLM
├── articles/
├── notes/
└── transcripts/
```

### Page Format

Every wiki page should have YAML frontmatter:

```yaml
---
title: "Page Title"
tags: [architecture, pipeline, v2]
sources: [docs/v2-project-runs.md]
updated: YYYY-MM-DD
---
```

Use `[[page-name]]` to cross-link pages (Obsidian-compatible). Cross-link liberally — the graph is the value.

### Operations

#### Ingest a new source

When the user drops content into `raw/` or pastes something and says "ingest this":
1. Read the source thoroughly
2. Discuss key takeaways with the user
3. Write a summary page in `wiki/` (or update the most relevant existing page)
4. Update `wiki/index.md` with any new pages
5. Touch every existing wiki page the source affects (update cross-refs, flag contradictions)
6. Append a log entry: `## [YYYY-MM-DD] ingest | Source Title`

#### Answer a query

When the user asks a project question:
1. Check `wiki/index.md` for relevant pages
2. Read the relevant pages
3. Synthesize an answer with citations (`[[page]]`, `docs/file.md`, `raw/file.md`)
4. If the answer is non-trivial (a comparison, an analysis, a decision), ask whether to file it back as a wiki page — good answers compound

#### Lint the wiki

When the user asks for a lint or health check:
1. Read all wiki pages
2. Flag: contradictions, stale claims, orphan pages, missing cross-refs, concepts mentioned but lacking their own page, data gaps worth a web search
3. Report a prioritized punch list — what to add, fix, or investigate

### Domain Glossary

| Term | Meaning |
|------|---------|
| **Segment / SegmentDef** | Contiguous time interval in source video, marked swap or keep. SegmentDef is the v2 version (reusable, shared across Runs) |
| **Run** | One character attempt on a Project: prompt + refs + model → one stitched result video |
| **Seedance** | AI video face-swap model accessed via kie.ai |
| **Gemini Omni** | Google's AI video model (alternative to Seedance, per-Run selectable) |
| **pre_roll / post_roll** | Extra seconds around a segment for Seedance context; trimmed from output |
| **analyze** | Pipeline phase: InsightFace detection → segment proposal |
| **process** | Pipeline phase: Seedance/Gemini submit → poll → stitch → deliver |
| **stitch** | FFmpeg concat of all segment results into one video with original audio |
| **buffalo_l** | InsightFace model (~300MB named Docker volume, persisted across restarts) |
