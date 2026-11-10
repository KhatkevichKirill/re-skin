# re-skin

A video re-skinning service. An operator uploads a source video; the system
detects faces (InsightFace), proposes swap/keep segments, lets the operator
review them, sends the swap segments to a video model (Seedance 2.0 / 2.5,
Gemini Omni, or a Fal-hosted model), stitches the results back into the
source and delivers the final to Google Drive.

The web UI is at `/v2` (Project → Runs). On top of plain face swaps it covers:

- **Localisation** projects: the hook is transcribed, translated by an LLM and
  re-spoken by the video model in the target language (`docs/localisation.md`).
- **Subtitle removal**, **reframe** (16:9 → 9:16) and **rebuild from a
  reference** project types (`docs/segment-caption-fill-design.md`,
  `docs/reframe-design.md`, `docs/rebuild-from-reference-design.md`).
- Post-processing on a delivered run: tail edits, burned captions
  (`docs/burned-captions.md`), overlays and PIP inserts (`docs/pip-insert.md`),
  in-place audio patches (`docs/run-audio-patch.md`), per-segment regeneration.
- GPT Image reference preprocessing before generation
  (`docs/gpt-image-2-preprocess.md`).
- An optional "by Assistant" autopilot that drives a project through its flow
  and asks the operator only for decisions (`docs/autopilot-design.md`).
- Optional exports of delivered creatives to a Google Sheet, and sheet-driven
  run automation with a completion webhook.

Optional subsystems (assistant, autopilot, knowledge base, Fal provider) are
off unless enabled in `.env`.

## Architecture

```
nginx :443 / :8847 (basic auth; TLS on 443)
  └─► api :8000            FastAPI + Jinja2/HTMX
        ├─ db              PostgreSQL 16
        └─ redis           RQ queues
  worker (x2)              analysis, generation submit/poll, stitch, delivery
  localise-worker          translate jobs (queue `localise`)
  brand-asr-worker         brand-word ASR check (queue `brand_asr`)
  knowledge-worker         knowledge-base capture (queue `knowledge`)
  autopilot-worker         "by Assistant" ticker
  fal-worker               Fal provider (compose profile `fal`, optional)
```

`api` and every worker share the `./data` bind mount (per-run media) and the
same image (`backend/Dockerfile`). Schema is managed with Alembic
(`backend/alembic`).

## Deploy

### 1. `.env`

```bash
cp .env.example .env
```

Set at least `KIE_API_KEY`, `GDRIVE_DEFAULT_FOLDER_ID`, `BASIC_AUTH_USER`,
`BASIC_AUTH_PASS`, `REDIS_PASSWORD`, `POSTGRES_PASSWORD`, `DATABASE_URL`
(the PostgreSQL line) and `PUBLIC_LINK_SECRET`. Every other variable is
documented in `.env.example` and `backend/app/config.py`; the optional
features stay off while their variables are empty.

### 2. Google service account

Put the service-account JSON at `./secrets/gdrive-sa.json` and share the
target Drive folder (and any export spreadsheet) with its `client_email` as
Editor. `./data` and `./secrets` must be owned by uid/gid 10001, the
container user (`deploy/container-hardening.md`).

### 3. TLS name

`deploy/nginx/nginx.conf` and `deploy/tls/renew-cert.sh` carry the TLS host
name literally (the example `203-0-113-10.sslip.io`). Replace it with your
own name and issue the certificate as described in `deploy/tls/README.md`.

### 4. ASR model

`brand-asr-worker` mounts the faster-whisper small model read-only from
`BRAND_ASR_MODEL_HOST_DIR` (default `/var/lib/reskin-assistant-runtime/asr-small`)
and does not start without it. Download it with
`scripts/download_assistant_asr.py` (`deploy/assistant/ASR-README.md`).

### 5. Migrate and start

```bash
docker compose up -d --build --scale worker=2
# The image does not contain the migrations; copy them in and apply them:
docker cp backend/alembic re-skin-api:/app/alembic
docker cp backend/alembic.ini re-skin-api:/app/alembic.ini
docker exec -w /app re-skin-api python3 -m alembic upgrade head
# nginx caches the upstream address, so restart it after every `up -d`:
docker compose restart nginx
```

`scripts/deploy.sh` is the gated production deploy (pauses new admissions,
waits for in-flight work to drain, backs up, migrates, restarts).

### Upgrading from the old public `main`

This release replaces the old v1 Job pipeline and SQLite setup. The old
public-only migration `b1f3d5a7c920` is not part of the new chain (the same
schema now arrives through several migrations), so an existing database
created from the old public `main` cannot be upgraded in place: start from a
fresh database, or export the data you need first.

## Development

Run the tests inside the app image (host Python packages may not match the
pinned versions):

```bash
cd backend
docker run --rm -u 0 -v "$PWD":/src:ro --entrypoint bash re-skin-api:latest -c \
  'pip install -q --require-hashes -r /src/requirements-dev.lock; cp -r /src /t; cd /t && python -m pytest tests -q'
```

Tests whose names end in `_pg` need a PostgreSQL server and skip without one.
A few `brand_check` tests need real-audio fixtures that are not published and
skip as well.

Logs: `docker compose logs -f api worker nginx`.

## Layout

```
backend/app/            FastAPI app, pipelines, provider clients, templates
backend/app/autopilot/  "by Assistant" flows
backend/app/knowledge/  knowledge-base storage and importer
backend/alembic/        migrations
backend/tests/          pytest suite
worker/                 worker entry points
deploy/                 nginx, TLS, ops gate, drain check, assistant stack
docs/                   design documents for each feature
scripts/                deploy, backup and probe scripts
```

Design documents mention internal ticket ids (`AICP-…`) and dates from the
original production deployment; they are kept as history.
