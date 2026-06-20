# re-skin

A video re-skinning tool that uses AI (Seedance via kie.ai) to swap faces/segments
in video clips, with a web UI for reviewing and approving segments before processing.

## Architecture

```
nginx:8847 (basic auth)
    └─► api:8000  (FastAPI — uvicorn)
            ├─ SQLite /app/data/app.db  ──┐  (bind-mounted ./data)
            └─ RQ enqueue via Redis        │
    worker        (RQ worker)            │
        ├─ SQLite /app/data/app.db  ◄──┘
        ├─ /app/data/jobs/<id>/clips/
        └─ /app/data/jobs/<id>/results/
redis:6379    (job queue)
```

Both `api` and `worker` containers share the `./data` bind mount, which holds
the SQLite database and all per-job media files.

## Deploy & Run

### 1. Fill in `.env`

```bash
cp .env.example .env
```

Edit `.env` and set at minimum:

| Variable | Description |
|---|---|
| `KIE_API_KEY` | kie.ai API key for Seedance video generation |
| `GDRIVE_DEFAULT_FOLDER_ID` | Google Drive folder ID for output delivery |
| `BASIC_AUTH_USER` | Username for the nginx basic-auth gate |
| `BASIC_AUTH_PASS` | Password for the nginx basic-auth gate |
| `APP_EXTERNAL_PORT` | Host port for nginx (default `8847`) |

The remaining variables have working defaults and usually do not need changing.

### 2. Place the Google Drive service account

```bash
# Obtain the service account JSON from GCP and place it here:
./secrets/gdrive-sa.json
```

Share the target Google Drive folder with the service account email address
(`client_email` field inside the JSON) as an **Editor**.

### 3. Start the stack

```bash
docker compose up -d --build
```

This builds a single image (used by both `api` and `worker`) that includes
ffmpeg and all Python dependencies.

> **Note — InsightFace model download:** On the first job the worker downloads
> the `buffalo_l` face-detection model (~300 MB) from the internet into
> `~/.insightface/models/` inside the worker container.  This is a one-time
> download per container lifetime.  Subsequent jobs reuse the cached model.
> If the container is replaced the download repeats unless you add a named
> volume for `~/.insightface`.

### 4. Verify services are up

```bash
docker compose ps
# All four services (redis, api, worker, nginx) should show "Up".
```

### 5. Browse

Open `http://<SERVER_IP>:8847` and authenticate with your `BASIC_AUTH_USER`
/ `BASIC_AUTH_PASS`.

## Stopping

```bash
docker compose down
```

Data in `./data/` persists on the host.

## Development

### Run tests

```bash
cd backend
pip install -r requirements.txt
pytest tests/
```

### Run locally without Docker

```bash
# Start Redis
redis-server

# Start API (from backend/)
uvicorn app.main:app --reload

# Start worker (from repo root — PYTHONPATH must include backend/)
PYTHONPATH=backend python -m worker.worker
```

### View logs

```bash
docker compose logs -f api
docker compose logs -f worker
docker compose logs -f nginx
```

## Project layout

```
.
├── backend/
│   ├── app/
│   │   ├── api.py          # REST API router (/api/*)
│   │   ├── config.py       # Settings from env vars
│   │   ├── db.py           # SQLAlchemy engine + session
│   │   ├── face.py         # InsightFace detection helpers
│   │   ├── gdrive_client.py
│   │   ├── kie_client.py   # Seedance API client
│   │   ├── main.py         # FastAPI app wiring
│   │   ├── media.py        # ffmpeg probing and cutting
│   │   ├── models.py       # SQLAlchemy ORM models
│   │   ├── pipeline.py     # analyze_job / process_job
│   │   ├── schemas.py      # Pydantic schemas
│   │   ├── state_machine.py
│   │   ├── storage.py      # Per-job path helpers
│   │   ├── tasks.py        # RQ task wrappers
│   │   ├── web.py          # HTMX/Jinja2 web UI
│   │   ├── static/
│   │   └── templates/
│   ├── Dockerfile
│   └── requirements.txt
├── worker/
│   └── worker.py           # RQ worker bootstrap
├── deploy/
│   └── nginx/
│       ├── Dockerfile
│       ├── entrypoint.sh
│       └── nginx.conf
├── data/                   # Runtime data (gitignored)
├── secrets/                # Service account (gitignored)
├── docker-compose.yml
├── .env                    # Local secrets (gitignored)
└── .env.example            # Template (safe to commit)
```

## Environment variables

See `.env.example` for the full list with comments.
