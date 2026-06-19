# re-skin

A video re-skinning tool that transforms video content with automated styling, color correction, and effects.

## Prerequisites

- Docker (26.1.3+)
- docker-compose v1.29.2+
- Python 3.10+ (for local development)

## Quick Start

1. **Configure environment:**
   ```bash
   cp .env.example .env
   # Edit .env and fill in required values:
   # - BASIC_AUTH_USER, BASIC_AUTH_PASS (for nginx basic auth)
   # - KIE_API_KEY (for Seedance API access)
   # - GDRIVE_DEFAULT_FOLDER_ID (for Google Drive output)
   ```

2. **Add secrets:**
   ```bash
   # Place your Google Drive service account JSON at:
   ./secrets/gdrive-sa.json
   ```

3. **Start services:**
   ```bash
   docker-compose up --build
   ```

4. **Access the app:**
   Open http://localhost:8847 in your browser.
   Login with credentials from BASIC_AUTH_USER/BASIC_AUTH_PASS in .env

## Services

- **API** (FastAPI): Runs on http://localhost:8000 (proxied through Nginx)
- **Worker** (RQ): Background job processing
- **Redis**: Message queue and cache
- **Nginx**: Reverse proxy with basic authentication on port 8847

## Project Structure

```
.
├── backend/                 # FastAPI application
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py         # FastAPI app with health & root endpoints
│   │   └── config.py       # Configuration from environment
│   ├── Dockerfile
│   └── requirements.txt
├── worker/                  # RQ background worker
│   ├── __init__.py
│   └── worker.py           # Worker bootstrap
├── deploy/
│   └── nginx/              # Reverse proxy + basic auth
│       ├── Dockerfile
│       ├── nginx.conf
│       └── entrypoint.sh
├── frontend/               # UI placeholder (T9)
├── docs/
├── scripts/
├── tasks/                  # Task management (todo.md)
├── docker-compose.yml
├── .env                    # Local secrets (not committed)
├── .env.example            # Template (safe to commit)
└── .gitignore
```

## Environment Variables

See `.env.example` for a complete list. Key variables:

- `BASIC_AUTH_USER`, `BASIC_AUTH_PASS`: Nginx authentication
- `KIE_API_KEY`: Seedance API key
- `REDIS_URL`: Redis connection (default: redis://redis:6379/0)
- `DATABASE_URL`: SQLite database path (default: ./data/app.db)

## Development

### Run locally (without Docker)

```bash
# Install dependencies
pip install -r backend/requirements.txt

# Set environment variables
export REDIS_URL=redis://localhost:6379/0
export BASIC_AUTH_USER=reskin
export BASIC_AUTH_PASS=dev-password

# Start Redis (requires separate Redis instance)
redis-server

# Run API
cd backend
uvicorn app.main:app --reload

# In another terminal, run worker
cd worker
python -m worker.worker
```

### Docker Compose Tips

```bash
# View logs
docker-compose logs -f api

# Stop services
docker-compose down

# Rebuild a specific service
docker-compose up --build api

# Remove volumes (useful for fresh data)
docker-compose down -v
```

## Endpoints

- `GET /health` — Health check
- `GET /` — Landing page (HTML)

## Notes

- Nginx listens on port 8847 (configurable via `APP_EXTERNAL_PORT`)
- All requests are protected by HTTP Basic Auth
- Worker connects to Redis and listens on the "default" queue
- Database files are stored in `./data/` (not committed)

## License

TBD
