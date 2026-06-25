"""
Configuration module for re-skin.
Loads settings from environment variables using python-dotenv.

Relative paths (GOOGLE_SERVICE_ACCOUNT_FILE, DATABASE_URL) are resolved against
BASE_DIR so they work correctly regardless of the process working directory.

Inside Docker containers, WORKDIR is /app and ./data is mounted at /app/data,
so a relative path like ./secrets/gdrive-sa.json resolves to /app/secrets/gdrive-sa.json
and sqlite:///./data/app.db resolves to sqlite:////app/data/app.db.

Locally, BASE_DIR is the backend/ directory (two levels up from this file),
which keeps the same structure as before.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env file (no-op in containers where vars are injected; harmless).
load_dotenv()

# Single authoritative base directory.  Override via APP_BASE_DIR if needed.
# Default: inside container => /app  (the WORKDIR); locally => backend/
BASE_DIR: Path = Path(
    os.environ.get("APP_BASE_DIR", str(Path(__file__).parent.parent))
).resolve()


def _resolve_path(raw: str) -> str:
    """Return *raw* as an absolute path, resolved against BASE_DIR if relative."""
    p = Path(raw)
    if p.is_absolute():
        return str(p)
    return str(BASE_DIR / p)


def _resolve_db_url(url: str) -> str:
    """Resolve the file path inside a sqlite:/// URL against BASE_DIR."""
    prefix = "sqlite:///"
    if url.startswith(prefix):
        rest = url[len(prefix):]
        if rest and rest != ":memory:":
            return f"{prefix}{_resolve_path(rest)}"
    return url


class Settings:
    """Application settings loaded from environment."""

    # kie.ai (Seedance)
    KIE_API_KEY: str = os.getenv("KIE_API_KEY", "")

    # Google Drive
    GOOGLE_SERVICE_ACCOUNT_FILE: str = _resolve_path(
        os.getenv("GOOGLE_SERVICE_ACCOUNT_FILE", "./secrets/gdrive-sa.json")
    )
    GDRIVE_DEFAULT_FOLDER_ID: str = os.getenv("GDRIVE_DEFAULT_FOLDER_ID", "")

    # Web access
    BASIC_AUTH_USER: str = os.getenv("BASIC_AUTH_USER", "reskin")
    BASIC_AUTH_PASS: str = os.getenv("BASIC_AUTH_PASS", "")

    # Secret used to sign unauthenticated public links (/public/...). Falls back
    # to BASIC_AUTH_PASS so it works out of the box; override to rotate links.
    PUBLIC_LINK_SECRET: str = os.getenv("PUBLIC_LINK_SECRET", "") or os.getenv(
        "BASIC_AUTH_PASS", ""
    )

    # Infrastructure
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://redis:6379/0")

    # DATABASE_URL — two supported forms:
    #   SQLite (default / tests):
    #     sqlite:///./data/app.db  (relative, resolved against BASE_DIR above)
    #     sqlite:///:memory:       (in-memory, used by the test suite)
    #   PostgreSQL (production):
    #     postgresql+psycopg2://<user>:<password>@<host>:5432/<dbname>
    #     e.g. postgresql+psycopg2://reskin:secret@db:5432/reskin
    #     Use the psycopg2 driver — it is pinned in requirements.txt.
    DATABASE_URL: str = _resolve_db_url(
        os.getenv("DATABASE_URL", "sqlite:///./data/app.db")
    )

    # App configuration
    DEFAULT_RESOLUTION: str = os.getenv("DEFAULT_RESOLUTION", "480p")
    MAX_REFERENCE_IMAGES: int = int(os.getenv("MAX_REFERENCE_IMAGES", "2"))
    SEGMENT_MAX_SECONDS: int = int(os.getenv("SEGMENT_MAX_SECONDS", "15"))

    # Upload size cap (MiB).  Applies to source video and reference image uploads.
    # Set high enough for 1080p source videos (several hundred MB typical).
    # Nginx client_max_body_size must be >= this value.
    MAX_UPLOAD_SIZE_MB: int = int(os.getenv("MAX_UPLOAD_SIZE_MB", "1024"))


settings = Settings()
