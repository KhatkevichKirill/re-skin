"""
Database infrastructure: engine, session factory, declarative base, and dependency.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine, event
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from .config import settings


def _ensure_data_dir(database_url: str) -> None:
    """Create the data/ directory for SQLite databases if it doesn't exist."""
    if database_url.startswith("sqlite:///"):
        # Strip the sqlite:/// prefix and get the path
        path = database_url[len("sqlite:///"):]
        if path and path != ":memory:":
            dirpath = os.path.dirname(os.path.abspath(path))
            os.makedirs(dirpath, exist_ok=True)


_ensure_data_dir(settings.DATABASE_URL)

# SQLite-specific: enable WAL mode and foreign keys via connection event
_is_sqlite = settings.DATABASE_URL.startswith("sqlite")

if _is_sqlite:
    _engine_kwargs: dict = {
        "connect_args": {"check_same_thread": False},
    }
else:
    # Postgres: use connection pooling suitable for multi-worker deployments.
    # pool_pre_ping sends a lightweight "SELECT 1" before handing out a connection
    # so stale connections after a Postgres restart are detected and recycled.
    _engine_kwargs = {
        "pool_pre_ping": True,
        "pool_size": 5,       # persistent connections per process
        "max_overflow": 10,   # extra connections allowed under load
    }

engine = create_engine(
    settings.DATABASE_URL,
    echo=False,
    **_engine_kwargs,
)

if _is_sqlite:
    @event.listens_for(engine, "connect")
    def _set_sqlite_pragmas(dbapi_conn, connection_record):  # noqa: ANN001
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()


SessionLocal = sessionmaker(
    bind=engine,
    autocommit=False,
    autoflush=False,
    expire_on_commit=False,
)


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""
    pass


@contextmanager
def get_session() -> Generator[Session, None, None]:
    """Context manager that yields a database session and handles commit/rollback."""
    session: Session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


# FastAPI dependency version
def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency that yields a database session."""
    with get_session() as session:
        yield session
