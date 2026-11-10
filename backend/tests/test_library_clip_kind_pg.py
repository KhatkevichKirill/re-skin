"""AICP-381 migration ``w2c4e6a8b0d3`` (library_clips.kind) on PostgreSQL.

Opt-in like tests/test_rebuild_pg.py: set RESKIN_TEST_PG_URL to a DISPOSABLE
Postgres server (never production — this creates and drops a database):

    docker run --rm -d --name reskin-kind-pgtest -e POSTGRES_PASSWORD=t \\
        -p 127.0.0.1:15512:5432 postgres:15
    RESKIN_TEST_PG_URL=postgresql://postgres:t@127.0.0.1:15512/postgres \\
        pytest tests/test_library_clip_kind_pg.py

The prod library as read 2026-11-05 (names + lengths) goes in at the previous
head; ``upgrade head`` must mark only the short ``CTA_*`` cards ``end_card``.
"""

from __future__ import annotations

import os
import sys
import tempfile
import uuid

import pytest

_PG_URL = os.environ.get("RESKIN_TEST_PG_URL")
pytestmark = pytest.mark.skipif(
    not _PG_URL, reason="RESKIN_TEST_PG_URL not set (needs a disposable PostgreSQL)")

os.environ.setdefault("DATA_DIR", tempfile.mkdtemp(prefix="reskin_kind_pg_"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

BACKEND = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
PREV = "w4c8e2a6d1f3"  # the previous head (release chain w2c6e8a0b4d7 -> w4c8e2a6d1f3 -> w2c4e6a8b0d3)

#: (name, duration_sec, expected kind) — prod's library_clips on 2026-11-05
#: plus edge cases.
LIBRARY = [
    ("PsychologyOfMoney_Bedroom_Style1", 23.3, "insert"),
    ("Bible_Home", 25.9, "insert"),
    ("BlankDemo_Phone-Laptop_1", 26.3, "insert"),
    ("BlankDemo_Jp_v1.mp4", 23.3, "insert"),
    ("CTA_Spanish", 7.0, "end_card"),
    ("CTA_Portuguese", 7.0, "end_card"),
    ("BlankDemo_POV-Train_1", 19.2, "insert"),
    ("CTA_English", 7.0, "end_card"),
    ("ip_reader", 27.8, "insert"),
    ("CTA_Poland", 5.2, "end_card"),
    ("CTA_Poland_2", 7.0, "end_card"),
    ("CTA_Spanish_Book", 24.6, "insert"),
    ("CTA_Portuguese_Book", 24.8, "insert"),
    ("Demo_French", 24.7, "insert"),
    ("TEMP autopilot ecb9ac03 es probe", 12.7, "insert"),
    ("CTAX_not_a_card", 5.0, "insert"),     # the underscore is literal
    ("CTA_unprobed", None, "insert"),       # no length: never guessed
]


def _db_url(name: str) -> str:
    from sqlalchemy.engine import make_url

    return make_url(_PG_URL).set(database=name).render_as_string(hide_password=False)


def _admin(sql: str) -> None:
    from sqlalchemy import create_engine, text

    admin = create_engine(_PG_URL, isolation_level="AUTOCOMMIT")
    with admin.connect() as c:
        c.execute(text(sql))
    admin.dispose()


def test_kind_migration_marks_short_cta_cards(monkeypatch):
    from alembic import command
    from alembic.autogenerate import compare_metadata
    from alembic.config import Config
    from alembic.migration import MigrationContext
    from sqlalchemy import create_engine, text
    from sqlalchemy.exc import IntegrityError

    from app.config import settings
    from app.db import Base

    name = f"kind_mig_{uuid.uuid4().hex[:8]}"
    _admin(f'DROP DATABASE IF EXISTS "{name}"')
    _admin(f'CREATE DATABASE "{name}"')
    url = _db_url(name)
    monkeypatch.setattr(settings, "DATABASE_URL", url)  # alembic/env.py reads it
    cfg = Config(os.path.join(BACKEND, "alembic.ini"))
    cfg.set_main_option("script_location", os.path.join(BACKEND, "alembic"))
    eng = create_engine(url)
    try:
        command.upgrade(cfg, PREV)
        with eng.begin() as c:
            for i, (n, dur, _k) in enumerate(LIBRARY):
                c.execute(text(
                    "INSERT INTO library_clips (id, name, local_path, duration_sec) "
                    "VALUES (:id, :n, '/x.mp4', :d)"), {"id": f"c{i}", "n": n, "d": dur})
        command.upgrade(cfg, "head")
        with eng.begin() as c:
            got = dict(c.execute(text("SELECT name, kind FROM library_clips")).all())
        assert got == {n: k for n, _d, k in LIBRARY}
        # The old image's INSERT (no kind) gets the server default.
        with eng.begin() as c:
            c.execute(text("INSERT INTO library_clips (id, name, local_path) "
                           "VALUES ('old', 'CTA_from_old_image', '/x.mp4')"))
            assert c.execute(text(
                "SELECT kind FROM library_clips WHERE id='old'")).scalar() == "insert"
        with pytest.raises(IntegrityError):
            with eng.begin() as c:
                c.execute(text("UPDATE library_clips SET kind='outro' WHERE id='old'"))
        # The ORM matches the migrated table.
        with eng.connect() as c:
            diffs = compare_metadata(MigrationContext.configure(c), Base.metadata)
        assert [d for d in diffs if "library_clips" in repr(d)] == []
        command.downgrade(cfg, PREV)
        with eng.begin() as c:
            cols = {r[0] for r in c.execute(text(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name='library_clips'"))}
        assert "kind" not in cols
        command.upgrade(cfg, "head")  # re-runnable after a downgrade
    finally:
        eng.dispose()
        _admin(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')
