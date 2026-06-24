#!/usr/bin/env python3
"""
sqlite_to_postgres.py — One-time ETL: copy all rows from a SQLite DB into a
PostgreSQL DB (which must already have the schema applied via alembic upgrade head).

Strategy: TRUNCATE all tables in dependency order, then bulk-INSERT from SQLite.
This makes the script safe to re-run (idempotent via truncate-then-load).

Usage:
    python3 scripts/sqlite_to_postgres.py \\
        --source sqlite:////path/to/app.db \\
        --target "postgresql+psycopg2://user:pass@host:5432/dbname"

The script NEVER modifies the source database (opens it read-only via SQLAlchemy).
To protect the live database, pass a COPY of it as --source, e.g.:
    cp data/app.db /tmp/app_copy.db
    python3 scripts/sqlite_to_postgres.py \\
        --source sqlite:////tmp/app_copy.db \\
        --target "postgresql+psycopg2://..."
"""

from __future__ import annotations

import argparse
import sys
import os
from datetime import datetime, timezone

# Allow running from project root or scripts/ directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import sessionmaker


# Tables in correct foreign-key insertion order (parents before children).
# Truncation is done in reverse order.
TABLE_ORDER = [
    "jobs",
    "segments",
    "video_projects",
    "segment_defs",
    "runs",
    "run_segments",
]


def _get_engine(url: str):
    if url.startswith("sqlite"):
        return create_engine(
            url,
            connect_args={"check_same_thread": False},
        )
    else:
        return create_engine(
            url,
            pool_pre_ping=True,
            pool_size=5,
            max_overflow=10,
        )


def _fetch_all_rows(src_engine, table_name: str) -> list[dict]:
    """Return all rows of *table_name* as a list of dicts."""
    with src_engine.connect() as conn:
        result = conn.execute(text(f"SELECT * FROM {table_name}"))
        rows = result.mappings().all()
        # Convert to plain dicts so they can be passed to Postgres insert
        return [dict(r) for r in rows]


def _truncate_tables(dst_engine, tables: list[str]) -> None:
    """Truncate tables in reverse dependency order (children first)."""
    with dst_engine.begin() as conn:
        # Disable FK checks temporarily to allow arbitrary truncation order
        if dst_engine.dialect.name == "postgresql":
            for table in reversed(tables):
                conn.execute(text(f"TRUNCATE TABLE {table} CASCADE"))
        else:
            conn.execute(text("PRAGMA foreign_keys=OFF"))
            for table in reversed(tables):
                conn.execute(text(f"DELETE FROM {table}"))
            conn.execute(text("PRAGMA foreign_keys=ON"))


def _insert_rows(dst_engine, table_name: str, rows: list[dict]) -> int:
    """Insert *rows* into *table_name* in the destination DB. Returns count inserted."""
    if not rows:
        return 0

    # Ensure datetime objects are timezone-aware (Postgres TIMESTAMP WITH TIME ZONE
    # is strict; plain TIMESTAMP accepts both naive and aware).
    # Our schema uses TIMESTAMP (without TZ), so naive datetimes are fine.
    # SQLite returns strings for datetime columns — convert them.
    import json as _json

    # Boolean columns: SQLite stores as INTEGER (0/1); Postgres needs True/False.
    BOOLEAN_COLUMNS = {"has_face"}

    # JSON columns: SQLite may return a Python list/dict (SQLAlchemy decodes JSON
    # columns automatically) or occasionally a string.  Postgres psycopg2 needs
    # these wrapped in psycopg2.extras.Json so they are serialised correctly
    # instead of being treated as Postgres arrays.
    try:
        from psycopg2.extras import Json as PGJson
        _use_pgjson = dst_engine.dialect.name == "postgresql"
    except ImportError:
        _use_pgjson = False

    JSON_COLUMNS = {
        "default_reference_image_urls",
        "reference_image_urls",
        "reference_image_urls_override",
    }

    def _coerce_row(row: dict) -> dict:
        cleaned = {}
        for k, v in row.items():
            if isinstance(v, str):
                # Parse datetime strings from SQLite
                if k in ("created_at", "updated_at"):
                    try:
                        for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
                            try:
                                v = datetime.strptime(v, fmt)
                                break
                            except ValueError:
                                pass
                    except Exception:
                        pass
                # Parse JSON strings from SQLite (stored as TEXT in old rows)
                elif k in JSON_COLUMNS:
                    try:
                        v = _json.loads(v)
                    except (ValueError, TypeError):
                        pass
            # Coerce SQLite INTEGER booleans → Python bool for Postgres
            if k in BOOLEAN_COLUMNS and isinstance(v, int):
                v = bool(v)
            # Wrap JSON-column values so psycopg2 sends them as JSON, not as
            # Postgres arrays.  None stays None (NULL).
            if k in JSON_COLUMNS and v is not None and _use_pgjson:
                v = PGJson(v)
            cleaned[k] = v
        return cleaned

    coerced = [_coerce_row(r) for r in rows]

    with dst_engine.begin() as conn:
        conn.execute(
            text(
                f"INSERT INTO {table_name} ({', '.join(coerced[0].keys())}) "
                f"VALUES ({', '.join(':' + k for k in coerced[0].keys())})"
            ),
            coerced,
        )
    return len(coerced)


def migrate(source_url: str, target_url: str, dry_run: bool = False) -> None:
    print(f"\n{'='*60}")
    print(f"SQLite → Postgres ETL")
    print(f"  Source: {source_url}")
    print(f"  Target: {target_url}")
    print(f"  Dry-run: {dry_run}")
    print(f"{'='*60}\n")

    src_engine = _get_engine(source_url)
    dst_engine = _get_engine(target_url)

    # Verify source tables exist
    src_inspector = inspect(src_engine)
    src_tables = set(src_inspector.get_table_names())
    missing = [t for t in TABLE_ORDER if t not in src_tables]
    if missing:
        print(f"[WARN] Source DB missing tables: {missing} — they will be skipped.")

    # Read all data from source
    print("Reading source data...")
    data: dict[str, list[dict]] = {}
    for table in TABLE_ORDER:
        if table not in src_tables:
            data[table] = []
        else:
            rows = _fetch_all_rows(src_engine, table)
            data[table] = rows
            print(f"  {table}: {len(rows)} rows")

    total_source = sum(len(v) for v in data.values())
    print(f"\nTotal source rows: {total_source}")

    if dry_run:
        print("\n[DRY RUN] — no changes written to target.")
        return

    # Truncate destination tables
    print("\nTruncating destination tables...")
    _truncate_tables(dst_engine, [t for t in TABLE_ORDER if t in src_tables])
    print("  Done.")

    # Insert rows
    print("\nInserting rows into destination...")
    total_inserted = 0
    for table in TABLE_ORDER:
        rows = data[table]
        if not rows:
            print(f"  {table}: 0 rows (skipped)")
            continue
        count = _insert_rows(dst_engine, table, rows)
        total_inserted += count
        print(f"  {table}: {count} rows inserted")

    print(f"\nTotal rows inserted: {total_inserted}")

    # Verify counts match
    print("\nVerifying row counts in destination...")
    with dst_engine.connect() as conn:
        for table in TABLE_ORDER:
            if table not in src_tables:
                continue
            dst_count = conn.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar()
            src_count = len(data[table])
            status = "OK" if dst_count == src_count else "MISMATCH"
            print(f"  {table}: src={src_count}  dst={dst_count}  [{status}]")
            if status == "MISMATCH":
                print(f"    [ERROR] Row count mismatch in {table}!")
                sys.exit(1)

    print("\nETL complete. All row counts match.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Copy all rows from a SQLite DB into a PostgreSQL DB.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--source",
        required=True,
        help="SQLAlchemy URL of the source SQLite DB, e.g. sqlite:////tmp/app_copy.db",
    )
    parser.add_argument(
        "--target",
        required=True,
        help="SQLAlchemy URL of the target Postgres DB, e.g. postgresql+psycopg2://user:pass@host:5432/db",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Read source data and print counts but do not write to target.",
    )
    args = parser.parse_args()
    migrate(args.source, args.target, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
