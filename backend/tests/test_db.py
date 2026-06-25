"""
Tests for database connection setup.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import app.db as db  # noqa: E402


def test_apply_sqlite_pragmas_sets_expected_values():
    class _Cursor:
        def __init__(self):
            self.statements = []
            self.closed = False

        def execute(self, sql):
            self.statements.append(sql)

        def close(self):
            self.closed = True

    class _Conn:
        def __init__(self):
            self.cursor_obj = _Cursor()

        def cursor(self):
            return self.cursor_obj

    conn = _Conn()

    db._apply_sqlite_pragmas(conn)

    assert conn.cursor_obj.statements == [
        "PRAGMA journal_mode=WAL",
        "PRAGMA synchronous=NORMAL",
        "PRAGMA busy_timeout=5000",
        "PRAGMA foreign_keys=ON",
    ]
    assert conn.cursor_obj.closed is True
