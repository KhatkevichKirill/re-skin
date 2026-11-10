"""
True-concurrency proof for the daily kie.ai budget (AICP-155) on PostgreSQL —
production's database. SQLite (the rest of the suite) serializes every writer
through one database lock, so it cannot show that the day-row lock is what makes
check+reserve atomic between the api and two workers.

Opt-in like test_pipeline_range_attempt_pg_concurrency.py: set
RESKIN_TEST_PG_URL to a DISPOSABLE Postgres database (never production — this
creates and drops tables), e.g.

    docker run --rm -d --name reskin-spend-pgtest -e POSTGRES_PASSWORD=t \\
        -p 127.0.0.1:15455:5432 postgres:15
    RESKIN_TEST_PG_URL=postgresql://postgres:t@127.0.0.1:15455/postgres \\
        pytest tests/test_spend_pg_concurrency.py
"""

from __future__ import annotations

import os
import sys
import tempfile
import threading
from contextlib import contextmanager
from datetime import datetime, timezone

import pytest

_PG_URL = os.environ.get("RESKIN_TEST_PG_URL")
pytestmark = [
    pytest.mark.skipif(not _PG_URL, reason="RESKIN_TEST_PG_URL not set (needs a disposable PostgreSQL)"),
    pytest.mark.spend_cap,
]

os.environ.setdefault("DATA_DIR", tempfile.mkdtemp(prefix="reskin_spend_pg_"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

DAY = datetime(2026, 10, 16, 12, 0, tzinfo=timezone.utc)


@pytest.fixture(scope="module")
def Session():
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from app.models import KieSpendDay, KieSpendDraw, KieSpendReservation

    tables = [KieSpendDay.__table__, KieSpendReservation.__table__, KieSpendDraw.__table__]
    eng = create_engine(_PG_URL, pool_size=40, max_overflow=0)
    for t in reversed(tables):
        t.drop(eng, checkfirst=True)
    for t in tables:
        t.create(eng)
    yield sessionmaker(bind=eng, autocommit=False, autoflush=False, expire_on_commit=False)
    for t in reversed(tables):
        t.drop(eng, checkfirst=True)
    eng.dispose()


@pytest.fixture(autouse=True)
def _clean(Session, monkeypatch):
    from app.config import settings
    from app.models import KieSpendDay, KieSpendDraw, KieSpendReservation

    monkeypatch.setattr(settings, "KIE_USD_PER_CREDIT", 0.005)
    s = Session()
    for m in (KieSpendDraw, KieSpendReservation, KieSpendDay):
        s.query(m).delete()
    s.commit()
    s.close()
    yield


def _total(Session):
    from app import spend

    s = Session()
    try:
        return spend.today_summary(s, DAY)["charged_credits"]
    finally:
        s.close()


def test_simultaneous_reservers_admit_exactly_what_fits(Session, monkeypatch):
    from app import spend
    from app.config import settings

    monkeypatch.setattr(settings, "KIE_DAILY_BUDGET_USD", 5.0)  # 1000 credits
    n = 30
    barrier = threading.Barrier(n)
    admitted, refused, errors = [], [], []

    def racer(i):
        s = Session()
        try:
            barrier.wait()
            spend.reserve(s, f"run:{i}", 100, kind="new run", now=DAY)
            s.commit()
            admitted.append(i)
        except spend.BudgetExceeded:
            s.rollback()
            refused.append(i)
        except Exception as exc:  # pragma: no cover
            s.rollback()
            errors.append(repr(exc))
        finally:
            s.close()

    threads = [threading.Thread(target=racer, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    assert len(admitted) == 10 and len(refused) == 20
    assert _total(Session) == pytest.approx(1000)


def test_reservers_and_drawers_interleave_without_losing_updates(Session, monkeypatch):
    """Workers drawing (in-flight overdraft) while the api reserves: the day
    total is exactly reservations + overdraft, no lost update, no deadlock."""
    from app import spend
    from app.config import settings

    monkeypatch.setattr(settings, "KIE_DAILY_BUDGET_USD", 0.0)  # count only
    s = Session()
    spend.reserve(s, "run:live", 1000, kind="new run", now=DAY)
    s.commit()
    s.close()

    @contextmanager
    def sessions():
        sess = Session()
        try:
            yield sess
            sess.commit()
        except Exception:
            sess.rollback()
            raise
        finally:
            sess.close()

    n = 20
    barrier = threading.Barrier(2 * n)
    errors = []

    def drawer(i):
        barrier.wait()
        # 20 draws x 100 on a 1000 reservation -> 1000 of overdraft.
        spend.record_draw("run:live", 100, kind="video", call_key=f"task-{i}", sessions=sessions, now=DAY)

    def reserver(i):
        sess = Session()
        try:
            barrier.wait()
            spend.reserve(sess, f"run:new{i}", 50, kind="new run", now=DAY)
            sess.commit()
        except Exception as exc:  # pragma: no cover
            sess.rollback()
            errors.append(repr(exc))
        finally:
            sess.close()

    threads = [threading.Thread(target=drawer, args=(i,)) for i in range(n)]
    threads += [threading.Thread(target=reserver, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    # 1000 reserved + 1000 overdraft (2000 drawn) + 20 x 50 new reservations.
    assert _total(Session) == pytest.approx(3000)


def test_duplicate_draws_racing_count_once(Session, monkeypatch):
    from app import spend

    @contextmanager
    def sessions():
        sess = Session()
        try:
            yield sess
            sess.commit()
        except Exception:
            sess.rollback()
            raise
        finally:
            sess.close()

    n = 10
    barrier = threading.Barrier(n)

    def dup():
        barrier.wait()
        spend.record_draw("run:x", 700, kind="video", call_key="same-task", sessions=sessions, now=DAY)

    threads = [threading.Thread(target=dup) for _ in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert _total(Session) == pytest.approx(700)
