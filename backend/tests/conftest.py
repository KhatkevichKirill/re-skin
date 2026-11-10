"""Suite-wide fixtures.

Kept import-free at module level on purpose: every test module sets
DATABASE_URL / DATA_DIR in os.environ BEFORE its first ``app`` import, and an
``app`` import here (conftest loads first) would freeze the settings early.
"""

import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "spend_cap: keep the daily kie.ai budget (KIE_DAILY_BUDGET_USD) enforced "
        "for this test instead of the suite default of 0 (disabled)",
    )


@pytest.fixture(autouse=True)
def _kie_daily_budget_disabled_by_default(request, monkeypatch):
    """The test modules share one database across many tests, so reservations
    from hundreds of synthetic runs pile up on "today" and would eventually trip
    the 250 USD default. Disable enforcement (accounting still runs) unless a
    test opts back in with ``@pytest.mark.spend_cap``."""
    if request.node.get_closest_marker("spend_cap") is not None:
        yield
        return
    from app.config import settings

    monkeypatch.setattr(settings, "KIE_DAILY_BUDGET_USD", 0.0)
    yield


TEST_PUBLIC_LINK_SECRET = "test-public-link-secret-0123456789abcdef"


@pytest.fixture(autouse=True)
def _public_link_secret(monkeypatch):
    """AICP-151: public links fail closed without a secret; give tests one."""
    from app.config import settings

    monkeypatch.setattr(settings, "PUBLIC_LINK_SECRET", TEST_PUBLIC_LINK_SECRET, raising=False)
    monkeypatch.setattr(settings, "PUBLIC_LINK_SECRET_PREVIOUS", "", raising=False)
