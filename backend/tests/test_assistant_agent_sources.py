"""Project inputs become immutable authorized donors, not selected results."""
from sqlalchemy import func, select
import pytest

from tests.test_assistant_service import setup
from app.assistant_access import AssistantAccessDenied, ProjectAllowlist
from app.assistant_agent_sources import import_original_source
from app.assistant_media import AssistantMediaCancelled
from app.assistant_models import AssistantArtifact, AssistantSelection, AssistantVersion
from app.assistant_store import verify_artifact
from app.models import VideoProject


def test_original_import_preserves_selection_and_pins_once(setup):
    policy = ProjectAllowlist([setup.base.project])
    with setup.factory.begin() as session:
        session.get(VideoProject, setup.base.project).source_local_path = str(setup.base.path)
        session.flush()
        before = session.get(AssistantSelection, setup.base.conversation).selected_version_id
        count = session.scalar(select(func.count()).select_from(AssistantVersion))
        args = dict(run_id=setup.base.run, base_version_id=setup.base.version,
                    storage=setup.storage, policy=policy)
        first = import_original_source(session, **args)
        setup.base.path.write_bytes(b"changed legacy source")
        again = import_original_source(session, **args)
        assert again == first
        assert first.label == "Original project source video"
        assert first.duration_sec == 3 and first.fps == "30"
        assert verify_artifact(session.get(AssistantArtifact, first.asset_id),
                               storage=setup.storage).read_bytes() != setup.base.path.read_bytes()
        assert session.get(AssistantSelection, setup.base.conversation).selected_version_id == before
        assert session.scalar(select(func.count()).select_from(AssistantVersion)) == count


def test_denied_project_cannot_pin_or_probe_source(setup, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("access denied before storage")
    monkeypatch.setattr(setup.storage, "pin", forbidden)
    with setup.factory.begin() as session, pytest.raises(AssistantAccessDenied):
        import_original_source(session, run_id=setup.base.run, base_version_id=setup.base.version,
                               storage=setup.storage, policy=ProjectAllowlist([]))


def test_cancelled_pin_leaves_no_object_or_staging_file(setup):
    before = set(setup.storage.root.rglob("*"))
    with pytest.raises(AssistantMediaCancelled):
        setup.storage.pin(setup.base.path, cancelled=lambda: True)
    assert set(setup.storage.root.rglob("*")) == before
