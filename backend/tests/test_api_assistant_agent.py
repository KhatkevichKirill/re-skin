"""Analysis history and retained candidates respect current source access."""
import uuid
from tests.test_api_assistant import api, root
from tests.test_assistant_turns import setup
from tests.test_assistant_agent_loop import fixture_loop
from app.api_assistant import get_assistant_service
from app.assistant_access import ProjectAllowlist
from app.assistant_turns import AssistantTurnService


def test_candidate_preview_is_private_and_never_exposes_storage(setup, api):
    client, app = api
    svc, turn_id, broker, media, supervisor = fixture_loop(setup)
    pinned = setup.storage.pin(setup.base.path)
    media.candidate = media.candidate.model_copy(update={"sha256": pinned.sha256, "size_bytes": pinned.size_bytes})
    media.persist_candidate = lambda _: {"storage_key": pinned.storage_key,
        "sha256": pinned.sha256, "size_bytes": pinned.size_bytes}
    supervisor.run_once()
    state = client.get(root(setup) + "/context")
    assert state.status_code == 200, state.text
    turn = next(item for item in state.json()["turns"] if item["id"] == turn_id)
    assert turn["status"] == "succeeded", turn
    url = turn["analysis"]["candidate_url"]
    assert "storage_key" not in state.text and "data_base64" not in state.text
    response = client.get(url)
    assert response.status_code == 200 and response.content == setup.base.path.read_bytes()
    assert response.headers["cache-control"] == "private, no-store"
    assert client.get(url.replace(media.candidate.candidate_id, str(uuid.uuid4()))).status_code == 404
    app.dependency_overrides[get_assistant_service] = lambda: AssistantTurnService(
        storage=setup.storage, policy=ProjectAllowlist([]), enabled=True)
    assert client.get(url).status_code == 404
