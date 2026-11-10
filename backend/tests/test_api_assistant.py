"""Standalone private router tests, isolated DB and media fixtures."""
import json
import pytest
from fastapi import FastAPI
import asyncio
import httpx
from sqlalchemy import func, select
from tests.test_assistant_turns import setup, service, accept, reply, NOW
from app.api_assistant import router, get_assistant_service
from app.assistant_access import ProjectAllowlist
from app.assistant_jobs import AssistantJob
from app.assistant_models import AssistantMessage
from app.assistant_turns import AssistantTurnService, claim_turn, fail_turn
from app.db import get_db


@pytest.fixture
def api(setup):
    app = FastAPI()
    app.include_router(router, prefix="/api/v2")
    def db():
        with setup.factory.begin() as session:
            yield session
    app.dependency_overrides[get_db] = db
    app.dependency_overrides[get_assistant_service] = lambda: service(setup)
    class Client:
        def request(self, method, path, **kwargs):
            async def perform():
                async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
                    return await client.request(method, path, **kwargs)
            return asyncio.run(perform())
        def get(self, path, **kwargs):
            return self.request("GET", path, **kwargs)
        def post(self, path, **kwargs):
            return self.request("POST", path, **kwargs)
    yield Client(), app


def root(setup):
    return f"/api/v2/assistant/runs/{setup.base.run}"


def start(setup, client, **kwargs):
    body = dict(base_version_id=setup.base.version, instruction="Trim ending", idempotency_key="request-1")
    return client.post(root(setup) + "/turns", json={**body, **kwargs})


@pytest.mark.parametrize('agent_enabled',[False,True])
def test_essential_context_over_agent_budget_is_fixed_422_without_acceptance(setup,api,monkeypatch,agent_enabled):
    from app.assistant_planner import PlannerSubtitleCue
    from app.assistant_turns import AssistantTurn
    client,app=api
    svc=service(setup,agent_enabled=agent_enabled)
    original=svc.version_context
    def pinned(*args,**kwargs):
        value=original(*args,**kwargs)
        cues=tuple(PlannerSubtitleCue(cue_id=f'cue-{i}',start_sec=0.,end_sec=1.,text='private caption '*110) for i in range(20))
        return {**value,'subtitle_cues':cues,'editable_subtitle_cue_ids':tuple(c.cue_id for c in cues)}
    monkeypatch.setattr(svc,'version_context',pinned)
    app.dependency_overrides[get_assistant_service]=lambda:svc
    response=start(setup,client)
    if agent_enabled:
        assert response.status_code==422
        assert response.json()=={'detail':'Assistant initial context exceeds the supported 32 KiB limit; reduce the request or selected context'}
    else:
        assert response.status_code==202  # Legacy admission keeps its original transport budget.
    with setup.factory() as session:
        count=0 if agent_enabled else 1
        assert session.scalar(select(func.count()).select_from(AssistantTurn))==count
        assert session.scalar(select(func.count()).select_from(AssistantMessage))==count


def test_start_reopen_complete_accept_and_download_keep_paths_private(setup, api):
    client, _ = api
    response = start(setup, client)
    assert response.status_code == 202
    turn_id = response.json()["id"]
    assert start(setup, client).json()["id"] == turn_id
    with setup.factory.begin() as session:
        lease = claim_turn(session, turn_id, owner="worker", now=NOW)
        service(setup).finish(session, lease, reply(setup), now=NOW)
    accepted = client.post(root(setup) + f"/turns/{turn_id}/accept")
    assert accepted.status_code == 202, accepted.text
    job_id = accepted.json()["job"]["id"]
    assert client.post(root(setup) + f"/turns/{turn_id}/accept").json()["job"]["id"] == job_id
    context = client.get(root(setup) + "/context")
    assert context.status_code == 200
    payload = context.json()
    assert [m["role"] for m in payload["messages"]] == ["operator", "assistant"]
    assert len(payload["versions"]) == 2 and payload["turns"][0]["edit_job_id"] == job_id
    assert str(setup.data) not in context.text and "storage_key" not in context.text and "context_snapshot" not in context.text
    assert client.get(root(setup) + f"/jobs/{job_id}").json()["status"] == "queued"
    media = client.get(root(setup) + f"/versions/{setup.base.version}/media/preview")
    assert media.status_code == 200 and media.content == setup.base.path.read_bytes()
    assert media.headers["cache-control"] == "private, no-store"
    with setup.factory() as session:
        assert session.scalar(select(func.count()).select_from(AssistantJob)) == 1
        assert session.scalar(select(func.count()).select_from(AssistantMessage)) == 3  # includes private admission audit


def test_all_routes_recheck_project_scope(setup, api):
    client, app = api
    turn = start(setup, client).json()
    app.dependency_overrides[get_assistant_service] = lambda: AssistantTurnService(storage=setup.storage, policy=ProjectAllowlist(), enabled=True)
    paths = ["/context", f"/turns/{turn['id']}", f"/versions/{setup.base.version}/media/full",
        f"/catalog?base_version_id={setup.base.version}"]
    for path in paths:
        assert client.get(root(setup) + path).status_code == 404
    assert start(setup, client).status_code == 404
    assert client.post(root(setup) + f"/turns/{turn['id']}/cancel").status_code == 404


@pytest.mark.parametrize("indexes", [None, [0], [1], [0, 1]])
def test_accept_selected_changes_creates_one_job_and_retries_only_that_selection(setup, api, indexes):
    from app.assistant_contracts import EditPlan
    from app.assistant_jobs import claim_job, fail_job
    client, _ = api
    turn_id = start(setup, client).json()["id"]
    proposal = reply(setup).model_copy(update={"plan": EditPlan.model_validate({
        "run_id": setup.base.run, "base_version_id": setup.base.version,
        "operations": [
            {"kind": "image_fade", "start_sec": 0.5, "end_sec": 1.0, "direction": "out"},
            {"kind": "image_fade", "start_sec": 1.0, "end_sec": 1.5, "direction": "in"}]})})
    with setup.factory.begin() as session:
        lease = claim_turn(session, turn_id, owner="worker", now=NOW)
        service(setup).finish(session, lease, proposal, now=NOW)
    url = root(setup) + f"/turns/{turn_id}/accept"
    body = {} if indexes is None else {"operation_indexes": indexes}
    for invalid in ([], [2], [-1], [0, 0], [1, 0], [True], [0.0], ["0"]):
        assert client.post(url, json={"operation_indexes": invalid}).status_code == 422
    response = client.post(url, json=body)
    assert response.status_code == 202, response.text
    expected_indexes = [0, 1] if indexes is None else indexes
    assert response.json()["turn"]["accepted_operation_indexes"] == expected_indexes
    assert client.post(url, json=body).json()["job"]["id"] == response.json()["job"]["id"]
    different = [1] if expected_indexes == [0] else [0]
    assert client.post(url, json={"operation_indexes": different}).status_code == 409
    expected = [proposal.plan.operations[index].model_dump(mode="json") for index in expected_indexes]
    with setup.factory() as session:
        assert session.scalar(select(func.count()).select_from(AssistantJob)) == 1
        job = session.get(AssistantJob, response.json()["job"]["id"])
        assert job.accepted_plan["operations"] == expected
    with setup.factory.begin() as session:
        lease = claim_job(session, response.json()["job"]["id"], owner="worker", now=NOW)
        fail_job(session, lease, error_code="fixture_failure", now=NOW)
    retried = client.post(root(setup) + f"/turns/{turn_id}/retry-edit", json={"idempotency_key": "selected-retry"})
    assert retried.status_code == 202, retried.text
    assert retried.json()["turn"]["accepted_operation_indexes"] == expected_indexes
    with setup.factory() as session:
        assert session.scalar(select(func.count()).select_from(AssistantJob)) == 2
        assert session.get(AssistantJob, retried.json()["job"]["id"]).accepted_plan["operations"] == expected


def test_wrong_run_and_denied_source_do_not_reveal_metadata(setup, api):
    client, _ = api
    assert start(setup, client, sources=[dict(asset_id=setup.denied.artifact, version_id=setup.denied.version)]).status_code == 404
    assert client.get(root(setup) + f"/versions/{setup.donor.version}/media/full").status_code == 404
    response = client.get(root(setup) + f"/catalog?base_version_id={setup.base.version}")
    assert response.status_code == 200
    assert setup.denied.artifact not in response.text and "storage_key" not in response.text


def test_cancel_retry_and_selection_epoch(setup, api):
    client, _ = api
    turn = start(setup, client).json()
    cancelled = client.post(root(setup) + f"/turns/{turn['id']}/cancel")
    assert cancelled.json()["turn"]["status"] == "cancelled"
    retried = client.post(root(setup) + f"/turns/{turn['id']}/retry", json={"idempotency_key": "retry"})
    assert retried.status_code == 202 and retried.json()["parent_turn_id"] == turn["id"]
    assert client.post(root(setup) + f"/turns/{turn['id']}/accept").status_code == 409
    chosen = client.post(root(setup) + "/select", json={"version_id": setup.base.version, "expected_epoch": 0})
    assert chosen.status_code == 200 and chosen.json()["selection_epoch"] == 1
    assert client.post(root(setup) + "/select", json={"version_id": setup.base.version, "expected_epoch": 0}).status_code == 409


def test_disabled_default_dependency_never_creates_storage(setup, api, monkeypatch):
    client, app = api
    app.dependency_overrides.pop(get_assistant_service)
    from app.config import settings
    monkeypatch.setattr(settings, "ASSISTANT_ENABLED", False, raising=False)
    assert client.get(root(setup) + "/context").status_code == 404


def test_request_bounds_and_extra_private_fields_rejected(setup, api):
    client, _ = api
    for kwargs in (dict(instruction="x" * 4001), dict(idempotency_key="../../bad"), dict(path="/root/private")):
        assert start(setup, client, **kwargs).status_code == 422


def test_custom_option_pins_exact_proposal_and_inherits_authorized_donors(setup, api):
    from app.assistant_contracts import EditPlan
    from app.assistant_planner import PlannerContext
    from app.assistant_turns import AssistantTurn
    client, app = api
    original = start(setup, client, sources=[{"asset_id":setup.donor.artifact,"version_id":setup.donor.version}]).json()["id"]
    proposed = reply(setup).model_copy(update={"plan":EditPlan.model_validate({
        "run_id":setup.base.run,"base_version_id":setup.base.version,"operations":[
            {"kind":"image_fade","direction":"out","start_sec":0.5,"end_sec":1.0},
            {"kind":"audio_splice","start_sec":1.0,"end_sec":2.0,
             "source":{"asset_id":setup.donor.artifact,"version_id":setup.donor.version}}]})})
    with setup.factory.begin() as session:
        lease = claim_turn(session, original, owner="worker", now=NOW)
        service(setup).finish(session, lease, proposed, now=NOW)
    body = dict(base_version_id=setup.base.version,instruction="Keep change 1, adjust change 2",
        idempotency_key="custom",proposal_turn_id=original)
    result = client.post(root(setup)+"/turns",json=body)
    assert result.status_code == 202, result.text
    assert client.post(root(setup)+"/turns",json=body).json()["id"] == result.json()["id"]
    changed = {key:value for key,value in body.items() if key != "proposal_turn_id"}
    assert client.post(root(setup)+"/turns",json=changed).status_code == 409
    with setup.factory() as session:
        context = PlannerContext.model_validate(session.get(AssistantTurn,result.json()["id"]).context_snapshot)
        assert context.reference_turn_id == original
        assert context.reference_plan == proposed.plan
        assert [(s.asset_id,s.version_id) for s in context.sources] == [(setup.donor.artifact,setup.donor.version)]
        assert session.scalar(select(func.count()).select_from(AssistantJob)) == 0
    assert client.post(root(setup)+"/turns",json={**body,"idempotency_key":"wrong-base",
        "base_version_id":setup.donor.version}).status_code == 409
    app.dependency_overrides[get_assistant_service] = lambda: AssistantTurnService(
        storage=setup.storage,policy=ProjectAllowlist([setup.base.project]),enabled=True)
    assert client.post(root(setup)+"/turns",json={**body,"idempotency_key":"revoked"}).status_code == 404
    assert client.post(root(setup)+"/turns",json=body).status_code == 404


def test_failed_edit_retry_has_new_version_same_plan_and_no_inference(setup, api):
    from app.assistant_jobs import claim_job, fail_job
    from app.assistant_turns import AssistantTurnEdit
    client, _ = api
    turn_id = start(setup, client).json()["id"]
    with setup.factory.begin() as session:
        lease = claim_turn(session, turn_id, owner="worker", now=NOW)
        service(setup).finish(session, lease, reply(setup), now=NOW)
    first = client.post(root(setup) + f"/turns/{turn_id}/accept").json()["job"]
    with setup.factory.begin() as session:
        lease = claim_job(session, first["id"], owner="worker", now=NOW)
        fail_job(session, lease, error_code="fixture_failure", now=NOW)
    response = client.post(root(setup) + f"/turns/{turn_id}/retry-edit", json={"idempotency_key": "retry-1"})
    assert response.status_code == 202, response.text
    retried = response.json()["job"]
    assert retried["id"] != first["id"] and retried["version_id"] != first["version_id"]
    assert retried["base_version_id"] == first["base_version_id"]
    assert client.post(root(setup) + f"/turns/{turn_id}/retry-edit", json={"idempotency_key": "retry-1"}).json()["job"] == retried
    assert client.post(root(setup) + f"/turns/{turn_id}/retry-edit", json={"idempotency_key": "retry-2"}).status_code == 409
    context = client.get(root(setup) + "/context").json()
    assert len(context["turns"]) == 1 and len(context["jobs"]) == 2
    with setup.factory() as session:
        assert session.scalar(select(func.count()).select_from(AssistantTurnEdit)) == 2


def test_source_media_reauthorizes_target_and_donor(setup, api):
    client, _ = api
    query = f"?base_version_id={setup.base.version}&version_id={setup.donor.version}"
    response = client.get(root(setup) + f"/assets/{setup.donor.artifact}/media" + query)
    assert response.status_code == 200 and response.content == setup.donor.path.read_bytes()
    denied = client.get(root(setup) + f"/assets/{setup.denied.artifact}/media?base_version_id={setup.base.version}&version_id={setup.denied.version}")
    assert denied.status_code == 404


def test_bulk_import_refuses_denied_project_before_scanning(setup, api):
    client, _ = api
    response = client.post(root(setup) + "/sources/import", json={"project_id": setup.denied.project, "limit": 5})
    assert response.status_code == 404 and setup.denied.run not in response.text
    assert client.post(root(setup) + "/sources/import", json={"limit": 11}).status_code == 422


def test_bulk_import_scans_only_allowed_done_local_runs_and_isolates_failure(setup, api, monkeypatch):
    from app import api_assistant
    from app.models import Run
    from app.assistant_store import import_run_version
    client, _ = api
    with setup.factory.begin() as session:
        good = Run(project_id=setup.base.project, status="done", result_local_path=str(setup.base.path), result_duration_sec=3)
        bad = Run(project_id=setup.base.project, status="done", result_local_path=str(setup.donor.path), result_duration_sec=3)
        denied = Run(project_id=setup.denied.project, status="done", result_local_path=str(setup.denied.path), result_duration_sec=3)
        session.add_all([good, bad, denied])
        session.flush()
        good_id, bad_id, denied_id = good.id, bad.id, denied.id
    calls = []
    def imported(session, run, **kwargs):
        calls.append(run.id)
        if run.id == bad_id:
            raise ValueError("private local failure")
        return import_run_version(session, run, actor=kwargs["actor"], storage=kwargs["storage"])
    monkeypatch.setattr(api_assistant, "import_legacy_run", imported)
    response = client.post(root(setup) + "/sources/import", json={"limit": 5})
    assert response.status_code == 200, response.text
    assert set(calls) == {good_id, bad_id} and denied_id not in response.text
    assert response.json()["imported"][0]["run_id"] == good_id
    assert response.json()["unavailable"] == [{"run_id": bad_id, "status": "unavailable"}]
    assert "private local failure" not in response.text


def test_selected_video_fps_comes_from_verified_probe_for_context_and_turn(setup, api, monkeypatch):
    from app import assistant_media
    monkeypatch.setattr(assistant_media, "probe_trusted_media", lambda *args, **kwargs:
        assistant_media.MediaMetadata(3, 3, 160, 120, "30000/1001", "1:1", True))
    client, _ = api
    response = client.get(root(setup) + "/context")
    assert response.status_code == 200, response.text
    context = response.json()
    assert context["base_context"]["selected"]["fps"] == "30000/1001"
    assert context["versions"][0]["fps"] == "30000/1001"
    turn_id = start(setup, client).json()["id"]
    from app.assistant_turns import AssistantTurn
    with setup.factory() as session:
        assert session.get(AssistantTurn, turn_id).context_snapshot["selected"]["fps"] == "30000/1001"


def test_structured_player_passage_is_bounded_and_part_of_request_identity(setup, api):
    client, _ = api
    passage = {"start_sec": .5, "end_sec": 2}
    first = start(setup, client, selected_passage=passage)
    assert first.status_code == 202 and first.json()["selected_passage"] == passage
    assert start(setup, client, selected_passage={"start_sec": .5, "end_sec": 2.5}).status_code == 409
    assert start(setup, client, idempotency_key="bad-range", selected_passage={"start_sec": 0, "end_sec": 4}).status_code == 422


def test_history_cursor_pages_preserve_current_selection_and_hide_other_conversation_cursor(setup, api):
    from datetime import timedelta
    from app.assistant_store import ArtifactInput, reserve_version, publish_version, select_version
    from tests.test_assistant_turns import plan
    client, _ = api
    with setup.factory.begin() as session:
        for index in range(4):
            version = reserve_version(session, plan(setup), idempotency_key=f"version-{index}", actor="operator")
            artifact = ArtifactInput(setup.base.path, metadata={"duration_sec": 2, "fps": "30"})
            publish_version(session, version.id, full=artifact, preview=artifact, storage=setup.storage)
        select_version(session, setup.base.conversation, setup.base.version, expected_epoch=0)
        turns = [accept(setup, session, idempotency_key=f"turn-{index}", now=NOW + timedelta(seconds=index)).id for index in range(4)]
        unrelated = service(setup).accept(session, run_id=setup.donor.run, base_version_id=setup.donor.version,
            instruction="Trim", idempotency_key="other-conversation", actor="operator", now=NOW).id
    first = client.get(root(setup) + "/context?limit=2")
    assert first.status_code == 200, first.text
    first = first.json()
    assert [version["sequence"] for version in first["versions"]] == [4, 5]
    assert [turn["id"] for turn in first["turns"]] == turns[2:]
    assert first["selected_version"]["id"] == setup.base.version
    assert first["latest_version"]["sequence"] == 5
    cursor = first["history_cursors"]
    assert cursor == {"before_version_sequence": 4, "before_turn_id": turns[2]}
    second = client.get(root(setup) + f"/context?limit=2&before_version_sequence=4&before_turn_id={turns[2]}").json()
    assert [version["sequence"] for version in second["versions"]] == [2, 3]
    assert [turn["id"] for turn in second["turns"]] == turns[:2]
    assert second["history_cursors"] == {"before_version_sequence": 2, "before_turn_id": None}
    assert len(second["messages"]) == 2 and len(second["jobs"]) == 0
    final = client.get(root(setup) + "/context?limit=2&before_version_sequence=2").json()
    assert [version["sequence"] for version in final["versions"]] == [1]
    assert final["history_cursors"]["before_version_sequence"] is None
    assert client.get(root(setup) + f"/context?before_turn_id={unrelated}").status_code == 404
    assert client.get(root(setup) + "/context?limit=101").status_code == 422


def test_existing_proposal_acceptance_is_fenced_by_unresolved_main_result(setup, api):
    from app.assistant_models import AssistantRunResultIntent
    client, _ = api
    turn_id = start(setup, client).json()["id"]
    with setup.factory.begin() as session:
        lease = claim_turn(session, turn_id, owner="worker", now=NOW)
        service(setup).finish(session, lease, reply(setup), now=NOW)
        session.add(AssistantRunResultIntent(conversation_id=setup.base.conversation,
            run_id=setup.base.run, version_id=setup.base.version, artifact_id=setup.base.artifact,
            idempotency_key="pending-main", request_sha256="0" * 64, actor="operator",
            request_json={}, before_json={}, after_json={}))
    response = client.post(root(setup) + f"/turns/{turn_id}/accept")
    assert response.status_code == 409
    assert start(setup, client, idempotency_key="new-planning").status_code == 409
    with setup.factory() as session:
        assert session.scalar(select(func.count()).select_from(AssistantJob)) == 0


def test_bound_and_visible_turn_capabilities_do_not_depend_on_version_page(setup, api, monkeypatch):
    from app import assistant_media
    from app.assistant_store import ArtifactInput, reserve_version, publish_version
    from tests.test_assistant_turns import plan
    client, _ = api
    ids = []
    with setup.factory.begin() as session:
        for index in range(5):
            version = reserve_version(session, plan(setup), idempotency_key=f"capability-version-{index}", actor="operator")
            metadata = {"duration_sec": 3, "fps": "30"}
            if index == 0:
                metadata.update(needs_editing_preparation=True, editing_preparation={"kind":"audio_tail",
                    "added_frames":3,"added_duration_sec":.1,"target_duration_sec":3.1,"audio_copied":True})
            artifact = ArtifactInput(setup.base.path, metadata=metadata)
            publish_version(session, version.id, full=artifact, preview=artifact, storage=setup.storage)
            ids.append(version.id)
        turn = service(setup).accept(session, run_id=setup.base.run, base_version_id=ids[1],
            instruction="Use this earlier base", idempotency_key="off-page-turn", actor="operator", now=NOW)
        turn_id = turn.id
    probes = []
    original = assistant_media.probe_trusted_media
    def probe(*args, **kwargs):
        probes.append(args[0])
        return original(*args, **kwargs)
    monkeypatch.setattr(assistant_media, "probe_trusted_media", probe)
    response = client.get(root(setup) + "/context", params=[("limit", "1"), ("bound_version_ids", ids[0])])
    assert response.status_code == 200, response.text
    payload = response.json()
    assert [version["id"] for version in payload["versions"]] == [ids[-1]]
    assert payload["turns"][0]["id"] == turn_id
    assert payload["selected_version_id"] == payload["latest_version_id"] == ids[-1]
    assert payload["version_capabilities"] == [
        {"id":ids[0],"sequence":2,"status":"succeeded","needs_editing_preparation":True},
        {"id":ids[1],"sequence":3,"status":"succeeded","needs_editing_preparation":False}]
    assert all(set(item) == {"id","sequence","status","needs_editing_preparation"} for item in payload["version_capabilities"])
    # The existing selected-base probe remains; compact off-page hints do not
    # add a media probe or serialize private artifact metadata.
    assert len(probes) == 1
    assert str(setup.data) not in json.dumps(payload["version_capabilities"])


@pytest.mark.parametrize("owner", ["donor", "denied", "missing"])
def test_bound_capabilities_reject_other_conversation_and_unknown_versions(setup, api, owner):
    import uuid
    client, _ = api
    identifier = str(uuid.uuid4()) if owner == "missing" else getattr(setup, owner).version
    response = client.get(root(setup) + "/context", params={"bound_version_ids":identifier})
    assert response.status_code == 404
    assert response.json() == {"detail":"Assistant resource unavailable"}


@pytest.mark.parametrize("identifier", ["", "*", "not-a-uuid", "00000000000000000000000000000000"])
def test_bound_capabilities_require_canonical_entity_ids(setup, api, identifier):
    client, _ = api
    assert client.get(root(setup) + "/context", params={"bound_version_ids":identifier}).status_code == 422


def test_bound_capability_query_is_bounded_to_eight_entries(setup, api):
    client, _ = api
    params = [("bound_version_ids", setup.base.version)] * 8
    accepted = client.get(root(setup) + "/context", params=params)
    assert accepted.status_code == 200, accepted.text
    assert len(accepted.json()["version_capabilities"]) == 1
    assert client.get(root(setup) + "/context", params=params + params[:1]).status_code == 422
