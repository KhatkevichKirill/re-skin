"""Explicit private-operator all-project scope; isolated DB/files, no models."""
from pathlib import Path
from types import SimpleNamespace
import time
from uuid import uuid4

import pytest
from sqlalchemy import delete, select

from app.assistant_access import AssistantAccessDenied, ProjectAllowlist
from app.assistant_models import AssistantConversation
from app.assistant_turns import AssistantTurnService
from app.models import Run, VideoProject
from tests.test_assistant_turns import setup
from tests.test_api_assistant import api


@pytest.mark.parametrize("flag", [None, False, "true", "false", "1", 1, 0, [], {}, object()])
def test_unknown_or_nonboolean_flags_never_enable_all_projects(flag):
    policy = ProjectAllowlist(all_projects=flag)
    assert policy.all_projects is False and policy.has_scope is False
    assert policy.allows_project(str(uuid4())) is False
    approved, other = str(uuid4()), str(uuid4())
    restricted = ProjectAllowlist([approved], all_projects=flag)
    assert restricted.has_scope and restricted.allows_project(approved)
    assert not restricted.all_projects and not restricted.allows_project(other)


@pytest.mark.parametrize("configuration", ["*", "all", "broken", "bad,", ["*"], True, {}, 123])
def test_malformed_scope_denies_everything_even_with_literal_all_flag(configuration):
    policy = ProjectAllowlist(configuration, all_projects=True)
    assert not policy.has_scope and not policy.all_projects
    assert not policy.allows_project(str(uuid4()))


def test_literal_all_with_empty_list_covers_future_canonical_projects_only():
    policy = ProjectAllowlist("", all_projects=True)
    assert policy.has_scope and policy.all_projects and policy.project_ids == frozenset()
    future = str(uuid4())
    assert policy.require_project(future.upper()) == future
    assert policy.allows_project(future)
    for invalid in ("", "*", "all", future.replace("-", ""), "../../file", None, True):
        assert not policy.allows_project(invalid)
        with pytest.raises(AssistantAccessDenied):
            policy.require_project(invalid)


def all_service(setup):
    return AssistantTurnService(storage=setup.storage,
        policy=ProjectAllowlist([setup.base.project], all_projects=True), enabled=True)


def test_shared_factory_preserves_default_restriction_and_explicit_all_mode(setup):
    config = SimpleNamespace(ASSISTANT_PROJECT_IDS=setup.base.project)
    restricted = ProjectAllowlist.from_settings(config)
    assert restricted.has_scope and restricted.allows_project(setup.base.project)
    assert not restricted.allows_project(setup.denied.project)
    config.ASSISTANT_ALL_PROJECTS = True
    expanded = ProjectAllowlist.from_settings(config)
    assert expanded.allows_project(setup.denied.project)
    config.ASSISTANT_PROJECT_IDS = "malformed"
    assert not ProjectAllowlist.from_settings(config).has_scope


def test_all_scope_service_catalog_and_admission_accept_third_project(setup):
    from app.assistant_service import AssistantAdmissionService
    from app.assistant_contracts import EditPlan
    from tests.test_assistant_turns import NOW
    svc = all_service(setup)
    with setup.factory() as session:
        assert svc.require_run(session, setup.denied.run).project_id == setup.denied.project
        assert svc.conversation(session, setup.denied.run).id == setup.denied.conversation
        target = svc.catalog(session).require_target(setup.denied.version)
        assert target.project_id == setup.denied.project
        artifact = svc.catalog(session).resolve_reference(target_version_id=setup.base.version,
            asset_id=setup.denied.artifact, version_id=setup.denied.version)
        assert artifact.run_id == setup.denied.run
        with pytest.raises(AssistantAccessDenied):
            svc.catalog(session).resolve_reference(target_version_id=setup.base.version,
                asset_id=setup.denied.artifact, version_id=setup.base.version)
        plan = EditPlan.model_validate({"run_id": setup.denied.run, "base_version_id": setup.denied.version,
            "operations": [{"kind": "trim", "start_sec": 0, "end_sec": 2}]})
        admitted = AssistantAdmissionService(storage=setup.storage, policy=svc.policy, enabled=True).admit(
            session, run_id=setup.denied.run, plan=plan, instruction="Trim ending", idempotency_key="all-scope",
            actor="operator", now=NOW)
        assert admitted.base_version_id == setup.denied.version
        session.rollback()


@pytest.mark.parametrize("break_owner", ["wrong_project", "deleted_run", "deleted_project"])
def test_all_scope_still_requires_actual_exact_owner_bindings(setup, break_owner):
    svc = all_service(setup)
    with setup.factory.begin() as session:
        if break_owner == "wrong_project":
            session.get(Run, setup.denied.run).project_id = setup.base.project
        elif break_owner == "deleted_run":
            session.execute(delete(Run).where(Run.id == setup.denied.run))
        else:
            session.execute(delete(VideoProject).where(VideoProject.id == setup.denied.project))
    with setup.factory() as session:
        with pytest.raises(AssistantAccessDenied):
            svc.catalog(session).require_target(setup.denied.version)
        if break_owner != "wrong_project":
            with pytest.raises(AssistantAccessDenied):
                svc.require_run(session, setup.denied.run)


def test_explicit_catalog_filter_remains_narrow_and_invalid_filters_do_not_expand(setup):
    svc = all_service(setup)
    with setup.factory() as session:
        catalog = svc.catalog(session)
        page = catalog.list_artifacts(target_version_id=setup.base.version, project_id=setup.denied.project)
        assert {row.project_id for row in page.artifacts} == {setup.denied.project}
        assert catalog.list_artifacts(target_version_id=setup.base.version, project_id=str(uuid4())).artifacts == ()
        assert len(catalog.list_artifacts(target_version_id=setup.base.version).artifacts) == 3
        for invalid in ("", "*", "all", "malformed"):
            with pytest.raises(AssistantAccessDenied):
                catalog.list_artifacts(target_version_id=setup.base.version, project_id=invalid)


def test_future_project_api_import_context_catalog_and_restricted_rollback(setup, api, monkeypatch):
    from app.api_assistant import get_assistant_service
    from app import assistant_legacy, assistant_media
    client, app = api
    svc = all_service(setup)
    app.dependency_overrides[get_assistant_service] = lambda: svc
    monkeypatch.setattr(assistant_legacy, "probe_trusted_media", assistant_media.probe_trusted_media)
    with setup.factory.begin() as session:
        project = VideoProject(source_type="upload", source_ref="future-private-fixture")
        session.add(project)
        session.flush()
        run = Run(project_id=project.id, status="done", result_local_path=str(setup.base.path), result_duration_sec=3)
        session.add(run)
        session.flush()
        future_run, future_project = run.id, project.id
    root = f"/api/v2/assistant/runs/{future_run}"
    assert client.post(root + "/import").status_code == 200
    context = client.get(root + "/context")
    assert context.status_code == 200 and context.json()["selected_version_id"]
    selected = context.json()["selected_version_id"]
    page = client.get(root + "/catalog", params={"base_version_id": selected, "project_id": setup.denied.project})
    assert page.status_code == 200 and {row["project_id"] for row in page.json()["artifacts"]} == {setup.denied.project}
    for invalid in ("", "*", "malformed"):
        assert client.get(root + "/catalog", params={"base_version_id": selected, "project_id": invalid}).status_code >= 400
    svc.policy = ProjectAllowlist([setup.base.project])
    assert client.get(root + "/context").status_code == 404
    assert client.get(root + f"/versions/{selected}/media/full").status_code == 404


def test_explicit_source_import_reaches_only_requested_third_project(setup, api, monkeypatch):
    from app.api_assistant import get_assistant_service
    from app import assistant_legacy, assistant_media
    client, app = api
    app.dependency_overrides[get_assistant_service] = lambda: all_service(setup)
    monkeypatch.setattr(assistant_legacy, "probe_trusted_media", assistant_media.probe_trusted_media)
    created = []
    with setup.factory.begin() as session:
        for project in (setup.base.project, setup.denied.project):
            run = Run(project_id=project, status="done", result_local_path=str(setup.base.path), result_duration_sec=3)
            session.add(run)
            session.flush()
            created.append(run.id)
    root = f"/api/v2/assistant/runs/{setup.base.run}/sources/import"
    response = client.post(root, json={"project_id": setup.denied.project})
    assert response.status_code == 200, response.text
    assert {item["run_id"] for item in response.json()["imported"]} == {created[1]}
    for invalid in ("", "*", "malformed"):
        assert client.post(root, json={"project_id": invalid}).status_code == 422
    with setup.factory() as session:
        assert session.scalar(select(AssistantConversation).where(AssistantConversation.run_id == created[0])) is None


def test_internal_project_inventory_includes_third_and_future_project_without_new_tools(setup):
    from tests.test_assistant_internal_api import read
    svc = all_service(setup)
    with setup.factory.begin() as session:
        future = VideoProject(source_type="upload", source_ref="future-fixture")
        session.add(future)
        session.flush()
        future_id = future.id
        page, _ = read(setup, svc, session, "list_projects")
        assert {row.project_id for row in page.records} == {
            setup.base.project, setup.donor.project, setup.denied.project, future_id}
        runs, _ = read(setup, svc, session, "list_runs", project_id=setup.denied.project)
        assert {row.run_id for row in runs.records} == {setup.denied.run}


def test_enrichment_discovers_and_imports_formerly_excluded_donor(setup, monkeypatch):
    from app.assistant_enrichment import AssistantContextEnricher, _Budget
    from app import assistant_legacy, assistant_media
    svc = all_service(setup)
    enricher = AssistantContextEnricher(setup.factory, service=svc, asr=None)
    with setup.factory.begin() as session:
        context = svc.build_context(session, run_id=setup.base.run, base_version_id=setup.base.version,
            instruction="Find source speech")
        run = Run(project_id=setup.denied.project, status="done", result_local_path=str(setup.denied.path), result_duration_sec=3)
        session.add(run)
        session.flush()
        unimported = run.id
    budget = _Budget(time.monotonic() + 30, None, time.monotonic)
    donors = enricher._donors(context, budget, [])
    assert setup.denied.project in {row.project_id for row in donors}
    monkeypatch.setattr(assistant_legacy, "probe_trusted_media", assistant_media.probe_trusted_media)
    enricher._import_donors(context, budget, [])
    with setup.factory() as session:
        assert session.scalar(select(AssistantConversation).where(AssistantConversation.run_id == unimported)) is not None


@pytest.mark.parametrize("all_projects", [False, True])
def test_actual_api_and_supervisor_factories_use_identical_trusted_mode(setup, monkeypatch, all_projects):
    from app import api_assistant, assistant_store, assistant_supervisor, assistant_broker_client, storage
    from app.config import settings
    import worker.assistant_supervisor as entrypoint
    monkeypatch.setattr(settings, "ASSISTANT_ENABLED", True)
    monkeypatch.setattr(settings, "ASSISTANT_PROJECT_IDS", setup.base.project)
    monkeypatch.setattr(settings, "ASSISTANT_ALL_PROJECTS", all_projects)
    monkeypatch.setattr(settings, "ASSISTANT_AGENT_ENABLED", False)
    monkeypatch.setattr(settings, "ASSISTANT_ASR_SOCKET", "")
    monkeypatch.setattr(api_assistant, "AssistantArtifactStorage", lambda _: setup.storage)
    monkeypatch.setattr(assistant_store, "AssistantArtifactStorage", lambda _: setup.storage)
    monkeypatch.setattr(storage, "data_dir_root", lambda: setup.data)
    monkeypatch.setattr(assistant_broker_client, "BrokerClient", lambda _: object())
    monkeypatch.setattr(entrypoint.signal, "signal", lambda *args: None)
    captured = []
    class Started(Exception): pass
    class Supervisor:
        def __init__(self, factory, **kwargs):
            captured.append(kwargs["service"].policy)
        def run_once(self):
            raise Started()
    monkeypatch.setattr(assistant_supervisor, "AssistantSupervisor", Supervisor)
    api_policy = api_assistant.get_assistant_service().policy
    with pytest.raises(Started):
        entrypoint.main()
    assert captured == [api_policy]
    assert api_policy.allows_project(setup.denied.project) is all_projects
    monkeypatch.setattr(settings, "ASSISTANT_PROJECT_IDS", "malformed")
    with pytest.raises(SystemExit, match="empty or invalid"):
        entrypoint.main()
    assert not api_assistant.get_assistant_service().policy.has_scope
    from fastapi import HTTPException
    monkeypatch.setattr(settings, "ASSISTANT_ENABLED", False)
    with pytest.raises(HTTPException) as error:
        api_assistant.get_assistant_service()
    assert error.value.status_code == 404


@pytest.mark.parametrize("all_projects,enabled,expected", [(False, True, False), (True, True, True), (True, False, False)])
def test_rendered_third_project_run_uses_same_visibility_policy(setup, monkeypatch, all_projects, enabled, expected):
    from starlette.requests import Request
    from app import web_v2
    from app.config import settings
    monkeypatch.setattr(settings, "ASSISTANT_PROJECT_IDS", setup.base.project)
    monkeypatch.setattr(settings, "ASSISTANT_ALL_PROJECTS", all_projects)
    monkeypatch.setattr(settings, "ASSISTANT_ENABLED", enabled)
    request = Request({"type": "http", "method": "GET", "path": "/v2/runs/fixture", "headers": [],
        "scheme": "http", "server": ("test", 80), "query_string": b""})
    with setup.factory() as session:
        response = web_v2.run_detail(setup.denied.run, request, db=session)
        assert response.context["assistant_available"] is expected
        assert ('id="run-assistant"' in response.body.decode()) is expected


def test_orphaned_run_never_displays_assistant_in_all_mode(setup, monkeypatch):
    from starlette.requests import Request
    from app import web_v2
    from app.config import settings
    monkeypatch.setattr(settings, "ASSISTANT_PROJECT_IDS", "")
    monkeypatch.setattr(settings, "ASSISTANT_ALL_PROJECTS", True)
    monkeypatch.setattr(settings, "ASSISTANT_ENABLED", True)
    request = Request({"type": "http", "method": "GET", "path": "/v2/runs/fixture", "headers": [],
        "scheme": "http", "server": ("test", 80), "query_string": b""})
    with setup.factory() as session:
        original = session.get
        def missing_project(model, identifier, *args, **kwargs):
            return None if model is VideoProject and identifier == setup.denied.project else original(model, identifier, *args, **kwargs)
        monkeypatch.setattr(session, "get", missing_project)
        response = web_v2.run_detail(setup.denied.run, request, db=session)
        assert response.context["assistant_available"] is False
        assert 'id="run-assistant"' not in response.body.decode()


@pytest.mark.parametrize("value,expected", [(None, False), ("false", False), ("1", False), ("all", False), ("true", True)])
def test_environment_mode_parser_requires_explicit_true(monkeypatch, value, expected):
    import runpy
    import dotenv
    monkeypatch.setattr(dotenv, "load_dotenv", lambda: None)
    if value is None:
        monkeypatch.delenv("ASSISTANT_ALL_PROJECTS", raising=False)
    else:
        monkeypatch.setenv("ASSISTANT_ALL_PROJECTS", value)
    config = runpy.run_path(str(Path(__file__).parents[1] / "app/config.py"))
    assert config["settings"].ASSISTANT_ALL_PROJECTS is expected
