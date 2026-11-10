"""Bounded public-history verification without weakening live authorization."""
import hashlib
import os
import time
from types import SimpleNamespace

import pytest

from app import assistant_store as store
from app.api_assistant import _turn, get_context, get_turn
from app.assistant_access import AssistantAccessDenied, ProjectAllowlist
from app.assistant_agent_contracts import AgentActionEnvelope
from app.assistant_media import AssistantMediaCancelled
from app.assistant_models import AssistantArtifact
from app.assistant_store import ArtifactIntegrityError, AssistantArtifactStorage
from app.assistant_turns import AssistantTurn
from app.models import Run
from tests.test_assistant_agent_loop import fixture_loop
from tests.test_assistant_turns import NOW, service, setup


def count_full_hashes(monkeypatch):
    calls = []
    original = hashlib.sha256

    def digest(*args, **kwargs):
        if not args and not kwargs:
            calls.append(True)
        return original(*args, **kwargs)

    # Only storage's streaming digest; context/JPEG fingerprints stay real.
    monkeypatch.setattr(store, "hashlib", SimpleNamespace(sha256=digest))
    return calls


def age_for_cache(*paths):
    # Keep mutation tests on real filesystem timestamps. Old immutable objects
    # may use the cache; objects in the current timestamp tick must rehash.
    wait = (max(path.stat().st_ctime_ns for path in paths) + 1_100_000_000 - time.time_ns()) / 1e9
    if wait > 0:
        time.sleep(min(wait, 1.2))


def age_objects(storage):
    age_for_cache(*(storage.root / "objects").glob("*/*"))


def inventory_loop(setup):
    svc, turn_id, broker, media, supervisor = fixture_loop(setup)
    scope = {"run_id": setup.base.run, "base_version_id": setup.base.version}
    broker.actions[:0] = [
        AgentActionEnvelope.model_validate({"action": dict(kind=kind, **scope, **fields)}).action
        for kind, fields in (
            ("list_projects", {}),
            ("list_runs", {"project_id": setup.donor.project}),
            ("list_versions", {"source_run_id": setup.donor.run}),
            ("list_assets", {"source_version_id": setup.donor.version}),
        )
    ]
    supervisor.run_once()
    return svc, turn_id


def test_multistep_public_state_hashes_once_and_discards_cache_after_return(setup, monkeypatch):
    svc, turn_id = inventory_loop(setup)
    age_objects(setup.storage)
    hashes = count_full_hashes(monkeypatch)
    with setup.factory() as session:
        turn = session.get(AssistantTurn, turn_id)
        assert turn.status == "succeeded"
        first = svc.agent_public_state(session, turn)
        assert len(first["steps"]) == 8
        assert len(hashes) == 1
        assert svc.agent_public_state(session, turn) == first
        assert len(hashes) == 2


@pytest.mark.parametrize("endpoint", ["context", "turn"])
def test_history_response_computes_sanitized_state_once_per_turn(setup, monkeypatch, endpoint):
    svc, turn_id = inventory_loop(setup)
    original = svc.agent_public_state
    calls = []

    def state(session, turn):
        calls.append(turn.id)
        return original(session, turn)

    monkeypatch.setattr(svc, "agent_public_state", state)
    with setup.factory() as session:
        def response():
            if endpoint == "context":
                return get_context(setup.base.run, db=session, service=svc)
            return get_turn(setup.base.run, turn_id, db=session, service=svc)
        first = response()
        assert calls == [turn_id]
        assert response() == first
        assert calls == [turn_id, turn_id]


def test_computed_none_and_invisible_or_other_turn_state_do_not_leak(setup, monkeypatch):
    svc = service(setup, agent_enabled=True)
    with setup.factory.begin() as session:
        turn = svc.accept(session, run_id=setup.base.run, base_version_id=setup.base.version,
            instruction="No steps yet", idempotency_key="no-steps", actor="operator", now=NOW)
        original = svc.agent_public_state
        calls = []

        def state(db, current):
            calls.append(current.id)
            return original(db, current)

        monkeypatch.setattr(svc, "agent_public_state", state)
        assert get_turn(setup.base.run, turn.id, db=session, service=svc)["analysis"] is None
        assert calls == [turn.id]
        assert _turn(turn, session, svc, visible=True,
            agent_states={setup.donor.version: {"private": "foreign"}})["analysis"] is None
        assert calls == [turn.id, turn.id]
        hidden = _turn(turn, session, svc, visible=False,
            agent_states={turn.id: {"private": "must be hidden"}})
        assert hidden["analysis"] is None and hidden["source_access_changed"]
        assert calls == [turn.id, turn.id]


@pytest.mark.parametrize("revocation", ["policy", "run_owner"])
def test_reused_bytes_never_mask_live_scope_changes(setup, revocation):
    svc = service(setup)
    with setup.factory.begin() as session:
        context = svc.build_context(session, run_id=setup.base.run, base_version_id=setup.base.version,
            instruction="Compare", source_refs=((setup.donor.artifact, setup.donor.version),))
        cache = {}
        age_objects(setup.storage)
        svc.reauthorize_context(session, context, verification_cache=cache)
        assert len(cache) == 2
        if revocation == "policy":
            svc.policy = ProjectAllowlist([setup.base.project])
        else:
            session.get(Run, setup.donor.run).project_id = setup.base.project
            session.flush()
        with pytest.raises(AssistantAccessDenied):
            svc.reauthorize_context(session, context, verification_cache=cache)


@pytest.fixture
def pinned(tmp_path):
    storage = AssistantArtifactStorage(tmp_path)
    source = tmp_path / "owned-input.bin"
    source.write_bytes(b"owned immutable bytes")
    item = storage.pin(source)
    return storage, item, storage.path_for(item.storage_key)


@pytest.mark.parametrize("fresh", [True, False])
def test_cache_rechecks_same_size_content_even_with_restored_mtime(pinned, monkeypatch, fresh):
    storage, item, path = pinned
    if not fresh:
        age_for_cache(path)
    hashes = count_full_hashes(monkeypatch)
    cache = {}
    storage.verify(item, verification_cache=cache)
    assert bool(cache) == (not fresh)
    before = path.stat()
    os.chmod(path, 0o600)
    path.write_bytes(b"X" * before.st_size)
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert path.stat().st_size == before.st_size and path.stat().st_mtime_ns == before.st_mtime_ns
    with pytest.raises(ArtifactIntegrityError):
        storage.verify(item, verification_cache=cache)
    assert len(hashes) == 2


@pytest.mark.parametrize("replacement", ["missing", "symlink", "new_inode"])
def test_cache_reopens_and_validates_file_identity(pinned, tmp_path, monkeypatch, replacement):
    storage, item, path = pinned
    age_for_cache(path)
    hashes = count_full_hashes(monkeypatch)
    cache = {}
    storage.verify(item, verification_cache=cache)
    replacement_file = tmp_path / "owned-replacement.bin"
    replacement_file.write_bytes(b"owned immutable bytes")
    path.unlink()
    if replacement == "symlink":
        path.symlink_to(replacement_file)
    elif replacement == "new_inode":
        os.replace(replacement_file, path)
    if replacement == "new_inode":
        assert storage.verify(item, verification_cache=cache) == path
        assert len(hashes) == 2
    else:
        with pytest.raises(ArtifactIntegrityError):
            storage.verify(item, verification_cache=cache)


def test_cache_hit_still_obeys_cancellation(pinned, monkeypatch):
    storage, item, path = pinned
    age_for_cache(path)
    hashes = count_full_hashes(monkeypatch)
    cache = {}
    storage.verify(item, verification_cache=cache)
    with pytest.raises(AssistantMediaCancelled):
        storage.verify(item, verification_cache=cache, cancelled=lambda: True)
    assert len(hashes) == 1


def test_fresh_or_future_ctime_never_enters_byte_cache(pinned, monkeypatch):
    storage, item, path = pinned
    hashes = count_full_hashes(monkeypatch)
    monkeypatch.setattr(store, "time", SimpleNamespace(time_ns=lambda: path.stat().st_ctime_ns - 1))
    cache = {}
    for _ in range(2):
        storage.verify(item, verification_cache=cache)
    assert cache == {} and len(hashes) == 2


def test_failed_hash_is_never_cached(pinned, monkeypatch):
    storage, item, path = pinned
    os.chmod(path, 0o600)
    path.write_bytes(b"X" * path.stat().st_size)
    hashes = count_full_hashes(monkeypatch)
    cache = {}
    for _ in range(2):
        with pytest.raises(ArtifactIntegrityError):
            storage.verify(item, verification_cache=cache)
    assert cache == {} and len(hashes) == 2

