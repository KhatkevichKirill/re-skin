"""AICP-351 (AICP-344 stage 2): paid submit attempts — the receipt before every
paid provider call (docs/assistant-drain-check-design.md §5, §9 item 2).

Unit-level: the single-send classification of the kie client, the attempt
protocol (begin / outcome / park / resolution + audit / sender release), the
LLM ledger (pre-call draw, one send per (attempt, call_no)), the durable
transcription job's bounded retry + reaper, the §9 item 2 drain counting, and
the drain SQL / ops_status parity. Pipeline restart tests per intent site live
in tests/test_pipeline_v2.py (TestPaidSubmitAttemptsRestart) and
tests/test_recovery.py (segment image job).
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from contextlib import contextmanager
from datetime import timedelta

_db_fd, _db_path = tempfile.mkstemp(suffix=".db")
os.close(_db_fd)
os.environ.setdefault("DATABASE_URL", f"sqlite:///{_db_path}")
os.environ.setdefault("DATA_DIR", tempfile.mkdtemp())
os.environ.setdefault("KIE_API_KEY", "fake-key-for-tests")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import httpx  # noqa: E402
import pytest  # noqa: E402
import respx  # noqa: E402
from sqlalchemy import create_engine, func, select, text  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402

from app import kie_client, localisation, localisation_jobs, ops_status, paid_attempts  # noqa: E402
from app import spend as spend_mod  # noqa: E402
from app.db import Base  # noqa: E402
from app.localisation_jobs_models import ProjectTranscription  # noqa: E402
from app.models import KieSpendDraw, KieSpendReservation, VideoProject  # noqa: E402
from app.paid_attempts_models import (  # noqa: E402
    OWNER_LLM_TRANSCRIBE,
    OWNER_RUN_SEGMENT,
    PaidSubmitAttempt,
    PaidSubmitAttemptAction,
    utcnow,
)
from app.state_machine import ProjectStatus  # noqa: E402

JOBS = "https://jobs.test"


@pytest.fixture(scope="module")
def SF():
    eng = create_engine(f"sqlite:///{_db_path}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(eng)
    yield sessionmaker(bind=eng, autocommit=False, autoflush=False, expire_on_commit=False)
    Base.metadata.drop_all(eng)


@pytest.fixture()
def sessions(SF):
    @contextmanager
    def cm():
        db = SF()
        try:
            yield db
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    return cm


def _client() -> kie_client.KieClient:
    return kie_client.KieClient(api_key="k", upload_base="https://up.test", jobs_base=JOBS,
                                sleep_fn=lambda s: None)


def _get(SF, aid) -> PaidSubmitAttempt:
    db = SF()
    try:
        return db.get(PaidSubmitAttempt, aid)
    finally:
        db.close()


def _draw(SF, key):
    db = SF()
    try:
        return db.execute(select(KieSpendDraw).where(KieSpendDraw.call_key == key)).scalar_one_or_none()
    finally:
        db.close()


def _begin(sessions, owner_id, **kw):
    return paid_attempts.begin(
        OWNER_RUN_SEGMENT, owner_id, kind="video", run_id=kw.pop("run_id", "run-x"),
        request={"r": owner_id}, credits=kw.pop("credits", 100.0), scope_key="run:run-x",
        draw_kind="video", sessions=sessions, **kw)


# ---------------------------------------------------------------------------
# kie client: one POST, classified (§5.1)
# ---------------------------------------------------------------------------


class TestSingleSendCreate:
    @respx.mock
    def test_accepted(self):
        route = respx.post(f"{JOBS}/api/v1/jobs/createTask").mock(
            return_value=httpx.Response(200, json={"code": 200, "data": {"taskId": "t1"}}))
        assert _client().create_image_task(prompt="p", input_urls=["u"], once=True) == "t1"
        assert route.call_count == 1

    @respx.mock
    def test_5xx_is_unknown_and_never_resent(self):
        route = respx.post(f"{JOBS}/api/v1/jobs/createTask").mock(
            return_value=httpx.Response(502, text="bad gateway"))
        with pytest.raises(kie_client.KieTaskUnknown):
            _client().create_image_task(prompt="p", input_urls=["u"], once=True)
        assert route.call_count == 1          # the retrying path would send 4

    @respx.mock
    def test_transport_error_is_unknown(self):
        route = respx.post(f"{JOBS}/api/v1/jobs/createTask").mock(
            side_effect=httpx.ReadTimeout("slow"))
        with pytest.raises(kie_client.KieTaskUnknown):
            _client().create_image_task(prompt="p", input_urls=["u"], once=True)
        assert route.call_count == 1

    @respx.mock
    def test_4xx_and_envelope_are_refused(self):
        respx.post(f"{JOBS}/api/v1/jobs/createTask").mock(side_effect=[
            httpx.Response(422, json={"code": 422, "msg": "bad"}),
            httpx.Response(200, json={"code": 422, "msg": "too long"}),
        ])
        for _ in range(2):
            with pytest.raises(kie_client.KieTaskRefused):
                _client().create_image_task(prompt="p", input_urls=["u"], once=True)

    @respx.mock
    def test_2xx_without_task_id_is_unknown(self):
        respx.post(f"{JOBS}/api/v1/jobs/createTask").mock(
            return_value=httpx.Response(200, json={"data": {}}))
        with pytest.raises(kie_client.KieTaskUnknown):
            _client().create_image_task(prompt="p", input_urls=["u"], once=True)


class TestLlmClassification:
    def _c(self, status, body):
        return kie_client.classify_llm_response(
            status, body if isinstance(body, str) else json.dumps(body),
            kie_client._extract_chat_content)

    def test_matrix(self):
        ok = {"choices": [{"message": {"content": "hi"}}]}
        assert self._c(200, ok).kind == kie_client.LLM_ACCEPTED
        assert self._c(500, "x").kind == kie_client.LLM_UNAVAILABLE
        assert self._c(429, "x").kind == kie_client.LLM_UNAVAILABLE
        assert self._c(422, {"code": 422}).kind == kie_client.LLM_UNSUPPORTED
        assert self._c(401, "no").kind == kie_client.LLM_REFUSED
        assert self._c(200, {"code": 500, "msg": "maintenance"}).kind == kie_client.LLM_UNAVAILABLE
        assert self._c(200, {"code": 422, "msg": "not supported"}).kind == kie_client.LLM_UNSUPPORTED
        assert self._c(200, {"code": 400, "msg": "bad"}).kind == kie_client.LLM_REFUSED
        assert self._c(200, {"choices": []}).kind == kie_client.LLM_UNKNOWN
        assert self._c(200, "not json").kind == kie_client.LLM_UNKNOWN

    @respx.mock
    def test_send_once_posts_once_on_5xx(self):
        route = respx.post(f"{JOBS}/m/v1/chat/completions").mock(
            return_value=httpx.Response(503, text="down"))
        out = _client().chat_completion_send_once(model="m", messages=[], timeout_sec=5)
        assert out.kind == kie_client.LLM_UNAVAILABLE and route.call_count == 1


# ---------------------------------------------------------------------------
# §5.1 attempt protocol
# ---------------------------------------------------------------------------


class TestAttemptProtocol:
    def test_begin_commits_started_and_estimated_draw_before_the_call(self, SF, sessions):
        aid = _begin(sessions, "rs-begin", predecessor_task_id="old-1")
        row = _get(SF, aid)
        assert row.status == "started" and row.provider_task_id is None
        assert row.predecessor_task_id == "old-1" and row.attempt_no == 1
        d = _draw(SF, paid_attempts.attempt_call_key(aid))
        assert d is not None and d.mcredits == 100_000

    def test_one_open_attempt_per_owner(self, SF, sessions):
        _begin(sessions, "rs-open")
        with pytest.raises(paid_attempts.AttemptBlocked):
            _begin(sessions, "rs-open")

    def test_lease_lost_sends_nothing(self, SF, sessions):
        lease = paid_attempts.SubmitLease(check=lambda: False, token="t")
        with pytest.raises(paid_attempts.AttemptBlocked):
            _begin(sessions, "rs-lease", lease=lease)
        db = SF()
        assert paid_attempts.attempt_count(db, OWNER_RUN_SEGMENT, "rs-lease") == 0
        db.close()

    def test_pre_send_check_after_takeover_closes_unbilled(self, SF, sessions):
        state = {"ok": True}
        lease = paid_attempts.SubmitLease(check=lambda: state["ok"], token="t")
        aid = _begin(sessions, "rs-presend", lease=lease)
        state["ok"] = False
        with pytest.raises(paid_attempts.AttemptBlocked):
            paid_attempts.pre_send_check(aid, lease, scope_key="run:run-x", kind="video",
                                         sessions=sessions)
        row = _get(SF, aid)
        assert row.status == "failed_unbilled" and row.sender_released_at is not None
        assert _draw(SF, paid_attempts.attempt_call_key(aid)).mcredits == 0

    def test_accepted_moves_the_draw_to_the_task_id(self, SF, sessions):
        aid = _begin(sessions, "rs-acc")
        with sessions() as db:
            assert paid_attempts.record_outcome(db, aid, outcome="accepted", task_id="T-acc",
                                                credits=100.0, scope_key="run:run-x",
                                                draw_kind="video")
        row = _get(SF, aid)
        assert row.status == "accepted" and row.provider_task_id == "T-acc"
        assert _draw(SF, "T-acc").mcredits == 100_000
        assert _draw(SF, paid_attempts.attempt_call_key(aid)).mcredits == 0

    @pytest.mark.parametrize("exc,status,kept", [
        (kie_client.KieTaskUnknown("502"), "parked", True),
        (kie_client.KieTaskRefused("422"), "failed_unbilled", False),
        (ValueError("duration"), "failed_unbilled", False),
        (RuntimeError("boom"), "parked", True),
    ])
    def test_close_after_error(self, SF, sessions, exc, status, kept):
        aid = _begin(sessions, f"rs-err-{status}-{type(exc).__name__}")
        paid_attempts.close_after_error(aid, exc, scope_key="run:run-x", kind="video",
                                        sessions=sessions)
        row = _get(SF, aid)
        assert row.status == status
        d = _draw(SF, paid_attempts.attempt_call_key(aid))
        assert d.mcredits == (100_000 if kept else 0)

    def test_late_receipt_on_a_parked_attempt_does_not_touch_the_owner(self, SF, sessions):
        aid = _begin(sessions, "rs-late")
        with sessions() as db:
            row = db.get(PaidSubmitAttempt, aid)
            paid_attempts.park_started(db, row, reason="recovered")
        with sessions() as db:
            assert not paid_attempts.record_outcome(db, aid, outcome="accepted", task_id="T-late",
                                                    credits=100.0, scope_key="run:run-x")
        row = _get(SF, aid)
        assert row.status == "parked" and row.provider_task_id == "T-late"
        assert "late receipt" in row.error and row.sender_released_at is not None


class TestResolution:
    def _parked(self, SF, sessions, owner, *, released):
        aid = _begin(sessions, owner)
        with sessions() as db:
            row = db.get(PaidSubmitAttempt, aid)
            row.sender_host = "abcdef012345"
            proof = (lambda r: "boot changed") if released else (lambda r: None)
            paid_attempts.park_started(db, row, reason="recovered", evidence=proof)
        return aid

    def test_unreleased_sender_blocks_resolution_and_counts_active(self, SF, sessions):
        aid = self._parked(SF, sessions, "rs-unrel", released=False)
        db = SF()
        try:
            with pytest.raises(paid_attempts.ActionRefused) as e:
                paid_attempts.abandon(db, aid, actor="op", note="n")
            assert e.value.status_code == 409 and "not proven stopped" in e.value.detail
            db.rollback()
            c = ops_status.counts(db)
            assert c["active"]["by_kind"].get("paid_submit_attempts", 0) >= 1
        finally:
            db.close()

    def test_attestation_then_abandon_is_audited(self, SF, sessions):
        aid = self._parked(SF, sessions, "rs-attest", released=False)
        with sessions() as db:
            with pytest.raises(paid_attempts.ActionRefused):
                paid_attempts.release_sender(db, aid, attest="ffff00000000:absent")
        with sessions() as db:
            paid_attempts.release_sender(db, aid, attest="abcdef012345:exited")
        with sessions() as db:
            paid_attempts.abandon(db, aid, actor="operator", note="kie dashboard: nothing")
        row = _get(SF, aid)
        assert row.status == "abandoned"
        db = SF()
        acts = [a.action for a in db.execute(select(PaidSubmitAttemptAction).where(
            PaidSubmitAttemptAction.attempt_id == aid).order_by(PaidSubmitAttemptAction.at)).scalars()]
        db.close()
        assert acts == ["release_sender", "abandon"]

    def test_released_parked_counts_parked_not_active(self, SF, sessions):
        aid = self._parked(SF, sessions, "rs-rel", released=True)
        db = SF()
        try:
            n_parked = db.execute(text(
                "SELECT count(*) " + dict(ops_status.PARKED)["paid_submit_attempts"]
                + " AND id = :i"), {"i": aid}).scalar_one()
            n_active = db.execute(text(
                "SELECT count(*) FROM (SELECT * " + dict(ops_status.ACTIVE)["paid_submit_attempts"]
                + ") t WHERE id = :i"), {"i": aid}).scalar_one()
            assert (n_parked, n_active) == (1, 0)
            row = paid_attempts.resubmit(db, aid, actor="op", note="checked")
            assert row.status == "resubmitted"
            db.commit()
        finally:
            db.close()


# ---------------------------------------------------------------------------
# §5.2 LLM ledger
# ---------------------------------------------------------------------------


def _llm_attempt(sessions, owner, *, max_calls=2, lease_sec=60, fence=None):
    return paid_attempts.begin_llm(OWNER_LLM_TRANSCRIBE, owner, max_calls=max_calls,
                                   lease_sec=lease_sec, fence_token="tok", fence=fence,
                                   sessions=sessions)


class TestLlmLedger:
    def test_pre_call_draw_then_one_send_per_call_no(self, SF, sessions):
        aid = _llm_attempt(sessions, "led-1")
        a = paid_attempts.LlmLedger(aid, scope_key="s", kind="transcribe", sessions=sessions)
        b = paid_attempts.LlmLedger(aid, scope_key="s", kind="transcribe", sessions=sessions)
        assert a.claim() == 1
        assert _draw(SF, paid_attempts.llm_call_key(aid, 1)) is not None
        # A second executor of the SAME logical call (its own counter says 1)
        # loses — it can never take the next number instead.
        with pytest.raises(paid_attempts.LedgerFenced):
            b.claim()
        assert a.claim() == 2
        with pytest.raises(paid_attempts.LedgerFenced):  # max_calls = 2
            a.claim()
        assert _get(SF, aid).calls_sent == 2

    def test_superseded_before_post_does_not_send(self, SF, sessions):
        aid = _llm_attempt(sessions, "led-2")
        led = paid_attempts.LlmLedger(aid, scope_key="s", kind="transcribe", sessions=sessions)
        n = led.claim()
        with sessions() as db:
            db.get(PaidSubmitAttempt, aid).status = "superseded"
        with pytest.raises(paid_attempts.LedgerFenced):
            led.before_post(n)
        assert _draw(SF, paid_attempts.llm_call_key(aid, n)).mcredits == 0

    def test_transcribe_with_ledger_draws_before_and_never_after(self, SF, sessions, monkeypatch):
        aid = _llm_attempt(sessions, "led-3", max_calls=2)
        led = paid_attempts.LlmLedger(aid, scope_key="s3", kind="transcribe", sessions=sessions)
        seen = []

        class Fake:
            def upload_file(self, *a, **k):
                return "https://up.test/v.mp4"

            def chat_completion_send_once(self, *, model, **kw):
                # The draw of THIS call is committed before the POST.
                seen.append((model, _draw(SF, paid_attempts.llm_call_key(aid, len(seen) + 1))))
                if len(seen) == 1:
                    return kie_client.LlmSend(kie_client.LLM_UNSUPPORTED, error="422")
                return kie_client.LlmSend(kie_client.LLM_ACCEPTED, content=json.dumps({
                    "source_language": "en", "lines": []}), body={})

        monkeypatch.setattr(localisation, "_upload_proxy", _null_proxy)
        monkeypatch.setattr(localisation.settings, "LOCALISATION_TRANSCRIBE_FALLBACK_MODELS", "m2")
        calls = []
        monkeypatch.setattr(localisation.spend, "record_draw", lambda *a, **k: calls.append(a))
        out = localisation.transcribe_video("/x.mp4", kie=Fake(), model="m1", ledger=led)
        assert out["model"] == "m2" and [m for m, _ in seen] == ["m1", "m2"]
        assert all(d is not None for _, d in seen)
        assert calls == []                                  # no post-call draw
        assert _draw(SF, paid_attempts.llm_call_key(aid, 1)).mcredits == 0   # 422: settled
        assert _draw(SF, paid_attempts.llm_call_key(aid, 2)).mcredits > 0

    def test_unknown_outcome_raises_the_retryable_error(self, SF, sessions, monkeypatch):
        aid = _llm_attempt(sessions, "led-4", max_calls=1)
        led = paid_attempts.LlmLedger(aid, scope_key="s4", kind="transcribe", sessions=sessions)

        class Fake:
            def upload_file(self, *a, **k):
                return "u"

            def chat_completion_send_once(self, **kw):
                return kie_client.LlmSend(kie_client.LLM_UNKNOWN, error="timeout")

        monkeypatch.setattr(localisation, "_upload_proxy", _null_proxy)
        monkeypatch.setattr(localisation.settings, "LOCALISATION_TRANSCRIBE_FALLBACK_MODELS", "")
        with pytest.raises(localisation.LocalisationUnknownOutcome):
            localisation.transcribe_video("/x.mp4", kie=Fake(), model="m1", ledger=led)
        assert _draw(SF, paid_attempts.llm_call_key(aid, 1)).mcredits > 0   # kept


@contextmanager
def _null_proxy(path):
    yield path


# ---------------------------------------------------------------------------
# Durable transcription: bounded retry, reaper, §9 item 2
# ---------------------------------------------------------------------------


def _project(SF, tmp_path):
    src = tmp_path / f"src-{os.urandom(3).hex()}.mp4"
    src.write_bytes(b"x")
    db = SF()
    p = VideoProject(source_type="upload", source_ref="t.mp4", source_local_path=str(src),
                     status=ProjectStatus.ready, project_type="localisation")
    db.add(p)
    db.commit()
    pid = p.id
    db.close()
    return pid


def _admit(SF, pid):
    db = SF()
    try:
        job = localisation_jobs.admit_transcription(db, db.get(VideoProject, pid), origin="manual")
        db.commit()
        return job.id
    finally:
        db.close()


def _attempts(SF, owner):
    db = SF()
    try:
        return list(db.execute(select(PaidSubmitAttempt).where(
            PaidSubmitAttempt.owner_id == owner).order_by(PaidSubmitAttempt.attempt_no)).scalars())
    finally:
        db.close()


class TestTranscriptionAttempts:
    def test_reservation_is_the_worst_case_and_is_released_at_the_end(self, SF, tmp_path, monkeypatch):
        pid = _project(SF, tmp_path)
        jid = _admit(SF, pid)
        db = SF()
        res = db.execute(select(KieSpendReservation).where(
            KieSpendReservation.scope_key == spend_mod.transcribe_job_scope(jid))).scalar_one()
        db.close()
        chain = localisation.transcribe_chain_len()
        assert res.reserved_mcredits == int(round(3 * chain * spend_mod.llm_credits() * 1000))

        def fake(path, kie=None, ledger=None, **kw):
            ledger.claim()
            return {"lines": [{"id": 1, "text": "hi"}], "source_language": "en"}

        monkeypatch.setattr(localisation, "transcribe_video", fake)
        assert localisation_jobs.run_transcription_job(jid, 1, session_factory=SF) == "succeeded"
        db = SF()
        res = db.execute(select(KieSpendReservation).where(
            KieSpendReservation.scope_key == spend_mod.transcribe_job_scope(jid))).scalar_one()
        db.close()
        assert res.reserved_mcredits == res.consumed_mcredits      # unused part released
        assert [a.status for a in _attempts(SF, jid)] == ["accepted"]

    def test_unknown_outcome_retried_at_most_twice_then_interrupted(self, SF, tmp_path, monkeypatch):
        pid = _project(SF, tmp_path)
        jid = _admit(SF, pid)
        n = {"calls": 0}

        def fake(path, kie=None, ledger=None, **kw):
            n["calls"] += 1
            ledger.claim()
            raise localisation.LocalisationUnknownOutcome("timeout")

        monkeypatch.setattr(localisation, "transcribe_video", fake)
        assert localisation_jobs.run_transcription_job(jid, 1, session_factory=SF) == "interrupted"
        assert n["calls"] == 3
        assert [a.status for a in _attempts(SF, jid)] == ["unknown"] * 3

    def test_definite_failure_is_not_retried(self, SF, tmp_path, monkeypatch):
        pid = _project(SF, tmp_path)
        jid = _admit(SF, pid)

        def fake(path, kie=None, ledger=None, **kw):
            ledger.claim()
            raise localisation.LocalisationError("bad answer")

        monkeypatch.setattr(localisation, "transcribe_video", fake)
        assert localisation_jobs.run_transcription_job(jid, 1, session_factory=SF) == "failed"
        assert [a.status for a in _attempts(SF, jid)] == ["failed"]

    def _dead_running_job(self, SF, tmp_path, *, lease_sec):
        """A claimed job whose worker made attempt 1, claimed a call and froze."""
        pid = _project(SF, tmp_path)
        jid = _admit(SF, pid)
        db = SF()
        job = db.get(ProjectTranscription, jid)
        job.status, job.claim_token, job.attempts = "running", "tok-1", 1
        db.commit()
        db.close()

        @contextmanager
        def cm():
            d = SF()
            try:
                yield d
            finally:
                d.close()

        fence = localisation_jobs._job_fence(ProjectTranscription, jid, "tok-1")
        aid = paid_attempts.begin_llm(OWNER_LLM_TRANSCRIBE, jid, max_calls=1, lease_sec=lease_sec,
                                      fence_token="tok-1", fence=fence, sessions=cm)
        led = paid_attempts.LlmLedger(aid, scope_key=spend_mod.transcribe_job_scope(jid),
                                      kind="transcribe", fence=fence, sessions=cm)
        n = led.claim()
        return pid, jid, aid, led, n

    def _reap(self, SF, *, evidence, later_min=10):
        return localisation_jobs.reap(session_factory=SF, is_live=lambda q, j: False,
                                      enqueue_fn=lambda job: None,
                                      now=utcnow() + timedelta(minutes=later_min), evidence=evidence)

    def test_reaper_waits_while_the_sender_may_live(self, SF, tmp_path):
        pid, jid, aid, led, n = self._dead_running_job(SF, tmp_path, lease_sec=24 * 3600)
        self._reap(SF, evidence=lambda r: None, later_min=0)     # observe
        self._reap(SF, evidence=lambda r: None)                  # past grace: still waits
        assert _get(SF, aid).status == "started"
        db = SF()
        assert db.get(ProjectTranscription, jid).status == "running"
        db.close()

    @pytest.mark.parametrize("outcome,job_status", [("failed", "failed"),
                                                    ("accepted", "interrupted")])
    def test_definite_outcome_then_death_is_never_rebilled(self, SF, tmp_path, outcome,
                                                           job_status):
        """Independent review: the worker committed a DEFINITE attempt outcome
        (a billed answer that failed validation, or an accepted answer) and died
        before writing the job row. The reaper ends the job with that outcome and
        starts no new attempt (§5.2: only an unknown outcome is retried)."""
        pid, jid, aid, led, n = self._dead_running_job(SF, tmp_path, lease_sec=24 * 3600)
        paid_attempts.finish_llm_own_session(aid, outcome, error="bad answer: not JSON",
                                             sessions=_cm(SF))
        sent = []
        kw = dict(session_factory=SF, is_live=lambda q, j: False, enqueue_fn=sent.append,
                  evidence=lambda r: None)
        localisation_jobs.reap(now=utcnow() + timedelta(minutes=10), **kw)
        localisation_jobs.reap(now=utcnow() + timedelta(minutes=20), **kw)
        assert not any(j.id == jid for j in sent)
        assert [a.status for a in _attempts(SF, jid)] == [outcome]
        db = SF()
        job = db.get(ProjectTranscription, jid)
        p = db.get(VideoProject, pid)
        db.close()
        assert (job.status, job.claim_token) == (job_status, None)
        assert p.transcript_status == "failed"
        if outcome == "failed":
            assert "bad answer" in job.error
        else:
            assert "billed" in job.error and "not re-run" in job.error

    def test_proof_retries_with_a_new_dispatch(self, SF, tmp_path):
        pid, jid, aid, led, n = self._dead_running_job(SF, tmp_path, lease_sec=24 * 3600)
        self._reap(SF, evidence=lambda r: "boot changed", later_min=0)
        out = self._reap(SF, evidence=lambda r: "boot changed")
        assert out.get("retried", 0) >= 1
        assert _get(SF, aid).status == "unknown"
        db = SF()
        job = db.get(ProjectTranscription, jid)
        assert (job.status, job.claim_token, job.dispatch_gen) == ("queued", None, 2)
        db.close()

    def test_9_2_superseded_sender_stays_active_after_the_job_ends(self, SF, tmp_path, monkeypatch):
        """Design §9 item 2: lease ran out, no proof -> attempt 1 superseded (still
        active); attempt 2 finishes the job; attempt 1 — paused after its
        permit — sends; drain-check / ops_status / GET /ops/admission keep it
        active until its own acknowledgement."""
        pid, jid, aid1, led1, n1 = self._dead_running_job(SF, tmp_path, lease_sec=1)
        self._reap(SF, evidence=lambda r: None, later_min=0)
        assert self._reap(SF, evidence=lambda r: None).get("retried", 0) >= 1
        assert _get(SF, aid1).status == "superseded"

        def fake(path, kie=None, ledger=None, **kw):
            ledger.claim()
            return {"lines": [], "source_language": "en"}

        monkeypatch.setattr(localisation, "transcribe_video", fake)
        assert localisation_jobs.run_transcription_job(jid, 2, session_factory=SF) == "succeeded"

        def active_now():
            db = SF()
            try:
                frag = dict(ops_status.ACTIVE)["paid_submit_attempts"]
                sql_n = db.execute(text(f"SELECT count(*) FROM (SELECT id {frag}) t "
                                        "WHERE t.id = :i"), {"i": aid1}).scalar_one()
                body = ops_status.admission_status(db, include_rq=False)
                return sql_n, body["active"]["by_kind"].get("paid_submit_attempts", 0), body["drained"]
            finally:
                db.close()

        sql_n, api_n, drained = active_now()
        assert sql_n == 1 and api_n >= 1 and drained is False
        # The stale worker's permit for call 1 was taken BEFORE the supersede:
        # it may still send it (bounded) — and it stays counted while it does.
        assert active_now()[0] == 1
        # Its NEXT claim is refused: nothing more is sent.
        with pytest.raises(paid_attempts.LedgerFenced):
            led1.claim()
        assert active_now()[0] == 1
        # Its own acknowledgement after the last I/O returned: stopped.
        paid_attempts.finish_llm_own_session(aid1, "unknown", sessions=_cm(SF))
        assert _get(SF, aid1).status == "stopped"
        assert active_now()[0] == 0

    def test_9_2_superseded_released_by_sweep_proof(self, SF, tmp_path):
        pid, jid, aid1, led1, n1 = self._dead_running_job(SF, tmp_path, lease_sec=1)
        self._reap(SF, evidence=lambda r: None, later_min=0)
        self._reap(SF, evidence=lambda r: None)
        assert _get(SF, aid1).status == "superseded"
        paid_attempts.sweep(sessions=_cm(SF), evidence=lambda r: None)
        assert _get(SF, aid1).status == "superseded"
        assert paid_attempts.sweep(sessions=_cm(SF), evidence=lambda r: "horse reaped")["stopped"] >= 1
        assert _get(SF, aid1).status == "stopped"

    def test_stale_worker_result_is_discarded(self, SF, tmp_path, monkeypatch):
        """The reaper superseded attempt 1 and re-dispatched; the stale worker's
        final write fails the claim fence and writes nothing."""
        pid, jid, aid1, led1, n1 = self._dead_running_job(SF, tmp_path, lease_sec=1)
        self._reap(SF, evidence=lambda r: None, later_min=0)
        self._reap(SF, evidence=lambda r: None)
        db = SF()
        try:
            assert localisation_jobs._fenced(db, ProjectTranscription, jid, "tok-1") is None
        finally:
            db.close()


def _cm(SF):
    @contextmanager
    def cm():
        d = SF()
        try:
            yield d
        finally:
            d.close()

    return cm


# ---------------------------------------------------------------------------
# Drain SQL parity
# ---------------------------------------------------------------------------


def test_drain_check_sql_has_the_attempt_rows():
    root = os.path.join(os.path.dirname(__file__), "..", "..", "deploy", "drain-check.sql")
    if not os.path.exists(root):
        pytest.skip("repo root not mounted")
    sql = open(root).read()
    for state, rows in (("active", ops_status.ACTIVE), ("parked", ops_status.PARKED)):
        frag = dict(rows)["paid_submit_attempts"]
        assert f"SELECT '{state}', 'paid_submit_attempts', count(*) {frag}" in sql


# ---------------------------------------------------------------------------
# Caption re-dispatch enqueue (Terra gate r2: RQ rejects positional args next
# to an explicit kwargs=)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dispatch,expected", [(None, {}), ("n1", {"dispatch": "n1"})])
def test_enqueue_process_caption_is_rq_valid(monkeypatch, dispatch, expected):
    import fakeredis
    from rq import Queue

    from app import tasks

    q = Queue("process", connection=fakeredis.FakeStrictRedis())
    monkeypatch.setattr(tasks, "_get_queue", lambda name: q)
    tasks.enqueue_process_caption("cap-1", dispatch=dispatch)
    (job,) = q.jobs
    assert job.args == ("cap-1",) and job.kwargs == expected
    assert job.func_name == "app.tasks.run_process_caption"
