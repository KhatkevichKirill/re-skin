"""Rebuild from reference — the integrated feature end to end (waves 1 + 2a + 2b).

One rebuild project is driven through the whole path on SQLite, with real
ffmpeg on tiny synthetic media and a fake kie.ai that COUNTS every POST:

  create (upload) -> analyze (real shot detection) -> storyboard -> keyframes
  + sheet -> regenerate one + select the older attempt -> keep brand cards ->
  launch through the project page's form fields (Kling default, and a Seedance
  run) -> chunk attempts -> fit -> QA (free part + automatic character check)
  -> stitch with the original soundtrack + the language-matched library end
  card -> delivery stubbed at the Drive client -> re-roll one chunk (delivered
  in place) -> a refused re-roll -> Retry (delivered in place).

Asserted: exactly one provider POST per paid row; every draw equals its row's
cost and every scope's consumed total equals its draws; the final has the
source's frame count; the v2v requests are byte-identical to main 730216b's
(tests/fixtures/rebuild_v2v_request_bytes_main730.json, recorded from a clean
export of main with the same recipe as test_rebuild_core).

Also the integration seams: the LLM dispatcher routes every kind, the paid-call
reaper re-drives a lost character-check row through it, upload names are
unique per row, and the project page's precondition list IS the server's.
"""

from __future__ import annotations

import html as html_mod
import importlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta

_db_fd, _db_path = tempfile.mkstemp(suffix=".db")
os.close(_db_fd)
os.environ.setdefault("DATABASE_URL", f"sqlite:///{_db_path}")
os.environ.setdefault("DATA_DIR", tempfile.mkdtemp())
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest  # noqa: E402
from sqlalchemy import create_engine, event, func, select  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402

from app import pipeline_rebuild as pr, rebuild_jobs as jobs, rebuild_launch  # noqa: E402
from app import rebuild_llm_steps, rebuild_paid_call as pc, rebuild_storyboard as sbmod  # noqa: E402
from app import spend  # noqa: E402
from app.config import settings  # noqa: E402
from app.db import Base  # noqa: E402
from app.kie_client import SendOutcome  # noqa: E402
from app.models import (  # noqa: E402
    KieSpendDraw,
    KieSpendReservation,
    LibraryClip,
    Run,
    RunSegment,
    SegmentDef,
    VideoProject,
)
from app.rebuild_models import (  # noqa: E402
    PAID_CALL_MODELS,
    ProjectStoryboard,
    RebuildChunkAttempt,
    RebuildKeyframe,
    RebuildLlmCall,
    RebuildShot,
)
from app.state_machine import ProjectStatus, RunStatus, SegmentStatus  # noqa: E402

BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FPS = 30
W, H = 180, 320
N_SHOTS = 10          # 2 s each; the last one is the brand card / end card
SHOT_SEC = 2.0
# Alternating dark / bright, so every hard cut is an unmistakable scene change.
COLORS = ("black", "white", "navy", "yellow", "maroon", "cyan", "darkgreen", "pink",
          "purple", "lime")

GLOBAL = {
    "style": "3D animated cartoon",
    "characters": [{"id": "C1", "name_hint": "Hero", "look": "tall, red cap",
                    "outfit": "blue jacket", "role": "main"}],
    "sheet_groups": [{"key": "street", "description": "street clothes",
                      "characters": ["C1"]}],
    "audio": {"kind": "voiceover", "lines": ["hello"]},
    "on_screen_text": ["SALE 50%"],
}


def ff(*args):
    subprocess.run(["ffmpeg", "-v", "error", "-y", *args], check=True)


def make_video(path, *, spans, fps=FPS, size=(W, H), audio=True):
    """Solid colours with hard cuts; *spans* = [(colour, seconds)]."""
    args = []
    for c, d in spans:
        args += ["-f", "lavfi", "-i", f"color=c={c}:s={size[0]}x{size[1]}:r={fps}:d={d}"]
    n = len(spans)
    flt = "".join(f"[{i}:v]" for i in range(n)) + f"concat=n={n}:v=1:a=0[v]"
    total = sum(d for _c, d in spans)
    if audio:
        args += ["-f", "lavfi", "-i", f"sine=frequency=440:duration={total}"]
    out = ["-filter_complex", flt, "-map", "[v]"]
    if audio:
        out += ["-map", f"{n}:a", "-c:a", "aac", "-shortest"]
    ff(*args, *out, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", str(fps), str(path))
    return str(path)


def frames_of(path):
    return pr.count_frames(path)


def has_audio(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries",
                          "stream=index", "-of", "csv=p=0", path], capture_output=True, text=True)
    return bool(out.stdout.strip())


def _text(messages) -> str:
    return " ".join(p.get("text", "") for p in messages[0]["content"]
                    if isinstance(p, dict) and p.get("type") == "text")


def accepted(obj) -> SendOutcome:
    content = obj if isinstance(obj, str) else "```json\n" + json.dumps(obj) + "\n```"
    return SendOutcome("accepted", content=content,
                       body={"choices": [{"message": {"content": content}}]}, http_status=200)


_SHOT_RE = re.compile(r"Shot \d+ \((\d+(?:\.\d+)?)-(\d+(?:\.\d+)?) s\)")
_RUNS_RE = re.compile(r"runs from (\d+(?:\.\d+)?) s to (\d+(?:\.\d+)?) s")


class FakeKie:
    """kie.ai for the whole feature: chat (storyboard + character checks),
    GPT Image tasks (keyframes), video tasks (chunks). Counts every POST and
    never touches the network. A video task's result is a real mp4 of the
    requested duration with hard cuts at the prompt's shot times."""

    VIDEO_CREDITS_PER_SEC = {"kling-4-0": 28.5, "seedance-2-5": 63.0}

    def __init__(self, workdir):
        self.workdir = workdir
        self.chats: list = []      # (model, kind, text)
        self.creates: list = []    # payloads
        self.uploads: list = []    # (path, upload_path, file_name)
        self.polls: list = []
        self.tasks: dict = {}
        self.refuse_next_video = 0

    # -- uploads (free) --
    def upload_file(self, path, upload_path="", file_name=None):
        assert os.path.exists(path), path
        self.uploads.append((path, upload_path, file_name))
        return f"https://up.example/{upload_path}/{file_name or os.path.basename(path)}"

    # -- chat --
    def chat_completion_once(self, model, messages, *, timeout, response_format=None):
        text = _text(messages)
        if "shot-by-shot remake" in text:
            self.chats.append((model, "global", text))
            return accepted(GLOBAL)
        if "contact sheet" in text:
            self.chats.append((model, "character_check", text))
            n = int(re.search(r"contact sheet of (\d+) row", text).group(1))
            shots = [{"shot": i, "present": ["C1"], "missing": [], "identity_drift": []}
                     for i in range(1, n + 1)]
            if n <= 2:  # the short last chunk: the hero is missing in its first shot
                shots[0] = {"shot": 1, "present": [], "missing": ["C1"], "identity_drift": []}
            return accepted({"shots": shots})
        self.chats.append((model, "shot", text))
        start = float(_RUNS_RE.search(text).group(1))
        brand = start >= (N_SHOTS - 1) * SHOT_SEC - 1e-6
        return accepted({"location": "street", "framing": "close-up", "camera": "slow push-in",
                         "characters": [] if brand else ["C1"],
                         "action": "The logo." if brand else "Hero walks.",
                         "speaking_on_camera": None, "is_brand_card": brand,
                         "sheet_key": "street"})

    # -- tasks --
    def create_task_once(self, payload, *, timeout):
        self.creates.append(payload)
        is_video = "duration" in (payload.get("input") or {})
        if is_video and self.refuse_next_video:
            self.refuse_next_video -= 1
            return SendOutcome("refused", http_status=200, body={"code": 422, "msg": "nope"})
        tid = f"T{len(self.creates)}"
        self.tasks[tid] = payload
        return SendOutcome("accepted", task_id=tid, http_status=200)

    def get_task(self, task_id):
        self.polls.append(task_id)
        payload = self.tasks[task_id]
        inp = payload.get("input") or {}
        if "duration" in inp:
            model = "kling-4-0" if "kling" in payload["model"] else "seedance-2-5"
            credits = round(self.VIDEO_CREDITS_PER_SEC[model] * int(inp["duration"]), 1)
            url = f"https://cdn.example/{task_id}.mp4"
        else:
            credits, url = 9, f"https://cdn.example/{task_id}.png"
        return {"state": "success", "creditsConsumed": credits,
                "resultJson": json.dumps({"resultUrls": [url]})}

    def download_result(self, url, dst):
        tid = os.path.splitext(os.path.basename(url))[0]
        payload = self.tasks[tid]
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if url.endswith(".png"):
            from PIL import Image

            Image.new("RGB", (36, 64), (10, 20, 30)).save(dst, format="PNG")
            return
        inp = payload["input"]
        dur = int(inp["duration"])
        cuts = [(float(a), float(b)) for a, b in _SHOT_RE.findall(inp["prompt"])]
        spans, t = [], 0.0
        for i, (_a, b) in enumerate(cuts):
            end = dur if i == len(cuts) - 1 else b
            spans.append((COLORS[i % len(COLORS)], round(end - t, 3)))
            t = end
        make_video(dst, spans=spans, fps=24, size=(144, 256), audio=True)


class FakeDrive:
    """The Drive boundary of delivery: a new upload, or content replaced in place."""

    def __init__(self):
        self.uploads: list = []
        self.updates: list = []

    def upload_file(self, path, folder_id=None, name=None):
        fid = f"drive-{len(self.uploads) + 1}"
        self.uploads.append((fid, frames_of(path), name))
        return {"id": fid, "webViewLink": f"https://drive.example/{fid}"}

    def update_file_content(self, file_id, path):
        self.updates.append((file_id, frames_of(path)))


# ---------------------------------------------------------------------------
# Infrastructure
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def SF():
    eng = create_engine(f"sqlite:///{_db_path}", connect_args={"check_same_thread": False})

    @event.listens_for(eng, "connect")
    def _fk(dbapi_conn, _):
        cur = dbapi_conn.cursor()
        cur.execute("PRAGMA foreign_keys=ON")
        cur.close()

    Base.metadata.drop_all(eng)
    Base.metadata.create_all(eng)
    yield sessionmaker(bind=eng, autocommit=False, autoflush=False, expire_on_commit=False)
    Base.metadata.drop_all(eng)
    eng.dispose()


class Harness:
    """Every enqueue of the feature recorded and run in order, as RQ would:
    paid-call steps through ``STEP_HANDLERS`` (the real dispatcher), the
    storyboard's advance jobs, the QA jobs."""

    def __init__(self, SF):
        self.SF = SF
        self.steps: list = []
        self.sb_jobs: list = []
        self.qa_jobs: list = []
        self.ran: list = []

    def drive(self, max_iter=2000):
        for _ in range(max_iter):
            if self.steps:
                table, row_id, gen, step = self.steps.pop(0)
                mod, fn = pc.STEP_HANDLERS[table].rsplit(".", 1)
                job_id = pc.step_job_id(table, row_id, gen, step,
                                        "s" if step in (pc.STEP_POLL, pc.STEP_FETCH) else None)
                self.ran.append((table, step))
                getattr(importlib.import_module(mod), fn)(row_id, gen, step, job_id)
                continue
            if self.sb_jobs:
                sb_id, gen, token = self.sb_jobs.pop(0)
                sbmod.run_storyboard_job(sb_id, gen, token, sf=self.SF)
                continue
            if self.qa_jobs:
                pr.run_attempt_qa(self.qa_jobs.pop(0))
                continue
            return
        raise AssertionError("the job chain did not settle")


def install_env(SF, monkeypatch, tmp_path) -> Harness:
    """Point every module of the feature at *SF* and the fakes (shared with
    tests/test_rebuild_browser.py)."""
    @contextmanager
    def sess():
        s = SF()
        try:
            yield s
            s.commit()
        except Exception:
            s.rollback()
            raise
        finally:
            s.close()

    from app import api_rebuild_run, db as db_mod, pipeline_v2

    for mod in (pr, pipeline_v2, api_rebuild_run, db_mod):
        monkeypatch.setattr(mod, "get_session", sess)
    h = Harness(SF)
    fake = FakeKie(str(tmp_path))
    monkeypatch.setattr(jobs, "session_factory", lambda: SF)
    monkeypatch.setattr(jobs, "kie_client", lambda: fake)
    monkeypatch.setattr(pr, "_kie", lambda: fake)
    monkeypatch.setattr(rebuild_llm_steps, "_session_factory", lambda: SF)
    monkeypatch.setattr(pc, "_BOOT_ID", "boot-e2e")
    monkeypatch.setattr(pc, "enqueue_step",
                        lambda table, row_id, gen, step, delay_sec=0, seq=None:
                        h.steps.append((table, row_id, int(gen), step)) or "jid")
    monkeypatch.setattr(sbmod, "_enqueue",
                        lambda sb_id, gen, token, jid, delay_sec=0:
                        h.sb_jobs.append((sb_id, int(gen), token)))
    monkeypatch.setattr(pr, "enqueue_attempt_qa", lambda aid: h.qa_jobs.append(aid))
    monkeypatch.setattr(settings, "REBUILD_ENABLED", True)
    monkeypatch.setattr(settings, "KIE_LLM_CALL_EST_CREDITS", 20.0)
    monkeypatch.setattr(settings, "LOCALISATION_TRANSCRIBE_MODEL", "gemini-3.1-pro")
    monkeypatch.setattr(settings, "LOCALISATION_TRANSCRIBE_FALLBACK_MODELS", "gemini-3-pro")
    monkeypatch.setattr(settings, "GDRIVE_DEFAULT_FOLDER_ID", "folder-e2e", raising=False)
    from app import web_v2

    monkeypatch.setattr(web_v2, "_cached_credit_balance", lambda *a, **k: None)
    h.fake, h.sess, h.tmp = fake, sess, tmp_path
    return h


@pytest.fixture()
def env(SF, monkeypatch, tmp_path):
    yield install_env(SF, monkeypatch, tmp_path)


@pytest.fixture()
def client(SF, env, monkeypatch):
    from fastapi.testclient import TestClient

    import app.api_v2 as api_v2_module
    from app.db import get_db
    from app.main import app

    analyses, runs = [], []
    monkeypatch.setattr(api_v2_module, "enqueue_analyze_project",
                        lambda pid, **kw: analyses.append((pid, kw)))
    monkeypatch.setattr(api_v2_module, "enqueue_process_run",
                        lambda rid, **kw: runs.append(rid))

    def override():
        s = SF()
        try:
            yield s
            s.commit()
        except Exception:
            s.rollback()
            raise
        finally:
            s.close()

    app.dependency_overrides[get_db] = override
    try:
        with TestClient(app, raise_server_exceptions=True) as c:
            c.analyses, c.runs = analyses, runs
            yield c
    finally:
        app.dependency_overrides.pop(get_db, None)


@pytest.fixture(autouse=True)
def _clean_libraries(SF):
    """The clip library is GLOBAL: never let one test's clip pick another's end card."""
    yield
    with SF() as s:
        for clip in s.execute(select(LibraryClip)).scalars():
            s.delete(clip)
        s.commit()


def _source(tmp_path):
    return make_video(tmp_path / "source.mp4",
                      spans=[(COLORS[i], SHOT_SEC) for i in range(N_SHOTS)])


def _library_end_card(SF, tmp_path, language="en"):
    path = make_video(tmp_path / f"outro_{language}.mp4", spans=[("white", SHOT_SEC)])
    with SF() as s:
        clip = LibraryClip(name=f"Outro {language}", local_path=path, language=language,
                           kind="end_card",
                           duration_sec=SHOT_SEC, width=W, height=H, fps=float(FPS))
        s.add(clip)
        s.commit()
        return clip.id


def form_fields(fragment_html: str) -> dict:
    """The launch form's fields exactly as rbLaunch() sends them by default:
    the hidden prompt / audio_mode, the selected model, the resolution the
    page's JS picks (the default 720p), the selected language; no name, no
    stitch_plan (End card = automatic), no chunk_sheets (auto picks)."""
    form = fragment_html[fragment_html.index('<form id="rb-launch-form"'):]
    form = form[:form.index("</form>")]

    def hidden(name):
        m = re.search(rf'name="{name}" value="([^"]*)"', form)
        return html_mod.unescape(m.group(1))

    def selected(select_id):
        block = form[form.index(f'id="{select_id}"'):]
        block = block[:block.index("</select>")]
        m = re.search(r'<option value="([^"]+)"\s+selected', block)
        return m.group(1)

    return {"prompt": hidden("prompt"), "audio_mode": hidden("audio_mode"),
            "model": selected("rb-model"), "resolution": "720p",
            "language": selected("rb-language")}


def _fragment(client, pid) -> str:
    r = client.get(f"/v2/projects/{pid}/rebuild-fragment")
    assert r.status_code == 200, r.text
    return r.text


def _rows(SF, model, **where):
    with SF() as s:
        q = select(model)
        for k, v in where.items():
            q = q.where(getattr(model, k) == v)
        return list(s.execute(q).scalars())


def _drive_run(env, client, rid, gdrive):
    """The run chain without the Redis lock (tasks.run_submit_run's branch and
    the observer's lock handover are tested in test_rebuild_run): submit, the
    attempt / QA / check jobs, the observer, then finalize."""
    from app import pipeline_v2

    pr.submit_images_only(rid)
    env.drive()
    verdict = pr.observe_run(rid)
    if verdict == pr.OBSERVE_FINALIZE:
        pipeline_v2.finalize_run(rid, gdrive=gdrive)
        env.drive()  # QA of the attempts admitted above (already drained) + nothing
    return verdict


# ---------------------------------------------------------------------------
# The whole path
# ---------------------------------------------------------------------------


def test_rebuild_end_to_end(client, env, SF, tmp_path):
    from app import pipeline_v2

    fake = env.fake
    src = _source(tmp_path)
    src_frames = frames_of(src)
    assert src_frames == int(N_SHOTS * SHOT_SEC * FPS)
    end_card_id = _library_end_card(SF, tmp_path, "en")

    # 1. Create (upload) + analyze (real shot detection, real publish).
    with open(src, "rb") as fh:
        r = client.post("/api/v2/projects", data={"project_type": "rebuild"},
                        files={"video_file": ("source.mp4", fh, "video/mp4")})
    assert r.status_code == 201, r.text
    pid = r.json()["project_id"] if "project_id" in r.json() else r.json()["id"]
    assert client.analyses and client.analyses[0][0] == pid
    pipeline_v2.analyze_project(pid, **client.analyses[0][1])
    with SF() as s:
        p = s.get(VideoProject, pid)
        assert p.status == ProjectStatus.ready
        shots = list(s.execute(select(RebuildShot).where(RebuildShot.project_id == pid)
                               .order_by(RebuildShot.index)).scalars())
        assert len(shots) == N_SHOTS
        assert all(abs(float(sh.start_sec) - i * SHOT_SEC) < 0.05 for i, sh in enumerate(shots))
    assert "build the storyboard" in _fragment(client, pid)

    # 2. Storyboard: one global + one call per shot, one POST each.
    r = client.post(f"/api/v2/projects/{pid}/rebuild/storyboard", data={"mode": "resume"})
    assert r.status_code == 202, r.text
    sb_id = r.json()["storyboard_id"]
    env.drive()
    with SF() as s:
        sb = s.get(ProjectStoryboard, sb_id)
        assert sb.status == "succeeded" and sb.applied_at is not None, sb.error
        brand = [sh for sh in s.execute(select(RebuildShot).where(
            RebuildShot.project_id == pid)).scalars() if sh.is_brand_card]
        assert len(brand) == 1 and float(brand[0].start_sec) > 17.5
    assert [k for _m, k, _t in fake.chats].count("global") == 1
    assert [k for _m, k, _t in fake.chats].count("shot") == N_SHOTS

    # 3. Keep the brand card as original footage (it becomes the end card).
    r = client.post(f"/api/v2/projects/{pid}/rebuild/keep-brand-cards", data={"keep": "true"})
    assert r.status_code == 200, r.text
    segs = r.json()["segments"]
    assert segs[-1]["action"] == "keep" and segs[-1]["start_sec"] == pytest.approx(18.0, abs=0.05)
    swaps = [sg for sg in segs if sg["action"] == "swap"]
    assert len(swaps) == 2

    # 4. Keyframes + the sheet: one GPT Image POST per row.
    r = client.post(f"/api/v2/projects/{pid}/rebuild/keyframes",
                    data={"shots": "all_missing", "sheets": "all_missing"})
    assert r.status_code == 202, r.text
    assert len(r.json()["keyframe_ids"]) == N_SHOTS + 1
    env.drive()
    with SF() as s:
        kfs = list(s.execute(select(RebuildKeyframe).where(
            RebuildKeyframe.project_id == pid)).scalars())
        assert all(k.status == "succeeded" and k.selected for k in kfs)
        first_shot = min((k for k in kfs if k.shot_id), key=lambda k: next(
            sh.index for sh in shots if sh.id == k.shot_id))
    # Regenerate one shot (auto-selects the new one), then select the older back.
    r = client.post(f"/api/v2/projects/{pid}/rebuild/keyframes",
                    data={"shots": first_shot.shot_id, "prompt": "sharper"})
    assert r.status_code == 202, r.text
    env.drive()
    with SF() as s:
        newest = s.execute(select(RebuildKeyframe).where(
            RebuildKeyframe.shot_id == first_shot.shot_id, RebuildKeyframe.id != first_shot.id)
        ).scalar_one()
        assert newest.selected and not s.get(RebuildKeyframe, first_shot.id).selected
    r = client.patch(f"/api/v2/rebuild/keyframes/{first_shot.id}", data={"selected": "true"})
    assert r.status_code == 200, r.text
    image_posts = [p for p in fake.creates if "duration" not in (p.get("input") or {})]
    assert len(image_posts) == N_SHOTS + 2

    # 5. The page: everything met, the launch button enabled (same list as the server).
    frag = _fragment(client, pid)
    assert 'id="rb-unmet"' not in frag and 'id="rb-launch-btn" >' in frag
    data = form_fields(frag)
    assert data == {"prompt": "Rebuild from reference (prompts are rendered per chunk)",
                    "audio_mode": "original", "model": "kling-4-0", "resolution": "720p",
                    "language": "en"}

    # 6. Launch: Kling (the form's defaults) and a Seedance variant.
    r = client.post(f"/api/v2/projects/{pid}/runs", data=data)
    assert r.status_code == 201, r.text
    kling = r.json()["run_id"]
    assert any("library end card" in w for w in r.json()["warnings"])
    r = client.post(f"/api/v2/projects/{pid}/runs", data=dict(data, model="seedance-2-5"))
    assert r.status_code == 201, r.text
    seedance = r.json()["run_id"]
    assert client.runs == [kling, seedance]
    with SF() as s:
        for rid in (kling, seedance):
            run = s.get(Run, rid)
            assert run.generation_mode == "images_only" and run.resolution == "720p"
            assert run.stitch_plan[-1] == {"kind": "library", "clip_id": end_card_id}
            assert len(run.rebuild_snapshot["chunks"]) == 2
            # the selected (older) keyframe is the one the run copied
            ch0 = run.rebuild_snapshot["chunks"][0]
            assert ch0["shots"][0]["keyframe_id"] == first_shot.id

    # 7. Run both: attempts -> fit -> QA + check -> stitch -> delivery.
    drive = FakeDrive()
    for rid in (kling, seedance):
        assert _drive_run(env, client, rid, drive) == pr.OBSERVE_FINALIZE
    with SF() as s:
        for rid in (kling, seedance):
            run = s.get(Run, rid)
            assert run.status == RunStatus.done, run.error_message
            assert frames_of(run.result_local_path) == src_frames
            assert has_audio(run.result_local_path)
            atts = list(s.execute(select(RebuildChunkAttempt).where(
                RebuildChunkAttempt.run_id == rid)).scalars())
            assert len(atts) == 2 and all(a.status == "succeeded" and a.selected for a in atts)
            for a in atts:
                assert a.qa.get("free_done_at") and a.qa.get("contact_sheet")
                assert isinstance(a.qa.get("character_check"), dict)
            flagged = [a for a in atts if a.qa["flags"]]
            assert len(flagged) == 1 and "shot 1: missing C1" in flagged[0].qa["flags"]
    assert [fid for fid, _n, _name in drive.uploads] == ["drive-1", "drive-2"]
    assert all(n == src_frames for _fid, n, _name in drive.uploads)
    with SF() as s:
        assert s.get(Run, kling).result_gdrive_file_id == "drive-1"
    # The run page renders the chunks with their QA flags.
    page = client.get(f"/v2/runs/{kling}")
    assert page.status_code == 200 and "shot 1: missing C1" in page.text
    assert 'id="rb-no-copy"' in page.text
    assert "Not available for a <strong>speech-only</strong>" not in page.text

    # 8. Re-roll chunk 2 of the Kling run with an edited prompt: delivered in place.
    with SF() as s:
        rs2 = s.execute(select(RunSegment).where(RunSegment.run_id == kling)
                        .order_by(RunSegment.index.desc())).scalars().first()
    r = client.post(f"/api/v2/runs/{kling}/segments/{rs2.id}/rerun",
                    data={"prompt": "Shot 1 (0.0-2.0 s) the hero walks in."})
    assert r.status_code == 200, r.text
    assert _drive_run(env, client, kling, drive) == pr.OBSERVE_FINALIZE
    with SF() as s:
        run = s.get(Run, kling)
        assert run.status == RunStatus.done and run.result_gdrive_file_id == "drive-1"
        atts = list(s.execute(select(RebuildChunkAttempt).where(
            RebuildChunkAttempt.run_segment_id == rs2.id)
            .order_by(RebuildChunkAttempt.attempt_no)).scalars())
        assert [a.attempt_no for a in atts] == [1, 2]
        assert [a.selected for a in atts] == [False, True]
        assert "the hero walks in" in atts[1].request["input"]["prompt"]
    assert drive.updates == [("drive-1", src_frames)]

    # 9. A re-roll the provider refuses -> incomplete; Retry -> done, in place.
    fake.refuse_next_video = 1
    r = client.post(f"/api/v2/runs/{kling}/segments/{rs2.id}/rerun", data={})
    assert r.status_code == 200, r.text
    assert _drive_run(env, client, kling, drive) == pr.OBSERVE_INCOMPLETE
    with SF() as s:
        assert s.get(Run, kling).status == RunStatus.incomplete
        assert s.get(RunSegment, rs2.id).status == SegmentStatus.failed
    r = client.post(f"/api/v2/runs/{kling}/retry")
    assert r.status_code == 200, r.text
    assert _drive_run(env, client, kling, drive) == pr.OBSERVE_FINALIZE
    with SF() as s:
        run = s.get(Run, kling)
        assert run.status == RunStatus.done and frames_of(run.result_local_path) == src_frames
        atts = list(s.execute(select(RebuildChunkAttempt).where(
            RebuildChunkAttempt.run_segment_id == rs2.id)
            .order_by(RebuildChunkAttempt.attempt_no)).scalars())
        assert [a.status for a in atts] == ["succeeded", "succeeded", "failed", "succeeded"]
        assert [a.selected for a in atts] == [False, False, False, True]
    assert drive.updates == [("drive-1", src_frames), ("drive-1", src_frames)]
    assert len(drive.uploads) == 2  # never a second file for the same run

    # 10. Exactly one provider POST per paid row that was sent.
    sent = {"sending", "submitted", "fetching", "succeeded", "unknown"}

    def sent_rows(model):
        return [r for r in _rows(SF, model) if r.status in sent
                or (r.status == "failed" and not (r.error or "").startswith("not sent"))]

    llm_rows = sent_rows(RebuildLlmCall)
    assert len(fake.chats) == len(llm_rows)
    assert sorted(c.kind for c in llm_rows) == sorted(k for _m, k, _t in fake.chats)
    # one check per succeeded attempt (all answered on the primary model)
    n_ok_attempts = len([a for a in _rows(SF, RebuildChunkAttempt) if a.status == "succeeded"])
    assert [c.kind for c in llm_rows].count("character_check") == n_ok_attempts
    video_rows = sent_rows(RebuildChunkAttempt)
    kf_rows = sent_rows(RebuildKeyframe)
    assert len(fake.creates) == len(video_rows) + len(kf_rows)
    posted = [json.dumps(p, sort_keys=True) for p in fake.creates]
    for row in video_rows + kf_rows:
        assert posted.count(json.dumps(row.request, sort_keys=True)) == 1, row.id

    # 11. Every paid row drew exactly its cost; every scope's books add up.
    with SF() as s:
        draws = {d.call_key: d for d in s.execute(select(KieSpendDraw)).scalars()}
        for row in llm_rows + video_rows + kf_rows:
            d = draws[spend.rebuild_call_key(row.id)]
            # an answered chat call reports no cost: it stays drawn at its estimate
            cost = row.cost_credits if row.cost_credits is not None else row.estimate_credits
            assert d.mcredits == round(float(cost) * 1000), row.id
        for res in s.execute(select(KieSpendReservation)).scalars():
            mine = [d for d in draws.values() if d.reservation_id == res.id]
            assert res.consumed_mcredits == sum(d.mcredits for d in mine), res.scope_key
            assert res.charged_mcredits == max(res.reserved_mcredits, res.consumed_mcredits)
        # Kling run: every sent take at 28.5 cr per requested second (the
        # refused one settled to 0), a check per succeeded take.
        k_atts = list(s.execute(select(RebuildChunkAttempt).where(
            RebuildChunkAttempt.run_id == kling)).scalars())
        assert len(k_atts) == 5  # 2 first takes, re-roll, refused re-roll, Retry
        want_video = sum(a.request_sec for a in k_atts if a.status == "succeeded") * 28.5
        run_draws = [d for d in draws.values() if d.reservation_id in {
            r.id for r in s.execute(select(KieSpendReservation).where(
                KieSpendReservation.scope_key == spend.run_scope(kling))).scalars()}]
        video_mc = sum(d.mcredits for d in run_draws if d.kind == "rebuild chunk")
        assert video_mc == round(want_video * 1000)
        llm_mc = sum(d.mcredits for d in run_draws if d.kind == "rebuild llm")
        assert llm_mc == 4 * 20 * 1000
        run = s.get(Run, kling)
        assert spend.estimate_images_only_run_credits(s, run) == 0.0

    # 12. The v2v requests are byte-identical to main 730216b's.
    _assert_v2v_bytes_identical()


def _assert_v2v_bytes_identical():
    import httpx
    import respx

    from app import kie_client, pipeline_v2

    golden = json.load(open(os.path.join(BACKEND, "tests", "fixtures",
                                         "rebuild_v2v_request_bytes_main730.json")))
    assert len(golden) == 15
    for key, want in golden.items():
        model, ga = key.split("|")
        ga = None if ga == "None" else ga == "True"
        recipe = dict(
            run_model=model, effective_prompt="swap the character",
            ref_urls=["https://x/r1.png", "https://x/r2.png"], clip_url="https://x/clip.mp4",
            clip_start=12.3, clip_end=19.65, run_resolution="720p",
            project_aspect_ratio="9:16", project_width=1080, project_height=1920,
            generate_audio=ga,
            audio_ref_urls=(["https://x/v.mp3"] if model == "seedance-2-5" and ga else None),
        )
        method, kwargs = pipeline_v2._swap_task_call(**recipe)
        assert method == want["method"], key
        with respx.mock() as m:
            route = m.post(want["url"]).mock(return_value=httpx.Response(
                200, json={"code": 200, "data": {"taskId": "t1"}}))
            getattr(kie_client.KieClient(api_key="k", sleep_fn=lambda s: None), method)(**kwargs)
        assert route.call_count == 1
        assert route.calls[0].request.content.decode("utf-8") == want["body"], key


# ---------------------------------------------------------------------------
# Seams
# ---------------------------------------------------------------------------


def _done_attempt(SF, tmp_path):
    """A rebuild project + run + one succeeded attempt with a contact sheet."""
    from PIL import Image

    with SF() as s:
        p = VideoProject(source_type="upload", source_ref="s", status=ProjectStatus.ready,
                         project_type="rebuild", width=W, height=H, fps=float(FPS))
        s.add(p)
        s.flush()
        sd = SegmentDef(project_id=p.id, index=0, start_sec=0, end_sec=4, action="swap")
        s.add(sd)
        s.flush()
        run = Run(project_id=p.id, model="kling-4-0", resolution="720p", status=RunStatus.done,
                  generation_mode="images_only",
                  rebuild_snapshot={"version": 1, "chunks": [{
                      "segment_def_id": sd.id, "index": 0, "shots": [
                          {"shot_id": "a", "rel_start": 0, "rel_end": 4, "characters": ["C1"]}],
                      "characters": [{"id": "C1", "name_hint": "Hero"}]}]})
        s.add(run)
        s.flush()
        rs = RunSegment(run_id=run.id, segment_def_id=sd.id, index=0,
                        status=SegmentStatus.completed)
        s.add(rs)
        s.flush()
        sheet = str(tmp_path / f"{uuid.uuid4().hex}.jpg")
        Image.new("RGB", (40, 40)).save(sheet)
        a = RebuildChunkAttempt(run_id=run.id, run_segment_id=rs.id, attempt_no=1,
                                model="kling-4-0", resolution="720p", request_sec=4,
                                chunk_frames=120, selected=True, status="succeeded",
                                dispatch_gen=1, estimate_credits=114.0, cost_credits=114.0,
                                qa={"free_done_at": "x", "contact_sheet": sheet},
                                finished_at=datetime.utcnow())
        s.add(a)
        s.commit()
        return run.id, a.id


def test_dispatcher_routes_a_check_and_the_reaper_redrives_a_lost_one(env, SF, tmp_path):
    """The paid-call reaper finds a character-check row whose claim job died
    (queued, then claimed) and re-drives it through STEP_HANDLERS
    ['rebuild_llm_calls'] -> the dispatcher -> character_check_step: ONE POST."""
    rid, aid = _done_attempt(SF, tmp_path)
    cid = pr.admit_check(aid, 0)
    assert cid and env.steps == [("rebuild_llm_calls", cid, 1, "claim")]
    env.steps.clear()  # the claim job is lost
    later = datetime.utcnow() + timedelta(seconds=pc.NOT_LIVE_GRACE_SEC + 5)
    common = dict(session_factory=SF, conn=object(), is_live=lambda q, p: False,
                  evidence=lambda row: None)
    pc.reap(now=datetime.utcnow(), **common)   # observe: not live since now
    pc.reap(now=later, **common)               # grace over: re-driven
    assert [(t, st) for t, _r, _g, st in env.steps] == [("rebuild_llm_calls", "claim")]
    # Run the claim only, then lose the send job too: still claimed.
    table, row_id, gen, step = env.steps.pop(0)
    rebuild_llm_steps.handle(row_id, gen, step, "j1")
    assert _rows(SF, RebuildLlmCall, id=cid)[0].status == "claimed"
    env.steps.clear()
    pc.reap(now=later + timedelta(seconds=1), **common)
    pc.reap(now=later + timedelta(seconds=pc.NOT_LIVE_GRACE_SEC + 10), **common)
    env.drive()
    call = _rows(SF, RebuildLlmCall, id=cid)[0]
    assert call.status == "succeeded"
    assert [k for _m, k, _t in env.fake.chats] == ["character_check"]  # ONE POST
    a = _rows(SF, RebuildChunkAttempt, id=aid)[0]
    assert a.qa["character_check"]["call_id"] == cid
    # The upload was named after the call (the host overwrites same names).
    assert env.fake.uploads[-1][2] == f"cc{cid.replace('-', '')}.jpg"


def test_chunk_uploads_are_named_per_attempt(env):
    names = {jobs.upload_name("ca", aid, f"/x/{c}/{n}.png", n)
             for aid in ("a-1", "a-2") for c in (0, 1) for n in (1, 2)}
    # two attempts x the same run-owned file names -> distinct remote names
    assert len({jobs.upload_name("ca", "a-1", "/x/0/1.png", 0),
                jobs.upload_name("ca", "a-2", "/x/1/1.png", 0)}) == 2
    assert all(re.fullmatch(r"ca[a-z0-9]+n\d\.png", n) for n in names)


def _page_vs_server(client, pid, model="kling-4-0", **extra):
    frag = _fragment(client, pid)
    m = re.search(rf'class="alert alert-warning rb-unmet" data-model="{re.escape(model)}"'
                  r'[^>]*>(.*?)</div>', frag, re.S)
    page = [html_mod.unescape(x) for x in re.findall(r"<li>(.*?)</li>", m.group(1))] if m else []
    data = dict(form_fields(frag), model=model, **extra)
    r = client.post(f"/api/v2/projects/{pid}/runs", data=data)
    return page, r


def test_project_page_preconditions_are_the_servers(client, env, SF, tmp_path):
    """The page's unmet list and the server's 409 are the same list (one
    implementation, rebuild_launch.preconditions), per model."""
    from tests.test_rebuild_2a import RANGES, make_project  # the 2a project builder

    pid = make_project(SF, ranges=RANGES)
    with SF() as s:
        s.get(VideoProject, pid).source_local_path = _source(tmp_path)
        s.commit()
    # No storyboard.
    page, r = _page_vs_server(client, pid)
    assert page and r.status_code == 409
    assert r.json()["detail"] == "Cannot launch the rebuild: " + "; ".join(page)
    # Storyboard applied, nothing generated yet: every keyframe + sheet missing.
    r0 = client.post(f"/api/v2/projects/{pid}/rebuild/storyboard", data={"mode": "resume"})
    assert r0.status_code == 202
    env.drive()
    page, r = _page_vs_server(client, pid)
    assert any("no selected keyframe for shot(s)" in u for u in page)
    assert any("no selected character sheet for group 'street'" in u for u in page)
    assert r.status_code == 409
    assert r.json()["detail"] == "Cannot launch the rebuild: " + "; ".join(page)
    # A storyboard call still `sending` after its storyboard ended blocks both.
    with SF() as s:
        sb_id = r0.json()["storyboard_id"]
        assert s.get(ProjectStoryboard, sb_id).status == "succeeded"
        s.add(RebuildLlmCall(project_id=pid, kind="shot", model="m", status="sending",
                             estimate_credits=20.0, storyboard_id=sb_id,
                             shot_id=str(uuid.uuid4()), global_call_id="g-late"))
        s.commit()
    page, r = _page_vs_server(client, pid)
    assert any("storyboard call(s) still running" in u for u in page)
    assert r.status_code == 409 and "storyboard call(s) still running" in r.json()["detail"]
    with SF() as s:
        for c in s.execute(select(RebuildLlmCall).where(
                RebuildLlmCall.project_id == pid, RebuildLlmCall.status == "sending")).scalars():
            c.status = "failed"
        s.commit()
    # Everything generated: both lists empty, the launch is admitted.
    client.post(f"/api/v2/projects/{pid}/rebuild/keyframes",
                data={"shots": "all_missing", "sheets": "all_missing"})
    env.drive()
    page, r = _page_vs_server(client, pid, model="seedance-2-5")
    assert page == [] and r.status_code == 201, r.text


# ---------------------------------------------------------------------------
# Rendered-HTML state matrix (project page and run page)
# ---------------------------------------------------------------------------


STAGES = ("analyzing", "no_storyboard", "storyboard_running", "storyboard_done",
          "keyframes_partial", "all_selected")


def stage_project(client, env, SF, tmp_path, until: str, *, speech=False) -> str:
    """Drive a fresh rebuild project through the real API and jobs up to
    *until* (one of :data:`STAGES`); returns its id. *speech*: the global
    answer says on-camera dialogue (the storyboard's speech warning)."""
    from app import pipeline_v2

    src = _source(tmp_path)
    with open(src, "rb") as fh:
        r = client.post("/api/v2/projects", data={"project_type": "rebuild"},
                        files={"video_file": ("source.mp4", fh, "video/mp4")})
    assert r.status_code == 201, r.text
    body = r.json()
    pid = body.get("project_id") or body["id"]
    if until == "analyzing":
        with SF() as s:
            s.get(VideoProject, pid).status = ProjectStatus.analyzing
            s.commit()
        return pid
    pipeline_v2.analyze_project(pid, **client.analyses[-1][1])
    if until == "no_storyboard":
        return pid
    if speech:
        GLOBAL_SPEECH = dict(GLOBAL, audio={"kind": "on_camera_dialogue", "lines": ["hi"]})
        orig = env.fake.chat_completion_once

        def chat(model, messages, **kw):
            if "shot-by-shot remake" in _text(messages):
                env.fake.chats.append((model, "global", ""))
                return accepted(GLOBAL_SPEECH)
            return orig(model, messages, **kw)

        env.fake.chat_completion_once = chat
    r = client.post(f"/api/v2/projects/{pid}/rebuild/storyboard", data={"mode": "resume"})
    assert r.status_code == 202, r.text
    if until == "storyboard_running":
        return pid
    env.drive()
    assert client.post(f"/api/v2/projects/{pid}/rebuild/keep-brand-cards",
                       data={"keep": "true"}).status_code == 200
    if until == "storyboard_done":
        return pid
    with SF() as s:
        anchors = [sh.id for sh in s.execute(select(RebuildShot).where(
            RebuildShot.project_id == pid).order_by(RebuildShot.index)).scalars()
            if sh.folded_into is None]
    if until == "keyframes_partial":
        r = client.post(f"/api/v2/projects/{pid}/rebuild/keyframes",
                        data={"shots": ",".join(anchors[:3])})
        assert r.status_code == 202, r.text
        env.drive()
        return pid
    r = client.post(f"/api/v2/projects/{pid}/rebuild/keyframes",
                    data={"shots": "all_missing", "sheets": "all_missing"})
    assert r.status_code == 202, r.text
    env.drive()
    assert until == "all_selected", until
    return pid


def test_project_page_state_matrix(client, env, SF, tmp_path, monkeypatch):
    seen = {}
    # analysing: the project page polls the analysis, no rebuild card yet
    pid = stage_project(client, env, SF, tmp_path, "analyzing")
    page = client.get(f"/v2/projects/{pid}").text
    assert "Analyzing video" in page and 'id="rebuild-content"' not in page
    seen["analyzing"] = True
    # analysed, no storyboard: the card loads (hx load), every launch blocked
    pid = stage_project(client, env, SF, tmp_path, "no_storyboard")
    page = client.get(f"/v2/projects/{pid}").text
    assert 'id="rebuild-content"' in page and f"/v2/projects/{pid}/rebuild-fragment" in page
    frag = _fragment(client, pid)
    assert 'id="rb-sb-none"' in frag and 'id="rb-sb-build"' in frag
    assert "build the storyboard" in frag and 'id="rb-launch-btn" disabled' in frag
    assert 'data-live="0"' in frag
    # storyboard running: progress, Cancel, self-polling (paused while typing)
    pid = stage_project(client, env, SF, tmp_path, "storyboard_running")
    frag = _fragment(client, pid)
    assert 'id="rb-sb-running"' in frag and 'id="rb-sb-cancel"' in frag
    assert 'data-live="1"' in frag and 'hx-trigger="every 4s [!(document.activeElement' in frag
    assert "the storyboard is still running" in frag
    env.drive()
    # storyboard done with the on-camera speech warning
    pid = stage_project(client, env, SF, tmp_path, "storyboard_done", speech=True)
    frag = _fragment(client, pid)
    assert 'id="rb-speech-warning"' in frag and "on-camera speech" in frag
    assert 'id="rb-sb-result"' in frag and 'id="rb-generate-missing"' in frag
    assert 'id="rb-end-card"' in frag  # the kept brand card is the trailing end card
    # keyframes partial: the missing shots listed, launch disabled
    pid = stage_project(client, env, SF, tmp_path, "keyframes_partial")
    frag = _fragment(client, pid)
    assert "no selected keyframe for shot(s)" in frag
    assert "no selected character sheet for group &#39;street&#39;" in frag
    assert 'id="rb-launch-btn" disabled' in frag
    # all selected: nothing unmet, launch enabled, the form's defaults
    pid = stage_project(client, env, SF, tmp_path, "all_selected")
    frag = _fragment(client, pid)
    assert 'id="rb-unmet"' not in frag and 'id="rb-launch-btn" >' in frag
    assert form_fields(frag)["model"] == "kling-4-0"
    # launch refused with the unmet list: a keyframe file vanished after render
    with SF() as s:
        kf = s.execute(select(RebuildKeyframe).where(
            RebuildKeyframe.project_id == pid, RebuildKeyframe.shot_id.is_not(None),
            RebuildKeyframe.selected.is_(True))).scalars().first()
        os.remove(kf.image_path)
    r = client.post(f"/api/v2/projects/{pid}/runs", data=form_fields(frag))
    assert r.status_code == 409
    assert r.json()["detail"].startswith("Cannot launch the rebuild: chunk 1 (")
    assert "no selected keyframe for shot(s) 1" in r.json()["detail"]
    frag = _fragment(client, pid)  # the page now lists it too
    assert "no selected keyframe for shot(s) 1" in frag and 'id="rb-launch-btn" disabled' in frag
    # flag off: read-only banner, nothing can be launched
    monkeypatch.setattr(settings, "REBUILD_ENABLED", False)
    frag = _fragment(client, pid)
    assert 'id="rb-flag-off"' in frag and 'id="rb-launch-btn" disabled' in frag
    assert 'id="rb-sb-build"' not in frag
    assert client.post(f"/api/v2/projects/{pid}/rebuild/storyboard",
                       data={"mode": "resume"}).status_code == 409


def test_run_page_state_matrix(client, env, SF, tmp_path, monkeypatch):
    from app import pipeline_v2

    pid = stage_project(client, env, SF, tmp_path, "all_selected")
    frag = _fragment(client, pid)
    r = client.post(f"/api/v2/projects/{pid}/runs", data=form_fields(frag))
    assert r.status_code == 201, r.text
    rid = r.json()["run_id"]

    def chunks_html():
        page = client.get(f"/v2/runs/{rid}")
        assert page.status_code == 200
        frag = client.get(f"/v2/runs/{rid}/status-fragment")
        assert frag.status_code == 200
        assert 'id="rebuild-chunks"' in page.text and 'id="rebuild-chunks"' in frag.text
        return page.text

    # pending: launched, nothing submitted
    html = chunks_html()
    assert html.count('data-status="pending"') == 2
    # generating: sent and polling (submit + claim + send only)
    pr.submit_images_only(rid)
    while env.steps and env.steps[0][3] in ("claim", "send"):
        table, row_id, gen, step = env.steps.pop(0)
        pr.chunk_attempt_step(row_id, gen, step, pc.step_job_id(table, row_id, gen, step))
    html = chunks_html()
    assert html.count('data-status="submitted"') == 2
    assert 'data-attempt-status="submitted"' in html
    # one chunk failed: the provider refuses its fetch result (download keeps failing)
    real = env.fake.download_result
    first_task = [t for t, p in env.fake.tasks.items() if "duration" in p["input"]][0]

    def flaky(url, dst):
        if url.endswith(f"/{first_task}.mp4"):
            raise RuntimeError("cdn closed the connection")
        return real(url, dst)

    env.fake.download_result = flaky
    env.drive()
    with SF() as s:
        bad = s.execute(select(RebuildChunkAttempt).where(
            RebuildChunkAttempt.run_id == rid, RebuildChunkAttempt.task_id == first_task)
        ).scalar_one()
        assert bad.status == "failed" and bad.fetch_attempts == pc.FETCH_MAX_ATTEMPTS
    assert pr.observe_run(rid) == pr.OBSERVE_INCOMPLETE
    html = chunks_html()
    assert 'data-status="failed"' in html and 'data-status="completed"' in html
    assert "js-reroll-btn" in html
    # Retry -> done with QA flags
    env.fake.download_result = real
    assert client.post(f"/api/v2/runs/{rid}/retry").status_code == 200
    assert _drive_run(env, client, rid, FakeDrive()) == pr.OBSERVE_FINALIZE
    html = chunks_html()
    assert html.count('data-status="completed"') == 2
    assert "shot 1: missing C1" in html and "js-use-attempt" in html
    with SF() as s:
        assert s.get(Run, rid).status == RunStatus.done


def test_v2v_pages_carry_no_rebuild_ui(client, env, SF, tmp_path):
    """A v2v project and run render without any rebuild element (their full
    HTML was also diffed against main 730216b: see the integration notes)."""
    with SF() as s:
        p = VideoProject(source_type="upload", source_ref="s", status=ProjectStatus.ready,
                         project_type="face_swap", width=W, height=H, fps=float(FPS),
                         duration_sec=6.0, source_local_path=_source(tmp_path))
        s.add(p)
        s.flush()
        sd = SegmentDef(project_id=p.id, index=0, start_sec=0, end_sec=6, action="swap")
        s.add(sd)
        run = Run(project_id=p.id, model="seedance", resolution="720p", status=RunStatus.done,
                  prompt="x", reference_image_urls=[])
        s.add(run)
        s.commit()
        pid, rid = p.id, run.id
    page = client.get(f"/v2/projects/{pid}").text
    assert "rebuild-content" not in page and "kling-4-0" not in page
    run_page = client.get(f"/v2/runs/{rid}").text
    assert "rebuild-chunks" not in run_page


# ---------------------------------------------------------------------------
# Integration gate round 1 regressions
# ---------------------------------------------------------------------------


def test_observer_never_fails_a_chunk_whose_fetch_commits_between_its_reads(
        client, env, SF, tmp_path):
    """The fetch commits attempt `succeeded` + segment `completed` in one
    transaction; the observer reads attempts and segments in two statements.
    The fetch is made to commit exactly BETWEEN them: the observer must wait
    (or finalize), never overwrite the paid result with `failed`."""
    from sqlalchemy import event as sa_event
    from sqlalchemy.orm import Session as OrmSession

    pid = stage_project(client, env, SF, tmp_path, "all_selected")
    r = client.post(f"/api/v2/projects/{pid}/runs", data=form_fields(_fragment(client, pid)))
    rid = r.json()["run_id"]
    pr.submit_images_only(rid)
    fetches = []
    while env.steps:
        table, row_id, gen, step = env.steps.pop(0)
        if step == "fetch":
            fetches.append((table, row_id, gen, step))
            continue
        pr.chunk_attempt_step(row_id, gen, step, pc.step_job_id(table, row_id, gen, step))
    assert len(fetches) == 2
    # Fetch the first chunk now; hold the second one for the race.
    t, row_id, gen, step = fetches[0]
    pr.chunk_attempt_step(row_id, gen, step, pc.step_job_id(t, row_id, gen, step, "s"))
    t, race_id, race_gen, step = fetches[1]
    reads = []

    def hook(state):
        ents = {d.get("entity") for d in state.statement.column_descriptions} \
            if state.is_select else set()
        hit = ents & {RunSegment, RebuildChunkAttempt}
        if hit:
            reads.append(hit)
            if len(reads) == 2:  # between the observer's two reads
                pr.chunk_attempt_step(race_id, race_gen, "fetch",
                                      pc.step_job_id(t, race_id, race_gen, "fetch", "s"))

    sa_event.listen(OrmSession, "do_orm_execute", hook)
    try:
        verdict = pr.observe_run(rid)
    finally:
        sa_event.remove(OrmSession, "do_orm_execute", hook)
    assert verdict in (pr.OBSERVE_WAIT, pr.OBSERVE_FINALIZE), verdict
    with SF() as s:
        a = s.get(RebuildChunkAttempt, race_id)
        rs = s.get(RunSegment, a.run_segment_id)
        assert a.status == "succeeded" and rs.status == SegmentStatus.completed
        assert s.get(Run, rid).status == RunStatus.processing
    assert pr.observe_run(rid) == pr.OBSERVE_FINALIZE


def test_qa_reaper_reruns_a_missing_contact_sheet_then_gives_the_check_a_verdict(env, SF,
                                                                                 tmp_path):
    rid, aid = _done_attempt(SF, tmp_path)
    old = datetime.utcnow() - timedelta(hours=1)
    with SF() as s:
        a = s.get(RebuildChunkAttempt, aid)
        a.qa = {"free_done_at": "x", "free_runs": 1, "errors": {"contact_sheet": "boom"}}
        a.finished_at = old
        s.commit()
    queued = []
    out = pr.reap_qa(is_live=lambda jid: False, enqueue=queued.append)
    assert queued == [aid] and out["qa"] == 1 and out["checks"] == 0
    # A QA job that keeps crashing before it finishes still counts (gate round
    # 2): free_runs is written when a run STARTS; a live job is left alone.
    with SF() as s:
        a = s.get(RebuildChunkAttempt, aid)
        a.qa = {"free_runs": pr.FREE_QA_MAX_RUNS}  # started 3 times, never finished
        s.commit()
    queued.clear()
    pr.reap_qa(is_live=lambda jid: True, enqueue=queued.append)
    with SF() as s:
        assert "character_check" not in s.get(RebuildChunkAttempt, aid).qa
    # Gate round 3: the give-up is re-checked under the row lock — a run that
    # finished meanwhile (sheet written / a check admitted) is never overwritten.
    with SF() as s:
        a = s.get(RebuildChunkAttempt, aid)
        sheet = str(tmp_path / "late-sheet.jpg")
        open(sheet, "wb").write(b"x")
        a.qa = {"free_runs": pr.FREE_QA_MAX_RUNS, "free_done_at": "y", "contact_sheet": sheet}
        s.commit()
    assert pr._give_up_check(aid) is False
    with SF() as s:
        a = s.get(RebuildChunkAttempt, aid)
        a.qa = {"free_runs": pr.FREE_QA_MAX_RUNS}
        c = RebuildLlmCall(project_id=s.get(Run, rid).project_id, attempt_id=aid,
                           kind="character_check", slot=0, model="m", status="queued",
                           estimate_credits=20.0)
        s.add(c)
        s.commit()
        cid = c.id
    assert pr._give_up_check(aid) is False
    with SF() as s:
        s.delete(s.get(RebuildLlmCall, cid))
        s.commit()
    pr.reap_qa(is_live=lambda jid: False, enqueue=queued.append)
    assert queued == []
    with SF() as s:
        assert s.get(RebuildChunkAttempt, aid).qa["character_check"] == pr.CHECK_UNAVAILABLE
        run = s.get(Run, rid)
    assert pr.admit_auto_check(aid) is None  # a verdict exists: never admitted later
    assert pr.admit_check(aid, 0) is None and pr.admit_check(aid, 1) is None
    with SF() as s:
        run = s.get(Run, rid)
        assert spend.estimate_images_only_run_credits(s, run) == 0.0


def test_estimator_keeps_the_fallback_of_an_unusable_primary_answer(env, SF, tmp_path):
    rid, aid = _done_attempt(SF, tmp_path)
    with SF() as s:
        s.add(RebuildLlmCall(project_id=s.get(Run, rid).project_id, attempt_id=aid,
                             kind="character_check", slot=0, model="m", status="succeeded",
                             estimate_credits=20.0))
        s.commit()
        run = s.get(Run, rid)
        # answered, but no verdict yet (e.g. no JSON): the fallback is still due
        assert spend.estimate_images_only_run_credits(s, run) == 20.0
        a = s.get(RebuildChunkAttempt, aid)
        a.qa = dict(a.qa, character_check={"shots": [], "flagged": False})
        s.commit()
        assert spend.estimate_images_only_run_credits(s, s.get(Run, rid)) == 0.0
