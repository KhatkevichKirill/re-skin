"""Rebuild wave 2b — the images-only run path (docs/rebuild-from-reference-design.md
§6, §7.2–7.3, §10), on SQLite with real ffmpeg on tiny synthetic media.

* launch through the real ``create_run`` (snapshot, run-owned keyframe
  copies, pending RunSegments, reservation last, the end-card rule, refusals);
* single-send: one createTask POST per attempt across duplicate deliveries,
  reaper re-drives and a recovery re-entry; the request is the pilot shape;
* the fit (exact frame counts, padding, too short) and the fetch step;
* the observer (finalize / incomplete / wait, a reaper-made ``unknown``);
* stitch with the original soundtrack (default plan, library end card, a
  silent source), frame-exact on the output grid;
* QA (cut timing, burned text, contact sheet, flags) and the character check
  (single send, fallback slot, unavailable);
* Retry / re-roll / use-attempt reserve + draw through the real endpoints;
* the v2v keep cut is untouched;
* the run page HTML in each state.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone

_db_fd, _db_path = tempfile.mkstemp(suffix=".db")
os.close(_db_fd)
os.environ.setdefault("DATABASE_URL", f"sqlite:///{_db_path}")
os.environ.setdefault("DATA_DIR", tempfile.mkdtemp())
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest  # noqa: E402
from sqlalchemy import create_engine, event, select  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402

from app import kie_client, pipeline_rebuild as pr, rebuild as rb, rebuild_launch  # noqa: E402
from app import rebuild_paid_call as pc, spend, storage  # noqa: E402
from app.config import settings  # noqa: E402
from app.db import Base  # noqa: E402
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
    ProjectStoryboard,
    RebuildChunkAttempt,
    RebuildKeyframe,
    RebuildLlmCall,
    RebuildShot,
)
from app.state_machine import ProjectStatus, RunStatus, SegmentStatus  # noqa: E402

FPS = 30
W, H = 180, 320


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


@pytest.fixture(autouse=True)
def env(SF, monkeypatch, tmp_path):
    """Sessions of every module under test on this database; step / QA
    enqueues recorded instead of sent; a boot id for the sender."""

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
    # The character check's claim/send go through app/rebuild_jobs (shared with
    # the storyboard since the integration): same database.
    from app import rebuild_jobs

    monkeypatch.setattr(rebuild_jobs, "session_factory", lambda: SF)
    steps, qa_jobs = [], []
    monkeypatch.setattr(pc, "enqueue_step",
                        lambda table, row_id, gen, step, delay_sec=0, seq=None:
                        steps.append((table, row_id, int(gen), step)))
    monkeypatch.setattr(pr, "enqueue_attempt_qa", lambda aid: qa_jobs.append(aid))
    monkeypatch.setattr(pc, "_BOOT_ID", "boot-test")
    monkeypatch.setattr(settings, "REBUILD_ENABLED", True)
    monkeypatch.setattr(settings, "KIE_LLM_CALL_EST_CREDITS", 20.0)
    monkeypatch.setattr(settings, "GDRIVE_DEFAULT_FOLDER_ID", "", raising=False)
    ns = type("Env", (), {})()
    ns.steps, ns.qa_jobs, ns.sess, ns.tmp = steps, qa_jobs, sess, tmp_path
    yield ns


def ff(*args):
    subprocess.run(["ffmpeg", "-v", "error", "-y", *args], check=True)


def make_video(path, *, seconds, fps=FPS, size=(W, H), audio=True, colors=("red",)):
    """Hard cuts between solid colours (one segment per colour)."""
    seg = seconds / len(colors)
    args = []
    for c in colors:
        args += ["-f", "lavfi", "-i", f"color=c={c}:s={size[0]}x{size[1]}:r={fps}:d={seg}"]
    n = len(colors)
    flt = "".join(f"[{i}:v]" for i in range(n)) + f"concat=n={n}:v=1:a=0[v]"
    if audio:
        args += ["-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}"]
    out = ["-filter_complex", flt, "-map", "[v]"]
    if audio:
        out += ["-map", f"{n}:a", "-c:a", "aac", "-shortest"]
    ff(*args, *out, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", str(fps), str(path))
    return str(path)


def make_png(path, color="blue", size=(W, H)):
    ff("-f", "lavfi", "-i", f"color=c={color}:s={size[0]}x{size[1]}:d=1", "-frames:v", "1",
       str(path))
    return str(path)


def frames_of(path):
    return pr.count_frames(path)


def has_audio(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries",
                          "stream=index", "-of", "csv=p=0", path], capture_output=True, text=True)
    return bool(out.stdout.strip())


@pytest.fixture(scope="module")
def media(tmp_path_factory):
    d = tmp_path_factory.mktemp("media")
    return {
        "source": make_video(d / "src.mp4", seconds=6, colors=("red", "green", "blue")),
        "silent": make_video(d / "silent.mp4", seconds=6, audio=False,
                             colors=("red", "green", "blue")),
        # A "raw take" of a 4 s chunk: 5 s at 24 fps, cut at 2 s.
        "raw5": make_video(d / "raw5.mp4", seconds=5, fps=24, size=(144, 256),
                           colors=("yellow", "purple"), audio=True),
        "raw395": make_video(d / "raw395.mp4", seconds=3.95, fps=24, size=(144, 256),
                             colors=("yellow",), audio=False),
        "raw3": make_video(d / "raw3.mp4", seconds=3.0, fps=24, size=(144, 256),
                           colors=("yellow",), audio=False),
        "outro": make_video(d / "outro.mp4", seconds=1, colors=("white",)),
        "png": make_png(d / "kf.png"),
    }


# ---------------------------------------------------------------------------
# A ready rebuild project: chunk 0 = swap 0-4 s (shots 0, 1), segment 1 =
# keep 4-6 s (shot 2, a brand card = the trailing end card).
# ---------------------------------------------------------------------------


def ready_project(SF, media, *, source="source", brand=True, sheet=True, keyframes=True,
                  middle_keep=False, language_clip="en"):
    s = SF()
    p = VideoProject(source_type="upload", source_ref="s.mp4", status=ProjectStatus.ready,
                     project_type="rebuild", source_local_path=media[source],
                     width=W, height=H, fps=float(FPS), duration_sec=6.0, aspect_ratio="9:16")
    s.add(p)
    s.flush()
    shot_ranges = [(0, 2), (2, 4), (4, 6)]
    shots = []
    for i, (a, b) in enumerate(shot_ranges):
        sh = RebuildShot(project_id=p.id, index=i, generation=1, start_sec=a, end_sec=b,
                         frame_sec=(a + b) / 2, framing="medium", camera="slow push-in",
                         action=f"the hero does thing {i}", characters=["C1"],
                         sheet_key="main", is_brand_card=(i == 2 and brand))
        s.add(sh)
        shots.append(sh)
    s.flush()
    if middle_keep:
        plan = [(0, 2, "swap"), (2, 4, "keep"), (4, 6, "keep")]
    else:
        plan = [(0, 4, "swap"), (4, 6, "keep")]
    defs = []
    for i, (a, b, act) in enumerate(plan):
        sd = SegmentDef(project_id=p.id, index=i, start_sec=a, end_sec=b, action=act)
        s.add(sd)
        defs.append(sd)
    sb = ProjectStoryboard(project_id=p.id, status="succeeded", shots_generation=1,
                           applied_at=datetime(2026, 11, 3), model="gemini-3.1-pro",
                           result={"style": "3D cartoon", "on_screen_text": ["CAPTION"],
                                   "characters": [{"id": "C1", "name_hint": "Hero",
                                                   "look": "red cap", "outfit": "blue jacket"}]})
    s.add(sb)
    s.flush()
    kdir = os.path.join(settings.DATA_DIR, "kf", p.id)
    os.makedirs(kdir, exist_ok=True)
    if keyframes:
        for sh in shots[:2]:
            path = os.path.join(kdir, f"{sh.id}.png")
            shutil.copyfile(media["png"], path)
            s.add(RebuildKeyframe(project_id=p.id, batch_id="b1", shot_id=sh.id,
                                  status="succeeded", selected=True, image_path=path,
                                  estimate_credits=10.0))
    if sheet:
        path = os.path.join(kdir, "sheet.png")
        shutil.copyfile(media["png"], path)
        s.add(RebuildKeyframe(project_id=p.id, batch_id="b1", shot_id=None, sheet_key="main",
                              status="succeeded", selected=True, image_path=path,
                              estimate_credits=10.0))
    if language_clip:
        s.add(LibraryClip(name=f"Outro {language_clip} {uuid.uuid4().hex[:4]}",
                          local_path=media["outro"], language=language_clip, kind="end_card",
                          duration_sec=1.0, width=W, height=H, fps=float(FPS)))
    s.commit()
    ids = {"pid": p.id, "defs": [d.id for d in defs], "shots": [sh.id for sh in shots]}
    s.close()
    return ids


@pytest.fixture()
def client(SF, monkeypatch):
    from fastapi.testclient import TestClient

    import app.api_v2 as api_v2_module
    from app.db import get_db

    enq = []
    monkeypatch.setattr(api_v2_module, "enqueue_process_run", lambda rid, **kw: enq.append(rid))
    from app.main import app

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
    with TestClient(app, raise_server_exceptions=True) as c:
        c.enqueued = enq
        yield c
    app.dependency_overrides.clear()


def launch(client, pid, **data):
    data.setdefault("prompt", "rebuild")
    data.setdefault("model", "kling-4-0")
    return client.post(f"/api/v2/projects/{pid}/runs", data=data)


# ---------------------------------------------------------------------------
# Launch (§6.1)
# ---------------------------------------------------------------------------


def test_launch_freezes_snapshot_copies_segments_and_reserves(client, SF, media):
    ids = ready_project(SF, media)
    r = launch(client, ids["pid"])
    assert r.status_code == 201, r.text
    rid = r.json()["run_id"]
    assert client.enqueued == [rid]
    assert any("library end card" in w for w in r.json()["warnings"])
    with SF() as s:
        run = s.get(Run, rid)
        assert run.generation_mode == "images_only" and run.model == "kling-4-0"
        assert run.resolution == "720p"  # Q1 default when the form sends none
        assert run.audio_mode == "original" and not run.preprocess_enabled
        assert run.status == RunStatus.queued
        assert [e["id"] for e in run.segment_snapshot] == ids["defs"]
        snap = run.rebuild_snapshot
        assert snap["version"] == 1 and snap["fps_out"] == 30.0 and snap["aspect_ratio"] == "9:16"
        (ch,) = snap["chunks"]
        assert ch["segment_def_id"] == ids["defs"][0]
        assert ch["request_sec"] == 4 and ch["chunk_frames"] == 120
        assert [sh["shot_id"] for sh in ch["shots"]] == ids["shots"][:2]
        assert ch["shots"][1]["rel_start"] == 2.0
        assert "Shot 1 (0.0-2.0 s), starts from @image_1" in ch["prompt"]
        assert "Shot 2 (2.0-4.0 s), starts from @image_2" in ch["prompt"]
        assert "character sheet @image_3" in ch["prompt"] and "Hero" in ch["prompt"]
        # Run-owned copies, recorded with their sha256, the sheet last.
        assert len(ch["images"]) == 3 and ch["images"][-1] == ch["sheet_path"]
        for sh in ch["shots"]:
            assert sh["keyframe_path"].startswith(storage.run_dir(rid, ids["pid"]))
            assert rebuild_launch.file_sha256(sh["keyframe_path"]) == sh["sha256"]
        # The trailing end card (brand-card keep) is replaced by the library clip.
        assert run.stitch_plan[0] == {"kind": "segment", "segment_def_id": ids["defs"][0]}
        assert run.stitch_plan[-1]["kind"] == "library" and len(run.stitch_plan) == 2
        (rs,) = s.execute(select(RunSegment).where(RunSegment.run_id == rid)).scalars().all()
        assert rs.status == SegmentStatus.pending and rs.segment_def_id == ids["defs"][0]
        res = s.execute(select(KieSpendReservation).where(
            KieSpendReservation.scope_key == spend.run_scope(rid))).scalar_one()
        assert res.reserved_mcredits == int((4 * 28.5 + 2 * 20.0) * 1000)
        # Regenerating / deleting a project keyframe never changes the run.
        kf = s.execute(select(RebuildKeyframe).where(
            RebuildKeyframe.shot_id == ids["shots"][0])).scalar_one()
        os.remove(kf.image_path)
        assert os.path.exists(ch["shots"][0]["keyframe_path"])


def test_launch_seedance_end_card_kept_and_resolution(client, SF, media):
    ids = ready_project(SF, media)
    r = launch(client, ids["pid"], model="seedance-2-5", resolution="1080p",
               keep_source_end_card="true")
    assert r.status_code == 201, r.text
    with SF() as s:
        run = s.get(Run, r.json()["run_id"])
        assert run.stitch_plan is None  # every segment in order, source end card kept
        ch = run.rebuild_snapshot["chunks"][0]
        assert "[Image 1]" in ch["prompt"] and "[Image 3]" in ch["prompt"]
        assert ch["request_sec"] == 4  # Seedance minimum is 4 s


def test_end_card_rule_middle_keep_and_no_matching_clip(client, SF, media):
    ids = ready_project(SF, media, middle_keep=True, language_clip="es")
    with SF() as s:
        shots = list(s.execute(select(RebuildShot).where(
            RebuildShot.project_id == ids["pid"]).order_by(RebuildShot.index)).scalars())
        defs = list(s.execute(select(SegmentDef).where(
            SegmentDef.project_id == ids["pid"]).order_by(SegmentDef.index)).scalars())
        # Only the trailing all-brand keep is the end card; the middle keep stays.
        assert rebuild_launch.trailing_end_card_ids(defs, shots) == [ids["defs"][2]]
        plan, end_ids, clip = rebuild_launch.default_end_card_plan(s, defs, shots, "es")
        assert clip is not None and end_ids == [ids["defs"][2]]
        assert [e.get("segment_def_id") for e in plan[:-1]] == ids["defs"][:2]
        assert plan[-1] == {"kind": "library", "clip_id": clip.id}
    # A Polish run has no matching clip: default plan, warning only.
    r = launch(client, ids["pid"], language="pl")
    assert r.status_code == 201, r.text
    assert any("end card is kept" in w for w in r.json()["warnings"])
    assert any("keeps its burned text" in w for w in r.json()["warnings"])
    with SF() as s:
        assert s.get(Run, r.json()["run_id"]).stitch_plan is None


def _lib_clip(SF, media, *, name, language, kind, duration=1.0, created=None, path=None):
    with SF() as s:
        clip = LibraryClip(name=name, local_path=path or media["outro"], language=language,
                           kind=kind, duration_sec=duration, width=W, height=H, fps=float(FPS))
        if created is not None:
            clip.created_at = created
        s.add(clip)
        s.commit()
        return clip.id


def test_default_end_card_is_an_end_card_never_a_newer_insert(client, SF, media):
    """AICP-381 / run 813b251f: a NEWER insert in the run language (the 27.8 s
    demo ``ip_reader``) must not displace the language's end card."""
    card = _lib_clip(SF, media, name="CTA_French", language="fr", kind="end_card",
                     duration=7.0, created=datetime(2026, 8, 21))
    _lib_clip(SF, media, name="ip_reader_fr", language="fr", kind="insert",
              duration=27.8, created=datetime(2026, 9, 28))
    # An end card in another language is never picked either.
    _lib_clip(SF, media, name="CTA_German", language="de", kind="end_card", duration=7.0)
    with SF() as s:
        assert rebuild_launch.language_end_card_clip(s, "fr").id == card
        # A newer end card whose file is gone is skipped.
        _gone = LibraryClip(name="CTA_French_gone", local_path="/nonexistent/x.mp4",
                            language="fr", kind="end_card", duration_sec=7.0)
        s.add(_gone)
        s.commit()
        assert rebuild_launch.language_end_card_clip(s, "fr").id == card
        auto = rebuild_launch.auto_end_card_by_language(s, ["fr", "ja"])
        assert auto == {"fr": {"id": card, "name": "CTA_French", "duration_sec": 7.0},
                        "ja": None}
    ids = ready_project(SF, media, language_clip=None)
    r = launch(client, ids["pid"], language="fr")
    assert r.status_code == 201, r.text
    assert any("'CTA_French' (fr, 7.0 s)" in w for w in r.json()["warnings"])
    with SF() as s:
        plan = s.get(Run, r.json()["run_id"]).stitch_plan
        assert plan[-1] == {"kind": "library", "clip_id": card}


def test_only_inserts_in_the_language_keep_the_source_card_with_a_warning(client, SF, media):
    _lib_clip(SF, media, name="Demo_Japanese", language="ja", kind="insert", duration=24.7)
    ids = ready_project(SF, media, language_clip=None)
    r = launch(client, ids["pid"], language="ja")
    assert r.status_code == 201, r.text
    warnings = r.json()["warnings"]
    assert any("end card is kept" in w and "no end card in the run language" in w
               for w in warnings), warnings
    with SF() as s:
        assert s.get(Run, r.json()["run_id"]).stitch_plan is None


def test_end_card_picker_lists_end_cards_only_and_shows_the_auto_pick(client, SF, media):
    card = _lib_clip(SF, media, name="CTA_Italian_picker", language="it", kind="end_card",
                     duration=6.5)
    insert = _lib_clip(SF, media, name="Demo_Italian_picker", language="it", kind="insert",
                       duration=23.0)
    ids = ready_project(SF, media, language_clip=None)
    from app import api_rebuild

    with SF() as s:
        ctx = api_rebuild.rebuild_context(s, s.get(VideoProject, ids["pid"]))
        listed = [c.id for c in ctx["library_clips"]]
        assert card in listed and insert not in listed
        assert all(c.kind == "end_card" for c in ctx["library_clips"])
        assert ctx["auto_end_card"]["it"] == {"id": card, "name": "CTA_Italian_picker",
                                              "duration_sec": 6.5}
    r = client.get(f"/v2/projects/{ids['pid']}/rebuild-fragment")
    assert r.status_code == 200, r.text
    html = r.text
    assert f'value="clip:{card}"' in html and "CTA_Italian_picker (it) — 6.5 s" in html
    assert f'value="clip:{insert}"' not in html and "Demo_Italian_picker" not in html
    assert 'id="rb-end-card-info"' in html and "RB_AUTO_END_CARD" in html


def test_launch_refusals_write_nothing(client, SF, media, monkeypatch):
    ids = ready_project(SF, media, keyframes=False)
    r = launch(client, ids["pid"])
    assert r.status_code == 409
    detail = r.json()["detail"]
    assert isinstance(detail, str)  # the 2a launch form shows `detail` as is
    # The same wording the project page lists (rebuild_launch.preconditions).
    assert "chunk 1 (0.00–4.00 s): no selected keyframe for shot(s) 1, 2" in detail
    ids2 = ready_project(SF, media, sheet=False)
    r = launch(client, ids2["pid"])
    assert r.status_code == 409 and "no selected character sheet" in r.text
    ids3 = ready_project(SF, media)
    assert launch(client, ids3["pid"], resolution="480p").status_code == 400  # Kling
    assert launch(client, ids3["pid"], reference_urls="https://x/a.png").status_code == 400
    assert launch(client, ids3["pid"], audio_mode="seedance").status_code == 400
    assert launch(client, ids3["pid"], chunk_sheets="[1]").status_code == 400
    r = launch(client, ids3["pid"], chunk_sheets=json.dumps({ids3["defs"][0]: "wedding"}))
    assert r.status_code == 409 and "wedding" in r.text
    monkeypatch.setattr(settings, "REBUILD_ENABLED", False)
    assert launch(client, ids3["pid"]).status_code == 409
    monkeypatch.setattr(settings, "REBUILD_ENABLED", True)
    with SF() as s:  # a live keyframe row refuses the launch
        s.add(RebuildKeyframe(project_id=ids3["pid"], batch_id="b2", shot_id=ids3["shots"][0],
                              status="claimed", estimate_credits=10.0))
        s.commit()
    r = launch(client, ids3["pid"])
    assert r.status_code == 409 and "in flight" in r.text
    with SF() as s:
        for pid in (ids["pid"], ids2["pid"], ids3["pid"]):
            assert not s.execute(select(Run).where(Run.project_id == pid)).first()
    # keep_source_end_card is a rebuild-only field.
    with SF() as s:
        fs = VideoProject(source_type="upload", source_ref="s", status=ProjectStatus.ready,
                          project_type="face_swap")
        s.add(fs)
        s.commit()
        fs_id = fs.id
    r = client.post(f"/api/v2/projects/{fs_id}/runs",
                    data={"prompt": "x", "model": "seedance", "keep_source_end_card": "true"})
    assert r.status_code == 400


@pytest.mark.spend_cap
def test_launch_over_budget_is_429_and_leaves_no_files(client, SF, media, monkeypatch):
    monkeypatch.setattr(settings, "KIE_DAILY_BUDGET_USD", 0.1)
    monkeypatch.setattr(settings, "KIE_USD_PER_CREDIT", 0.005)
    ids = ready_project(SF, media)
    runs_dir = os.path.join(storage.project_dir(ids["pid"]), "runs")
    before = set(os.listdir(runs_dir)) if os.path.exists(runs_dir) else set()
    r = launch(client, ids["pid"])
    assert r.status_code == 429, r.text
    with SF() as s:
        assert not s.execute(select(Run).where(Run.project_id == ids["pid"])).first()
    after = set(os.listdir(runs_dir)) if os.path.exists(runs_dir) else set()
    assert after == before


# ---------------------------------------------------------------------------
# Submit + attempt steps: single send, pilot payloads (§6.4, §7.2)
# ---------------------------------------------------------------------------


class FakeKie:
    def __init__(self, raw=None, outcome=None, state=None, chat=None):
        self.raw = raw
        self.outcome = outcome or kie_client.SendOutcome("accepted", task_id="task-1")
        self.state = state or {"state": "success", "creditsConsumed": 114,
                               "resultJson": json.dumps({"resultUrls": ["https://cdn/r.mp4"]})}
        self.chat = chat
        self.uploads, self.creates, self.chats, self.polls = [], [], [], []

    def upload_file(self, path, upload_path="x", file_name=None):
        self.uploads.append(path)
        return f"https://up.example/{os.path.basename(os.path.dirname(path))}/{os.path.basename(path)}"

    def create_task_once(self, payload, *, timeout):
        self.creates.append(payload)
        return self.outcome

    def get_task(self, task_id):
        self.polls.append(task_id)
        return self.state

    def download_result(self, url, dst):
        shutil.copyfile(self.raw, dst)

    def chat_completion_once(self, model, messages, *, timeout, response_format=None):
        self.chats.append((model, messages))
        return self.chat(model) if callable(self.chat) else self.chat


def launched(client, SF, media, **kw):
    ids = ready_project(SF, media, **{k: v for k, v in kw.items() if k in (
        "source", "brand", "middle_keep", "language_clip")})
    r = launch(client, ids["pid"], **{k: v for k, v in kw.items() if k in (
        "model", "resolution", "keep_source_end_card")})
    assert r.status_code == 201, r.text
    ids["rid"] = r.json()["run_id"]
    return ids


def drain(env, *, only=None, limit=50):
    """Run the recorded step jobs in order (job ids as rq would give them)."""
    ran = []
    while env.steps and limit:
        if only and env.steps[0][3] not in only:
            break  # leave the next step queued
        limit -= 1
        table, row_id, gen, step = env.steps.pop(0)
        job_id = pc.step_job_id(table, row_id, gen, step, "s" if step in ("poll", "fetch") else None)
        if table == pr.TABLE:
            pr.chunk_attempt_step(row_id, gen, step, job_id)
        else:
            pr.character_check_step(row_id, gen, step, job_id)
        ran.append((table, step))
    return ran


def test_submit_admits_once_and_is_idempotent(client, SF, media, env):
    ids = launched(client, SF, media)
    assert pr.submit_images_only(ids["rid"]) is True
    assert pr.submit_images_only(ids["rid"]) is True  # recovery re-entry
    with SF() as s:
        run = s.get(Run, ids["rid"])
        assert run.status == RunStatus.processing
        (a,) = s.execute(select(RebuildChunkAttempt).where(
            RebuildChunkAttempt.run_id == ids["rid"])).scalars().all()
        assert (a.status, a.attempt_no, a.request_sec, a.chunk_frames) == ("queued", 1, 4, 120)
        assert a.estimate_credits == 4 * 28.5
        rs = s.get(RunSegment, a.run_segment_id)
        assert rs.status == SegmentStatus.uploading
    assert [st[3] for st in env.steps] == ["claim"]


def _terminalise_live_attempts(SF):
    """Other tests' live rows share this database; the reaper pass below
    would re-drive them too."""
    with SF() as s:
        for a in s.execute(select(RebuildChunkAttempt).where(
                RebuildChunkAttempt.status.in_(pc.PAID_CALL_LIVE_STATUSES))).scalars():
            a.status = "failed"
        s.commit()


def test_one_post_per_attempt_across_duplicates_reaper_and_recovery(client, SF, media, env,
                                                                    monkeypatch):
    _terminalise_live_attempts(SF)
    ids = launched(client, SF, media)
    kie = FakeKie(raw=media["raw5"])
    monkeypatch.setattr(pr, "_kie", lambda: kie)
    pr.submit_images_only(ids["rid"])
    (claim,) = env.steps
    aid = claim[1]
    drain(env, only={"claim"})
    assert env.steps == [(pr.TABLE, aid, 1, "send")]
    # The reaper finds the row `claimed` with no live job past the grace:
    # back to `queued` under generation 2 (nothing was sent).
    t0 = datetime(2026, 11, 3, 12, 0)
    pc.reap(session_factory=SF, is_live=lambda q, p: False, evidence=lambda r: None,
            enqueue=lambda *a: env.steps.append(a), now=t0, models=(RebuildChunkAttempt,))
    pc.reap(session_factory=SF, is_live=lambda q, p: False, evidence=lambda r: None,
            enqueue=lambda *a: env.steps.append(a), now=t0 + timedelta(seconds=200),
            models=(RebuildChunkAttempt,))
    # The stale generation-1 send is fenced; generation 2 sends exactly once.
    drain(env)
    assert len(kie.creates) == 1
    # A duplicate delivery of the generation-2 send, and a recovery re-entry.
    pr.chunk_attempt_step(aid, 2, "send", "rb:dup")
    assert pr.submit_images_only(ids["rid"]) is True
    assert len(kie.creates) == 1
    with SF() as s:
        a = s.get(RebuildChunkAttempt, aid)
        assert a.status == "succeeded" and a.dispatch_gen == 2 and a.task_id == "task-1"
        assert len(s.execute(select(RebuildChunkAttempt).where(
            RebuildChunkAttempt.run_id == ids["rid"])).all()) == 1
        draw = s.execute(select(KieSpendDraw).where(
            KieSpendDraw.call_key == spend.rebuild_call_key(aid))).scalar_one()
        assert s.get(KieSpendReservation, draw.reservation_id).scope_key == spend.run_scope(ids["rid"])


def test_request_is_the_pilot_shape_through_the_real_client(client, SF, media, env, monkeypatch):
    import httpx
    import respx

    for model, expect in (("kling-4-0", "kling"), ("seedance-2-5", "seedance")):
        ids = launched(client, SF, media, model=model)
        real = kie_client.KieClient(api_key="k", sleep_fn=lambda s: None)
        fake = FakeKie()

        class Mixed:
            upload_file = fake.upload_file
            create_task_once = real.create_task_once

        monkeypatch.setattr(pr, "_kie", lambda: Mixed())
        env.steps.clear()
        pr.submit_images_only(ids["rid"])
        drain(env, only={"claim"})
        with respx.mock() as m:
            route = m.post(url__regex=r".*/api/v1/jobs/createTask").mock(
                return_value=httpx.Response(200, json={"code": 200, "data": {"taskId": "t9"}}))
            drain(env, only={"send"})
        assert route.call_count == 1
        body = json.loads(route.calls[0].request.content)
        with SF() as s:
            run = s.get(Run, ids["rid"])
            ch = run.rebuild_snapshot["chunks"][0]
            urls = [f"https://up.example/0/{n}" for n in ("1.png", "2.png", "sheet.png")]
            if expect == "kling":
                assert body == {"model": "kling-4.0/reference-to-video", "input": {
                    "prompt": ch["prompt"], "duration": 4, "resolution": "720p",
                    "aspect_ratio": "9:16", "image_urls": urls}}
            else:
                from app import ai_models

                assert body == {"model": ai_models.AI_MODELS["seedance-2-5"].kie_model_id,
                                "input": {"prompt": ch["prompt"], "reference_image_urls": urls,
                                          "reference_video_urls": [], "resolution": "720p",
                                          "aspect_ratio": "9:16", "duration": 4,
                                          "generate_audio": False}}
            a = s.execute(select(RebuildChunkAttempt).where(
                RebuildChunkAttempt.run_id == ids["rid"])).scalar_one()
            assert a.request == body and a.status == "submitted"


@pytest.mark.parametrize("outcome,status,seg", [
    (kie_client.SendOutcome("refused", error="HTTP 422"), "failed", SegmentStatus.failed),
    (kie_client.SendOutcome("unknown", error="timeout"), "unknown", SegmentStatus.failed),
])
def test_send_refused_or_unknown_is_terminal_never_resent(client, SF, media, env, monkeypatch,
                                                         outcome, status, seg):
    ids = launched(client, SF, media)
    kie = FakeKie(outcome=outcome)
    monkeypatch.setattr(pr, "_kie", lambda: kie)
    pr.submit_images_only(ids["rid"])
    drain(env)
    assert len(kie.creates) == 1 and env.steps == []
    with SF() as s:
        a = s.execute(select(RebuildChunkAttempt).where(
            RebuildChunkAttempt.run_id == ids["rid"])).scalar_one()
        assert a.status == status
        assert s.get(RunSegment, a.run_segment_id).status == seg
    assert pr.submit_images_only(ids["rid"]) is True  # a failed segment is not re-admitted
    assert len(kie.creates) == 1
    assert pr.observe_run(ids["rid"]) == pr.OBSERVE_INCOMPLETE


def test_missing_run_owned_keyframe_fails_unsent(client, SF, media, env, monkeypatch):
    ids = launched(client, SF, media)
    kie = FakeKie()
    monkeypatch.setattr(pr, "_kie", lambda: kie)
    with SF() as s:
        ch = s.get(Run, ids["rid"]).rebuild_snapshot["chunks"][0]
    with open(ch["images"][0], "ab") as fh:  # altered after launch
        fh.write(b"x")
    pr.submit_images_only(ids["rid"])
    drain(env)
    assert kie.creates == []
    with SF() as s:
        a = s.execute(select(RebuildChunkAttempt).where(
            RebuildChunkAttempt.run_id == ids["rid"])).scalar_one()
        assert a.status == "failed" and "changed since launch" in a.error
        assert not s.execute(select(KieSpendDraw).where(
            KieSpendDraw.call_key == spend.rebuild_call_key(a.id))).first()


# ---------------------------------------------------------------------------
# Fit (real ffmpeg, exact frame counts) and fetch
# ---------------------------------------------------------------------------


def test_fit_exact_frames_padding_and_too_short(media, tmp_path):
    out = str(tmp_path / "fitted.mp4")
    q = pr.fit_attempt(media["raw5"], out, 120, 30.0)
    assert frames_of(out) == 120 and q["padded_frames"] == 0 and q["raw_frames"] == 150
    assert not has_audio(out)
    q = pr.fit_attempt(media["raw395"], out, 120, 30.0)  # 118 frames: padded by 2
    assert q["padded_frames"] == 2 and frames_of(out) == 120
    with pytest.raises(pr.FitRefused, match="result too short"):
        pr.fit_attempt(media["raw3"], out, 120, 30.0)
    # A fractional grid: 29.97 fps, a chunk of 1.7-5.4 s.
    n = rb.chunk_frames(1.7, 5.4, 30000 / 1001)
    pr.fit_attempt(media["raw5"], out, n, 30000 / 1001)
    assert frames_of(out) == n


def test_fetch_fits_selects_completes_and_settles(client, SF, media, env, monkeypatch):
    ids = launched(client, SF, media)
    kie = FakeKie(raw=media["raw5"])
    monkeypatch.setattr(pr, "_kie", lambda: kie)
    pr.submit_images_only(ids["rid"])
    drain(env)
    with SF() as s:
        a = s.execute(select(RebuildChunkAttempt).where(
            RebuildChunkAttempt.run_id == ids["rid"])).scalar_one()
        rs = s.get(RunSegment, a.run_segment_id)
        assert a.status == "succeeded" and a.selected and a.cost_credits == 114
        assert rs.status == SegmentStatus.completed and rs.local_result_path == a.fitted_path
        assert frames_of(a.fitted_path) == 120 and os.path.exists(a.raw_path)
        assert a.qa["duration"]["chunk_frames"] == 120 and a.qa["flags"] == []
        draw = s.execute(select(KieSpendDraw).where(
            KieSpendDraw.call_key == spend.rebuild_call_key(a.id))).scalar_one()
        assert draw.mcredits == 114 * 1000
    assert env.qa_jobs == [a.id]
    assert pr.observe_run(ids["rid"]) == pr.OBSERVE_FINALIZE


def test_too_short_take_fails_final_and_download_error_retries(client, SF, media, env,
                                                              monkeypatch):
    ids = launched(client, SF, media)
    kie = FakeKie(raw=media["raw3"])
    monkeypatch.setattr(pr, "_kie", lambda: kie)
    pr.submit_images_only(ids["rid"])
    drain(env)
    with SF() as s:
        a = s.execute(select(RebuildChunkAttempt).where(
            RebuildChunkAttempt.run_id == ids["rid"])).scalar_one()
        assert a.status == "failed" and "result too short" in a.error
        assert s.get(RunSegment, a.run_segment_id).status == SegmentStatus.failed
    assert len(kie.creates) == 1  # no automatic resubmit
    # A download error re-schedules the fetch only.
    ids2 = launched(client, SF, media)
    kie2 = FakeKie(raw="/nonexistent.mp4")
    monkeypatch.setattr(pr, "_kie", lambda: kie2)
    env.steps.clear()
    pr.submit_images_only(ids2["rid"])
    drain(env, limit=4)
    assert env.steps and env.steps[-1][3] == "fetch"
    assert len(kie2.creates) == 1
    with SF() as s:
        a = s.execute(select(RebuildChunkAttempt).where(
            RebuildChunkAttempt.run_id == ids2["rid"])).scalar_one()
        assert a.status == "fetching" and a.fetch_attempts == 1


# ---------------------------------------------------------------------------
# Observer + the tasks chain
# ---------------------------------------------------------------------------


def test_observer_marks_reaper_unknown_segment_failed(client, SF, media, env, monkeypatch):
    ids = launched(client, SF, media)
    pr.submit_images_only(ids["rid"])
    assert pr.observe_run(ids["rid"]) == pr.OBSERVE_WAIT
    with SF() as s:  # the reaper released a `sending` row: no step touched the segment
        a = s.execute(select(RebuildChunkAttempt).where(
            RebuildChunkAttempt.run_id == ids["rid"])).scalar_one()
        a.status, a.error = "unknown", pc.UNKNOWN_MESSAGE
        s.commit()
    assert pr.observe_run(ids["rid"]) == pr.OBSERVE_INCOMPLETE
    with SF() as s:
        run = s.get(Run, ids["rid"])
        assert run.status == RunStatus.incomplete and "[0]" in run.error_message
        rs = s.execute(select(RunSegment).where(RunSegment.run_id == ids["rid"])).scalar_one()
        assert rs.status == SegmentStatus.failed and "may have billed" in rs.error_message
    assert pr.observe_run(ids["rid"]) == pr.OBSERVE_STOP


def test_run_submit_run_branches_and_observer_hands_the_lock(client, SF, media, env,
                                                            monkeypatch):
    import fakeredis
    from rq import Queue

    from app import tasks

    conn = fakeredis.FakeStrictRedis()
    q = Queue("process", connection=conn)
    monkeypatch.setattr(tasks, "_get_redis", lambda: conn)
    monkeypatch.setattr(tasks, "_get_queue", lambda name=None: q)
    called = []
    monkeypatch.setattr("app.pipeline_v2.submit_run", lambda *a, **k: called.append(a))
    ids = launched(client, SF, media)
    tasks.run_submit_run(ids["rid"])
    assert called == []  # the v2v submit never saw the run
    from rq.registry import ScheduledJobRegistry

    (jid,) = ScheduledJobRegistry("process", connection=conn).get_job_ids()
    from rq.job import Job

    job = Job.fetch(jid, connection=conn)
    assert job.func_name == "app.tasks.run_rebuild_observe" and job.args == (ids["rid"],)
    lock_key, token = job.kwargs["lock_key"], job.kwargs["lock_token"]
    assert conn.get(lock_key).decode() == token
    # Chunk completed meanwhile -> the observer hands the lock to finalize.
    kie = FakeKie(raw=media["raw5"])
    monkeypatch.setattr(pr, "_kie", lambda: kie)
    drain(env)
    tasks.run_rebuild_observe(ids["rid"], lock_key=lock_key, lock_token=token)
    fin = [j for j in q.get_jobs() if j.func_name == "app.tasks.run_finalize_run"]
    assert len(fin) == 1 and fin[0].kwargs["lock_token"] == token
    # A stale observer (lock taken over) does nothing.
    conn.set(lock_key, "other")
    tasks.run_rebuild_observe(ids["rid"], lock_key=lock_key, lock_token=token)
    assert len([j for j in q.get_jobs() if j.func_name == "app.tasks.run_finalize_run"]) == 1


# ---------------------------------------------------------------------------
# Stitch with the original soundtrack (§6.6)
# ---------------------------------------------------------------------------


def _complete_run(client, SF, media, env, monkeypatch, **kw):
    ids = launched(client, SF, media, **kw)
    kie = FakeKie(raw=media["raw5"])
    monkeypatch.setattr(pr, "_kie", lambda: kie)
    pr.submit_images_only(ids["rid"])
    drain(env)
    assert pr.observe_run(ids["rid"]) == pr.OBSERVE_FINALIZE
    return ids


@pytest.mark.parametrize("source,keep_card,frames", [
    ("source", "true", 180),   # default plan: chunk 120 + frame-exact keep 60
    ("source", None, 150),     # end card -> library clip (30 frames, own audio)
    ("silent", "true", 180),   # a silent source still delivers (silence)
])
def test_finalize_stitches_on_the_frame_grid_with_original_audio(client, SF, media, env,
                                                                monkeypatch, source,
                                                                keep_card, frames):
    from app import pipeline_v2

    kw = {"source": source}
    if keep_card:
        kw["keep_source_end_card"] = keep_card
    ids = _complete_run(client, SF, media, env, monkeypatch, **kw)
    pipeline_v2.finalize_run(ids["rid"], gdrive=None)
    with SF() as s:
        run = s.get(Run, ids["rid"])
        assert run.status == RunStatus.done, run.error_message
        final = run.result_local_path
    assert frames_of(final) == frames
    assert has_audio(final)
    clips = os.listdir(storage.run_clips_dir(ids["rid"], ids["pid"]))
    if keep_card:
        assert any(c.startswith("rbkeep_0001_120_180") for c in clips)


def test_v2v_keep_cut_is_untouched(SF, media, monkeypatch, tmp_path):
    """A v2v run's keep segment still goes through media.cut_clip at
    storage.keep_clip_path (never the frame-exact images-only cut)."""
    from app import media as media_mod, pipeline_v2

    s = SF()
    p = VideoProject(source_type="upload", source_ref="s", status=ProjectStatus.ready,
                     project_type="face_swap", source_local_path=media["source"])
    s.add(p)
    s.flush()
    sd0 = SegmentDef(project_id=p.id, index=0, start_sec=0, end_sec=4, action="swap")
    sd1 = SegmentDef(project_id=p.id, index=1, start_sec=4, end_sec=6, action="keep")
    s.add_all([sd0, sd1])
    s.flush()
    run = Run(project_id=p.id, model="seedance", resolution="480p", status=RunStatus.processing,
              segment_snapshot=pipeline_v2.build_segment_snapshot([sd0, sd1]))
    s.add(run)
    s.flush()
    res = str(tmp_path / "res.mp4")
    make_video(res, seconds=4, audio=False)
    s.add(RunSegment(run_id=run.id, segment_def_id=sd0.id, index=0,
                     status=SegmentStatus.completed, local_result_path=res))
    s.commit()
    rid = run.id
    s.close()
    cuts, frame_cuts = [], []
    real_cut = media_mod.cut_clip
    monkeypatch.setattr(media_mod, "cut_clip",
                        lambda *a, **k: (cuts.append(a), real_cut(*a, **k))[1])
    monkeypatch.setattr(pr, "cut_keep_frames", lambda *a, **k: frame_cuts.append(a))
    pipeline_v2.finalize_run(rid, gdrive=None)
    assert len(cuts) == 1 and frame_cuts == []
    with SF() as s2:
        assert s2.get(Run, rid).status == RunStatus.done


# ---------------------------------------------------------------------------
# QA + character check (§6.5)
# ---------------------------------------------------------------------------


def _check_answer(missing=()):
    body = {"choices": [{"message": {"content": "```json\n" + json.dumps({"shots": [
        {"shot": 1, "present": ["C1"], "missing": [], "identity_drift": []},
        {"shot": 2, "present": [], "missing": list(missing), "identity_drift": []},
    ]}) + "\n```"}}]}
    return kie_client.SendOutcome("accepted", content=body["choices"][0]["message"]["content"],
                                  body=body)


def test_free_qa_cut_timing_text_band_contact_sheet(client, SF, media, env, monkeypatch):
    from app import qa_tools

    ids = _complete_run(client, SF, media, env, monkeypatch)
    monkeypatch.setattr(qa_tools, "locate_captions",
                        lambda path, **k: {"state": "located", "candidate_band_y": [0.6, 0.7]})
    (aid,) = env.qa_jobs
    qa = pr.free_qa(aid)
    # raw5 cuts at 2.5 s (5 s, two colours) after the 4 s fit: 1 cut vs planned 2.0.
    ct = qa["cut_timing"]
    assert ct["cuts"] and abs(ct["cuts"][0]["planned"] - 2.0) < 1e-6
    assert abs(ct["cuts"][0]["detected"] - 2.5) < 0.1 and not ct["flagged"]
    assert qa["text_band"]["candidate_band_y"] == [0.6, 0.7]
    assert "burned text (located)" in qa["flags"]
    assert os.path.exists(qa["contact_sheet"]) and qa["contact_sheet"].endswith("sheet.jpg")
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=width,height",
                          "-of", "csv=p=0", qa["contact_sheet"]], capture_output=True, text=True)
    w, h = map(int, out.stdout.strip().split(","))
    assert h == 2 * pr.SHEET_CELL_LONG and w > h / 2  # two rows of generated | keyframe
    assert pr.qa_flags({"cut_timing": rb.cut_timing_report([1.0], [3.0])}) == [
        "cut timing (1 missing, 1 extra)"]
    assert pr.qa_flags({"cut_timing": rb.cut_timing_report([1.0], [1.7])})[0].startswith(
        "cut timing (max |Δ| 0.70s")


def test_character_check_single_send_fallback_and_unavailable(client, SF, media, env,
                                                             monkeypatch):
    from app import qa_tools

    monkeypatch.setattr(qa_tools, "locate_captions", lambda path, **k: {"state": "none"})
    ids = _complete_run(client, SF, media, env, monkeypatch)
    (aid,) = env.qa_jobs
    kie = FakeKie(chat=lambda model: _check_answer(missing=["C1"]))
    monkeypatch.setattr(pr, "_kie", lambda: kie)
    pr.run_attempt_qa(aid)
    drain(env)
    pr.character_check_step(env.steps[0][1] if env.steps else "none", 1, "send", "dup")
    assert len(kie.chats) == 1
    with SF() as s:
        a = s.get(RebuildChunkAttempt, aid)
        (call,) = s.execute(select(RebuildLlmCall).where(RebuildLlmCall.attempt_id == aid)).scalars()
        assert call.status == "succeeded" and call.slot == 0
        assert a.qa["character_check"]["flagged"] is True
        assert "shot 2: missing C1" in a.qa["flags"]
        draw = s.execute(select(KieSpendDraw).where(
            KieSpendDraw.call_key == spend.rebuild_call_key(call.id))).scalar_one()
        assert s.get(KieSpendReservation, draw.reservation_id).scope_key == spend.run_scope(ids["rid"])
    # Only the free part is re-enqueued by the QA reaper; the check is not re-sent.
    requeued = []
    pr.reap_qa(now=datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(hours=1),
               is_live=lambda jid: jid != pr.qa_job_id(aid), enqueue=requeued.append)
    assert aid not in requeued and len(kie.chats) == 1
    with SF() as s:
        assert len(s.execute(select(RebuildLlmCall).where(
            RebuildLlmCall.attempt_id == aid)).all()) == 1
    # A lost free-QA job is re-enqueued (only the free part).
    with SF() as s:
        a = s.get(RebuildChunkAttempt, aid)
        a.qa = {k: v for k, v in a.qa.items() if k != "free_done_at"}
        s.commit()
    pr.reap_qa(now=datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(hours=1),
               is_live=lambda jid: jid != pr.qa_job_id(aid), enqueue=requeued.append)
    assert requeued == [aid] and len(kie.chats) == 1
    # A refused primary admits the fallback once; a refused fallback -> unavailable.
    ids2 = _complete_run(client, SF, media, env, monkeypatch)
    aid2 = env.qa_jobs[-1]
    kie2 = FakeKie(chat=lambda model: kie_client.SendOutcome("refused", error="HTTP 400"))
    monkeypatch.setattr(pr, "_kie", lambda: kie2)
    pr.run_attempt_qa(aid2)
    drain(env)
    assert [m for m, _ in kie2.chats] == pr.check_models()[:2]
    with SF() as s:
        a2 = s.get(RebuildChunkAttempt, aid2)
        assert a2.qa["character_check"] == "unavailable"
        assert sorted(c.slot for c in s.execute(select(RebuildLlmCall).where(
            RebuildLlmCall.attempt_id == aid2)).scalars()) == [0, 1]
        run = s.get(Run, ids2["rid"])
        # Both slots drawn and settled to 0 (refused): nothing left to reserve.
        assert spend.estimate_images_only_run_credits(s, run) == 0.0


def test_check_kind_handler_is_registered_for_the_dispatcher():
    from app import rebuild_llm_steps

    assert rebuild_llm_steps.KIND_HANDLERS["character_check"] == \
        "app.pipeline_rebuild.character_check_step"
    assert pc.STEP_HANDLERS["rebuild_chunk_attempts"] == "app.pipeline_rebuild.chunk_attempt_step"
    # Integration: the table's handler is 2a's dispatcher, which routes by kind.
    assert pc.STEP_HANDLERS["rebuild_llm_calls"] == "app.rebuild_llm_steps.handle"


# ---------------------------------------------------------------------------
# Retry / re-roll / use-attempt: reserve + draw through the endpoints (§7.3)
# ---------------------------------------------------------------------------


def _reservation(s, rid):
    return sum(r.reserved_mcredits for r in s.execute(select(KieSpendReservation).where(
        KieSpendReservation.scope_key == spend.run_scope(rid))).scalars()) / 1000


def test_retry_after_failed_attempt_resets_and_reserves(client, SF, media, env, monkeypatch):
    ids = launched(client, SF, media)
    kie = FakeKie(outcome=kie_client.SendOutcome("refused", error="HTTP 422"))
    monkeypatch.setattr(pr, "_kie", lambda: kie)
    pr.submit_images_only(ids["rid"])
    drain(env)
    assert pr.observe_run(ids["rid"]) == pr.OBSERVE_INCOMPLETE
    client.enqueued.clear()
    r = client.post(f"/api/v2/runs/{ids['rid']}/retry")
    assert r.status_code == 200, r.text
    assert client.enqueued == [ids["rid"]]
    with SF() as s:
        run = s.get(Run, ids["rid"])
        assert run.status == RunStatus.queued
        rs = s.execute(select(RunSegment).where(RunSegment.run_id == ids["rid"])).scalar_one()
        assert rs.status == SegmentStatus.pending
        # The new attempt (114) + its check allowance (40), re-baselined: the
        # refused attempt settled to 0.
        assert _reservation(s, ids["rid"]) == pytest.approx(114 + 40)
    kie.outcome = kie_client.SendOutcome("accepted", task_id="t2")
    kie.raw = media["raw5"]
    pr.submit_images_only(ids["rid"])
    drain(env)
    with SF() as s:
        nos = sorted(a.attempt_no for a in s.execute(select(RebuildChunkAttempt).where(
            RebuildChunkAttempt.run_id == ids["rid"])).scalars())
        assert nos == [1, 2]
    assert len(kie.creates) == 2


@pytest.mark.spend_cap
def test_rerun_at_the_budget_limit(client, SF, media, env, monkeypatch):
    ids = _complete_run(client, SF, media, env, monkeypatch)
    from app import pipeline_v2

    pipeline_v2.finalize_run(ids["rid"], gdrive=None)
    with SF() as s:
        rs_id = s.execute(select(RunSegment.id).where(RunSegment.run_id == ids["rid"])).scalar()
        charged = spend.today_summary(s)["charged_credits"]
    monkeypatch.setattr(settings, "KIE_USD_PER_CREDIT", 0.005)
    monkeypatch.setattr(settings, "KIE_DAILY_BUDGET_USD", (charged + 100) * 0.005)
    r = client.post(f"/api/v2/runs/{ids['rid']}/segments/{rs_id}/rerun",
                    data={"prompt": "an edited prompt"})
    assert r.status_code == 429, r.text
    with SF() as s:  # nothing changed
        rs = s.get(RunSegment, rs_id)
        assert rs.status == SegmentStatus.completed and rs.prompt_override is None
        assert s.get(Run, ids["rid"]).status == RunStatus.done
    monkeypatch.setattr(settings, "KIE_DAILY_BUDGET_USD", (charged + 1000) * 0.005)
    r = client.post(f"/api/v2/runs/{ids['rid']}/segments/{rs_id}/rerun",
                    data={"prompt": "an edited prompt"})
    assert r.status_code == 200, r.text
    with SF() as s:
        rs = s.get(RunSegment, rs_id)
        assert rs.status == SegmentStatus.pending and rs.prompt_override == "an edited prompt"
        assert rs.local_result_path is None
        assert not s.execute(select(RebuildChunkAttempt).where(
            RebuildChunkAttempt.run_segment_id == rs_id,
            RebuildChunkAttempt.selected.is_(True))).first()
    # The re-roll sends the edited prompt, once.
    kie = FakeKie(raw=media["raw5"], outcome=kie_client.SendOutcome("accepted", task_id="t3"))
    monkeypatch.setattr(pr, "_kie", lambda: kie)
    pr.submit_images_only(ids["rid"])
    drain(env)
    assert [c["input"]["prompt"] for c in kie.creates] == ["an edited prompt"]


def test_reroll_refusals_and_use_attempt(client, SF, media, env, monkeypatch):
    from app import pipeline_v2

    ids = _complete_run(client, SF, media, env, monkeypatch)
    pipeline_v2.finalize_run(ids["rid"], gdrive=None)
    with SF() as s:
        rs_id = s.execute(select(RunSegment.id).where(RunSegment.run_id == ids["rid"])).scalar()
        first = s.execute(select(RebuildChunkAttempt.id).where(
            RebuildChunkAttempt.run_id == ids["rid"])).scalar()
    base = f"/api/v2/runs/{ids['rid']}"
    r = client.post(f"{base}/segments/{rs_id}/rerun", data={"reference_urls": "https://x/a.png"})
    assert r.status_code == 400
    monkeypatch.setattr(settings, "REBUILD_ENABLED", False)
    assert client.post(f"{base}/segments/{rs_id}/rerun", data={"prompt": ""}).status_code == 409
    monkeypatch.setattr(settings, "REBUILD_ENABLED", True)
    for path, data in ((f"{base}/copy", {}), (f"{base}/segments/restore", None)):
        rr = client.post(path, data=data) if data is not None else client.post(path, json={})
        assert rr.status_code in (409, 422), (path, rr.text)
    assert client.post(f"{base}/segments/{rs_id}/rerun", data={"prompt": ""}).status_code == 200
    kie = FakeKie(raw=media["raw395"], outcome=kie_client.SendOutcome("accepted", task_id="t4"))
    monkeypatch.setattr(pr, "_kie", lambda: kie)
    pr.submit_images_only(ids["rid"])
    # While the re-roll is live, use-attempt / another rerun are refused.
    assert client.post(f"{base}/rebuild/segments/{rs_id}/use-attempt/{first}",
                       data={"discard_audio_patch": "false"}).status_code == 409
    drain(env)
    assert pr.observe_run(ids["rid"]) == pr.OBSERVE_FINALIZE
    pipeline_v2.finalize_run(ids["rid"], gdrive=None)
    r = client.post(f"{base}/rebuild/segments/{rs_id}/use-attempt/{first}",
                    data={"discard_audio_patch": "false"})
    assert r.status_code == 200, r.text
    with SF() as s:
        a1 = s.get(RebuildChunkAttempt, first)
        rs = s.get(RunSegment, rs_id)
        assert a1.selected and rs.local_result_path == a1.fitted_path
        assert rs.status == SegmentStatus.completed
        run = s.get(Run, ids["rid"])
        assert run.status == RunStatus.queued and run.result_local_path is None
        assert len([a for a in s.execute(select(RebuildChunkAttempt).where(
            RebuildChunkAttempt.run_segment_id == rs_id,
            RebuildChunkAttempt.selected.is_(True))).scalars()]) == 1


def test_attempt_files_are_served(client, SF, media, env, monkeypatch):
    ids = _complete_run(client, SF, media, env, monkeypatch)
    (aid,) = env.qa_jobs
    for kind in ("raw", "fitted"):
        r = client.get(f"/api/v2/runs/{ids['rid']}/rebuild/attempts/{aid}/{kind}")
        assert r.status_code == 200 and r.headers["content-type"].startswith("video/")
    assert client.get(f"/api/v2/runs/{ids['rid']}/rebuild/attempts/{aid}/sheet").status_code == 404
    assert client.get(f"/api/v2/runs/{ids['rid']}/rebuild/attempts/{aid}/x").status_code == 404


# ---------------------------------------------------------------------------
# Run page HTML (§10) — each state
# ---------------------------------------------------------------------------


def _page(client, rid):
    full = client.get(f"/v2/runs/{rid}")
    frag = client.get(f"/v2/runs/{rid}/status-fragment")
    assert full.status_code == 200 and frag.status_code == 200, (full.text[:500], frag.text[:500])
    return full.text, frag.text


def test_run_page_states(client, SF, media, env, monkeypatch):
    from app import pipeline_v2, qa_tools

    monkeypatch.setattr(qa_tools, "locate_captions", lambda path, **k: {"state": "none"})
    # pending: launched, nothing admitted yet.
    ids = launched(client, SF, media)
    full, frag = _page(client, ids["rid"])
    for html in (full, frag):
        assert 'id="rebuild-chunks"' in html and 'data-status="pending"' in html
        assert "Per-Segment Overrides" not in html
        assert "Re-roll chunk #" not in html  # not idle
        assert "Shot 1 (0.0-2.0 s), starts from @image_1" in html
    # generating: the attempt is with the provider.
    kie = FakeKie(raw=media["raw5"], state={"state": "generating"})
    monkeypatch.setattr(pr, "_kie", lambda: kie)
    pr.submit_images_only(ids["rid"])
    drain(env, limit=3)
    _full, frag = _page(client, ids["rid"])
    assert 'data-status="submitted"' in frag and 'data-attempt-status="submitted"' in frag
    assert "generating" in frag and "~114" in frag  # label + the estimate drawn at send
    # one chunk failed: incomplete, the re-roll offered.
    ids2 = launched(client, SF, media)
    kie2 = FakeKie(outcome=kie_client.SendOutcome("refused", error="HTTP 422 bad image"))
    monkeypatch.setattr(pr, "_kie", lambda: kie2)
    pr.submit_images_only(ids2["rid"])
    drain(env)
    pr.observe_run(ids2["rid"])
    _full, frag = _page(client, ids2["rid"])
    assert 'data-status="failed"' in frag and "HTTP 422 bad image" in frag
    assert "Re-roll chunk #0 (~154 cr)" in frag
    assert "Run Incomplete" in frag
    # done with flags + an older usable attempt.
    ids3 = _complete_run(client, SF, media, env, monkeypatch)
    pipeline_v2.finalize_run(ids3["rid"], gdrive=None)
    with SF() as s:
        a = s.execute(select(RebuildChunkAttempt).where(
            RebuildChunkAttempt.run_id == ids3["rid"])).scalar_one()
        old = RebuildChunkAttempt(run_id=a.run_id, run_segment_id=a.run_segment_id,
                                  attempt_no=0, model=a.model, resolution=a.resolution,
                                  request_sec=4, chunk_frames=120, status="succeeded",
                                  fitted_path=a.fitted_path, raw_path=a.raw_path,
                                  estimate_credits=114, cost_credits=114)
        s.add(old)
        a.qa = dict(a.qa, cut_timing=rb.cut_timing_report([2.0], [2.9]),
                    character_check={"shots": [{"shot": 2, "missing": ["C1"],
                                                "identity_drift": []}], "flagged": True})
        a.qa = dict(a.qa, flags=pr.qa_flags(a.qa))
        s.commit()
    full, frag = _page(client, ids3["rid"])
    for html in (full, frag):
        assert 'data-status="completed"' in html
        assert "⚑ cut timing (max |Δ| 0.90s)" in html and "shot 2: missing C1" in html
        assert "Use this attempt</button>" in html and "▶ raw take" in html and "▶ fitted" in html
        assert "in use" in html and "Re-roll chunk #0" in html
    # flag off: readable, no re-roll.
    monkeypatch.setattr(settings, "REBUILD_ENABLED", False)
    full, frag = _page(client, ids3["rid"])
    for html in (full, frag):
        assert "js-rebuild-disabled" in html and "Re-roll chunk #" not in html
        assert 'id="rebuild-chunks"' in html


def test_v2v_run_page_has_no_chunk_card(client, SF):
    with SF() as s:
        p = VideoProject(source_type="upload", source_ref="s", status=ProjectStatus.ready,
                         project_type="face_swap")
        s.add(p)
        s.flush()
        run = Run(project_id=p.id, model="seedance", resolution="480p", status=RunStatus.queued)
        s.add(run)
        s.commit()
        rid = run.id
    full, frag = _page(client, rid)
    assert 'id="rebuild-chunks"' not in full and 'id="rebuild-chunks"' not in frag


def test_qa_reaper_folds_a_check_the_paid_call_reaper_released(client, SF, media, env,
                                                               monkeypatch):
    """A check call released by the paid-call REAPER (`sending -> unknown`)
    never ran its handler's tail: the QA reaper admits the fallback once, and
    after the fallback ends the same way the check is `unavailable`."""
    from app import qa_tools

    monkeypatch.setattr(qa_tools, "locate_captions", lambda path, **k: {"state": "none"})
    _complete_run(client, SF, media, env, monkeypatch)
    aid = env.qa_jobs[-1]
    pr.free_qa(aid)
    later = datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(hours=1)
    first = pr.admit_auto_check(aid)
    with SF() as s:
        s.get(RebuildLlmCall, first).status = "unknown"
        s.commit()
    pr.reap_qa(now=later, is_live=lambda jid: True, enqueue=lambda a: None)
    with SF() as s:
        calls = {c.slot: c for c in s.execute(select(RebuildLlmCall).where(
            RebuildLlmCall.attempt_id == aid)).scalars()}
        assert sorted(calls) == [0, 1] and calls[1].status == "queued"
        calls[1].status = "unknown"
        s.commit()
    pr.reap_qa(now=later, is_live=lambda jid: True, enqueue=lambda a: None)
    with SF() as s:
        assert s.get(RebuildChunkAttempt, aid).qa["character_check"] == "unavailable"
        assert len(s.execute(select(RebuildLlmCall).where(
            RebuildLlmCall.attempt_id == aid)).all()) == 2


def test_stale_fetch_overlapping_a_redrive_never_touches_the_winners_files(
        client, SF, media, env, monkeypatch):
    """Terra 2b round 1: a fetch whose generation the reaper bumped while it
    downloaded must not overwrite (or corrupt) the files the newer generation
    selected; it writes its own paths and removes them when fenced out."""
    _terminalise_live_attempts(SF)
    ids = launched(client, SF, media)
    kie = FakeKie(raw=media["raw5"])
    monkeypatch.setattr(pr, "_kie", lambda: kie)
    pr.submit_images_only(ids["rid"])
    drain(env, only={"claim", "send", "poll"})
    (fetch,) = env.steps
    env.steps.clear()
    aid = fetch[1]
    real_download = kie.download_result
    state = {"done": False}

    def download(url, dst):
        real_download(url, dst)
        if not state["done"]:  # the reaper re-drives meanwhile; gen 2 completes first
            state["done"] = True
            with SF() as s:
                a = s.get(RebuildChunkAttempt, aid)
                a.dispatch_gen = 2
                s.commit()
            assert pr.chunk_attempt_step(aid, 2, "fetch", "rb:g2") == "succeeded"

    kie.download_result = download
    assert pr.chunk_attempt_step(aid, 1, "fetch", "rb:g1") is None  # fenced out
    with SF() as s:
        a = s.get(RebuildChunkAttempt, aid)
        rs = s.get(RunSegment, a.run_segment_id)
        assert a.status == "succeeded" and "-g2-" in a.fitted_path and "-g2-" in a.raw_path
        assert rs.local_result_path == a.fitted_path and frames_of(a.fitted_path) == 120
        files = sorted(os.listdir(os.path.dirname(a.fitted_path)))
    assert not [f for f in files if "-g1-" in f]  # the stale execution cleaned up
    assert len(kie.creates) == 1


def test_non_json_check_answer_falls_back_and_manual_check(client, SF, media, env,
                                                          monkeypatch):
    from app import qa_tools

    monkeypatch.setattr(qa_tools, "locate_captions", lambda path, **k: {"state": "none"})
    ids = _complete_run(client, SF, media, env, monkeypatch)
    aid = env.qa_jobs[-1]
    prose = kie_client.SendOutcome("accepted", content="looks fine to me",
                                   body={"choices": [{"message": {"content": "looks fine"}}]})
    kie = FakeKie(chat=lambda model: prose)
    monkeypatch.setattr(pr, "_kie", lambda: kie)
    pr.run_attempt_qa(aid)
    drain(env)
    assert len(kie.chats) == 2  # the primary's prose answer admitted the fallback once
    with SF() as s:
        assert s.get(RebuildChunkAttempt, aid).qa["character_check"] == "unavailable"
        before = _reservation(s, ids["rid"])
    _full, frag = _page(client, ids["rid"])
    assert "Run character check</button>" in frag
    r = client.post(f"/api/v2/runs/{ids['rid']}/rebuild/attempts/{aid}/character-check")
    assert r.status_code == 202, r.text
    assert r.json()["slot"] == 2
    with SF() as s:
        assert _reservation(s, ids["rid"]) == pytest.approx(before + 20.0)
        assert "character_check" not in s.get(RebuildChunkAttempt, aid).qa
    kie.chat = lambda model: _check_answer()
    drain(env)
    assert len(kie.chats) == 3
    with SF() as s:
        cc = s.get(RebuildChunkAttempt, aid).qa["character_check"]
        assert isinstance(cc, dict) and cc["flagged"] is False
    # A second manual check while one is live is refused.
    r = client.post(f"/api/v2/runs/{ids['rid']}/rebuild/attempts/{aid}/character-check")
    assert r.status_code == 202
    r2 = client.post(f"/api/v2/runs/{ids['rid']}/rebuild/attempts/{aid}/character-check")
    assert r2.status_code == 409


def test_fetch_sweeps_older_generation_files(tmp_path):
    for name in ("raw-g1-aaaaaaaa.mp4", "fitted-g1-bbbbbbbb.mp4.cccccccc.part.mp4",
                 "raw-g1-dddddddd.mp4.part", "raw-g2-eeeeeeee.mp4", "sheet.jpg"):
        (tmp_path / name).write_bytes(b"x")
    pr._sweep_stale_fetch_files(str(tmp_path), 2)
    assert sorted(os.listdir(tmp_path)) == ["raw-g2-eeeeeeee.mp4", "sheet.jpg"]


def test_a_losing_concurrent_fold_never_marks_the_check_unavailable(client, SF, media, env,
                                                                   monkeypatch):
    from app import qa_tools

    monkeypatch.setattr(qa_tools, "locate_captions", lambda path, **k: {"state": "none"})
    _complete_run(client, SF, media, env, monkeypatch)
    aid = env.qa_jobs[-1]
    pr.free_qa(aid)
    first = pr.admit_auto_check(aid)
    with SF() as s:
        s.get(RebuildLlmCall, first).status = "failed"
        s.commit()
    pr._check_terminal(first)      # the handler's fold admits the fallback
    pr._check_terminal(first)      # the QA reaper's concurrent fold loses the slot race
    with SF() as s:
        assert "character_check" not in (s.get(RebuildChunkAttempt, aid).qa or {})
        assert len(s.execute(select(RebuildLlmCall).where(
            RebuildLlmCall.attempt_id == aid)).all()) == 2


def test_create_run_called_directly_like_the_autopilot_still_works(SF, monkeypatch):
    """app/autopilot/actions.py calls api_v2.create_run as a function without
    the rebuild-only fields (they hold their Form() defaults): a v2v launch
    must be unaffected."""
    import app.api_v2 as api_v2_module

    monkeypatch.setattr(api_v2_module, "enqueue_process_run", lambda rid, **kw: None)
    with SF() as s:
        p = VideoProject(source_type="upload", source_ref="s", status=ProjectStatus.ready,
                         project_type="face_swap")
        s.add(p)
        s.flush()
        s.add(SegmentDef(project_id=p.id, index=0, start_sec=0, end_sec=5, action="swap"))
        s.commit()
        resp = api_v2_module.create_run(
            p.id, prompt="swap", name=None, model="seedance", resolution="480p",
            audio_mode="original", language=None, gdrive_folder_id=None, reference_files=[],
            reference_urls=None, segment_prompts=None, stitch_plan=None,
            preprocess_enabled=False, preprocess_mode="per_image",
            preprocess_image_prompts=None, preprocess_reference_file=None,
            preprocess_reference_url=None, voice_clip_id=None, voice_file=None, db=s,
        )
        s.commit()
        run = s.get(Run, resp.run_id)
        assert run.generation_mode == "v2v" and run.resolution == "480p"
        assert run.status == RunStatus.queued


def test_launch_accepts_the_2a_form_contract(client, SF, media):
    """The 2a launch form (#rb-launch-form): prompt text (ignored), language
    `en`, audio_mode `original`, an explicit every-segment plan for "keep
    source end card", and `chunk_sheets`."""
    ids = ready_project(SF, media)
    plan = json.dumps([{"kind": "segment", "segment_def_id": d} for d in ids["defs"]])
    r = client.post(f"/api/v2/projects/{ids['pid']}/runs", data={
        "prompt": "Rebuild from reference (prompts are rendered per chunk)",
        "model": "kling-4-0", "resolution": "720p", "language": "en",
        "audio_mode": "original", "name": "rb", "stitch_plan": plan,
        "chunk_sheets": json.dumps({ids["defs"][0]: "main"})})
    assert r.status_code == 201, r.text
    with SF() as s:
        run = s.get(Run, r.json()["run_id"])
        assert run.stitch_plan is None  # every segment in order: the source end card kept
        assert run.prompt == "" and run.rebuild_snapshot["chunks"][0]["sheet_key"] == "main"
    lib = {"kind": "library", "clip_id": None}
    with SF() as s:
        lib["clip_id"] = s.execute(select(LibraryClip.id)).scalars().first()
    r = client.post(f"/api/v2/projects/{ids['pid']}/runs", data={
        "prompt": "x", "model": "seedance-2-5", "language": "en", "audio_mode": "original",
        "stitch_plan": json.dumps([{"kind": "segment", "segment_def_id": ids["defs"][0]}, lib])})
    assert r.status_code == 201, r.text
    with SF() as s:
        assert s.get(Run, r.json()["run_id"]).stitch_plan[-1] == lib
