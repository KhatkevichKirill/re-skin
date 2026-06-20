"""v2 orchestrator verification: real end-to-end on Erewhon (480p).

Project (analyze once) → Run (one character) → process → deliver to GDrive.
Run from project root:  python3 scripts/e2e_v2_erewhon.py
Uses real InsightFace + kie.ai/Seedance (spends credits) + Google Drive.
"""
import sys, os, time
# Resolve relative ./secrets and ./data against the project root when run on the
# host (in Docker APP_BASE_DIR=/app; here it must point at the repo root, else the
# SA file / sqlite db resolve under backend/ and Drive delivery fails).
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("APP_BASE_DIR", _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "backend"))

from app.db import engine, Base, SessionLocal
from app import models
from app.models import VideoProject, Run
from app.state_machine import ProjectStatus, RunStatus, transition
from app import pipeline_v2

SRC = "/root/Downloads/Full_HowDoYou_Erewhon_JC_V1.mp4"
REF = "https://res.cloudinary.com/dtfdgwupa/image/upload/v1781901629/Q-00017_dn7jos.png"
PROMPT = ("Replace the main person in the reference video with the person shown in the reference image. "
          "Keep their face and identity consistent with the reference image throughout. Change only the "
          "character — keep everything else exactly the same: the phone or tablet screen and its contents, "
          "all on-screen text and captions, the background, lighting, framing, and the original motion and lip movements.")

Base.metadata.create_all(engine)

with SessionLocal() as s:
    proj = VideoProject(
        source_type="upload", source_ref=os.path.basename(SRC),
        source_local_path=SRC, status=ProjectStatus.created,
    )
    s.add(proj); s.commit(); project_id = proj.id
print(f"[e2e-v2] project_id={project_id}")

t0 = time.time()
print("[e2e-v2] analyze_project ...")
pipeline_v2.analyze_project(project_id)
with SessionLocal() as s:
    proj = s.get(VideoProject, project_id)
    print(f"[e2e-v2] project status={proj.status} dur={proj.duration_sec} {proj.width}x{proj.height}@{proj.fps}")
    for sd in proj.segments:
        print(f"   def[{sd.index}] {sd.start_sec:.2f}-{sd.end_sec:.2f} {sd.action}")

with SessionLocal() as s:
    run = Run(
        project_id=project_id, name="Redhead woman", prompt=PROMPT,
        reference_image_urls=[REF], resolution="480p", status=RunStatus.created,
    )
    s.add(run); s.flush()
    transition(run, RunStatus.queued)
    s.commit(); run_id = run.id
print(f"[e2e-v2] run_id={run_id} (queued)")

print("[e2e-v2] process_run ... (real Seedance)")
pipeline_v2.process_run(run_id)
with SessionLocal() as s:
    run = s.get(Run, run_id)
    print(f"[e2e-v2] FINAL run status={run.status}")
    print(f"[e2e-v2] result_local_path={run.result_local_path}")
    print(f"[e2e-v2] result_gdrive_file_id={run.result_gdrive_file_id}")
    print(f"[e2e-v2] error={run.error_message}")
    for rs in run.run_segments:
        print(f"   runseg[{rs.index}] status={rs.status} task={rs.seedance_task_id} result={bool(rs.local_result_path)}")
print(f"[e2e-v2] total {time.time()-t0:.0f}s")
