"""Orchestrator verification: real end-to-end pipeline run on the Erewhon sample (480p).

Run from project root:  python3 scripts/e2e_erewhon.py
Uses real InsightFace, real kie.ai/Seedance (spends credits), real Google Drive.
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from app.db import engine, Base, SessionLocal
from app import models
from app.models import Job
from app.state_machine import JobStatus
from app import pipeline

SRC = "/root/Downloads/Full_HowDoYou_Erewhon_JC_V1.mp4"
REF = "https://res.cloudinary.com/dtfdgwupa/image/upload/v1781901629/Q-00017_dn7jos.png"
PROMPT = ("Take the person in the reference image and replace the girl in the reference "
          "video with her. Change only the character, do not touch any other elements, "
          "especially the phone screen, on-screen texts, captions, and background.")

Base.metadata.create_all(engine)

with SessionLocal() as s:
    job = Job(
        source_type="upload",
        source_ref=os.path.basename(SRC),
        source_local_path=SRC,
        default_prompt=PROMPT,
        default_reference_image_urls=[REF],
        resolution="480p",
        gdrive_folder_id=None,  # use settings default
        status=JobStatus.created,
    )
    s.add(job); s.commit()
    job_id = job.id
print(f"[e2e] job_id={job_id}")

t0 = time.time()
print("[e2e] analyze_job ...")
pipeline.analyze_job(job_id)
with SessionLocal() as s:
    job = s.get(Job, job_id)
    print(f"[e2e] after analyze: status={job.status} dur={job.duration_sec} "
          f"{job.width}x{job.height}@{job.fps} aspect={job.aspect_ratio}")
    for seg in job.segments:
        print(f"   seg[{seg.index}] {seg.start_sec:.2f}-{seg.end_sec:.2f} "
              f"action={seg.action} status={seg.status}")
    # operator "submits": review -> queued
    from app.state_machine import transition
    transition(job, JobStatus.queued)
    s.commit()

print("[e2e] process_job ... (real Seedance, sequential)")
pipeline.process_job(job_id)
with SessionLocal() as s:
    job = s.get(Job, job_id)
    print(f"[e2e] FINAL status={job.status}")
    print(f"[e2e] result_local_path={job.result_local_path}")
    print(f"[e2e] result_gdrive_file_id={job.result_gdrive_file_id}")
    print(f"[e2e] error_message={job.error_message}")
    for seg in job.segments:
        print(f"   seg[{seg.index}] action={seg.action} status={seg.status} "
              f"task={seg.seedance_task_id} result={bool(seg.local_result_path)}")
print(f"[e2e] total {time.time()-t0:.0f}s")
