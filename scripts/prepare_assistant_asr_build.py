"""Prepare the minimal offline ASR Docker context from hash-locked wheels."""
import argparse
import email
import hashlib
from pathlib import Path
import re
import shutil
import zipfile


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("wheels",type=Path)
    parser.add_argument("destination",type=Path)
    args=parser.parse_args()
    repo=Path(__file__).resolve().parents[1]
    lock=repo/"backend/requirements-asr-cp312.lock"
    expected={}
    normalize=lambda name:re.sub(r"[-_.]+","-",name).lower()
    for line in lock.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        match=re.fullmatch(r"([A-Za-z0-9_.-]+)==([^ ]+) --hash=sha256:([0-9a-f]{64})",line)
        if match is None:
            raise ValueError("invalid ASR hash lock line")
        expected[(normalize(match[1]),match[2])]=match[3]
    found={}
    for wheel in args.wheels.glob("*.whl"):
        with zipfile.ZipFile(wheel) as archive:
            metadata=email.message_from_bytes(archive.read(next(n for n in archive.namelist() if n.endswith(".dist-info/METADATA"))))
        identity=(normalize(metadata["Name"]),metadata["Version"])
        digest=hashlib.sha256(wheel.read_bytes()).hexdigest()
        if identity not in expected or digest!=expected[identity] or identity in found:
            raise ValueError("wheel set differs from approved exact hash lock")
        found[identity]=wheel
    if set(found)!=set(expected):
        raise ValueError("ASR wheel set is incomplete")
    args.destination.mkdir(parents=True,exist_ok=False)
    (args.destination/"media-source.sha256").write_text(
        hashlib.sha256((repo/"backend/sandbox/media_exec.c").read_bytes()).hexdigest()+"\n")
    (args.destination/"wheels").mkdir()
    for wheel in found.values():
        shutil.copyfile(wheel,args.destination/"wheels"/wheel.name)
    for source,target in ((lock,"requirements-asr-cp312.lock"),
        (repo/"deploy/assistant/asr.Dockerfile","Dockerfile"),
        (repo/"deploy/assistant/asr-small-manifest.json","asr-small-manifest.json"),
        (repo/"backend/app/assistant_asr_worker.py","assistant_asr_worker.py")):
        shutil.copyfile(source,args.destination/target)
    (args.destination/"asrapp").mkdir()
    for name in ("assistant_asr.py","assistant_asr_worker.py","assistant_asr_service.py",
                 "assistant_audio.py","assistant_subtitles.py","assistant_media.py","assistant_media_sandbox.py","assistant_contracts.py"):
        shutil.copyfile(repo/"backend/app"/name,args.destination/"asrapp"/name)
    print(f"Prepared {len(found)} verified wheels and trusted ASR modules at {args.destination}")


if __name__=="__main__":
    main()
