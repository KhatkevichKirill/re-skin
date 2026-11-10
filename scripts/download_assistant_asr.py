"""Fetch only the pinned public small ASR model; never called at app runtime."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

REPOSITORY = "Systran/faster-whisper-small"
REVISION = "536b0662742c02347bc0e980a01041f333bce120"
FILES = ("config.json", "model.bin", "tokenizer.json", "vocabulary.txt")
WEIGHTS_SHA256 = "3e305921506d8872816023e4c273e75d2419fb89b24da97b4fe7bce14170d671"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("destination",type=Path)
    parser.add_argument("--manifest",type=Path,required=True)
    args = parser.parse_args()
    args.destination.mkdir(parents=True,exist_ok=True)
    total, manifest = 0, {"repository":REPOSITORY,"revision":REVISION,"files":{}}
    for name in FILES:
        path = args.destination/name
        digest, size = hashlib.sha256(),0
        request = urllib.request.Request(f"https://huggingface.co/{REPOSITORY}/resolve/{REVISION}/{name}")
        with urllib.request.urlopen(request,timeout=60) as response, path.open("xb") as output:
            while chunk := response.read(1024*1024):
                total += len(chunk)
                size += len(chunk)
                if total>600_000_000:
                    raise RuntimeError("model download exceeded600MB budget")
                digest.update(chunk)
                output.write(chunk)
        sha = digest.hexdigest()
        if name=="model.bin" and sha!=WEIGHTS_SHA256:
            raise RuntimeError("model weights do not match pinned LFS SHA256")
        manifest["files"][name] = {"sha256":sha,"size_bytes":size}
        print(name,size,sha,flush=True)
    args.manifest.write_text(json.dumps(manifest,indent=2)+"\n")


if __name__=="__main__":
    main()
