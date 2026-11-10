"""Credential-free, serial Unix ASR sidecar over immutable artifact IDs only.

Run in a dedicated network-none container capped at 2 GiB/2 CPUs. Mount model
and assistant artifact objects read-only; share only its private Unix socket
directory with the trusted backend. The model child additionally denies network
syscalls and receives only a bounded decoded chunk, no application credentials.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import fcntl
import json
import os
from pathlib import Path
import re
import socket
import struct
import stat
import sys
import time

from .assistant_asr import ASRTranscript, LocalASR
from . import assistant_media as media

MAX_REQUEST=16000
MAX_RESPONSE=1_000_000


def _peer(sock,expected_uid):
    _,uid,_=struct.unpack("3i",sock.getsockopt(socket.SOL_SOCKET,socket.SO_PEERCRED,12))
    if uid!=expected_uid:
        raise ValueError("ASR socket peer denied")


def _receive(sock,limit):
    data=bytearray()
    while True:
        chunk=sock.recv(min(65536,limit+1-len(data)))
        if not chunk:
            break
        data.extend(chunk)
        if len(data)>limit:
            raise ValueError("ASR message exceeds size limit")
        if b"\n" in data:
            head,tail=bytes(data).split(b"\n",1)
            if tail:
                raise ValueError("ASR request contains trailing data")
            return head
    return bytes(data)


@dataclass(frozen=True)
class ASRSidecarClient:
    socket_path: Path
    artifact_root: Path
    peer_uid: int = 10001

    def transcribe(self,path,*,start_sec=0,end_sec,language=None,timeout_sec=120,cancelled=None):
        budget=media._budget(timeout_sec,cancelled)
        source=media._input(path)
        relative=source.relative_to(self.artifact_root.resolve()).as_posix()
        if re.fullmatch(r"objects/([0-9a-f]{2})/([0-9a-f]{64})",relative) is None:
            raise media.AssistantMediaError("ASR sidecar requires a pinned immutable artifact")
        digest=media._hash(source,budget)
        if source.name!=digest:
            raise media.AssistantMediaError("ASR source hash does not match object identity")
        request={"storage_key":relative,"sha256":digest,"size_bytes":source.stat().st_size,
                 "start_sec":start_sec,"end_sec":end_sec,"language":language}
        with socket.socket(socket.AF_UNIX,socket.SOCK_STREAM) as sock:
            sock.settimeout(.25)
            sock.connect(str(self.socket_path))
            _peer(sock,self.peer_uid)
            sock.sendall(json.dumps(request,allow_nan=False).encode()+b"\n")
            data=bytearray()
            while True:
                budget.check()
                try:
                    chunk=sock.recv(65536)
                except socket.timeout:
                    continue
                if not chunk:
                    break
                data.extend(chunk)
                if len(data)>MAX_RESPONSE:
                    raise media.AssistantMediaError("ASR sidecar response too large")
        response=json.loads(data)
        if response.get("status")!="succeeded":
            raise media.AssistantMediaError("ASR sidecar unavailable for this input")
        transcript=ASRTranscript.model_validate(response["transcript"])
        if (transcript.network_denied is not True or not 0<transcript.max_rss_kib<=2*1024**2
                or abs(transcript.source_start_sec-start_sec)>1e-6
                or abs(transcript.source_end_sec-end_sec)>1e-6
                or transcript.source_duration_sec+1e-6<end_sec
                or any(w.start_sec<start_sec or w.end_sec<w.start_sec or w.end_sec>end_sec+.1 for w in transcript.words)):
            raise media.AssistantMediaError("ASR sidecar returned invalid coverage or sandbox evidence")
        return transcript


def serve(*,socket_path: Path,artifact_root: Path,asr: LocalASR,peer_uid=10001,stop_requested=None):
    artifact_root=artifact_root.resolve(strict=True)
    socket_path.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
    lock=os.open(socket_path.parent/"asr.lock",os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW,0o600)
    try:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except Exception:
        os.close(lock)
        raise ValueError("ASR sidecar already active") from None
    if socket_path.exists():
        info=socket_path.lstat()
        if not stat.S_ISSOCK(info.st_mode) or info.st_uid!=os.getuid():
            os.close(lock)
            raise ValueError("ASR stale socket is unsafe")
        socket_path.unlink()
    with socket.socket(socket.AF_UNIX,socket.SOCK_STREAM) as server:
        server.bind(str(socket_path))
        os.chmod(socket_path,0o600)
        server.listen(1)
        server.settimeout(.25)
        try:
            while not (stop_requested and stop_requested()):
                try:
                    connection,_=server.accept()
                except socket.timeout:
                    continue
                with connection:
                    try:
                        _peer(connection,peer_uid)
                        connection.settimeout(2)
                        request=json.loads(_receive(connection,MAX_REQUEST))
                        if not isinstance(request,dict) or set(request)!={"storage_key","sha256","size_bytes","start_sec","end_sec","language"}:
                            raise ValueError("invalid ASR request")
                        key=request["storage_key"]
                        match=re.fullmatch(r"objects/([0-9a-f]{2})/([0-9a-f]{64})",key) if isinstance(key,str) else None
                        if (match is None or match[1]!=match[2][:2]
                                or request["sha256"]!=match[2] or type(request["size_bytes"]) is not int
                                or not 0<request["size_bytes"]<=2*1024**3):
                            raise ValueError("invalid ASR object identity")
                        source=(artifact_root/key).resolve(strict=True)
                        source.relative_to(artifact_root)
                        budget=media._budget(120,None)
                        if (not source.is_file() or source.name!=request["sha256"]
                                or source.stat().st_size!=request["size_bytes"]
                                or media._hash(source,budget)!=request["sha256"]):
                            raise ValueError("invalid ASR source integrity")
                        connection.setblocking(False)
                        def disconnected():
                            if stop_requested and stop_requested():
                                return True
                            try:
                                return connection.recv(1,socket.MSG_PEEK)==b""
                            except BlockingIOError:
                                return False
                        result=asr.transcribe(source,start_sec=request["start_sec"],end_sec=request["end_sec"],
                                               language=request["language"],timeout_sec=120,cancelled=disconnected)
                        if media._hash(source,budget)!=request["sha256"]:
                            raise ValueError("ASR source changed")
                        response={"status":"succeeded","transcript":result.model_dump(mode="json")}
                    except Exception:
                        response={"status":"unavailable"}
                    try:
                        connection.settimeout(2)
                        connection.sendall(json.dumps(response).encode())
                    except (BrokenPipeError,ConnectionResetError):
                        pass
        finally:
            socket_path.unlink(missing_ok=True)
            os.close(lock)


def main():
    # Configuration belongs to the trusted container supervisor, never requests.
    import signal,threading
    stopped=threading.Event()
    signal.signal(signal.SIGTERM,lambda *_:stopped.set())
    signal.signal(signal.SIGINT,lambda *_:stopped.set())
    serve(socket_path=Path(os.environ.get("ASR_SOCKET","/socket/asr.sock")),stop_requested=stopped.is_set,
          artifact_root=Path("/artifacts"),peer_uid=10001,
          asr=LocalASR(Path(sys.executable),Path("/model"),Path("/opt/asr-small-manifest.json"),Path("/opt/asr")))


if __name__=="__main__":
    main()
