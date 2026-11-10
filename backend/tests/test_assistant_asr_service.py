"""Private Unix protocol tests; fake ASR never downloads or runs a model."""
import hashlib
import json
import os
from pathlib import Path
import socket
import threading
import time

import pytest

from app.assistant_asr import ASRTranscript
from app.assistant_asr_service import ASRSidecarClient,serve
from app import assistant_media as media


@pytest.fixture
def service(tmp_path):
    with socket.socket(socket.AF_UNIX,socket.SOCK_STREAM) as probe:
        try:
            probe.bind(str(tmp_path/"probe.sock"))
        except PermissionError:
            pytest.skip("execution sandbox denies local Unix sockets")
    (tmp_path/"probe.sock").unlink()
    root=tmp_path/"artifacts"
    content=b"trusted disposable ASR source"
    sha=hashlib.sha256(content).hexdigest()
    path=root/"objects"/sha[:2]/sha
    path.parent.mkdir(parents=True)
    path.write_bytes(content)
    stopped=threading.Event()
    class Fake:
        calls=[]
        def transcribe(self,source,**kwargs):
            self.calls.append((source,kwargs))
            if kwargs.get("language")=="zz":
                while not kwargs["cancelled"]():
                    time.sleep(.01)
                raise media.AssistantMediaCancelled("disconnected")
            return ASRTranscript(language="en",language_probability=.9,words=[],network_denied=True,max_rss_kib=100,
                source_start_sec=kwargs["start_sec"],source_end_sec=kwargs["end_sec"],source_duration_sec=3)
    fake=Fake()
    sock=tmp_path/"socket"/"asr.sock"
    thread=threading.Thread(target=serve,kwargs=dict(socket_path=sock,artifact_root=root,asr=fake,
                peer_uid=os.getuid(),stop_requested=stopped.is_set))
    thread.start()
    for _ in range(100):
        if sock.exists():break
        time.sleep(.01)
    yield ASRSidecarClient(sock,root,peer_uid=os.getuid()),path,fake
    stopped.set()
    thread.join(3)
    assert not thread.is_alive()


def test_private_client_resolves_pinned_object_and_preserves_explicit_coverage(service):
    client,path,fake=service
    result=client.transcribe(path,start_sec=.2,end_sec=2.7)
    assert result.source_start_sec == .2 and result.source_end_sec == 2.7
    assert fake.calls[0][0] == path


def test_sidecar_rejects_arbitrary_paths_extra_keys_and_changed_source(service):
    client,path,fake=service
    with socket.socket(socket.AF_UNIX,socket.SOCK_STREAM) as sock:
        sock.connect(str(client.socket_path))
        sock.sendall(json.dumps({"storage_key":"../../private","shell":"true"}).encode()+b"\n")
        assert json.loads(sock.recv(1000))["status"] == "unavailable"
    assert not fake.calls
    path.write_bytes(b"tampered")
    with pytest.raises(media.AssistantMediaError):
        client.transcribe(path,start_sec=0,end_sec=1)


def test_peer_uid_is_checked_by_client(service):
    client,path,_=service
    wrong=ASRSidecarClient(client.socket_path,client.artifact_root,peer_uid=os.getuid()+1)
    with pytest.raises(ValueError,match="peer denied"):
        wrong.transcribe(path,start_sec=0,end_sec=1)


def test_client_cancellation_interrupts_active_transcription_and_service_recovers(service):
    client,path,fake=service
    started=time.monotonic()
    with pytest.raises(media.AssistantMediaCancelled):
        client.transcribe(path,start_sec=0,end_sec=1,language="zz",cancelled=lambda:time.monotonic()-started>.2)
    result=client.transcribe(path,start_sec=0,end_sec=1)
    assert result.language == "en"
