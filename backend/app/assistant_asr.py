"""Optional bounded offline word localization; transcript is not pronunciation QA."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import time
from typing import Annotated

from pydantic import Field

from . import assistant_media as media
from .assistant_contracts import ContractModel, Seconds


class ASRWord(ContractModel):
    text: Annotated[str,Field(min_length=1,max_length=200)]
    start_sec: Seconds
    end_sec: Seconds
    probability: Annotated[float,Field(ge=0,le=1,allow_inf_nan=False)]


class ASRTranscript(ContractModel):
    language: Annotated[str,Field(pattern=r"^[a-z]{2,3}$")]
    language_probability: Annotated[float,Field(ge=0,le=1,allow_inf_nan=False)]
    words: Annotated[tuple[ASRWord,...],Field(max_length=2000)]
    network_denied: bool
    max_rss_kib: int
    source_start_sec: Seconds = 0.0
    source_end_sec: Seconds = 0.0
    source_duration_sec: Seconds = 0.0


def transcribe_authorized_artifact(session, *, catalog, storage, target_version_id,
                                   asset_id, version_id, asr, start_sec=0, end_sec,
                                   language=None,timeout_sec=120,cancelled=None):
    """Recheck authorization and pinned bytes around the optional local ASR."""
    from .assistant_models import AssistantArtifact
    from .assistant_store import verify_artifact
    descriptor=catalog.resolve_editable_reference(target_version_id=target_version_id,
                                          asset_id=asset_id,version_id=version_id)
    if descriptor.media_kind not in ("audio","video") or descriptor.has_audio is False:
        raise media.AssistantMediaError("ASR requires a source soundtrack")
    artifact=session.get(AssistantArtifact,descriptor.asset_id,populate_existing=True)
    path=verify_artifact(artifact,storage=storage)
    transcript=asr.transcribe(path,start_sec=start_sec,end_sec=end_sec,language=language,
                              timeout_sec=timeout_sec,cancelled=cancelled)
    catalog.resolve_editable_reference(target_version_id=target_version_id,asset_id=asset_id,version_id=version_id)
    verify_artifact(artifact,storage=storage)
    return transcript


def phrase_windows(transcript: ASRTranscript, phrase: str):
    """Return approximate word-aligned candidates, never pronunciation verdicts."""
    import unicodedata,re
    def normalize(value):
        return re.sub(r"[^\w]+","",unicodedata.normalize("NFKC",value).casefold())
    needle=normalize(phrase)
    if not needle or len(phrase)>200:
        raise ValueError("expected bounded nonempty phrase")
    matches=[]
    for index,word in enumerate(transcript.words):
        combined=""
        for end in range(index,min(index+20,len(transcript.words))):
            combined+=normalize(transcript.words[end].text)
            if combined==needle:
                words=transcript.words[index:end+1]
                matches.append({"start_sec":word.start_sec,"end_sec":words[-1].end_sec,
                    "text":"".join(w.text for w in words),"minimum_word_probability":min(w.probability for w in words),
                    "timing_precision":"estimated_word","pronunciation_verified":False,"speaker_verified":False,
                    "review_required":True})
                break
            if len(combined)>len(needle):
                break
    return tuple(matches)


@dataclass(frozen=True)
class LocalASR:
    python_path: Path
    model_dir: Path
    manifest_path: Path
    package_path: Path | None = None

    def _verify(self, budget):
        manifest=json.loads(self.manifest_path.read_text())
        if manifest.get("revision")!="536b0662742c02347bc0e980a01041f333bce120":
            raise media.AssistantMediaError("ASR model revision is not approved")
        for name in ("config.json","model.bin","tokenizer.json","vocabulary.txt"):
            item=manifest["files"][name]
            path=self.model_dir/name
            if path.stat().st_size!=item["size_bytes"] or media._hash(path,budget)!=item["sha256"]:
                raise media.AssistantMediaError("ASR model manifest integrity failure")

    def transcribe(self, path, *, start_sec=0, end_sec, language=None, timeout_sec=120,cancelled=None):
        """Transcribe exactly the requested <=60s chunk, times on source timeline.

        No automatic clipping or model download. Caller authorizes/pins path.
        Word times are estimates and always need listening review before edits.
        """
        budget=media._budget(timeout_sec,cancelled)
        source=media._input(path)
        start,end=media._number(start_sec,"start_sec"),media._number(end_sec,"end_sec")
        if not 0<=start<end or end-start>60:
            raise media.AssistantMediaError("ASR requires an explicit chunk within (0,60] seconds")
        from .assistant_audio import probe_trusted_audio
        source_info=probe_trusted_audio(source,timeout_sec=min(30,timeout_sec),cancelled=cancelled)
        if end>source_info.duration_sec+0.000001:
            raise media.AssistantMediaError("ASR chunk exceeds source duration")
        if language is not None and (not isinstance(language,str) or not language.isalpha() or len(language) not in (2,3)):
            raise media.AssistantMediaError("invalid ASR language")
        self._verify(budget)
        with tempfile.TemporaryDirectory(prefix="assistant-asr-") as temp:
            audio=Path(temp)/"audio.wav"
            media._run([media.FFMPEG,"-v","error","-nostdin","-n","-ss",f"{start:.9f}","-i",str(source),
                "-t",f"{end-start:.9f}","-vn","-ac","1","-ar","16000","-c:a","pcm_s16le",str(audio)],budget,inputs=(source,),outputs=(audio,))
            if audio.stat().st_size>2_000_000:
                raise media.AssistantMediaError("ASR chunk exceeds decoded audio size limit")
            request=json.dumps({"audio_path":str(audio),"model_dir":str(self.model_dir),"language":language,
                "parent_pid":os.getpid()}).encode()
            env={"PATH":"/usr/bin:/bin","HOME":temp,"HF_HOME":temp,"HF_HUB_OFFLINE":"1","TRANSFORMERS_OFFLINE":"1",
                 "OMP_NUM_THREADS":"2","OPENBLAS_NUM_THREADS":"2","MKL_NUM_THREADS":"2","TOKENIZERS_PARALLELISM":"false"}
            with tempfile.TemporaryFile() as out,tempfile.TemporaryFile() as err:
                worker=str(Path(__file__).with_name("assistant_asr_worker.py"))
                argv=[str(self.python_path),"-I",worker]
                if self.package_path is not None:
                    argv=[str(self.python_path),"-I","-c",
                        "import sys,runpy;sys.path.insert(0,sys.argv[1]);runpy.run_path(sys.argv[2],run_name='__main__')",
                        str(self.package_path),worker]
                process=subprocess.Popen(argv,
                    stdin=subprocess.PIPE,stdout=out,stderr=err,env=env,start_new_session=True)
                try:
                    process.stdin.write(request)
                    process.stdin.close()
                    while process.poll() is None:
                        budget.check()
                        if out.tell()>1_000_000 or err.tell()>1_000_000:
                            raise media.AssistantMediaError("ASR output exceeds size limit")
                        time.sleep(.1)
                    budget.check()
                    if process.returncode:
                        err.seek(0)
                        # Return controlled failure; private local paths/stderr
                        # never enter operator chat or application responses.
                        raise media.AssistantMediaError("offline ASR failed its resource/runtime checks")
                    out.seek(0)
                    transcript=ASRTranscript.model_validate_json(out.read(1_000_001))
                finally:
                    if process.poll() is None:
                        os.killpg(process.pid,signal.SIGKILL)
                        process.wait()
            if not transcript.network_denied or any(w.end_sec<w.start_sec or w.end_sec>end-start+.1 for w in transcript.words):
                raise media.AssistantMediaError("ASR returned invalid word timing or sandbox evidence")
            return transcript.model_copy(update={"source_start_sec":start,"source_end_sec":end,
                "source_duration_sec":source_info.duration_sec,"words":tuple(w.model_copy(update={
                "start_sec":w.start_sec+start,"end_sec":w.end_sec+start}) for w in transcript.words)})
