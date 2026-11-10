"""Cancellable local soundtrack edits and authorized existing-asset discovery.

Donor audio is the complete mix. Replacing speech also replaces any music/noise
inside that window. Correct words in a transcript do not prove pronunciation.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import json
import math
import os
import struct
from pathlib import Path
import tempfile
import time
import unicodedata
import numpy as np

from . import assistant_media as media


class AudioFitRejected(media.AssistantMediaError):
    """Safe timing fit needs a different endpoint or operator listening review."""


@dataclass(frozen=True)
class AudioMetadata:
    duration_sec: float
    sample_rate: int
    channels: int
    channel_layout: str


@dataclass(frozen=True)
class AudioFitMetadata:
    donor_duration_sec: float
    target_duration_sec: float
    speed_ratio: float
    pad_samples: int
    trim_samples: int
    sample_rate: int
    review_required: bool = True


@dataclass(frozen=True)
class AudioEditResult(media.MediaEditResult):
    audio_fit: AudioFitMetadata | None = None


def _fit_donor(source, path, *, donor_start, donor_duration, target_samples, layout, rate,channels,budget):
    """Bounded float PCM tempo fit; never silently trim an audible word tail."""
    target_duration=target_samples/rate
    ratio=donor_duration/target_duration
    if not 0.8-1e-9<=ratio<=1.25+1e-9 or target_duration>60:
        raise media.AssistantMediaError("audio duration fit requires a <=60s target and ratio within [0.8,1.25]")
    fmt=f"aformat=sample_fmts=fltp:sample_rates={rate}:channel_layouts={layout}"
    filters=f"atrim=start={donor_start:.12f}:duration={donor_duration:.12f},asetpts=PTS-STARTPTS,{fmt}"
    if abs(ratio-1)>1e-12:
        filters+=f",atempo={ratio:.12f}"
    cap=128*1024**2
    def decode(dst,af):
        media._run([media.FFMPEG,"-v","error","-nostdin","-n","-filter_threads","1","-i",str(source),
            "-vn","-af",af,"-c:a","pcm_f32le","-ar",str(rate),"-ac",str(channels),
            "-fs",str(cap),"-f","f32le",str(dst)],budget,inputs=(source,),outputs=(dst,))
        if dst.stat().st_size>=cap or dst.stat().st_size%(4*channels):
            raise media.AssistantMediaError("duration-fit PCM exceeded its size or sample bounds")
    decode(path,filters)
    actual=path.stat().st_size//(4*channels)
    residual=actual-target_samples
    if abs(residual)>round(.05*rate) or not actual:
        raise AudioFitRejected("duration fit needs more than50ms residual adjustment; choose another source window")
    silence_peak=10**(-50/20)
    if residual>0:
        with path.open("rb") as handle:
            handle.seek(target_samples*channels*4)
            tail=np.frombuffer(handle.read(),dtype="<f4")
        if not np.isfinite(tail).all() or float(np.max(np.abs(tail),initial=0))>silence_peak:
            raise AudioFitRejected("duration fit would trim audible donor audio; choose a wider or quieter ending")
    elif residual<0:
        # Padding an active word would extend its apparent ending unnaturally.
        # Inspect the original requested donor end, rather than assuming a
        # tempo algorithm's shorter output means the source ended in silence.
        endpoint=path.with_name("donor-end.f32")
        tail_duration=min(.02,donor_duration)
        decode(endpoint,f"atrim=start={donor_start+donor_duration-tail_duration:.12f}:duration={tail_duration:.12f},asetpts=PTS-STARTPTS,{fmt}")
        tail=np.fromfile(endpoint,dtype="<f4")
        if not len(tail) or not np.isfinite(tail).all() or float(np.max(np.abs(tail)))>silence_peak:
            raise AudioFitRejected("duration fit needs padding at an audible donor ending; choose a window ending in silence")
    budget.check()
    return AudioFitMetadata(donor_duration,target_duration,ratio,max(0,-residual),max(0,residual),rate)


@dataclass(frozen=True)
class AudioDonorCandidate:
    asset_id: str
    version_id: str
    cue_id: str
    start_sec: float
    end_sec: float
    text: str
    evidence: str = "editable_cue_text_match"
    timing_precision: str = "cue_only"
    pronunciation_verified: bool = False
    review_required: bool = True
    boundary_rms_dbfs: float | None = None
    level_rms_dbfs: float | None = None
    clipped_fraction: float | None = None
    speaker_verified: bool = False
    minimum_word_probability: float | None = None


@dataclass(frozen=True)
class AudioSearchReport:
    candidates: tuple[AudioDonorCandidate,...]
    limitations: tuple[str,...]
    next_after_asset_id: str | None


def discover_audio_donors(session, *, catalog,storage,target_version_id,phrase,asr=None,
                           limit=10,scan_limit=30,max_asr_assets=3,timeout_sec=120,cancelled=None):
    """Search real cue/ASR word windows and rank measured editability hints.

    Explicit coverage limitations accompany every bounded search. Existing cue
    text and ASR are evidence for finding a listening window, never a verdict on
    pronunciation or speaker identity. Longer sources need an explicit chunk
    through transcribe_authorized_artifact rather than silently ignoring tails.
    """
    from .assistant_store import get_version_artifact,verify_artifact
    from .assistant_asr import transcribe_authorized_artifact,phrase_windows
    from .assistant_subtitles import get_subtitle_context
    if type(max_asr_assets) is not int or not 0<=max_asr_assets<=3:
        raise ValueError("ASR donor limit must be within [0,3]")
    budget=media._budget(timeout_sec,cancelled)
    candidates=list(find_audio_donor_candidates(session,catalog=catalog,storage=storage,
        target_version_id=target_version_id,phrase=phrase,limit=limit,scan_limit=scan_limit,
        timeout_sec=timeout_sec,cancelled=cancelled))
    limitations=["Pronunciation, speaker identity, lip sync and mixed background require listening review."]
    page=catalog.list_editable_artifacts(target_version_id=target_version_id,limit=scan_limit)
    used=0
    def check():
        budget.check()
        return False
    for descriptor in page.artifacts:
        budget.check()
        if descriptor.media_kind!="video" or descriptor.has_audio is False:
            continue
        full=get_version_artifact(session,descriptor.version_id)
        if full.id!=descriptor.asset_id or get_subtitle_context(session,descriptor.version_id,storage=storage) is not None:
            continue
        if asr is None:
            limitations.append("Offline ASR is not configured; assets without editable cues were not localized.")
            break
        if used>=max_asr_assets:
            limitations.append("Reached bounded ASR asset scan limit; remaining assets were not transcribed.")
            break
        path=verify_artifact(full,storage=storage)
        info=_probe_audio(path,budget)
        if info.duration_sec>60:
            limitations.append(f"Asset {descriptor.asset_id} exceeds60s; choose an explicit chunk for word localization.")
            continue
        used+=1
        transcript=transcribe_authorized_artifact(session,catalog=catalog,storage=storage,
            target_version_id=target_version_id,asset_id=descriptor.asset_id,version_id=descriptor.version_id,
            asr=asr,start_sec=0,end_sec=info.duration_sec,timeout_sec=max(.1,budget.deadline-time.monotonic()),cancelled=check)
        for index,match in enumerate(phrase_windows(transcript,phrase)):
            metrics={}
            if 0<match["end_sec"]-match["start_sec"]<=10:
                metrics=audio_window_metrics(path,start_sec=match["start_sec"],end_sec=match["end_sec"],_shared_budget=budget)
            candidates.append(AudioDonorCandidate(descriptor.asset_id,descriptor.version_id,f"asr{index+1}",
                match["start_sec"],match["end_sec"],match["text"],evidence="offline_asr_word_match",
                timing_precision="estimated_word",minimum_word_probability=match["minimum_word_probability"],**metrics))
    candidates.sort(key=lambda c:(c.timing_precision!="estimated_word",
        c.clipped_fraction if c.clipped_fraction is not None else 1,
        c.boundary_rms_dbfs if c.boundary_rms_dbfs is not None else 0,
        -(c.minimum_word_probability or 0),c.version_id,c.start_sec))
    if page.next_after_asset_id:
        limitations.append("Catalog scan is partial; additional authorized assets remain after the returned cursor.")
    return AudioSearchReport(tuple(candidates[:limit]),tuple(dict.fromkeys(limitations)),page.next_after_asset_id)


def find_audio_donor_candidates(session, *, catalog, storage, target_version_id,
                                phrase: str, limit: int = 10, scan_limit: int = 100,
                                timeout_sec: float = 60, cancelled=None):
    """Search authorized existing cue tracks; never claim acoustic correctness.

    This supports candidate discovery when a transcript is already available.
    Cue windows are listening/navigation hints, not word alignment. No ASR,
    synthesis, download or paid provider fallback is invoked. Missing tracks
    produce no candidates rather than invented timings.
    """
    from .assistant_store import get_version_artifact, verify_artifact
    from .assistant_subtitles import get_subtitle_context
    if not isinstance(phrase,str) or not phrase.strip() or len(phrase)>200:
        raise ValueError("donor phrase must be nonempty and at most 200 characters")
    if type(limit) is not int or not 1<=limit<=25 or type(scan_limit) is not int or not 1<=scan_limit<=100:
        raise ValueError("invalid candidate search bounds")
    normalize = lambda text: " ".join(unicodedata.normalize("NFKC",text).casefold().split())
    needle, candidates = normalize(phrase), []
    budget = media._budget(timeout_sec,cancelled)
    page = catalog.list_editable_artifacts(target_version_id=target_version_id,limit=scan_limit)
    for descriptor in page.artifacts:
        budget.check()
        if descriptor.media_kind != "video" or descriptor.has_audio is False:
            continue
        full = get_version_artifact(session,descriptor.version_id)
        if full.id != descriptor.asset_id:
            continue
        if descriptor.has_audio is None:
            try:
                _probe_audio(verify_artifact(full,storage=storage),budget)
            except media.AssistantMediaError:
                continue
        track = get_subtitle_context(session,descriptor.version_id,storage=storage)
        if track is None:
            continue
        for cue in track.cues:
            if needle in normalize(cue.text):
                # Bound decoded output and work; cue-only hints longer than10s
                # remain text candidates without invented acoustic metrics.
                metrics = {}
                if cue.end_sec-cue.start_sec <= 10:
                    metrics = audio_window_metrics(verify_artifact(full,storage=storage),
                        start_sec=cue.start_sec,end_sec=cue.end_sec,_shared_budget=budget)
                candidates.append(AudioDonorCandidate(descriptor.asset_id,descriptor.version_id,
                    cue.cue_id,cue.start_sec,cue.end_sec,cue.text,**metrics))
                if len(candidates)>=min(50,limit*3):
                    break
        if len(candidates)>=min(50,limit*3):
            break
    # Exact cue matches first, then unclipped audio and quieter seam regions.
    # These are editability hints only; none measures pronunciation/speaker.
    candidates.sort(key=lambda c:(normalize(c.text)!=needle,
        c.clipped_fraction if c.clipped_fraction is not None else 1,
        c.boundary_rms_dbfs if c.boundary_rms_dbfs is not None else 0,
        c.version_id,c.start_sec))
    return tuple(candidates[:limit])


def audio_window_metrics(path, *, start_sec, end_sec, timeout_sec=30, cancelled=None,
                          _shared_budget=None):
    """Bounded RMS/clip/seam metrics; no speech or pronunciation judgement."""
    budget = _shared_budget or media._budget(timeout_sec,cancelled)
    source = media._input(path)
    start,end = media._number(start_sec,"start_sec"),media._number(end_sec,"end_sec")
    if not 0<=start<end or end-start>10:
        raise media.AssistantMediaError("audio metrics require a window within (0,10] seconds")
    with tempfile.TemporaryDirectory(prefix="assistant-metrics-") as temp:
        pcm = Path(temp)/"samples.f32"
        media._run([media.FFMPEG,"-v","error","-nostdin","-n","-ss",f"{start:.9f}","-i",str(source),
            "-t",f"{end-start:.9f}","-vn","-ac","1","-ar","16000","-f","f32le",str(pcm)],budget,inputs=(source,),outputs=(pcm,))
        if pcm.stat().st_size>650000:
            raise media.AssistantMediaError("audio metrics exceeded decoded size limit")
        samples = np.fromfile(pcm,dtype="<f4").astype(np.float64)
        if len(samples)<1 or not np.isfinite(samples).all():
            raise media.AssistantMediaError("audio window has no finite samples")
        db = lambda x: round(20*math.log10(max(1e-8,float(np.sqrt(np.mean(x*x))))),3)
        edge = min(480,len(samples))
        return {"boundary_rms_dbfs":max(db(samples[:edge]),db(samples[-edge:])),
                "level_rms_dbfs":db(samples),
                "clipped_fraction":round(float(np.mean(np.abs(samples)>=.999)),6)}


def _probe_audio(path: Path, budget: media._Budget) -> AudioMetadata:
    try:
        data = json.loads(media._run([
            media.FFPROBE, "-v", "error", "-show_streams", "-of", "json", str(path),
        ], budget,inputs=(path,)))
        tracks = [s for s in data["streams"] if s["codec_type"] == "audio"]
        if len(tracks) != 1:
            raise ValueError("requires exactly one audio track")
        audio = tracks[0]
        duration = float(audio["duration"])
        rate, channels = int(audio["sample_rate"]), int(audio["channels"])
        if (not math.isfinite(duration) or duration <= 0 or not 8000 <= rate <= 192000
                or channels not in (1, 2) or abs(float(audio.get("start_time", 0))) > 0.00001
                or audio["codec_name"] not in ("aac", "pcm_s16le", "pcm_s24le", "flac")):
            raise ValueError("requires zero-start mono/stereo AAC, PCM WAV or FLAC")
        return AudioMetadata(duration, rate, channels, "mono" if channels == 1 else "stereo")
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise media.AssistantMediaError(f"unsupported audio: {exc}") from exc


def probe_trusted_audio(path, *, timeout_sec=30, cancelled=None) -> AudioMetadata:
    return _probe_audio(media._input(path), media._budget(timeout_sec, cancelled))


def splice_audio(
    src, donor, dst, *, start_sec: float, end_sec: float,
    donor_start_sec: float = 0, crossfade_sec: float = 0,
    donor_duration_sec: float | None = None,
    gain_db: float = 0, timeout_sec: float = 120, cancelled=None,
    _pad_base_to_window_end: bool = False,
) -> media.MediaEditResult:
    """Replace a donor window without moving the base picture or timeline.

    Boundaries must align to base audio samples. Complementary per-sample
    envelopes preserve timing; gain changes apply only to the donor. An explicit
    zero crossfade is supported. Video packets are copied without re-encoding.
    Optional donor_duration_sec permits gentle tempo fitting with explicitly
    measured near-silent residual trim/pad. Audible residuals are rejected.
    """
    budget = media._budget(timeout_sec, cancelled)
    source, replacement, destination = media._input(src), media._input(donor), media._destination(dst)
    info = media._probe(source, budget)
    if not info.has_audio:
        raise media.AssistantMediaError("audio replacement requires a base soundtrack")
    base_audio, donor_audio = _probe_audio(source, budget), _probe_audio(replacement, budget)
    start, end = media._span(start_sec, end_sec, info, frames=False)
    donor_start = media._number(donor_start_sec, "donor_start_sec")
    fade, gain = media._number(crossfade_sec, "crossfade_sec"), media._number(gain_db, "gain_db")
    if not 0 <= fade <= min(2, (end-start)/2) or not -24 <= gain <= 12:
        raise media.AssistantMediaError("crossfade must fit the window and gain must be within [-24,12] dB")
    donor_duration=end-start if donor_duration_sec is None else media._number(donor_duration_sec,"donor_duration_sec")
    if donor_duration<=0 or donor_start < 0 or donor_start + donor_duration > donor_audio.duration_sec + 0.000001:
        raise media.AssistantMediaError("donor does not cover the required audio window")
    if type(_pad_base_to_window_end) is not bool or (_pad_base_to_window_end and (start != 0 or end != info.video_duration_sec)):
        raise media.AssistantMediaError("full-track padding requires the complete picture window")
    rate = base_audio.sample_rate
    if any(abs(t*rate-round(t*rate)) > 0.0001 for t in (start, end, fade)):
        raise media.AssistantMediaError("audio edit boundaries must align to base audio samples")
    a, b, f = (round(t*rate) for t in (start, end, fade))
    # aeval uses decoded sample number n. No millisecond delay or frame-sized
    # gain blocks: even sub-millisecond windows have the requested placement.
    if f:
        envelope = f"if(lt(n,{a}),0,if(lt(n,{a+f}),(n-{a})/{f},if(lt(n,{b-f}),1,if(lt(n,{b}),({b}-n)/{f},0))))"
    else:
        envelope = f"if(between(n,{a},{b-1}),1,0)"
    fmt = f"aformat=sample_fmts=fltp:sample_rates={rate}:channel_layouts={base_audio.channel_layout}"
    provenance = tuple(media.SourceProvenance(str(p), media._hash(p, budget)) for p in (source, replacement))
    with tempfile.TemporaryDirectory(prefix=".assistant-audio-", dir=destination.parent) as temp:
        fit=None
        donor_path = replacement
        donor_input=["-i",str(replacement)]
        donor_trim=f"atrim=start={donor_start:.12f}:duration={end-start:.12f},asetpts=PTS-STARTPTS,{fmt}"
        if donor_duration_sec is not None:
            pcm=Path(temp)/"donor-fit.f32"
            fit=_fit_donor(replacement,pcm,donor_start=donor_start,donor_duration=donor_duration,
                target_samples=b-a,layout=base_audio.channel_layout,rate=rate,channels=base_audio.channels,budget=budget)
            # Keep the global safe demuxer policy unchanged: raw f32le is not
            # an accepted external format. Wrap our own bounded float samples
            # in a fixed IEEE-float WAV header without requantizing them.
            wav=Path(temp)/"donor-fit.wav"
            size=pcm.stat().st_size
            with wav.open("wb") as target,pcm.open("rb") as original:
                target.write(struct.pack("<4sI4s4sIHHIIHH4sI",b"RIFF",36+size,b"WAVE",b"fmt ",16,3,
                    base_audio.channels,rate,rate*base_audio.channels*4,base_audio.channels*4,32,b"data",size))
                while chunk:=original.read(1024*1024):
                    budget.check()
                    target.write(chunk)
            donor_input=["-i",str(wav)]
            donor_path = wav
            donor_trim=f"apad=whole_len={b-a},atrim=end_sample={b-a},asetpts=PTS-STARTPTS,{fmt}"
        base_length = f",apad=whole_len={b},atrim=end_sample={b}" if _pad_base_to_window_end else ""
        filters = (
            f"[0:a:0]{fmt}{base_length},aeval=exprs='val(ch)*(1-({envelope}))':c=same[base];"
            f"[1:a:0]{donor_trim},"
            f"adelay={a}S:all=1,apad,aeval=exprs='val(ch)*({envelope})*{10**(gain/20):.12f}':c=same[donor];"
            "[base][donor]amix=inputs=2:duration=first:normalize=0[a]"
        )
        stage = Path(temp) / "candidate.mp4"
        media._run([
            media.FFMPEG, "-hide_banner", "-loglevel", "error", "-nostdin", "-n",
            "-filter_complex_threads", "1", "-i", str(source), *donor_input,
            "-filter_complex", filters, "-map", "0:v:0", "-map", "[a]",
            "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-map_metadata", "-1",
            "-movflags", "+faststart", str(stage),
        ], budget,inputs=(source, donor_path),outputs=(stage,))
        output = media._probe(stage, budget)
        if output != info:
            # AAC packet padding can change container duration slightly.
            if (output.width, output.height, output.fps, output.sample_aspect_ratio,
                    output.video_duration_sec, output.has_audio) != (
                    info.width, info.height, info.fps, info.sample_aspect_ratio,
                    info.video_duration_sec, True):
                raise media.AssistantMediaError("audio render changed picture metadata")
        for item in provenance:
            if media._hash(Path(item.path), budget) != item.sha256:
                raise media.AssistantMediaError("source changed while rendering")
        digest = media._hash(stage, budget)
        budget.check()
        try:
            os.link(stage, destination)
        except FileExistsError as exc:
            raise media.AssistantMediaError("output already exists") from exc
    return AudioEditResult(
        "audio_splice", provenance, str(destination), digest, output, info.video_duration_sec,
        (media.TimelineSpan(0, info.video_duration_sec, 0),), "replace_mixed_audio",
        "Cue timing unchanged; review caption wording, pronunciation and lip sync against the replacement mix.",
        fit,
    )


def replace_full_track(src, donor, dst, *, donor_start_sec=0, gain_db=0, timeout_sec=120, cancelled=None):
    info = media.probe_trusted_media(src, timeout_sec=timeout_sec, cancelled=cancelled)
    result=splice_audio(src, donor, dst, start_sec=0, end_sec=info.video_duration_sec,
                        donor_start_sec=donor_start_sec, gain_db=gain_db,
                        timeout_sec=timeout_sec, cancelled=cancelled, _pad_base_to_window_end=True)
    return replace(result,operation="full_track_audio")
