"""Lossless editable subtitle tracks and a dedicated safe ASS renderer.

Unlike the legacy transcript cleaner this renderer never extends/drops short
consecutive cues, drops text tails, or guesses away overlapping/invalid cues.
Burning requires separately proven clean picture; no text removal is attempted.
"""
from __future__ import annotations

import math
import re
from pathlib import Path
import tempfile
from typing import Annotated, Literal

from pydantic import Field, model_validator

from . import assistant_media as media
from .assistant_contracts import ContractModel, CueId, SubtitleStyle, UpdateSubtitleCue, SubtitleEditOperation
from .captions import escape_text, _looks_cjk


def get_subtitle_context(session, version_id, *, storage, cancelled=None):
    """Return verified editable data; private paths stay in backend resolvers."""
    from .assistant_store import get_version_artifact, verify_artifact, AssistantNotFound
    from .assistant_models import AssistantArtifact
    try:
        track = get_version_artifact(session, version_id, role="captions")
        clean = get_version_artifact(session, version_id, role="clean")
    except AssistantNotFound:
        return None
    if track.media_kind != "subtitle" or clean.media_kind != "video":
        raise media.AssistantMediaError("invalid editable subtitle lineage")
    path = verify_artifact(track, storage=storage, **({"cancelled":cancelled} if cancelled is not None else {}))
    if track.size_bytes > 4*1024*1024:
        raise media.AssistantMediaError("subtitle track exceeds size limit")
    parsed = SubtitleTrack.model_validate_json(path.read_bytes())
    verify_artifact(clean, storage=storage, **({"cancelled":cancelled} if cancelled is not None else {}))
    return parsed


class SubtitleCue(UpdateSubtitleCue):
    pass


class SubtitleTrack(ContractModel):
    schema_version: Literal[1] = 1
    cues: Annotated[tuple[SubtitleCue, ...], Field(max_length=5000)] = ()
    style: SubtitleStyle = SubtitleStyle()

    @model_validator(mode="after")
    def valid_track(self):
        seen, last_end = set(), 0
        for cue in self.cues:
            if cue.cue_id in seen or cue.start_sec < last_end:
                raise ValueError("subtitle cues must have unique IDs and ordered nonoverlapping times")
            # ASS resolution is centiseconds. Reject collapse instead of loss.
            if round(cue.end_sec*100) <= round(cue.start_sec*100):
                raise ValueError("subtitle cue must occupy at least one ASS centisecond")
            seen.add(cue.cue_id)
            last_end = cue.end_sec
        return self


def parse_legacy_ass(text: str, *, width: int, height: int) -> SubtitleTrack:
    """Parse only the known single-style ASS subset, never executable overrides.

    Root import adapter must separately prove these exact ASS bytes rendered the
    selected legacy output from the proposed clean source. Mutable transcript
    rows are not accepted as historical render evidence.
    """
    if not isinstance(text,str) or len(text.encode())>4*1024*1024:
        raise media.AssistantMediaError("legacy ASS exceeds bounded text size")
    from .captions import _styles_section
    expected_style_format = _styles_section("DejaVu Sans",width=width,height=height).splitlines()[1]
    expected_events = "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text"
    info, styles, events, section = {}, [], [], None
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith(";"):
            continue
        if line.startswith("["):
            if line not in ("[Script Info]","[V4+ Styles]","[Events]"):
                raise media.AssistantMediaError("unsupported ASS section")
            section = line
        elif section=="[Script Info]":
            key,sep,value = line.partition(":")
            if not sep or key in info:
                raise media.AssistantMediaError("invalid ASS script info")
            info[key]=value.strip()
        elif section=="[V4+ Styles]":
            if line.startswith("Format:"):
                if line != expected_style_format:
                    raise media.AssistantMediaError("unsupported ASS style format")
            elif line.startswith("Style:"):
                styles.append(line.partition(":")[2].strip().split(","))
            else:
                raise media.AssistantMediaError("unsupported ASS style entry")
        elif section=="[Events]":
            if line.startswith("Format:"):
                if line!=expected_events:
                    raise media.AssistantMediaError("unsupported ASS event format")
            elif line.startswith("Dialogue:"):
                events.append(line.partition(":")[2].strip().split(",",9))
            else:
                raise media.AssistantMediaError("unsupported ASS event entry")
        else:
            raise media.AssistantMediaError("invalid ASS section ordering")
    if (info.get("ScriptType")!="v4.00+" or info.get("WrapStyle")!="0"
            or info.get("ScaledBorderAndShadow")!="yes"
            or info.get("PlayResX")!=str(width) or info.get("PlayResY")!=str(height)
            or len(styles)!=1 or len(styles[0])!=23):
        raise media.AssistantMediaError("legacy ASS does not match supported frame/style contract")
    f=styles[0]
    if (f[0]!="Default" or f[3]!=f[4] or f[6].upper()!="&H00000000" or f[7] not in ("0","-1")
            or f[8:15]!=["0","0","0","100","100","0","0"] or f[19]!=f[20] or f[22]!="1"
            or f[18] not in ("2","8")):
        raise media.AssistantMediaError("legacy ASS uses unsupported style properties")
    def color(value):
        if re.fullmatch(r"&H[0-9a-fA-F]{8}",value) is None:
            raise media.AssistantMediaError("unsupported ASS color")
        return "#"+value[8:10]+value[6:8]+value[4:6],int(value[2:4],16)
    primary,alpha = color(f[3])
    outline,outline_alpha = color(f[5])
    if alpha:
        raise media.AssistantMediaError("unsupported translucent ASS text")
    style = SubtitleStyle(font_family=f[1],font_size_ratio=float(f[2])/height,color_hex=primary,
        outline_px=float(f[16]),shadow_px=float(f[17]),bold=f[7]=="-1",border_style=int(f[15]),
        outline_color_hex=outline,outline_alpha=outline_alpha,alignment="bottom" if f[18]=="2" else "top",
        margin_horizontal_ratio=float(f[19])/width,margin_ratio=float(f[21])/height)
    def seconds(value):
        match = re.fullmatch(r"(\d+):([0-5]\d):([0-5]\d)\.(\d{2})",value)
        if match is None:
            raise media.AssistantMediaError("invalid ASS timestamp")
        h,m,s,c=map(int,match.groups())
        return h*3600+m*60+s+c/100
    cues=[]
    for i,event in enumerate(events):
        if (len(event)!=10 or event[0]!="0" or event[3]!="Default"
                or event[4:9]!=["","0","0","0",""] or "{" in event[9] or "}" in event[9]
                or "\\" in event[9].replace("\\N","")):
            raise media.AssistantMediaError("legacy ASS event contains unsupported overrides")
        cues.append(SubtitleCue(cue_id=f"legacy{i+1:06}",start_sec=seconds(event[1]),end_sec=seconds(event[2]),text=event[9].replace("\\N","\n")))
    return SubtitleTrack(cues=tuple(cues),style=style)


def edit_track(track: SubtitleTrack, operation: SubtitleEditOperation, *, duration_sec: float) -> SubtitleTrack:
    changes = {change.cue_id: change for change in operation.changes}
    if not set(changes) <= {cue.cue_id for cue in track.cues}:
        raise media.AssistantMediaError("subtitle cue is absent from editable track")
    result = []
    for cue in track.cues:
        change = changes.get(cue.cue_id)
        if change is None:
            result.append(cue)
        elif isinstance(change, UpdateSubtitleCue):
            result.append(SubtitleCue.model_validate(change.model_dump()))
    result.sort(key=lambda cue: (cue.start_sec, cue.cue_id))
    style=track.style
    if operation.style is not None:
        style=SubtitleStyle.model_validate({**track.style.model_dump(),**operation.style.model_dump(exclude_none=True)})
    output = SubtitleTrack(cues=tuple(result), style=style)
    validate_duration(output, duration_sec)
    return output


def validate_duration(track: SubtitleTrack, duration_sec: float):
    if not math.isfinite(duration_sec) or duration_sec <= 0 or any(c.end_sec > duration_sec+0.000001 for c in track.cues):
        raise media.AssistantMediaError("subtitle cue exceeds picture duration")


def remap_track(track: SubtitleTrack, spans: tuple[media.TimelineSpan, ...]) -> SubtitleTrack:
    """Clip/split cues against retained source spans and shift every kept part."""
    output = []
    for cue in track.cues:
        parts = []
        for span in spans:
            start, end = max(cue.start_sec, span.source_start_sec), min(cue.end_sec, span.source_end_sec)
            if end-start > 1e-9:
                a = span.output_start_sec+start-span.source_start_sec
                b = span.output_start_sec+end-span.source_start_sec
                if round(b*100) <= round(a*100):
                    raise media.AssistantMediaError("edit leaves subtitle fragment shorter than ASS resolution; review boundary")
                parts.append((a,b))
        for index, (start,end) in enumerate(parts):
            cue_id = cue.cue_id if len(parts)==1 else f"{cue.cue_id[:54]}_part{index+1}"
            output.append(SubtitleCue(cue_id=cue_id, start_sec=start, end_sec=end, text=cue.text))
    output.sort(key=lambda cue: (cue.start_sec, cue.cue_id))
    return SubtitleTrack(cues=tuple(output), style=track.style)


def _timestamp(value: float) -> str:
    ticks = round(value*100)
    return f"{ticks//360000}:{ticks//6000%60:02}:{ticks//100%60:02}.{ticks%100:02}"


def build_ass(track: SubtitleTrack, *, width: int, height: int, duration_sec: float) -> str:
    track = SubtitleTrack.model_validate(track.model_dump())
    validate_duration(track, duration_sec)
    style = track.style
    color = style.color_hex[1:]
    color = "&H00"+color[4:6]+color[2:4]+color[:2]
    align = 2 if style.alignment == "bottom" else 8
    margin = round(height*style.margin_ratio)
    outline = style.outline_color_hex[1:]
    outline = f"&H{style.outline_alpha:02X}"+outline[4:6]+outline[2:4]+outline[:2]
    # libass wraps text using frame geometry. Each cue is one event containing
    # its complete escaped text, with no approximate duration-based text split.
    header = (
        f"[Script Info]\nScriptType: v4.00+\nPlayResX: {width}\nPlayResY: {height}\nWrapStyle: 0\nScaledBorderAndShadow: yes\n"
        "[V4+ Styles]\nFormat: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
        f"Style: Default,{style.font_family},{height*style.font_size_ratio:.3f},{color},{color},{outline},&H00000000,{-1 if style.bold else 0},0,0,0,100,100,0,0,{style.border_style},{style.outline_px},{style.shadow_px},{align},{round(width*style.margin_horizontal_ratio)},{round(width*style.margin_horizontal_ratio)},{margin},1\n"
        "[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )
    return header+"".join(f"Dialogue: 0,{_timestamp(c.start_sec)},{_timestamp(c.end_sec)},Default,,0,0,0,,{escape_text(c.text)}\n" for c in track.cues)


def burn_track(src, dst, *, track: SubtitleTrack, clean_source_verified: bool = False, timeout_sec=120, cancelled=None,
               exact_audio_timing: bool = False):
    if type(exact_audio_timing) is not bool:
        raise media.AssistantMediaError('exact audio timing must be a boolean')
    if clean_source_verified is not True:
        raise media.AssistantMediaError("subtitle rendering requires proven clean source lineage")
    budget = media._budget(timeout_sec, cancelled)
    if track.cues:
        if any(_looks_cjk(c.text) for c in track.cues) and track.style.font_family != "Noto Sans CJK JP":
            raise media.AssistantMediaError("CJK subtitles require the Noto Sans CJK JP font style")
        family = media._run(["fc-match", "-f", "%{family}", track.style.font_family], budget)
        if track.style.font_family.casefold() not in family.casefold():
            raise media.AssistantMediaError("requested subtitle font is unavailable; refusing silent font substitution")
    source, destination = media._input(src), media._destination(dst)
    info = media._probe(source, budget)
    ass = build_ass(track, width=info.width, height=info.height, duration_sec=info.video_duration_sec)
    with tempfile.TemporaryDirectory(prefix="assistant-captions-", dir=destination.parent) as temporary:
        path = Path(temporary)/"track.ass"
        path.write_text(ass, encoding="utf-8")
        # Path is generated by trusted code, escaped for FFmpeg filter syntax.
        escaped = str(path).replace("\\", "\\\\").replace(":", "\\:").replace("'", "'\\''")
        return media._render(
            "subtitle_edit", (source,), destination, info, f"[0:v:0]subtitles=filename='{escaped}'[v]",
            info.video_duration_sec, (media.TimelineSpan(0, info.video_duration_sec, 0),),
            "stream_copy_base", "Editable captions rendered from verified clean picture; every cue retained.", budget,
            auxiliary_inputs=(path,),
            **({'exact_audio_timing':True} if exact_audio_timing else {}),
        )
