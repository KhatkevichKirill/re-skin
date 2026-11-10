"""Provider content-filter refusals of a swap segment.

A video provider sometimes refuses a segment because of what is IN its frames
or soundtrack, not because of a transient fault: Gemini Omni's "Request
blocked: The input content was flagged by safety filters ..." (a phone screen
showing the word "suicide", a prominent public figure), or Seedance's output
moderation ("the output video may contain sensitive information", a celebrity
photo on a phone; "copyright restrictions"; ``InputAudioSensitiveContentDetected``).
Retrying the same frames repeats the refusal (memory: reskin-celebrity-moderation),
so these are told apart from transient failures here, once, by the provider's
own message as ``pipeline_v2`` records it in ``RunSegment.error_message``
("failed after N attempt(s): <failMsg>").

Every text below was seen on production run segments (2026-11-02 survey). The
match is deliberately narrow (design review R1-10): an unknown refusal text
keeps the ordinary retry route, and the raw message is always shown next to the
category. kie.ai gives no structured code beyond its failMsg / failCode text.
"""

from __future__ import annotations

from typing import Optional

# (category, lower-case substring of the provider message)
_PATTERNS: tuple[tuple[str, str], ...] = (
    ("safety_filter", "flagged by safety filters"),
    ("sensitive_output", "may contain sensitive information"),
    ("sensitive_audio", "sensitivecontentdetected"),
    ("copyright", "copyright restrictions"),
)

LABELS = {
    "safety_filter": "the provider's safety filter refused the frames",
    "sensitive_output": "the provider's output moderation refused the segment (sensitive content)",
    "sensitive_audio": "the provider refused the segment's audio (sensitive content)",
    "copyright": "the provider refused the segment (copyright restrictions)",
}


def content_block(message: Optional[str]) -> Optional[str]:
    """The refusal category of a provider *message*, or None (not a content
    block: a transient error, a timeout, a short result ...)."""
    if not message:
        return None
    low = str(message).lower()
    for category, needle in _PATTERNS:
        if needle in low:
            return category
    return None


def _status(v) -> str:
    return v.value if hasattr(v, "value") else str(v)


def segment_block(rs) -> Optional[str]:
    """The content-block category of a FAILED run segment (never a source
    fallback, never a segment with a result), else None."""
    if rs is None or _status(rs.status) != "failed" or getattr(rs, "source_fallback", False):
        return None
    return content_block(rs.error_message)
