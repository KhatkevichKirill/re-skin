"""The one descriptor every final publisher returns (AICP-299 §3.2, §4).

Imported by pipeline modules: no database, no I/O, no knowledge imports. A
publisher builds it from objects it has already committed; its caller hands
it to ``capture.link_publication`` as the last statement of the job/request.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

PublicationKind = Literal["stitch", "range", "audio", "assistant"]

# kind -> the Run case version reason the capture queue records
REASON_FOR_KIND = {"stitch": "restitch", "range": "range_attempt", "audio": "audio_patch",
                   "assistant": "assistant_promotion"}


@dataclass(frozen=True)
class PublishedFinal:
    run_id: str
    project_id: str
    path: str
    kind: PublicationKind
    publication_id: str  # "stitch:<run>:<inv>" | "range_attempt:<id>" | "audio_patch:<id>" | "assistant_promotion:<id>"


__all__ = ["PublishedFinal", "PublicationKind", "REASON_FOR_KIND"]
