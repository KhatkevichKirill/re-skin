"""Final-preservation hardlinks (AICP-299 §3.2): layout, naming, lookup.

Pure filesystem; no database. A link is a directory entry pointing at the
published inode, so it copies no data blocks:

    <DATA_DIR>/knowledge/finals/<project_id>/<run_id>/
        <time_ns>__<kind>__<quoted publication id>__<dev>-<ino>-<mtime_ns>__s<scope8>.mp4

``kind`` is a publication kind (``stitch``, ``range``, ``audio``,
``assistant``) for point (i), or ``superseded`` for point (ii).
"""
from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from typing import Iterable, Optional
from urllib.parse import quote, unquote

PUBLICATION_KINDS = ("stitch", "range", "audio", "assistant")
SUPERSEDED = "superseded"
_NAME = re.compile(r"^(?P<t>\d+)__(?P<kind>[a-z]+)__(?P<pub>[^_/]+)__(?P<dev>\d+)-(?P<ino>\d+)-(?P<mtime>\d+)"
                   r"__s(?P<scope>[0-9a-f]{0,16})\.mp4$")


def knowledge_root() -> str:
    from ..storage import data_dir_root

    return os.path.join(data_dir_root(), "knowledge")


def finals_dir(project_id: str, run_id: str) -> str:
    return os.path.join(knowledge_root(), "finals", project_id, run_id)


@dataclass(frozen=True)
class LinkName:
    path: str
    time_ns: int
    kind: str
    publication_id: str
    st_dev: int
    st_ino: int
    st_mtime_ns: int
    scope8: str

    @property
    def identity(self) -> str:
        """The store's ``file_identity`` format of the published inode."""
        return f"{self.st_dev}:{self.st_ino}:{self.st_mtime_ns}"

    @property
    def publishable(self) -> bool:
        return self.kind in PUBLICATION_KINDS


def link_name(kind: str, publication_id: str, st: os.stat_result, scope8: str,
              time_ns: Optional[int] = None) -> str:
    pub = quote(publication_id, safe=":-.").replace("_", "%5F")  # "__" is the field separator
    return (f"{time_ns if time_ns is not None else time.time_ns()}__{kind}__{pub}__"
            f"{st.st_dev}-{st.st_ino}-{st.st_mtime_ns}__s{scope8}.mp4")


def parse_link(path: str) -> Optional[LinkName]:
    m = _NAME.match(os.path.basename(path))
    if m is None:
        return None
    return LinkName(path=path, time_ns=int(m["t"]), kind=m["kind"], publication_id=unquote(m["pub"]),
                    st_dev=int(m["dev"]), st_ino=int(m["ino"]), st_mtime_ns=int(m["mtime"]),
                    scope8=m["scope"])


def make_link(source: str, project_id: str, run_id: str, kind: str, publication_id: str, scope8: str,
              st: Optional[os.stat_result] = None) -> str:
    """Hardlink *source* into the Run's finals dir; returns the link path.

    ``st`` is the caller's ``os.stat(source)`` (taken first, so the identity is
    known whatever fails next). Raises ``OSError`` on any failure.
    """
    st = st if st is not None else os.stat(source)
    d = finals_dir(project_id, run_id)
    os.makedirs(d, exist_ok=True)
    target = os.path.join(d, link_name(kind, publication_id, st, scope8))
    os.link(source, target)
    return target


def iter_links(project_id: str, run_id: str) -> list[LinkName]:
    d = finals_dir(project_id, run_id)
    try:
        names = os.listdir(d)
    except OSError:
        return []
    out = [parse_link(os.path.join(d, n)) for n in names]
    return sorted((x for x in out if x is not None), key=lambda x: x.time_ns)


def iter_run_dirs() -> Iterable[tuple[str, str]]:
    """(project_id, run_id) for every Run that has a finals dir."""
    base = os.path.join(knowledge_root(), "finals")
    try:
        projects = os.listdir(base)
    except OSError:
        return
    for pid in sorted(projects):
        try:
            runs = os.listdir(os.path.join(base, pid))
        except OSError:
            continue
        for rid in sorted(runs):
            yield pid, rid


def live_link_lookup(project_id: Optional[str], run_id: str, live_path: Optional[str]) -> Optional[LinkName]:
    """The newest publishable link whose inode is the live final's (S14), or None."""
    if not project_id or not live_path:
        return None
    try:
        st = os.stat(live_path)
    except OSError:
        return None
    matches = [x for x in iter_links(project_id, run_id)
               if x.publishable and (x.st_dev, x.st_ino) == (st.st_dev, st.st_ino)]
    return matches[-1] if matches else None


__all__ = ["LinkName", "PUBLICATION_KINDS", "SUPERSEDED", "finals_dir", "iter_links", "iter_run_dirs",
           "knowledge_root", "link_name", "live_link_lookup", "make_link", "parse_link"]
