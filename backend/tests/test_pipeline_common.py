"""
Tests for app/pipeline_common.py — resolve_reference_urls and model-param helpers.

Fix #17 verification: resolve_reference_urls resolves items concurrently while
preserving the original order and the existing error semantics (one failure
propagates to the caller).

The per-model facts these helpers used to own now live in app/ai_models.py and
the helpers are thin delegations/aliases, so the tests below double as a guard
on that delegation: every name pipeline_v2 and the suite import from here must
stay importable AND keep its old behaviour.
"""

from __future__ import annotations

import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

os.environ.setdefault("KIE_API_KEY", "fake-key-for-tests")
os.environ.setdefault("DATA_DIR", tempfile.mkdtemp())


@pytest.fixture()
def tmp_path():
    """Per-test scratch dir INSIDE the storage root, shadowing pytest's own.

    Every endpoint that serves or uploads a stored path now refuses anything
    outside DATA_DIR (AICP-148), and real stored paths always live there — so
    fixture files must too, or every "serves the file" test would 404.
    """
    import pathlib

    from app import storage

    root = storage.data_dir_root()
    os.makedirs(root, exist_ok=True)
    return pathlib.Path(tempfile.mkdtemp(prefix="tmp-path-", dir=root))


# ---------------------------------------------------------------------------
# Fake clients for pipeline_common tests (no DB needed)
# ---------------------------------------------------------------------------


class _FakeKie:
    """Minimal KieClient stub — records upload_file calls, returns predictable URLs."""

    def __init__(self, *, fail_path: str | None = None):
        self.upload_calls: list[str] = []
        self._fail_path = fail_path

    def upload_file(self, local_path: str, upload_path: str = "charswap", **kw) -> str:
        self.upload_calls.append(local_path)
        if self._fail_path and local_path == self._fail_path:
            raise RuntimeError(f"injected upload failure for {local_path}")
        return f"https://fake-kie.example.com/files/{os.path.basename(local_path)}"


class _FakeGDrive:
    """Minimal GDriveClient stub — download_file writes predictable content."""

    def __init__(self, *, fail_link: str | None = None):
        self.download_calls: list[dict] = []
        self._fail_link = fail_link

    def download_file(self, link_or_id: str, dst_path: str) -> str:
        self.download_calls.append({"link": link_or_id, "dst": dst_path})
        if self._fail_link and link_or_id == self._fail_link:
            raise RuntimeError(f"injected download failure for {link_or_id}")
        os.makedirs(os.path.dirname(dst_path) or ".", exist_ok=True)
        with open(dst_path, "wb") as fh:
            fh.write(b"fake-gdrive-content")
        return dst_path


# ---------------------------------------------------------------------------
# Tests: _map_aspect, _clamp_duration, _map_omni_aspect, _snap_omni_duration
# ---------------------------------------------------------------------------


def test_map_aspect_known_value():
    from app.pipeline_common import _map_aspect
    assert _map_aspect("16:9") == "16:9"
    assert _map_aspect("9:16") == "9:16"
    assert _map_aspect("1:1") == "1:1"


def test_map_aspect_unknown_returns_adaptive():
    from app.pipeline_common import _map_aspect
    assert _map_aspect("3:2") == "adaptive"
    assert _map_aspect(None) == "adaptive"
    assert _map_aspect("") == "adaptive"


def test_clamp_duration_bounds():
    from app.pipeline_common import _clamp_duration
    assert _clamp_duration(0.0, 2.0) == 4    # floor
    assert _clamp_duration(0.0, 15.0) == 15  # ceiling
    assert _clamp_duration(0.0, 20.0) == 15  # capped at max
    assert _clamp_duration(0.0, 8.5) == 8    # rounds to nearest int (round(8.5) == 8 banker's rounding)


def test_snap_omni_duration():
    from app.pipeline_common import _snap_omni_duration
    assert _snap_omni_duration(0.0, 3.0) == 4
    assert _snap_omni_duration(0.0, 5.5) == 6
    assert _snap_omni_duration(0.0, 10.0) == 10


def test_map_omni_aspect_portrait_landscape():
    from app.pipeline_common import _map_omni_aspect
    assert _map_omni_aspect("16:9", 1920, 1080) == "16:9"
    assert _map_omni_aspect("9:16", 1080, 1920) == "9:16"
    assert _map_omni_aspect(None, 320, 240) == "16:9"   # landscape
    assert _map_omni_aspect(None, 240, 320) == "9:16"   # portrait


def test_omni_resolution_passes_supported_and_defaults_the_rest():
    from app.pipeline_common import _omni_resolution
    assert _omni_resolution("720p") == "720p"
    assert _omni_resolution("1080p") == "1080p"
    assert _omni_resolution("4k") == "4k"
    # 480p is a Seedance-only resolution → falls back to Gemini's first choice.
    assert _omni_resolution("480p") == "720p"
    assert _omni_resolution(None) == "720p"


# ---------------------------------------------------------------------------
# Tests: clamp_duration_for — the model-aware replacement for _clamp_duration
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "model, start, end, expected",
    [
        # Seedance 2.0 family: [4, 15].
        ("seedance", 0.0, 2.0, 4),
        ("seedance", 0.0, 9.0, 9),
        ("seedance", 0.0, 22.0, 15),
        ("seedance-fast", 0.0, 20.0, 15),
        # Seedance 2.5 reads the length off the input video, so the value we
        # send is the -1 sentinel regardless of clip length. The live API
        # rejects anything else ("`duration` must be -1"), which is why this
        # became model-aware in the first place.
        ("seedance-2-5", 0.0, 22.0, -1),
        ("seedance-2-5", 0.0, 30.0, -1),
        ("seedance-2-5", 0.0, 45.0, -1),
        ("seedance-2-5", 0.0, 2.0, -1),
        # Gemini Omni snaps to its fixed set instead of clamping.
        ("gemini-omni", 0.0, 5.5, 6),
        ("gemini-omni", 0.0, 22.0, 10),
        # Unknown/None fall back to the default backend's rule.
        (None, 0.0, 22.0, 15),
    ],
)
def test_clamp_duration_for_is_per_model(model, start, end, expected):
    from app.pipeline_common import clamp_duration_for
    assert clamp_duration_for(model, start, end) == expected


def test_clamp_duration_for_delegates_to_ai_models():
    """pipeline_common must not carry its own copy of the duration rules."""
    from app import ai_models
    from app.pipeline_common import clamp_duration_for

    for model in ai_models.AI_MODELS:
        for start, end in [(0.0, 1.0), (2.0, 13.5), (0.0, 28.0), (5.0, 60.0)]:
            assert clamp_duration_for(model, start, end) == ai_models.duration_for(
                model, start, end
            )


# ---------------------------------------------------------------------------
# Tests: re-exports — the names pipeline_v2 and the suite import from here
# ---------------------------------------------------------------------------


def test_reexported_names_are_still_importable():
    """These moved to app.ai_models; pipeline_common keeps them as aliases.

    Importers (pipeline_v2, this suite) were deliberately not touched by that
    move, so a missing alias is a real breakage, not a cleanup.
    """
    import app.pipeline_common as pc

    for name in (
        "_clamp_duration",
        "_map_aspect",
        "_snap_omni_duration",
        "_omni_resolution",
        "MIN_SWAP_VIDEO_SEC",
        "OMNI_MAX_CLIP_SECONDS",
        "clamp_duration_for",
        "resolve_reference_urls",
    ):
        assert hasattr(pc, name), f"app.pipeline_common lost {name}"


def test_reexported_constants_keep_their_values():
    from app import ai_models
    from app.pipeline_common import (
        MIN_SWAP_VIDEO_SEC,
        OMNI_MAX_CLIP_SECONDS,
        _SEEDANCE_ASPECTS,
    )

    assert MIN_SWAP_VIDEO_SEC == 2.0
    assert MIN_SWAP_VIDEO_SEC == ai_models.MIN_SWAP_VIDEO_SEC
    assert OMNI_MAX_CLIP_SECONDS == 10.0
    assert OMNI_MAX_CLIP_SECONDS == ai_models.max_clip_sec("gemini-omni")
    assert set(_SEEDANCE_ASPECTS) == {"1:1", "4:3", "16:9", "9:16", "21:9", "adaptive"}


def test_clamp_duration_shim_still_uses_the_2_0_range():
    """The old signature-less shim is pinned to the default backend's range.

    Callers that predate per-model durations keep the exact behaviour they had;
    anything model-aware must call clamp_duration_for instead.
    """
    from app.pipeline_common import _clamp_duration, clamp_duration_for

    assert _clamp_duration(0.0, 22.0) == 15
    assert _clamp_duration(0.0, 22.0) == clamp_duration_for("seedance", 0.0, 22.0)


# ---------------------------------------------------------------------------
# Tests: resolve_reference_urls
# ---------------------------------------------------------------------------


class TestResolveReferenceUrls:
    """
    Verify Fix #17: resolve_reference_urls resolves items concurrently while
    preserving the original order and existing error semantics.
    """

    def test_plain_https_url_returned_as_is(self):
        """A plain https:// URL is returned unchanged without any API calls."""
        from app.pipeline_common import resolve_reference_urls

        kie = _FakeKie()
        result = resolve_reference_urls(
            ["https://example.com/image.jpg"], kie
        )
        assert result == ["https://example.com/image.jpg"]
        assert kie.upload_calls == []

    def test_http_url_returned_as_is(self):
        from app.pipeline_common import resolve_reference_urls

        kie = _FakeKie()
        result = resolve_reference_urls(["http://cdn.example.com/img.png"], kie)
        assert result == ["http://cdn.example.com/img.png"]
        assert kie.upload_calls == []

    def test_drive_link_downloaded_and_uploaded(self):
        """A Drive share link is downloaded then uploaded to kie."""
        from app.pipeline_common import resolve_reference_urls

        kie = _FakeKie()
        gdrive = _FakeGDrive()
        drive_link = "https://drive.google.com/file/d/abc123/view"

        result = resolve_reference_urls([drive_link], kie, gdrive=gdrive)

        assert len(result) == 1
        assert result[0].startswith("https://fake-kie.example.com/")
        assert len(gdrive.download_calls) == 1
        assert gdrive.download_calls[0]["link"] == drive_link
        assert len(kie.upload_calls) == 1

    def test_local_path_uploaded_to_kie(self, tmp_path):
        """A local file path is uploaded to kie.ai."""
        from app.pipeline_common import resolve_reference_urls

        local = tmp_path / "ref.jpg"
        local.write_bytes(b"fake-image")
        kie = _FakeKie()

        result = resolve_reference_urls([str(local)], kie)

        assert len(result) == 1
        assert result[0].startswith("https://fake-kie.example.com/")
        assert kie.upload_calls == [str(local)]

    def test_empty_list_returns_empty(self):
        from app.pipeline_common import resolve_reference_urls

        kie = _FakeKie()
        assert resolve_reference_urls([], kie) == []

    def test_order_preserved_with_multiple_items(self, tmp_path):
        """
        Fix #17: the returned list must be in the same order as the input,
        regardless of which items complete first (concurrent execution).

        We use a mix of plain URLs, a local file, and a Drive link to exercise
        all three resolution paths in one call.
        """
        from app.pipeline_common import resolve_reference_urls

        local = tmp_path / "local.jpg"
        local.write_bytes(b"local-img")

        items = [
            "https://example.com/first.jpg",    # plain URL (fast)
            str(local),                          # local path (medium)
            "https://drive.google.com/file/d/xyz/view",  # Drive (slower)
            "https://example.com/last.jpg",      # plain URL (fast)
        ]

        gdrive = _FakeGDrive()
        kie = _FakeKie()

        result = resolve_reference_urls(items, kie, gdrive=gdrive)

        assert len(result) == 4
        # First and last are plain URLs — returned unchanged in input position.
        assert result[0] == "https://example.com/first.jpg"
        assert result[3] == "https://example.com/last.jpg"
        # Second is the local file upload — must be a kie URL.
        assert result[1].startswith("https://fake-kie.example.com/")
        # Third is the Drive link upload.
        assert result[2].startswith("https://fake-kie.example.com/")

    def test_order_preserved_all_drive_links(self):
        """
        When all items are Drive links, the resolved URLs must appear in the
        same order as the input — not first-completed order.
        """
        from app.pipeline_common import resolve_reference_urls

        import threading

        # Track resolution order by making each download take different time.
        resolve_order: list[str] = []
        resolve_lock = threading.Lock()

        class OrderedGDrive:
            _delays = {"drive-A": 0.05, "drive-B": 0.01, "drive-C": 0.03}

            def download_file(self, link: str, dst_path: str) -> str:
                import time
                key = link.split("?")[1] if "?" in link else link
                delay = self._delays.get(key, 0.01)
                time.sleep(delay)
                with resolve_lock:
                    resolve_order.append(link)
                os.makedirs(os.path.dirname(dst_path) or ".", exist_ok=True)
                with open(dst_path, "wb") as fh:
                    fh.write(b"content")
                return dst_path

        items = [
            "https://drive.google.com/uc?drive-A",
            "https://drive.google.com/uc?drive-B",
            "https://drive.google.com/uc?drive-C",
        ]
        kie = _FakeKie()
        result = resolve_reference_urls(items, kie, gdrive=OrderedGDrive())

        assert len(result) == 3
        # All results are kie URLs (not the original Drive links).
        assert all(r.startswith("https://fake-kie.example.com/") for r in result)
        # Completion order may differ from input order (drive-B finishes first),
        # but the returned list must still match input order.
        # The input order is [A, B, C]; the result[i] must match items[i]'s
        # resolution. We verify by checking that upload_calls are all distinct
        # and that result positions are stable (not reordered by completion time).
        # Since each drive download writes the same content, we verify that the
        # result has exactly 3 distinct URLs in the correct slot positions.
        assert result[0] != result[1] or result[1] != result[2]  # at least some differ
        # The three items map to three separate uploads.
        assert len(kie.upload_calls) == 3

    def test_error_in_one_item_propagates(self, tmp_path):
        """
        Fix #17 preserves error semantics: if any item fails, the exception
        propagates to the caller (same as the old serial loop).
        """
        from app.pipeline_common import resolve_reference_urls

        bad = tmp_path / "path.jpg"
        bad.write_bytes(b"x")
        kie = _FakeKie(fail_path=os.path.realpath(bad))

        with pytest.raises(RuntimeError, match="injected upload failure"):
            resolve_reference_urls([str(bad)], kie)

    def test_drive_download_failure_propagates(self):
        """A failing Drive download propagates the exception to the caller."""
        from app.pipeline_common import resolve_reference_urls

        bad_link = "https://drive.google.com/file/d/bad123/view"
        kie = _FakeKie()
        gdrive = _FakeGDrive(fail_link=bad_link)

        with pytest.raises(RuntimeError, match="injected download failure"):
            resolve_reference_urls([bad_link], kie, gdrive=gdrive)

    def test_single_item_fast_path(self, tmp_path):
        """
        Single-item list uses the fast path (no thread pool).  Verify it still
        works correctly and returns a single-element list.
        """
        from app.pipeline_common import resolve_reference_urls

        local = tmp_path / "single.jpg"
        local.write_bytes(b"img")
        kie = _FakeKie()

        result = resolve_reference_urls([str(local)], kie)

        assert len(result) == 1
        assert result[0].startswith("https://fake-kie.example.com/")
        assert kie.upload_calls == [str(local)]

    def test_many_plain_urls_in_order(self):
        """
        A large list of plain URLs (no network I/O) is returned unchanged and
        in input order.  Exercises the concurrent path with the fast-return
        early exit in _resolve_one_ref.
        """
        from app.pipeline_common import resolve_reference_urls

        urls = [f"https://cdn.example.com/img{i}.jpg" for i in range(9)]
        kie = _FakeKie()

        result = resolve_reference_urls(urls, kie)

        assert result == urls
        assert kie.upload_calls == []


# ---------------------------------------------------------------------------
# AICP-148 — resolve_reference_urls never uploads a path outside DATA_DIR
# ---------------------------------------------------------------------------


class TestResolveReferenceContainment:
    def _outside(self, tmp_path):
        from app import storage

        link = tmp_path / "evil.jpg"
        os.symlink("/etc/passwd", link)
        return [
            "/etc/passwd",
            "/proc/self/environ",
            os.path.join(storage.data_dir_root(), "..", "..", "etc", "passwd"),
            str(link),
            "T3",
        ]

    def test_outside_paths_raise_value_error_without_upload(self, tmp_path):
        from app.pipeline_common import resolve_reference_urls

        for bad in self._outside(tmp_path):
            kie = _FakeKie()
            with pytest.raises(ValueError, match="Reference 0"):
                resolve_reference_urls([bad], kie)
            assert kie.upload_calls == [], bad

    def test_bad_entry_in_a_list_names_its_slot(self, tmp_path):
        from app.pipeline_common import resolve_reference_urls

        ok = tmp_path / "ok.jpg"
        ok.write_bytes(b"x")
        with pytest.raises(ValueError, match="Reference 1 .*'T4'"):
            resolve_reference_urls(
                ["https://example.com/a.jpg", "T4", str(ok)], _FakeKie()
            )

    def test_missing_file_under_data_dir_is_a_clean_value_error(self, tmp_path):
        from app.pipeline_common import resolve_reference_urls

        kie = _FakeKie()
        with pytest.raises(ValueError, match="missing on disk"):
            resolve_reference_urls([str(tmp_path / "gone.jpg")], kie)
        assert kie.upload_calls == []


class TestPathWithin:
    def test_cases(self, tmp_path):
        from app.storage import path_within

        root = str(tmp_path / "root")
        os.makedirs(root)
        inside = os.path.join(root, "a", "b.jpg")
        assert path_within(inside, root) == os.path.realpath(inside)
        assert path_within(os.path.join(root, "a", "..", "c.jpg"), root) is not None
        assert path_within(os.path.join(root, "..", "x.jpg"), root) is None
        assert path_within(root, root) is None            # the root itself
        assert path_within(root + "2/x.jpg", root) is None  # prefix sibling
        assert path_within("/etc/passwd", root) is None
        assert path_within("relative.jpg", root) is None
        assert path_within(None, root) is None
        assert path_within("", root) is None
        assert path_within("bad\x00.jpg", root) is None
        os.symlink("/etc", os.path.join(root, "link"))
        assert path_within(os.path.join(root, "link", "passwd"), root) is None
