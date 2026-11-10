"""AICP-160: sheet formula injection, ffmpeg input allow-lists, Drive size cap."""

import csv
import io
import os
import shutil
import subprocess
from unittest.mock import MagicMock, patch

import pytest

from app import gdrive_client, media
from app.sheets_client import SheetsClient, build_final_creatives_row, neutralise_formula


# --- Sheets / CSV ----------------------------------------------------------


@pytest.mark.parametrize("v", ["=IMPORTXML(\"http://x\",\"//a\")", "+1+1", "-2+3", "@SUM(A1)", "\t=1", "\r=1"])
def test_formula_prefixed_values_neutralised(v):
    assert neutralise_formula(v) == "'" + v


@pytest.mark.parametrize("v", ["Normal name", "2026-10-16", "https://drive.google.com/x", "", "a=b", 5, 1.5, None])
def test_other_values_untouched(v):
    assert neutralise_formula(v) == v


def test_append_row_neutralises_user_cells_and_keeps_numbers():
    svc = MagicMock()
    client = SheetsClient(service=svc)
    client.append_row("sid", "Tab", ["=HYPERLINK(\"http://evil\")", "ok", 3, "@x"])
    kwargs = svc.spreadsheets().values().append.call_args.kwargs
    assert kwargs["valueInputOption"] == "USER_ENTERED"  # dates/numbers keep parsing
    assert kwargs["body"] == {"values": [["'=HYPERLINK(\"http://evil\")", "ok", 3, "'@x"]]}


def test_final_creatives_row_values_intact_for_normal_data():
    header = ["Date", "Job ID", "Creative Link", "Idea"]
    row = build_final_creatives_row(header, fields={
        "Date": "2026-10-16", "Job ID": "1736426f-ja",
        "Creative Link": "https://drive.google.com/file/d/abc/view", "Idea": "proj_run_SD2",
    })
    assert [neutralise_formula(c) for c in row] == row


# --- ffmpeg argv -------------------------------------------------------------


def test_harden_ffprobe_argv():
    out = media.harden_media_argv(["/usr/bin/ffprobe", "-v", "error", "in.mp4"])
    assert out[:5] == ["/usr/bin/ffprobe", "-protocol_whitelist", "file,pipe", "-format_whitelist", media.FORMAT_WHITELIST]
    assert out[-1] == "in.mp4"


def test_harden_ffmpeg_argv_before_every_input():
    cmd = ["ffmpeg", "-y", "-i", "a.mp4", "-f", "lavfi", "-i", "anullsrc", "-c", "copy", "out.mp4"]
    out = media.harden_media_argv(cmd)
    assert out.count("-protocol_whitelist") == 2
    for i, tok in enumerate(out):
        if tok == "-i":
            assert out[i - 4:i - 2] == ["-protocol_whitelist", "file,pipe"]
    assert media.harden_media_argv(out) == out  # idempotent
    assert cmd[2] == "-i" and "-protocol_whitelist" not in cmd  # input not mutated


def test_non_media_argv_untouched():
    assert media.harden_media_argv(["ls", "-i"]) == ["ls", "-i"]


def test_run_applies_whitelist():
    with patch("app.media.subprocess.run") as run:
        run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        media._run([media.FFPROBE, "-v", "error", "x.mp4"])
    argv = run.call_args.args[0]
    assert "-protocol_whitelist" in argv and "file,pipe" in argv


@pytest.mark.skipif(not shutil.which("ffprobe") or not shutil.which("ffmpeg"), reason="needs ffmpeg")
def test_crafted_hls_playlist_named_mp4_is_rejected(tmp_path):
    seg = tmp_path / "seg.ts"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc=d=1:s=64x64", "-c:v", "libx264", "-f", "mpegts", str(seg)],
        check=True,
    )
    evil = tmp_path / "evil.mp4"
    evil.write_text("#EXTM3U\n#EXT-X-TARGETDURATION:2\n#EXTINF:1,\nseg.ts\n#EXT-X-ENDLIST\n")
    with pytest.raises(media.MediaError):
        media.probe(str(evil))


@pytest.mark.skipif(not shutil.which("ffprobe") or not shutil.which("ffmpeg"), reason="needs ffmpeg")
@pytest.mark.parametrize("ext", ["mp4", "mkv", "mov", "avi"])
def test_legit_video_still_probes(tmp_path, ext):
    p = tmp_path / f"ok.{ext}"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc=d=1:s=64x64", "-f", "lavfi",
         "-i", "sine=d=1", "-shortest", "-c:v", "libx264", "-c:a", "aac" if ext != "avi" else "mp3", str(p)],
        check=True,
    )
    info = media.probe(str(p))
    assert info.width == 64


# --- Drive size cap ------------------------------------------------------------


def _client(size):
    svc = MagicMock()
    svc.files().get().execute.return_value = {} if size is None else {"size": str(size)}
    # resolve via the module: another test reloads app.gdrive_client, so names
    # imported at collection time can be stale classes
    return gdrive_client.GDriveClient(service=svc), svc


FILE_ID = "1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgVE2upms"


def test_oversized_file_refused_before_download(tmp_path):
    client, svc = _client(10_000)
    dst = str(tmp_path / "o.mp4")
    with patch("app.gdrive_client.MediaIoBaseDownload") as dl:
        with pytest.raises(gdrive_client.GDriveFileTooLargeError):
            client.download_file(FILE_ID, dst, max_bytes=1000)
    dl.assert_not_called()
    svc.files().get_media.assert_not_called()
    assert not os.path.exists(dst)


def test_default_cap_comes_from_max_upload_size_mb(tmp_path):
    from app.config import settings

    client, _ = _client((settings.MAX_UPLOAD_SIZE_MB << 20) + 1)
    with pytest.raises(gdrive_client.GDriveFileTooLargeError):
        client.download_file(FILE_ID, str(tmp_path / "o.mp4"))


def test_stream_aborted_and_partial_removed_when_metadata_lies(tmp_path):
    client, _ = _client(None)  # no size in metadata
    dst = str(tmp_path / "o.mp4")
    writes = []

    class Fake:
        def __init__(self, fh, request, chunksize=None):
            self._fh = fh

        def next_chunk(self, num_retries=0):
            self._fh.write(b"x" * 600)
            writes.append(1)
            return None, False  # never says done

    with patch("app.gdrive_client.MediaIoBaseDownload", Fake):
        with pytest.raises(gdrive_client.GDriveFileTooLargeError):
            client.download_file(FILE_ID, dst, max_bytes=1000)
    assert len(writes) == 2  # stopped right after passing the cap
    assert not os.path.exists(dst)


def test_download_under_cap_unchanged(tmp_path):
    client, _ = _client(5)
    dst = str(tmp_path / "o.mp4")

    class Fake:
        def __init__(self, fh, request, chunksize=None):
            self._fh = fh

        def next_chunk(self, num_retries=0):
            self._fh.write(b"hello")
            return None, True

    with patch("app.gdrive_client.MediaIoBaseDownload", Fake):
        assert client.download_file(FILE_ID, dst, max_bytes=1000) == dst
    assert open(dst, "rb").read() == b"hello"


def test_overlay_editor_renders_asset_name_via_textcontent():
    src = open(os.path.join(os.path.dirname(__file__), "..", "app", "templates", "overlay", "editor.html")).read()
    assert "ov.asset_path.split" not in src.split("card.innerHTML")[1].split("`;")[0]
    assert "textContent = String(ov.asset_path)" in src
