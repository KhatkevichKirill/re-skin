"""Manifest/publication checks; native attack acceptance lives separately."""
import os
from pathlib import Path
import stat

import pytest

from app import assistant_media_sandbox as sandbox


@pytest.fixture
def helper(monkeypatch):
    # Bridge tests don't execute this immutable root-owned fixture, and work
    # as UID10001 too. Native acceptance exercises the actual policy installer.
    path = Path("/usr/bin/true")
    monkeypatch.setattr(sandbox, "MEDIA_EXEC", path)
    return path


def test_explicit_manifest_rewrites_only_private_output_and_scrubs_env(helper, tmp_path, monkeypatch):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"immutable")
    output = tmp_path / "result.mp4"
    monkeypatch.setenv("DATABASE_URL", "synthetic-private-canary")
    monkeypatch.setenv("LD_PRELOAD", "/private/canary")
    with sandbox.prepare_media_command(
        ["/usr/bin/ffmpeg", "-n", "-i", str(source), str(output)],
        inputs=(source,), outputs=(output,),
    ) as command:
        assert str(output) not in command.argv
        assert "--input" in command.argv and str(source) in command.argv
        assert command.argv.count("-y") == 1 and "-n" not in command.argv
        assert "DATABASE_URL" not in command.env and "LD_PRELOAD" not in command.env
        stage = Path(command.argv[-1])
        assert stage.parent == Path(command.cwd) and stage.parent.parent == tmp_path
        assert stage.exists() and stat.S_IMODE(stage.stat().st_mode) == 0o600
        assert not output.exists()
        stage.write_bytes(b"rendered")
    assert output.read_bytes() == b"rendered"
    assert source.read_bytes() == b"immutable"
    assert not stage.parent.exists()


def test_exception_discards_unpublished_output(helper, tmp_path):
    output = tmp_path / "result.mp4"
    with pytest.raises(RuntimeError, match="cancelled"):
        with sandbox.prepare_media_command(["ffmpeg", "-n", str(output)], outputs=(output,)) as command:
            Path(command.argv[-1]).write_bytes(b"partial")
            work = Path(command.cwd)
            raise RuntimeError("cancelled")
    assert not output.exists() and not work.exists()


def test_concurrent_destination_is_never_overwritten(helper, tmp_path):
    output = tmp_path / "result.mp4"
    with pytest.raises(sandbox.MediaSandboxError, match="published"):
        with sandbox.prepare_media_command(["ffmpeg", "-n", str(output)], outputs=(output,)) as command:
            Path(command.argv[-1]).write_bytes(b"candidate")
            output.write_bytes(b"concurrent")
    assert output.read_bytes() == b"concurrent"


@pytest.mark.parametrize("replacement", ["symlink", "hardlink", "inode"])
def test_replaced_output_inode_or_links_refuse_publication(helper, tmp_path, replacement):
    output = tmp_path / "result.mp4"
    outside = tmp_path / "outside"
    outside.write_bytes(b"private")
    with pytest.raises(sandbox.MediaSandboxError, match="confinement"):
        with sandbox.prepare_media_command(["ffmpeg", "-n", str(output)], outputs=(output,)) as command:
            stage = Path(command.argv[-1])
            if replacement == "hardlink":
                os.link(stage, stage.with_suffix(".extra"))
            else:
                stage.unlink()
                if replacement == "symlink":
                    stage.symlink_to(outside)
                else:
                    stage.write_bytes(b"substituted")
    assert outside.read_bytes() == b"private" and not output.exists()


def test_no_outputs_have_no_write_grants(helper):
    with sandbox.prepare_media_command(["fc-match", "-f", "%{family}", "DejaVu Sans"]) as command:
        assert "--output" not in command.argv
        assert "--input" not in command.argv
        assert command.argv[-1] == "DejaVu Sans"


@pytest.mark.parametrize("tool", ["/bin/sh", "/tmp/ffmpeg", "python", "/usr/local/bin/ffprobe"])
def test_other_executables_are_refused(helper, tool):
    with pytest.raises(sandbox.MediaSandboxError, match="approved"):
        with sandbox.prepare_media_command([tool]):
            pytest.fail("unapproved tool prepared")


def test_absent_helper_is_fail_closed(tmp_path, monkeypatch):
    monkeypatch.setattr(sandbox, "MEDIA_EXEC", tmp_path / "missing")
    with pytest.raises(sandbox.MediaSandboxError, match="unavailable"):
        with sandbox.prepare_media_command(["ffprobe", "-version"]):
            pytest.fail("missing confinement prepared")


def test_directories_cannot_be_input_grants(helper, tmp_path):
    with pytest.raises(sandbox.MediaSandboxError):
        with sandbox.prepare_media_command(["ffprobe", str(tmp_path)], inputs=(tmp_path,)):
            pytest.fail("directory input prepared")
