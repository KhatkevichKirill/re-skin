"""Explicit file grants for the native, unprivileged media sandbox.

Only trusted adapters construct this manifest. No command/filter inspection
can expand it. Output placeholders are private and already exist before the
decoder starts, so Landlock never grants filesystem creation rights.
"""
from __future__ import annotations

from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
import os
from pathlib import Path
import stat
import tempfile

MEDIA_EXEC = Path("/usr/local/libexec/reskin-media-exec")
MAX_INPUTS = 64
MAX_OUTPUTS = 4
MAX_FILE_BYTES = 512 * 1024 * 1024
TOOLS = {name: f"/usr/bin/{name}" for name in ("ffmpeg", "ffprobe", "fc-match")}


class MediaSandboxError(RuntimeError):
    """Safe failure; never invoke an unconfined fallback."""


@dataclass(frozen=True)
class PreparedMediaCommand:
    argv: list[str]
    env: dict[str, str]
    cwd: str


def _regular(path):
    path = Path(path)
    if not path.is_absolute():
        raise MediaSandboxError("media grant must be an absolute regular file")
    try:
        # Resolve trusted immutable inputs before giving the kernel inode grants.
        path = path.resolve(strict=True)
        info = path.stat()
        if not stat.S_ISREG(info.st_mode):
            raise MediaSandboxError("media grant must be an absolute regular file")
        return path
    except (OSError, RuntimeError, ValueError) as exc:
        raise MediaSandboxError("media input is unavailable") from exc


@contextmanager
def prepare_media_command(argv, *, inputs=(), outputs=()):
    """Stage explicit outputs and publish only after the caller succeeds.

    The caller must check process exit, cancellation and deadline *inside*
    this context. A raised exception discards every unpublished placeholder.
    Existing requested destinations are never overwritten.
    """
    if (not isinstance(argv, (tuple, list)) or not argv or len(argv) > 1024
            or any(not isinstance(a, str) or "\0" in a for a in argv)):
        raise MediaSandboxError("invalid media command")
    name = Path(argv[0]).name
    tool = TOOLS.get(name)
    if tool is None or argv[0] not in (name, tool):
        raise MediaSandboxError("media executable is not approved")
    try:
        helper_info = MEDIA_EXEC.lstat()
        if (not stat.S_ISREG(helper_info.st_mode) or helper_info.st_uid != 0
                or helper_info.st_mode & 0o022 or not os.access(MEDIA_EXEC, os.X_OK)):
            raise MediaSandboxError("media confinement is unavailable")
    except OSError as exc:
        raise MediaSandboxError("media confinement is unavailable") from exc
    inputs, outputs = tuple(inputs), tuple(outputs)
    if len(inputs) > MAX_INPUTS or len(outputs) > MAX_OUTPUTS:
        raise MediaSandboxError("media file grant limit exceeded")
    read_paths = tuple(dict.fromkeys(_regular(p) for p in inputs))
    destinations = []
    replacements = {}
    try:
        for output in outputs:
            original = os.fspath(output)
            path = Path(output)
            if not path.is_absolute() or os.path.lexists(path):
                raise MediaSandboxError("media output must be a new absolute file")
            path = path.parent.resolve(strict=True) / path.name
            if path in destinations or original not in argv[1:]:
                raise MediaSandboxError("media output must have one explicit argument")
            if argv[1:].count(original) != 1:
                raise MediaSandboxError("media output must have one explicit argument")
            destinations.append(path)
            replacements[original] = None
        if destinations and any(p.parent != destinations[0].parent for p in destinations):
            raise MediaSandboxError("media outputs must share their staging filesystem")
    except (OSError, ValueError, RuntimeError) as exc:
        if isinstance(exc, MediaSandboxError):
            raise
        raise MediaSandboxError("media output is unavailable") from exc
    # Output staging shares the destination filesystem for exclusive hard-link
    # publication. Neither this directory nor its parent gets Landlock rights.
    base = destinations[0].parent if destinations else None
    with tempfile.TemporaryDirectory(prefix=".media-jail-", dir=base) as directory, ExitStack() as handles:
        work = Path(directory)
        private = []
        identities = []
        for index, destination in enumerate(destinations):
            stage = work / f"output-{index}{destination.suffix}"
            fd = os.open(stage, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
            # Retain a parent-only reference until publication so removing the
            # pathname cannot recycle its inode and defeat the identity check.
            # Popen(close_fds=True) never passes this writable capability on.
            handles.callback(os.close, fd)
            info = os.fstat(fd)
            identities.append((info.st_dev, info.st_ino))
            private.append(stage)
        for original, stage in zip(replacements, private):
            replacements[original] = str(stage)
        command = [replacements.get(a, a) for a in argv[1:]]
        if destinations:
            # -y only touches newly created private placeholders. Requested
            # destinations retain exclusive, no-overwrite publication below.
            command = ["-y" if a == "-n" else a for a in command]
            if "-y" not in command:
                command.insert(0, "-y")
        helper = [str(MEDIA_EXEC), "--tool", tool, "--parent-pid", str(os.getpid())]
        for path in read_paths:
            helper += ["--input", str(path)]
        for path in private:
            helper += ["--output", str(path)]
        environment = {
            "PATH": "/usr/bin:/bin", "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8",
            "HOME": directory, "TMPDIR": directory,
            "FONTCONFIG_FILE": "/etc/fonts/fonts.conf",
            "XDG_CACHE_HOME": directory, "OMP_NUM_THREADS": "2",
            "OPENBLAS_NUM_THREADS": "2",
        }
        yield PreparedMediaCommand(helper + ["--"] + command, environment, directory)
        for stage, identity in zip(private, identities):
            try:
                info = stage.lstat()
                if (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1
                        or (info.st_dev, info.st_ino) != identity
                        or info.st_uid != os.geteuid() or info.st_size > MAX_FILE_BYTES):
                    raise MediaSandboxError("media output failed confinement checks")
            except OSError as exc:
                raise MediaSandboxError("media output is unavailable") from exc
        published = []
        try:
            for stage, destination in zip(private, destinations):
                os.link(stage, destination, follow_symlinks=False)
                published.append(destination)
        except OSError as exc:
            for destination in published:
                destination.unlink()
            raise MediaSandboxError("media output cannot be published") from exc
