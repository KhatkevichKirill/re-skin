"""The predeploy gate must validate the actual remote dump before cleanup."""
import hashlib
import sys
import types
from pathlib import Path

import pytest


@pytest.mark.parametrize("remote_state", ["valid", "wrong_size", "wrong_hash", "missing_id"])
def test_backup_verifies_remote_dump(tmp_path, monkeypatch, capsys, remote_state):
    script = Path(__file__).resolve().parents[2] / "scripts" / "backup_db.sh"
    text = script.read_text()
    invocation, program = text.split("<<'PY'\n", 1)
    assert 'docker exec -i "$API_CONTAINER" python3 -' in invocation
    program = program.split("\nPY\n", 1)[0]
    payload = b"representative database archive bytes"
    dump = tmp_path / "backup.dump"
    dump.write_bytes(payload)
    metadata = {"size": str(len(payload)), "md5Checksum": hashlib.md5(payload).hexdigest()}
    if remote_state == "wrong_size":
        metadata["size"] = "0"
    if remote_state == "wrong_hash":
        metadata["md5Checksum"] = "0" * 32

    class Drive:
        service = property(lambda self: self)

        def upload_file(self, path, **kwargs):
            assert Path(path).read_bytes() == payload
            return {} if remote_state == "missing_id" else {"id": "test-backup"}

        def files(self):
            return self

        def get(self, **kwargs):
            assert kwargs == {"fileId": "test-backup", "fields": "id,size,md5Checksum", "supportsAllDrives": True}
            return self

        def execute(self):
            return metadata

    monkeypatch.setitem(sys.modules, "app.gdrive_client", types.SimpleNamespace(GDriveClient=Drive))
    monkeypatch.setattr(sys, "argv", ["-", str(dump), "test-folder", dump.name])
    if remote_state == "valid":
        exec(compile(program, str(script), "exec"), {})
        assert "verified remote size and checksum" in capsys.readouterr().out
    else:
        with pytest.raises(RuntimeError):
            exec(compile(program, str(script), "exec"), {})
        assert "uploaded id=" not in capsys.readouterr().out
    assert dump.read_bytes() == payload
