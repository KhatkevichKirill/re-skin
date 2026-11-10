"""Attack the production native policy using disposable, owned canaries.

The probe includes the exact policy source and runs hostile syscalls after its
installer. FFmpeg's own argument/protocol validation cannot make these pass.
Qualification requires Linux, a C compiler and libseccomp development headers.
"""
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest
from tests.test_assistant_turns import setup

@pytest.mark.parametrize("revocation", ["policy", "run_owner"])
def test_metadata_revocation_hides_completed_analysis_in_api_and_future_dialogue(setup, revocation):
    import json
    from app.api_assistant import get_context, get_turn
    from app.assistant_access import ProjectAllowlist
    from app.assistant_agent_contracts import AgentActionEnvelope, FinalAction
    from app.assistant_planner import ClarifyReply
    from app.assistant_turns import AssistantTurn
    from app.models import Run
    from tests.test_assistant_agent_loop import fixture_loop
    from tests.test_assistant_turns import NOW

    svc, turn_id, broker, _, supervisor = fixture_loop(setup)
    scope = {"run_id": setup.base.run, "base_version_id": setup.base.version}
    metadata_reply = "acceptance-owned-derived-metadata-explanation"
    broker.actions = [
        AgentActionEnvelope.model_validate({"action": dict(kind=kind, **scope, **fields)}).action
        for kind, fields in (
            ("list_projects", {}),
            ("list_runs", {"project_id": setup.donor.project}),
            ("list_versions", {"source_run_id": setup.donor.run}),
            ("list_assets", {"source_version_id": setup.donor.version}),
        )
    ] + [FinalAction(kind="final", **scope, reply=ClarifyReply(
        kind="clarify", **scope, explanation=metadata_reply, question="Which listed result?"))]
    supervisor.run_once()
    with setup.factory.begin() as session:
        assert session.get(AssistantTurn, turn_id).status == "succeeded"
        visible = get_turn(setup.base.run, turn_id, db=session, service=svc)
        assert len([step for step in visible["analysis"]["steps"]
                    if (step.get("observation") or {}).get("internal_read")]) == 4
        if revocation == "policy":
            svc.policy = ProjectAllowlist([setup.base.project])
        else:
            session.get(Run, setup.donor.run).project_id = setup.base.project
            session.flush()
        hidden = get_turn(setup.base.run, turn_id, db=session, service=svc)
        assert hidden["source_access_changed"] is True
        assert hidden["reply"] is None and hidden["analysis"] is None
        history = get_context(setup.base.run, db=session, service=svc)
        assert metadata_reply not in json.dumps(history, default=str)
        followup = svc.accept(session, run_id=setup.base.run, base_version_id=setup.base.version,
            instruction="Continue", idempotency_key="acceptance-clean-continuation", actor="operator", now=NOW)
        assert metadata_reply not in json.dumps(followup.context_snapshot)


_BACKEND = Path(__file__).resolve().parents[1]
_SOURCE = _BACKEND / "sandbox" / "media_exec.c"
_ATTACK_SOURCE = r'''
#define _GNU_SOURCE
#define main reskin_launcher_main
#include "@POLICY_SOURCE@"
#undef main
#include <pthread.h>
#include <sys/socket.h>
#include <sys/uio.h>
#include <sys/xattr.h>
#include <linux/openat2.h>

static void *acceptance_thread(void *unused) { return unused; }
static int acceptance_denied(long result) {
    return result == -1 && (errno == EPERM || errno == EACCES);
}
static int acceptance_read(const char *path) {
    char byte;
    int fd = open(path, O_RDONLY | O_CLOEXEC);
    if (fd < 0) return 0;
    int ok = read(fd, &byte, 1) == 1;
    close(fd);
    return ok;
}
int main(int argc, char **argv) {
    if (argc != 8) return 120;
    const char *profile=argv[1], *op=argv[2], *input=argv[3];
    const char *output=argv[4], *secret=argv[5], *model=argv[6], *extra=argv[7];
    const char *inputs[]={input}, *outputs[]={output};
    pid_t own_pid=getpid();
    uid_t own_uid=getuid();
    gid_t own_gid=getgid();
    int inherited=open(secret,O_RDONLY|O_CLOEXEC);
    if (inherited < 3) return 121;
    if (!strcmp(profile,"asr")) {
        if (reskin_install_asr_policy(input,model,getppid())) return 122;
    } else if (install_media_policy("/usr/bin/ffmpeg",inputs,1,outputs,1)) {
        return 123;
    }
    int ok=0,fd=-1;
    errno=0;
    if (!strcmp(op,"read_input")) ok=acceptance_read(input);
    else if (!strcmp(op,"read_secret")) ok=acceptance_denied(open(secret,O_RDONLY));
    else if (!strcmp(op,"read_env")) ok=acceptance_denied(open("/proc/self/environ",O_RDONLY));
    else if (!strcmp(op,"read_model")) ok=acceptance_read(extra);
    else if (!strcmp(op,"inherited_fd")) {
        char byte;
        ok=read(inherited,&byte,1)==-1 && errno==EBADF;
    }
    else if (!strcmp(op,"write_output")) {
        fd=open(output,O_WRONLY|O_TRUNC);
        ok=fd>=0 && write(fd,"candidate",9)==9;
    }
    else if (!strcmp(op,"deny_write_output")) ok=acceptance_denied(open(output,O_WRONLY));
    else if (!strcmp(op,"write_input")) ok=acceptance_denied(open(input,O_WRONLY));
    else if (!strcmp(op,"truncate_readonly")) ok=acceptance_denied(open(input,O_RDONLY|O_TRUNC));
    else if (!strcmp(op,"openat_truncate")) ok=acceptance_denied(openat(AT_FDCWD,input,O_RDONLY|O_TRUNC));
    else if (!strcmp(op,"truncate")) ok=acceptance_denied(truncate(input,0));
    else if (!strcmp(op,"ftruncate_input")) {
        fd=open(input,O_RDONLY);
        int result=fd<0 ? 0 : ftruncate(fd,0);
        ok=fd>=0 && result==-1 && (errno==EPERM || errno==EACCES || errno==EBADF || errno==EINVAL);
    }
    else if (!strcmp(op,"chmod")) ok=acceptance_denied(chmod(input,0400));
    else if (!strcmp(op,"fchmod")) {
        fd=open(input,O_RDONLY);
        ok=fd>=0 && acceptance_denied(fchmod(fd,0400));
    }
    else if (!strcmp(op,"chown")) ok=acceptance_denied(chown(input,own_uid,own_gid));
    else if (!strcmp(op,"utime")) {
        struct timespec times[2]={{100,0},{100,0}};
        ok=acceptance_denied(utimensat(AT_FDCWD,input,times,0));
    }
    else if (!strcmp(op,"xattr")) ok=acceptance_denied(setxattr(input,"user.acceptance","x",1,0));
    else if (!strcmp(op,"unlink")) ok=acceptance_denied(unlink(input));
    else if (!strcmp(op,"rename")) ok=acceptance_denied(rename(input,extra));
    else if (!strcmp(op,"hardlink")) ok=acceptance_denied(link(input,extra));
    else if (!strcmp(op,"symlink")) ok=acceptance_denied(symlink(secret,extra));
    else if (!strcmp(op,"create")) ok=acceptance_denied(open(extra,O_CREAT|O_WRONLY,0600));
    else if (!strcmp(op,"inet")) ok=acceptance_denied(socket(AF_INET,SOCK_STREAM,0));
    else if (!strcmp(op,"inet6")) ok=acceptance_denied(socket(AF_INET6,SOCK_DGRAM,0));
    else if (!strcmp(op,"unix")) ok=acceptance_denied(socket(AF_UNIX,SOCK_STREAM,0));
    else if (!strcmp(op,"socketpair")) {
        int pair[2];
        ok=acceptance_denied(socketpair(AF_UNIX,SOCK_STREAM,0,pair));
    }
    else if (!strcmp(op,"fork")) {
        pid_t child=fork();
        if (child==0) _exit(119);
        ok=acceptance_denied(child);
    }
    else if (!strcmp(op,"clone")) ok=acceptance_denied(syscall(SYS_clone,0,0,0,0,0));
    else if (!strcmp(op,"exec")) {
        char *args[]={"/bin/sh","-c","exit 118",NULL}, *env[]={NULL};
        ok=acceptance_denied(execve(args[0],args,env));
    }
    else if (!strcmp(op,"exec_allowed_tool")) {
        char *args[]={"/usr/bin/ffmpeg","-version",NULL}, *env[]={NULL};
        ok=acceptance_denied(execve(args[0],args,env));
    }
    else if (!strcmp(op,"thread")) {
        pthread_t thread;
        ok=pthread_create(&thread,NULL,acceptance_thread,NULL)==0;
        if (ok) ok=pthread_join(thread,NULL)==0;
    }
    else if (!strcmp(op,"process_read")) {
        char local='x',copy=0;
        struct iovec from={&local,1},to={&copy,1};
        ok=acceptance_denied(process_vm_readv(own_pid,&to,1,&from,1,0));
    }
    else if (!strcmp(op,"process_write")) {
        char local='x',copy=0;
        struct iovec from={&local,1},to={&copy,1};
        ok=acceptance_denied(process_vm_writev(own_pid,&from,1,&to,1,0));
    }
#ifdef SYS_io_uring_setup
    else if (!strcmp(op,"io_uring")) ok=acceptance_denied(syscall(SYS_io_uring_setup,1,NULL));
#endif
#ifdef SYS_openat2
    else if (!strcmp(op,"openat2")) {
        struct open_how how={.flags=O_RDONLY};
        ok=acceptance_denied(syscall(SYS_openat2,AT_FDCWD,input,&how,sizeof(how)));
    }
#endif
    if (fd>=0) close(fd);
    return ok ? 0 : 117;
}
'''


@pytest.fixture(scope="module")
def native_policy(tmp_path_factory):
    compiler = shutil.which("cc") or shutil.which("gcc")
    if not compiler:
        pytest.skip("native security acceptance requires a C compiler")
    assert _SOURCE.is_file(), "production native confinement source is missing"
    directory = tmp_path_factory.mktemp("native-policy-acceptance")
    source = directory / "attack.c"
    source.write_text(_ATTACK_SOURCE.replace("@POLICY_SOURCE@", str(_SOURCE)))
    probe, library = directory / "attack", directory / "policy.so"
    for argv in (
        [compiler, "-std=c11", "-O2", "-pthread", str(source), "-lseccomp", "-o", str(probe)],
        [compiler, "-std=c11", "-O2", "-shared", "-fPIC", str(_SOURCE), "-lseccomp", "-o", str(library)],
    ):
        result = subprocess.run(argv, capture_output=True, text=True, timeout=60)
        assert result.returncode == 0, result.stderr
    return probe, library


@pytest.fixture
def canaries(tmp_path):
    source = tmp_path / "input.bin"
    source.write_bytes(b"immutable source bytes")
    output = tmp_path / "output.bin"
    output.write_bytes(b"reserved output")
    other = tmp_path / "other-job"
    other.mkdir()
    secret = other / ".env"
    secret.write_bytes(b"acceptance-owned-credential-canary")
    model = tmp_path / "model"
    model.mkdir()
    model_file = model / "config.json"
    model_file.write_text('{"acceptance":"approved model"}')
    return source, output, secret, model, model_file, tmp_path / "must-not-exist"


def _attack(native_policy, canaries, operation, *, profile="media", extra=None):
    probe, _ = native_policy
    source, output, secret, model, _, absent = canaries
    before = source.stat()
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    result = subprocess.run(
        [str(probe), profile, operation, str(source), str(output), str(secret), str(model), str(extra or absent)],
        capture_output=True, timeout=10, env={"PATH": "/usr/bin:/bin", "ACCEPTANCE_ONLY": "owned-process-env"},
    )
    assert result.returncode == 0, (operation, result.returncode, result.stderr.decode(errors="replace"))
    after = source.stat()
    assert hashlib.sha256(source.read_bytes()).hexdigest() == digest
    assert (after.st_mode, after.st_uid, after.st_gid, after.st_mtime_ns) == (
        before.st_mode, before.st_uid, before.st_gid, before.st_mtime_ns)
    assert secret.read_bytes() == b"acceptance-owned-credential-canary"
    assert not absent.exists() and not absent.is_symlink()


def test_asr_child_dies_when_its_owning_process_exits(native_policy, canaries):
    import select
    import signal

    if not hasattr(os, "pidfd_open") or not hasattr(signal, "pidfd_send_signal"):
        pytest.skip("ASR lifecycle acceptance requires Linux pidfds")
    _, library = native_policy
    source, _, _, model, _, _ = canaries
    child_script = r"""
import ctypes, os, sys, time
library=ctypes.CDLL(sys.argv[1],use_errno=True)
install=library.reskin_install_asr_policy
install.argtypes=[ctypes.c_char_p,ctypes.c_char_p,ctypes.c_int]
install.restype=ctypes.c_int
assert install(os.fsencode(sys.argv[2]),os.fsencode(sys.argv[3]),os.getppid())==0
print('READY',flush=True)
time.sleep(60)
"""
    owner_script = r"""
import os, subprocess, sys
child=subprocess.Popen([sys.executable,'-I','-c',*sys.argv[1:]],
                       stdout=subprocess.PIPE,stderr=subprocess.PIPE)
if child.stdout.readline()!=b'READY\n':
    raise RuntimeError(child.stderr.read().decode(errors='replace'))
print(child.pid,flush=True)
sys.stdin.buffer.readline()
os._exit(0)
"""
    owner = subprocess.Popen(
        [sys.executable, "-I", "-c", owner_script, child_script, str(library), str(source), str(model)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        env={"PATH": "/usr/bin:/bin", "PYTHONDONTWRITEBYTECODE": "1"},
    )
    pidfd = None
    try:
        assert select.select([owner.stdout], [], [], 10)[0], "ASR child did not activate its policy"
        pid = owner.stdout.readline().strip()
        assert pid.isdigit(), owner.stderr.read().decode(errors="replace")
        pidfd = os.pidfd_open(int(pid))
        owner.stdin.write(b"exit\n")
        owner.stdin.flush()
        owner.wait(timeout=5)
        poll = select.poll()
        poll.register(pidfd, select.POLLIN)
        assert poll.poll(5000), "confined ASR child survived its owner"
    finally:
        if owner.poll() is None:
            owner.kill()
            owner.wait(timeout=5)
        if pidfd is not None:
            try:
                signal.pidfd_send_signal(pidfd, signal.SIGKILL)
            except ProcessLookupError:
                pass
            os.close(pidfd)
        for handle in (owner.stdin, owner.stdout, owner.stderr):
            handle.close()


def test_asr_policy_rejects_a_changed_parent(native_policy, canaries):
    _, library = native_policy
    source, _, _, model, _, _ = canaries
    script = r"""
import ctypes, os, sys
library=ctypes.CDLL(sys.argv[1],use_errno=True)
install=library.reskin_install_asr_policy
install.argtypes=[ctypes.c_char_p,ctypes.c_char_p,ctypes.c_int]
install.restype=ctypes.c_int
for parent in (-1,0,os.getppid()+1):
    assert install(os.fsencode(sys.argv[2]),os.fsencode(sys.argv[3]),parent)!=0
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", script, str(library), str(source), str(model)],
        capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == 0, result.stderr


_COMMON_ATTACKS = [
    "read_input", "read_secret", "read_env", "inherited_fd",
    "write_input", "truncate_readonly", "openat_truncate", "truncate",
    "chmod", "fchmod", "chown", "utime", "xattr", "unlink", "rename", "hardlink",
    "symlink", "create", "inet", "inet6", "unix", "socketpair", "fork", "clone",
    "exec", "thread", "process_read", "process_write", "io_uring", "openat2", "ftruncate_input",
]


@pytest.mark.parametrize("operation", [*_COMMON_ATTACKS, "write_output"])
def test_compromised_media_decoder_cannot_expand_capabilities(native_policy, canaries, operation):
    _attack(native_policy, canaries, operation)


@pytest.mark.parametrize("operation", [*_COMMON_ATTACKS, "deny_write_output", "exec_allowed_tool"])
def test_asr_native_child_gets_no_other_job_or_socket_write_capability(native_policy, canaries, operation):
    _attack(native_policy, canaries, operation, profile="asr")


def test_asr_can_read_its_selected_model(native_policy, canaries):
    _attack(native_policy, canaries, "read_model", profile="asr", extra=canaries[4])


def test_media_policy_rejects_symlink_and_multiply_linked_output(native_policy, canaries):
    probe, _ = native_policy
    source, output, secret, model, _, absent = canaries
    output.unlink()
    output.symlink_to(secret)
    for kind in ("symlink", "hardlink"):
        if kind == "hardlink":
            output.unlink()
            os.link(secret, output)
        result = subprocess.run(
            [str(probe), "media", "write_output", str(source), str(output), str(secret), str(model), str(absent)],
            capture_output=True, timeout=10,
        )
        assert result.returncode == 123, (kind, result.returncode)
        assert secret.read_bytes() == b"acceptance-owned-credential-canary"


def test_asr_worker_fails_closed_without_shared_policy(tmp_path):
    # Use an isolated interpreter: restrict() changes process resource limits.
    worker = _BACKEND / "app" / "assistant_asr_worker.py"
    script = """
import importlib.util, os, sys
spec=importlib.util.spec_from_file_location('owned_asr_worker',sys.argv[1])
worker=importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)
worker._POLICY_LIBRARY=sys.argv[2]
try:
    worker.restrict(audio_path=sys.argv[3],model_dir=sys.argv[4],parent_pid=os.getppid())
except RuntimeError as error:
    assert str(error)=='ASR filesystem/syscall sandbox unavailable'
else:
    raise AssertionError('missing confinement library was accepted')
assert 'faster_whisper' not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", script, str(worker),
         str(tmp_path / "missing-policy.so"), str(tmp_path / "audio.wav"), str(tmp_path / "model")],
        capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == 0, result.stderr

def test_python_asr_child_restricts_before_loading_decoder_packages(native_policy, canaries, tmp_path):
    _, library = native_policy
    source, _, secret, model, model_file, absent = canaries
    script = r"""
import ctypes, os, pathlib, socket, sys, threading
sys.dont_write_bytecode=True
library=ctypes.CDLL(sys.argv[1],use_errno=True)
install=library.reskin_install_asr_policy
install.argtypes=[ctypes.c_char_p,ctypes.c_char_p,ctypes.c_int]
install.restype=ctypes.c_int
assert install(os.fsencode(sys.argv[2]),os.fsencode(sys.argv[3]),os.getppid())==0
assert pathlib.Path(sys.argv[2]).read_bytes()==b'immutable source bytes'
assert 'approved model' in pathlib.Path(sys.argv[4]).read_text()
for path in (sys.argv[5],'/proc/self/environ'):
    try:
        pathlib.Path(path).read_bytes()
    except PermissionError:
        pass
    else:
        raise AssertionError('foreign bytes remained readable')
for call in (lambda: pathlib.Path(sys.argv[5]).unlink(),
             lambda: pathlib.Path(sys.argv[6]).write_bytes(b'escape'),
             lambda: socket.socket(socket.AF_UNIX,socket.SOCK_STREAM),
             lambda: os.execve('/usr/bin/ffmpeg',['ffmpeg','-version'],{})):
    try:
        call()
    except PermissionError:
        pass
    else:
        raise AssertionError('ASR child expanded its capabilities')
completed=[]
thread=threading.Thread(target=lambda: completed.append(True))
thread.start()
thread.join()
assert completed==[True]
assert 'faster_whisper' not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", script, str(library), str(source), str(model),
         str(model_file), str(secret), str(absent)],
        capture_output=True, text=True, timeout=10,
        env={"PATH": "/usr/bin:/bin", "PYTHONDONTWRITEBYTECODE": "1"},
    )
    assert result.returncode == 0, result.stderr
    assert not absent.exists()
    assert secret.read_bytes() == b"acceptance-owned-credential-canary"


