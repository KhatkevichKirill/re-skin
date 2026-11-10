"""Offline subprocess implementation, invoked only by trusted assistant_asr."""
import ctypes
import json
import os
import resource
import socket
import sys


_POLICY_LIBRARY = "/usr/local/libexec/reskin-media-policy.so"


def restrict(*, audio_path, model_dir, parent_pid):
    from pathlib import Path
    memory=Path("/sys/fs/cgroup/memory.max")
    cpu=Path("/sys/fs/cgroup/cpu.max")
    cgroup_bounded=False
    if memory.exists() and cpu.exists():
        memory_limit=memory.read_text().strip()
        quota,period=cpu.read_text().strip().split()
        cgroup_bounded=(memory_limit!="max" and int(memory_limit)<=2*1024**3
                        and quota!="max" and int(quota)/int(period)<=2)
    if not cgroup_bounded:
        # Virtual-address fallback is conservative and may reject MKL allocation
        # even with low RSS. Never drop the memory bound to make a probe pass.
        resource.setrlimit(resource.RLIMIT_AS,(2*1024**3,2*1024**3))
    resource.setrlimit(resource.RLIMIT_CPU,(120,120))
    resource.setrlimit(resource.RLIMIT_FSIZE,(4*1024**2,4*1024**2))
    resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    _,descriptor_max=resource.getrlimit(resource.RLIMIT_NOFILE)
    descriptor_limit=64 if descriptor_max==resource.RLIM_INFINITY else min(64,descriptor_max)
    resource.setrlimit(resource.RLIMIT_NOFILE,(descriptor_limit,descriptor_limit))
    # The model child gets the decoded chunk and verified model only. In
    # particular, the sidecar's artifact and socket mounts are not capabilities
    # of this process. Install before importing any native media/model package.
    # The same native policy also denies exec, process inspection, async I/O,
    # network and filesystem mutation, and closes descriptors above stderr.
    # Its fixed administrator-owned path cannot be selected by the request.
    if (not isinstance(audio_path,str) or not isinstance(model_dir,str)
            or not Path(audio_path).is_absolute() or not Path(model_dir).is_absolute()
            or "\x00" in audio_path or "\x00" in model_dir
            or type(parent_pid) is not int or not 0 < parent_pid <= 2147483647):
        raise RuntimeError("ASR sandbox input is invalid")
    sys.dont_write_bytecode=True
    try:
        lib=ctypes.CDLL(_POLICY_LIBRARY,use_errno=True)
        install=lib.reskin_install_asr_policy
        install.argtypes=[ctypes.c_char_p,ctypes.c_char_p,ctypes.c_int]
        install.restype=ctypes.c_int
        if install(os.fsencode(audio_path),os.fsencode(model_dir),parent_pid) != 0:
            raise RuntimeError("ASR filesystem/syscall sandbox unavailable")
    except (OSError,AttributeError):
        raise RuntimeError("ASR filesystem/syscall sandbox unavailable") from None
    try:
        socket.socket()
    except PermissionError:
        pass
    else:
        raise RuntimeError("ASR network denial self-test failed")


def main():
    request=json.loads(sys.stdin.buffer.read(10000))
    if not isinstance(request,dict) or set(request)!={"audio_path","model_dir","language","parent_pid"}:
        raise RuntimeError("ASR request is invalid")
    restrict(audio_path=request["audio_path"],model_dir=request["model_dir"],parent_pid=request["parent_pid"])
    from faster_whisper import WhisperModel
    model=WhisperModel(request["model_dir"],device="cpu",compute_type="int8",cpu_threads=2,
                       num_workers=1,local_files_only=True)
    segments,info=model.transcribe(request["audio_path"],beam_size=3,word_timestamps=True,
        condition_on_previous_text=False,vad_filter=False,language=request.get("language"))
    words=[]
    for segment in segments:
        for word in segment.words or ():
            words.append({"text":word.word,"start_sec":word.start,"end_sec":word.end,"probability":word.probability})
            if len(words)>2000:
                raise RuntimeError("ASR word budget exceeded")
    print(json.dumps({"language":info.language,"language_probability":info.language_probability,
        "words":words,"network_denied":True,"max_rss_kib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}))


if __name__=="__main__":
    main()
