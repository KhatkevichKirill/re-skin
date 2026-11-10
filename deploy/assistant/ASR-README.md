# Offline word localization

The optional ASR sidecar transcribes existing authorized immutable audio. It
does not synthesize speech, contact a provider, or verify pronunciation/speaker
identity. Word timestamps and confidence are estimates for listening review.
No ASR dependency is added to the API/media worker production requirements.

Model: `Systran/faster-whisper-small`, revision
`536b0662742c02347bc0e980a01041f333bce120`. The model is about 486 MB. Its exact
file sizes/SHA256 are in `asr-small-manifest.json`; the published LFS weights
hash is independently fixed in `scripts/download_assistant_asr.py`.
`backend/requirements-asr-cp312.lock` pins every binary dependency with SHA256.
PyAV 14.2 is intentional: the newly resolved PyAV 19 rejects the `metadata_errors`
argument used by faster-whisper 1.2.1.

The standalone image accepts only a private Unix socket. Mount the model and
`assistant_artifacts` root read-only, and a private socket directory owned by
UID/GID 10001 read/write. Start with `--network none --read-only --cap-drop ALL
--security-opt no-new-privileges --memory 2g --memory-swap 2g --cpus 2
--pids-limit 128`, and a 64 MB temporary filesystem. Do not mount production
environment files, credentials, Docker socket, root home, or general media.
The supervisor must run the client as UID 10001 and share only the private socket
directory. The sidecar serializes requests and checks Unix peer UID 10001.

The trusted backend first authorizes the target/donor with
`transcribe_authorized_artifact`. The sidecar request carries only immutable
storage key, SHA256/size, explicit start/end, and optional language. The source
and model hashes are verified; the chunk must be within the actual source and
at most 60 seconds. Sources longer than 60 seconds require explicit chunk choices;
search reports state incomplete coverage. Raw paths, URLs, commands, environment
overrides and arbitrary model selection are rejected. The response includes
coverage, estimated words/confidence and sandbox/resource evidence. No raw
transcript is logged by the service.

The control process launches a clean-environment child. It uses CPU int8,
2 threads/1 worker, a 120 second deadline, CPU/file/output limits, and kernel seccomp
denial of socket/connect/send syscalls. An actual socket-denial self-test must
pass. Client disconnect/cancellation stops the child; singleton locking permits
safe restart after a stale socket. If the intended cgroup limits are absent,
the child retains a 2 GiB virtual-address fallback, which currently fails MKL
allocation and returns unavailable. Do not remove memory limits to make it run.

Build a context containing `wheels/` downloaded for CPython 3.12 x86_64 using the
hash lock, the lock as `requirements-asr-cp312.lock`, the manifest, the worker
script, and `asrapp/` containing the reviewed assistant ASR/media/audio/subtitle/
contract modules. Build `asr.Dockerfile` with network disabled and an explicitly
reviewed immutable `ASR_BASE_IMAGE` built from the current application Dockerfile.
The prepared context includes the expected native sandbox source digest; the
ASR build refuses a base without the matching helper, shared policy library and
`media-source.sha256`. Historical bases below are evidence only and cannot be
used for the confined runtime. The original development qualification used the
existing base image ID
`sha256:7c3fdcbe368a264af5c5321e74655bb96abb46762a3c14b7a291e1b710394d3b`
under an isolated test-only local tag. Model bytes/wheels are not committed.
`scripts/prepare_assistant_asr_build.py WHEELS FRESH_DESTINATION` verifies the
complete wheel set against the lock and creates that minimal context. Download
wheels in a clean Python3.12 environment with `pip download --only-binary=:all:
--require-hashes -r backend/requirements-asr-cp312.lock --dest WHEELS`.

Development qualification on 2026-10-23 recognized synthetic Flite speech with
word timestamps under 2 GiB/2 CPU/network-none limits, seccomp denial passed, and
peak child RSS was 765372 KiB. This is compatibility/resource evidence for that
short synthetic input, not a multilingual quality or production-capacity claim.
The local virtual-address-only attempt failed MKL allocation; the first image
with PyAV 19 failed its decoding API check. Neither failure was treated as a
successful transcription. Live deployment/activation remains separately owned.

The complete UID10001 Unix RPC path also passed with disposable authorized
media copies under the same container limits. The qualified sidecar image ID
was `sha256:9301030d76c4c3b9e8524e85e68674de5252a2a811e1bdfa1782bf4358bd06d2`.
Only sanitized counts and resource summaries were recorded:

| Input | Covered audio | Detected language | Words | Words with probability below 0.6 | Elapsed | Peak child RSS |
|---|---:|---|---:|---:|---:|---:|
| Synthetic Flite | 3.690s | English | 11 | 1 | 20.31s | 764288KiB |
| Authorized caption copy | 29.728s | French | 87 | 11 | 25.03s | 765188KiB |
| Authorized ed75 fixture | 29.728s | Polish | 71 | 10 | 26.73s | 781416KiB |
| Authorized faff fixture | 29.728s | French | 87 | 11 | 24.19s | 765600KiB |

All four responses carried full requested source coverage and successful
network-denial evidence. Language/content quality was not manually accepted;
confidence is an estimate, and low-confidence words require review. The
disposable sidecar was stopped after qualification. No production service or
credential was accessed. Four fake-model Unix tests separately verify peer UID,
path/schema/integrity rejection, disconnect cancellation and subsequent reuse.

Primary references: [faster-whisper API and CPU benchmark](https://github.com/SYSTRAN/faster-whisper),
[pinned model revision](https://huggingface.co/Systran/faster-whisper-small/tree/536b0662742c02347bc0e980a01041f333bce120).
