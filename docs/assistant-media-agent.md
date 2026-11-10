# Video inspection and candidate review

Tracking: AICP-276, with
runtime AICP-277, media tools AICP-278, durable loop AICP-279 and rollout AICP-280.

`ASSISTANT_AGENT_ENABLED=true` admits new requests as workflow revision 2.
Existing revision 1 turns retain their original semantics. Apply migration
`i9d4e1f6a803` before enabling the flag or deploying the new ORM models.

The application owns the loop. Claude returns one typed action per invocation:
`inspect_frames`, `transcribe_window`, `search_sources`, `render_candidate`, or
`final`. Read-only QA actions (`detect_captions`, `check_alignment`, `compare_frames`,
`script_captions`) and the agent-only `caption_lines` reply are described in
[assistant-qa-tools.md](assistant-qa-tools.md) (AICP-345). Its native shell, filesystem tools, MCP and connectors remain disabled.
Frame JPEGs are delivered as actual image content blocks. The pinned Claude Code
2.1.309 requires both stream-json input and stream-json output for multimodal
input; only a validated successful terminal event can authorize the next action.
Changing this adapter requires independent runtime qualification.

The supervisor reauthorizes the pinned request and all discovered sources before
each inference and media action. Each invocation has its own durable UUID and
exact context fingerprint. Input/result records and image evidence are immutable;
repeated frames are reconstructed from earlier evidence, with hash verification,
rather than copied into every SQL snapshot. Unknown invocation outcomes are
fenced/cancelled using the same UUID. They never automatically start fresh calls.

The original project input is available through source discovery. The trusted
importer resolves the Run's project, checks its allowlist, and pins its real input
as a separate artifact. It does not create or select a video version. Model
requests cannot name a filesystem path. Imported originals retain their own
duration/timeline; these can differ from the generated Run.

Rendering candidates shares the accepted executor's media and subtitle handling.
The turn must inspect frames of its pinned selected asset/version before rendering.
A premature render returns `source_frames_required`; the agent can inspect that
source and request the render again within the existing 12 decisions and three
render attempts. Refused renders consume attempts. Operations ending at the video
end must copy `selected.duration_sec` exactly, without rounding up or recomputing
it from frame counts; strict source-duration validation remains in force. The
explicit `extend_ending` operation below is the only exception: its output end
must exceed the selected picture duration.
Candidates cover the full proposed output timeline; a cut or trim therefore
changes candidate coordinates. They cannot publish a version or advance selection.
The final candidate must match the exact proposed plan and must have been supplied
back to Claude as image evidence. Audio changes additionally require bounded
candidate transcription coverage. These checks establish inspected evidence, not
proof of perceptual correctness or proper pronunciation.

The selected final candidate is retained as content-addressed bytes. Its private
storage reference stays on the final step; the authenticated candidate route
checks the entire turn's source dependencies before delivery. The UI shows a trial
player and analysis steps. Applying the proposal uses normal admission and renders
a new immutable version. Applying a subset does not inherit the complete plan's
inspection status. Custom requests remain bound to the original proposal.

Fresh agent turns reserve tool-observation headroom by limiting initial context
to 32 KiB of expanded UTF-8 JSON. Older whole dialogue entries may be omitted;
the current request, pinned context and retained message provenance stay intact,
and stored conversation history is unchanged. Essential context above that limit
is refused with a fixed 422 response. This reserve does not guarantee every tool
combination fits the existing 64 KiB aggregate text budget. Existing pinned retries
retain their original context and do not automatically receive this reserve.

Limits are bounded: 12 model decisions, 3 candidate attempts, 3 speech windows,
2 source searches, 32 requested frames and 8 MiB of frame bytes per turn; model
requests retain at most 32 frames and 2 MiB decoded image data. Fresh requested
frames take priority, with remaining space balanced across source comparisons and
candidate inspection. The loop retains the full bounded evidence history so
separate stateless model invocations can compare earlier and later passages. Use 3–4 frames per source for
initial comparisons, leaving capacity to inspect the edited candidate. Sparse frames
do not establish complete video coverage. Tools admit video up to 120 seconds,
1920 pixels on the long edge and 2,073,600 pixels, with 512 MiB scratch space.
The loop has a 1,200-second deadline and individual media operations at most 600
seconds. Completed evidence survives failure; temporary candidates are discarded,
while the selected final candidate remains available for operator review.

Whisper runs locally through the existing offline sidecar. Its words, confidence
and estimated timing can locate speech and expose missing text. They do not verify
speaker identity, pronunciation, background continuity or lip synchronization.
Unsupported repairs produce limitations or alternatives; the agent must not invent
missing speech or silently invoke generation. Subscription exhaustion stops work.
The exact two-project beta scope and included Claude Team usage policy still apply.

## Bounded ending extension

Tracking: AICP-290.
This capability adds a trusted local operation, not general CLI access.

`extend_ending` replaces an existing suffix and extends the output. It must be
the sole operation in a plan. Its fields are `start_sec`, `end_sec`, a top-level
authorized audio `source`, `donor_start_sec`, `visual`, and `fade_out_sec`.
`visual` is either `{kind: "black"}` or
`{kind: "freeze_selected", timestamp_sec: ...}`. No path, command, filter, URL,
provider, or executable is accepted from the model.

The operation preserves a nonempty prefix: `0 < start < base_duration < end`.
The entire replaced/added suffix and the added duration are each at most ten
seconds; total output is at most 120 seconds. Picture boundaries must align to
the verified frame grid. A frozen frame must precede the replacement start.
A nonzero fade is allowed only with a frozen picture, lasts at least two frames
and at most two seconds, and fits inside the ending. Other operations retain
their original bounds and composition rules.

The donor supplies the entire `[start,end)` soundtrack at normal speed, including
speech, music and ambience. Its actual audio must cover the required sample
window. There is no donor padding, tempo fitting, synthesis or paid fallback.
Frame and audio boundaries are resolved once to frame/sample positions; output
timing and input hashes are checked. Picture and AAC audio are re-encoded, so
prefix content and timing are preserved without promising identical encoded
bytes. Original and previous version files remain immutable.

With verified editable captions, rendering uses clean picture and retains only
caption portions before `start`. Replaced ending cues are removed and new donor
captions are not invented. The extended clean picture and clipped cue track are
retained for subsequent edits. Retained cue ends are rounded down to ASS
centiseconds at the replacement boundary, which can omit less than 0.01 seconds
of caption display. Fragments
that collapse at this precision are rejected. The stored timing preserves this
bound during later subtitle edits and reburns. The clean artifact records the
precise MP4 audio timing profile; subtitle-only children retain it so their full
and clean versions remain compatible for further edits. A newly rendered clean
picture receives its own profile. Unknown burned text remains in
frozen pixels; black hides the whole replaced picture, including its text. The proposal UI
discloses these effects before acceptance.

Extension proposals require the candidate-capable workflow and exact durable
candidate/plan binding. Inspection must cover the selected ending and any frozen
source frame, the candidate seam, added time beyond the old end, and the new
ending/fade. Donor and candidate transcription must cover the changed speech
windows. These are bounded observations, not pronunciation, speaker identity,
lip-sync or complete visual QA guarantees.

The saved preview shows the edited ending only. The original cannot supply a
same-length comparison beyond its old end; compare it separately with the new
result. The full candidate and full immutable result retain the extended ending.
The existing sandbox, source authorization, cancellation/resource limits,
two-project allowlist and subscription policy are unchanged.

## Internal context actions and execution confinement

`list_projects`, `list_runs`, `list_versions` and `list_assets` call a finite
internal application service, not a general HTTP client. They return typed IDs,
ownership and approved scalar metadata only. Each call rechecks the pinned target
and every returned owner; inventory does not grant permission to use media.
There are at most four reads per turn, twenty rows and 8 KiB per page, and 24 KiB
of inventory observations within the existing 64 KiB text budget. Cursors are
bound to a preceding page of the same resource and owner. No implicit original
import, probe, generation, export or administrative write occurs during a read.

Metadata ownership dependencies remain in durable context. They are reauthorized
before another inference, tool execution, final result, public history or use in a
later conversation turn. This also covers observations in failed/cancelled turns
and transitive dialogue history. Revocation therefore cannot be bypassed by
reusing an earlier metadata response. The beta still uses the shared operator
login and the exact configured two-project allowlist, not per-user membership.

Public history assembles each turn's sanitized analysis once. Within that single
analysis inspection, successful byte hashes may be reused while live ownership,
status and project policy are checked again. Every reuse reopens the object with
`O_NOFOLLOW` and compares its descriptor's device, inode, size, nanosecond mtime
and ctime against the file actually hashed, bound to its expected key/hash/size.
Objects with ctime less than one second before verification starts, or in the
future, always take the full hash path to avoid same-tick timestamp collisions.
This cache ends with the inspection; it is not shared across turns or requests.

All assistant FFmpeg, FFprobe and font discovery calls use the mandatory native
launcher in `backend/sandbox/media_exec.c`. Trusted adapters explicitly declare
input files and output roles. Landlock allows exact input files read-only, fixed
public runtime libraries/fonts, and only precreated private output files for
writing. It grants no directory creation or arbitrary output-parent access.
The Python bridge publishes successful outputs with exclusive hard links;
failures discard staging files and cannot overwrite an earlier result.

The default-deny seccomp policy compensates for Landlock ABI1 gaps, including
read-only truncating opens and path metadata mutation. It denies sockets, async
I/O submission, process inspection, additional processes and namespace changes.
Only threads and self-directed thread signals are permitted. The launcher clears
environment variables, closes inherited descriptors, enforces resource limits,
and dies if its supervising process disappears. Cancellation terminates and reaps
the process group. Missing or unsupported confinement fails closed; there is no
direct-execution fallback. Raw subprocess diagnostics never reach tool results.

Whisper installs the same filesystem/syscall policy before importing native
model/decoder packages, with read-only access to its decoded audio chunk, approved
model and runtime packages. It cannot read the sidecar's other artifact objects
or socket directory. It has no execution or file-write grants. Existing outer
network-none, 2 GiB/2 CPU and offline model restrictions remain in place.

The producer validates JPEG pixels through confined FFmpeg. The supervisor and
Claude broker use bounded pure-Python envelope/dimension checks rather than
Pillow decoding. Claude's vendor client still processes image content; its
existing credential-isolated container and disabled native tools remain required.
Landlock ABI1 does not hide all filesystem metadata such as `stat` results. The
application's authorization and immutable artifact store remain trusted, as do
the kernel and fixed runtime; this is not a claim that every decoder or kernel
vulnerability is eliminated.

Release requires production-lock regression and PostgreSQL migration checks,
independent Sol-6.1 CLI review, real inline-image qualification and the complete
Run 7c29c40f analysis/candidate/acceptance journey. Record current evidence in Linear.
Follow the existing verified Drive backup, migration-before-deploy, two-worker,
overlay and nginx restart procedure in `deploy/assistant/README.md`. Rollback
disables new agent admission and preserves additive history and pinned media.
