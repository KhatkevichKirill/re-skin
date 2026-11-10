# Run assistant editing: contracts and release decisions

Status: foundational proposal implemented for plan validation; runtime, identity,
database, UI and launch remain gated. Date: 2026-10-23. Tracking: Re-skin
AICP-261 and contracts child
AICP-262.

This document records the immutable editing boundary for a run-scoped assistant.
The initial implementation is only `app.assistant_contracts`: pure Pydantic
proposal models and source-manifest validation. It does not enqueue work, provide
authorization, invoke a model, render, deliver, or expose a production endpoint.
Scenario coverage, fixture requirements and proposed measurable release targets
are defined in [assistant-acceptance.md](assistant-acceptance.md).

## Evidence and reasons for a separate editing layer

Starting code revision: `42f6c3fbf55b37ad0e11934d6aef0eb96b831da8`.

| Existing component | Verified behavior | Design implication |
|---|---|---|
| [Run model](../backend/app/models.py) | Run owns mutable `result_local_path`, `result_gdrive_file_id`, segment snapshot and generation settings. No shared creative-version or chat entities exist. | Import current successful output into an immutable version; do not use the Run's current pointer as an execution input. |
| [Audio patch](../backend/app/pipeline_audio_patch.py), `validate_audio_patch_request`, `_deliver_with_retry`, `_promote` | Donor must be a sibling Run. Promotion replaces content behind the existing Drive ID and changes the Run result pointer. | Reuse the audio splice primitive, not this publication lifecycle. Cross-project donor use needs its own access boundary. |
| [Range attempt](../backend/app/pipeline_range_attempt.py), `_promote` | Candidate artifacts and phase claims are isolated, but successful promotion changes the Run timeline, segment and result. | Reuse claim/isolation patterns; publish an immutable child version instead of promoting into the parent Run. |
| [Caption pipeline](../backend/app/pipeline_captions.py), `_find_transcript_donor` | Transcript reuse matches decoded-audio fingerprint and language globally. Existing manual transcript handling avoids another transcription call. | Scope cache reuse to authorized sources. Preserve reusable transcripts and never overwrite a user's correction with fresh transcription. |
| [API](../backend/app/api_v2.py), resource helpers and `delete_run` | Resource helpers fetch by ID without application identity. Deletion removes the Run's full disk subtree. | Access enforcement and reference-aware asset retention are prerequisites for donor sharing and durable ancestry. |
| [Deployment](../docker-compose.yml) and [nginx](../deploy/nginx/nginx.conf) | Authentication is nginx Basic auth; API has no independent user identity. Redis has no configured persistent volume/AOF. API/workers receive production environment and Drive credentials. | Decide the audience/identity boundary; persist jobs in SQL, and isolate the assistant runtime from existing worker credentials. |
| [Provider client](../backend/app/kie_client.py), `_create_task_with_retry` | A paid POST retries transport failures and HTTP 5xx without a provider idempotency parameter. | A worker lease alone cannot prevent duplicate paid generation after an accepted request loses its response. |
| [Spend ledger](../backend/app/spend.py) | SQL reservations enforce admission caps; admitted retries are measured and can exceed the original reservation. | Reuse reservations, but define bounded assistant/provider retries and an operation-level cost policy explicitly. |
| [Public delivery](../backend/app/public.py) | Resource-purpose HMAC links have no expiry and Run result links resolve the mutable Run pointer. | New version delivery must bind immutable artifacts and the selected access policy. |

`version_id` on existing range/audio attempt rows is an optimistic concurrency
counter. It is not a creative version ID. The older wiki architecture page
contains historical v1 and migration details; current code and operational
[CLAUDE.md](../CLAUDE.md) govern this work.

## Decisions

### 1. Persist the conversation and its resolved context

Create one durable conversation for a Run. Persist ordered human/assistant
messages, validated plans, tool-call arguments and results, job links, terminal
events, actor attribution and context revision. Reopening the Run must restore
the conversation and outstanding work without depending on a browser tab or
an LLM subprocess session.

The server assembles each turn's context from the authorized project's source,
Run settings and references, selected version's frozen recipe/timeline,
version ancestry, previous messages, known transcripts/cues, prior job results
and relevant evidence. Keep large media outside the prompt: supply bounded
metadata/transcript excerpts and selected frames through controlled asset IDs.
Record the exact context manifest and base version used for the plan. Never
include credentials, arbitrary repository files or unrestricted filesystem paths.

### 2. Make every successful edit a new immutable version

Recommended persistent entities are:

| Entity | Minimum contract |
|---|---|
| CreativeVersion | Immutable UUID, Run ID, parent version UUID, monotonic per-Run sequence, creator/message, schema revision, frozen operation recipe, source asset IDs and frozen timeline/cue/style metadata. |
| VersionArtifact | Immutable asset UUID, owner/version UUID, media kind, storage key, content hash, probe metadata and availability. Separate preview and full-result artifacts. |
| Conversation/Message/Event | Run scope, ordered durable sequence, actor, content, context revision, plan/job links; no credentials. |
| EditJob/PhaseAttempt | Version/plan reference, lifecycle, cancellation request, claims, phase checkpoints, error class, bounded retry count and terminal event. |
| ProviderCall | Stable logical request ID, provider capability/idempotency data, committed submit intent, task ID, ambiguity state and estimated/actual cost. |
| Outbox/DeliveryIntent | Committed work request, dispatch state, stable external-delivery identity and reconciliation result. |

Do not mutate an existing version's recipe or replace bytes behind an existing
version's local/Drive artifact. Lifecycle and availability records are mutable
execution metadata; completed artifact content and semantic provenance are not.
Upload each delivered version as a new Drive file. Sheet export identity must
identify that version rather than silently repointing an already exported Run.

An explicit operator publication is a separate action from editing or version
delivery. As requested on2026-10-27, **Use as Run result** may replace the main
Run pointer and the content of its existing Drive file, retaining that link and
Run export identity. It must first preserve the actual previous main result,
retain all immutable version artifacts, and append a publication receipt. Neither
rendering, choosing an editing base nor a Claude tool may publish implicitly.
See the [Run publication contract](assistant-run-result.md) for durable intent
recovery, concurrency, history and downstream regeneration limits. A Run without
an existing Drive result is published locally only; this action does not create
a version delivery or a new sheet row.

Import V1 from the currently available successful Run result with its hash and
known settings. Historical in-place Drive replacements cannot be reconstructed
from the current Run row. Label incomplete legacy provenance honestly; do not
invent earlier versions. Copy/pin bytes that old endpoints may still overwrite.
Legacy in-place mutation paths must be isolated from version-owned artifacts
before the assistant can execute edits.

Keep source and donor artifacts alive while referenced by versions or jobs.
Deletion becomes soft deletion/reference-aware retention, followed by separately
authorized garbage collection. A source Run deletion must not remove a donor
already pinned into an accepted edit.

### 3. Resolve the base before planning or enqueueing

The visible default is the latest successful version; failed, pending and
cancelled work never becomes the default. Resolve `latest` server-side and
persist the explicit `base_version_id` at command acceptance. Planning and
execution use that ID throughout. A newer result arriving later must not change
the accepted command's source.

Selecting an earlier version creates a child of that version. Allocate version
sequences under a Run lock and define latest as the highest successful sequence,
so an older slow job cannot displace a newer successful version on late finish.
UI selection is separate from the latest-version pointer. A base conflict or
unavailable pinned source is a clear failure requiring reselection, not an
automatic retry against a different version.

A successful media version requires a validated durable full output plus
preview/inspection metadata. External delivery has a separately visible state:
retrying an upload must not repeat a paid generation or alter output content.
The UI must show which version a preview, full result and delivery link refer to.

### 4. Use typed, restricted edit proposals

The planner emits `EditPlan(schema_version=1, run_id, base_version_id,
operations)`. All IDs are canonical UUID strings. Every object rejects unknown
fields, and models are frozen with tuple collections. Proposal objects accept no
filesystem path, URL, executable filter, command, environment override or model
selection. Storage paths exist only after trusted resolution outside this module.

All intervals are half-open `[start_sec, end_sec)` on the pinned base timeline.
Times must be finite numbers, are bounded to 24 hours, and cannot be booleans or
numeric strings. A plan has 1–32 operations. These limits are an initial contract,
not evidence that the VPS can safely render every allowed duration; runtime
admission must impose its measured resource limits.

| Kind / Python model | Semantics and limits |
|---|---|
| `trim` / TrimOperation | Keep the selected interval, shortening picture and audio together. |
| `image_fade` / ImageFadeOperation | Fade picture to/from black over the interval; direction is `in` or `out`; preserve audio and total duration. Burned-in text is part of the picture and can fade too. |
| `cut` / CutOperation | Remove the selected interval from picture and audio; output must remain nonempty. |
| `replace_footage` / ReplaceFootageOperation | Replace picture with an equal-duration window starting at `donor_start_sec` from `source=AssetReference(asset_id, version_id)`; preserve the base audio. Video source required. |
| `audio_splice` / AudioSpliceOperation | Replace mixed soundtrack over the selected interval with an equal-duration donor window; copy picture. No stretch, alignment guarantee, speech generation or stem separation. Crossfade is finite, at most 2 seconds and at most half the window. |
| `full_track_audio` / FullTrackAudioOperation | Replace the complete soundtrack from a donor offset. Donor must cover the full base duration. Shape/duration validity is not evidence of acoustic or lip-sync suitability. |
| `subtitle_edit` / SubtitleEditOperation | Up to 500 explicit `update` or `delete` changes to existing editable cue IDs. Update supplies full text and times. IDs are unique per plan. No hidden transcription, style change or burned-text reconstruction. |

Trim and cut must initially be the only operation in their plan; additional
edits operate on the resulting child version. This avoids implicit timeline
remapping. Overlapping operations on the same picture/audio channel are
rejected. Adjacent intervals and simultaneous picture/audio edits are allowed.
Subtitle changes must be grouped into one operation. Rendering order and the
clean picture/subtitle composition model must be explicit in runtime adapters.
Contract vocabulary is broader than any individual renderer's capabilities:
runtime must reject an unsupported kind, fade direction, interval or operation
mix before enqueueing, with a specific limitation. The initial ending-fade
adapter may support only fade-out ending at the video's end; it must not silently
map a middle fade or fade-in to that behavior. Middle-artifact concealment remains
required feature work tracked by the acceptance contract, not an implemented
capability or a dropped requirement.

The initial local [media primitives](../backend/app/assistant_media.py) are a
separate foundation, not assistant dispatch. They currently expose these
capabilities; the runtime capability registry must preserve their narrower
limits until implementation and verification support more:

| Primitive | Current supported subset |
|---|---|
| `image_only_fade` | Ending fade-out only, with `end_sec` at picture duration; no fade-in or middle concealment. |
| `trim_video` | Frame-aligned kept interval. |
| `cut_section` | Strictly internal frame-aligned interval; edge removal must be proposed explicitly as trim. |
| `replace_footage` | Frame-aligned picture replacement; matching donor geometry/frame rate/SAR; preserve base audio. |
| Audio and subtitle operations | No new assistant adapter yet. Existing audio splice and caption primitives require the immutable orchestration, suitability and subtitle/style work described here. |

These local primitives require normalized zero-start CFR H.264/yuv420p MP4,
with at most one zero-start AAC audio track spanning picture within 50 ms;
unsupported tracks, rotation and VFR are rejected. They are not a capability
claim for arbitrary uploaded video. Their tests/evidence remain separate from
the pure contract checks below.

Subtitle text is inert data, including shell-looking words. Reject control
characters and raw ASS override blocks; the trusted renderer must still escape
ASS/HTML text and must never interpolate captions into commands. These checks
do not turn arbitrary text into executable authority.

`validate_plan_sources` checks a plan against trusted `VersionManifest` and
`AssetManifest` objects: exact base version/Run binding, successful available
base, interval bounds, nonempty cut, editable cue membership, source asset/version
binding, source availability/media/audio capability and donor duration coverage.
It performs no I/O and grants no access. Never accept caller-supplied manifests
as proof. The caller must authorize, verify hashes/probe data, pin bytes, check
resources/policy and revalidate immediately before execution.

Explicit limits of this foundation: no semantic donor ranking, word alignment,
subtitle style snapshots, overlapping cue repair, full-track lip-sync evidence,
codec/frame/sample rounding or background preservation policy is implemented.
Burned-only subtitle correction is blocked unless a clean source or a separately
validated removal workflow exists. A correct schema is not permission to render.

### 5. Persist execution and distinguish retry from new generation

SQL is the job source of truth; use a transactional outbox so DB commit and
Redis enqueue cannot silently diverge. Reconstruct queue work from durable
checkpoints after Redis loss. Use leased/fenced phase ownership and
attempt-specific staging directories; verify ownership before publishing a
complete artifact by atomic rename. Recovery must not override cancellation or
publish a stale claim's output.

Accept a stable client/operation idempotency key and reject mismatched reuse.
An identical retried request returns the same job/version identity. Retrying a
render/download/upload resumes from checkpoints; a new generation request is
an explicit new paid attempt with policy admission and visible provenance.

Persist submit intent before any paid call. Use provider idempotency and
lookup-by-request when available. Otherwise, an accepted request whose response
is lost enters `needs_reconciliation`; do not retry the paid POST automatically.
Exactly-once billing cannot be promised without provider cooperation. The
generic retry decorator on current `createTask` is unsuitable for this boundary
and must be adapted before any assistant-paid regeneration is enabled.

Persist cancellation requests and check them before paid calls, between phases
and under the final publication transaction. Terminate local process groups;
cancel remote jobs only where supported. Remote work already submitted may still
incur cost. Late results can be retained for reconciliation but must not publish
as a cancelled job or become the default version. Emit one durable terminal
result per job, with cancellation and already-incurred costs visible.

Use the existing spend reservation/draw machinery for provider work, plus
assistant usage/admission accounting. Set operation cost estimates, maximum
attempts, budget owner and quota wait behavior. No automatic API fallback,
purchased credits, new model, speech synthesis, video regeneration or whole-run
regeneration is implied by a deterministic edit request.

### 6. Restrict donor search and tools on the VPS

Search only completed accessible version/artifact metadata and authorized
transcripts. Return provenance and exact source version/interval, not an
unrestricted local path. Apply the same scope to cached transcript reuse,
direct-ID reads, previews and workers. Recheck donor access when accepting and
executing an edit. Pin/copy selected donor bytes before queued work depends on
them. Cross-project use is allowed only through this boundary.

Text match alone cannot certify pronunciation, speaker identity, delivery,
background or timing. Return insufficient-evidence/no-suitable-donor outcomes
without generating replacement speech. For mixed-track splices, disclose that
music and ambience in the replaced window also change; do not claim stem
preservation without actual stems or a separately validated process.

The assistant runtime receives sanitized job snapshots and bounded job files.
Its broker exposes narrow tools such as inspect authorized asset metadata,
retrieve selected transcript/frame evidence, propose a plan and inspect job
status. Backend code owns validation, authorization, budget admission, enqueue,
render and publication. No arbitrary shell/filter execution, production
repository, Docker socket, root home, developer MCP servers, Drive/provider
credentials or broad shared-media mount belongs in the planner environment.
The production worker's existing environment is not a safe planner environment.

## Unresolved release gates

### Confirmed audience and remaining access implementation

The operator selected a private beta on2026-10-23 using the existing dashboard
login and a two-project allowlist. On2026-10-27 the operator expanded this scope
to all current and future dashboard projects through the trusted explicit
`ASSISTANT_ALL_PROJECTS=true` switch. See [scope contract](assistant-access-scope.md).
This remains a single trusted operator scope, without named-user isolation.
Absent/false all mode retains strict project allowlisting; empty/malformed lists
deny access. Malformed nonempty CSV also denies all mode. Apply the same policy
to targets/donors, caches, preview delivery, admission and execution while
retaining exact ownership/existence checks. Named-user/public rollout requires
a separate identity design. A pure manifest validator is not this boundary.

New artifact preview/delivery grants must follow that decision and bind exact
immutable assets. Existing public links retain their current legacy contract
until an explicit migration/rotation decision; silently reusing them for new
version ACLs is insufficient.

### Assistant runtime and account provisioning

The operator subsequently requested evaluation of an existing corporate Claude
account as the application runtime. Claude Code/Agent SDK is now the leading
candidate to qualify; Codex CLI with subscription authentication remains an
alternative. Sol-6.1 remains the development and review agent model. No runtime
has yet been adopted or measured as an application dependency. Verify the actual
corporate plan, permitted single-operator use, admin settings and billing before
implementation depends on them. Do not infer suitability from a corporate label.
Use an independently provisioned app-only login/auth
store under an approved existing subscription if permitted; a separate non-human
account is not a prerequisite. Do not copy or concurrently reuse the developer's
credential cache. Establish approved account ownership, permitted audience,
model entitlement, quota, refreshed-cache persistence, one serialized owner per
auth store and an operator reauthentication runbook.

Do not infer VPS account access to a model from the model running this review.
Verify the intended model from the actual service identity/image, with no silent
substitution. Prove bounded schema-valid output, latency/capacity, restart,
expired/revoked auth handling, resource limits, process-group cancellation and
denial of credential/other-job files and unapproved network/MCP/plugin access.
A read-only filesystem sandbox alone does not protect readable secrets.

Current provider-specific evidence is in
[runtime qualification](assistant-runtime-qualification.md). The original Codex
candidate's official entry points are
[non-interactive execution](https://learn.chatgpt.com/docs/non-interactive-mode),
[authentication](https://learn.chatgpt.com/docs/auth),
[auth ownership/refresh](https://learn.chatgpt.com/docs/auth/ci-cd-auth) and
[security](https://learn.chatgpt.com/docs/agent-approvals-security).
These links are context for the required spike, not live entitlement or capacity
verification by this contracts task.

### Integration, verification and rollout

Before enabling execution: implement migrations/V1 pinning, identity scope,
durable jobs/outbox/cancellation, storage retention, trusted resolvers, restricted
media adapters and truthful preview/full-result delivery. Validate all journeys
in [assistant-acceptance.md](assistant-acceptance.md), including no suitable donor,
earlier-version branching, burned-only subtitles, CTA-crossing edits, lost paid
responses, two workers, cancellation during finalization and Redis loss.

Require PostgreSQL concurrency/crash evidence, real-FFmpeg media invariants,
browser flows, independent repository-mandated design/code review and an isolated
migration/restore rehearsal. Rollout remains a separately authorized action under
the existing backup, migration, worker drain, rebuild and HTTPS smoke-test rules
in [CLAUDE.md](../CLAUDE.md). Completing contracts does not authorize or establish
production readiness.
A narrower pilot must state its supported operations and audience; it cannot
claim completion of the master while one of its required workflows is unsupported.

## Foundational validation evidence

Targeted test environment: host Python 3.10.12, Pydantic 2.13.4, pytest 7.4.3.
The production dependency lock specifies Pydantic 2.14.1; tests on this host are
not evidence of production-lock compatibility or a production-wide green suite.
Run `cd backend && PYTHONPATH=. python3 -m pytest tests/test_assistant_contracts.py -q`
for the pure contract checks. The acceptance document tracks integration/media
evidence separately.
