# Private operator assistant: runtime qualification and rollout

Tracking: AICP-263,
AICP-273.
This is the preparation and maintenance runbook; current deployment and pilot
verification evidence is recorded in AICP-273. The private two-project beta was
first deployed on 2026-10-23 from reviewed code `2e51372` (merge `2682caa`). Read
[`CLAUDE.md`](../../CLAUDE.md), the
[runtime qualification record](../../docs/assistant-runtime-qualification.md)
and [acceptance corpus](../../docs/assistant-acceptance.md) first.

## Applying a proposal

The chat lists proposed changes by number. A single operation uses **Render this
edit**. For multiple operations, **Apply all changes** submits the complete
proposal; **Apply only change N** submits only that change. The chosen
operations are pinned in the accepted job, shown in conversation history, and
retained by **Retry edit**. A repeated acceptance cannot change the choice or
create another job. **Custom option** prepares a follow-up request on the
proposal's original base version; the operator edits and sends it normally.

Compatible picture, audio and subtitle changes are processed together and
published as one immutable version only after all selected work succeeds.
Trim and cut change the timeline and currently require their own proposal;
continue editing their result to combine them with other work. Each fade is
limited to its stated interval: picture outside it and the soundtrack remain
unchanged apart from disclosed encoding effects. Adjacent fade-out/fade-in
changes produce a dip to black; choosing either alone restores the original
picture immediately outside that selected interval.

The local capability validator is shared by planning completion and admission.
Media and source compatibility are checked again against trusted probes before
rendering. Unsupported requests show an explanation; inline action errors remain
visible through polling. Long multi-change previews identify truncated coverage;
the full result includes every accepted change. Audio comparisons play the same
bounded source window before and after, so the delivered comparison duration is
twice that window.

## Publishing a chat version as the Run result

Choose a completed version under **Continue editing from**, review its full
result, then use **Use as Run result**. This explicit operator action changes
the main Run player/download and updates the content behind an existing Run
Drive link. If no Drive result exists, publication is local-only and is labeled
as such. Choosing a version or rendering an edit alone does not publish it.

Previous main results remain downloadable through publication history; every
chat version and original artifact is retained. The selected full file is copied
without reencoding into a new Run delivery path. Main/public result links carry
a new revision and revalidate, although previously cached old immutable public
URLs cannot be evicted from remote clients.

An interrupted request keeps its exact saved identity. Use its explicit retry
to reconcile the same version; it never switches to a newly selected version.
An uncertain Drive transfer blocks other Run replacements until resolved. An
expired/unqueryable upload session requires manual reconciliation, not a new
upload session. Session URLs remain private runtime state.

After a successful publication, generate changed segments in a copied/new Run.
The original segment pipeline cannot reconstruct later chat edits. Create fresh
direct Run-source derived jobs instead of retrying/rerendering retained ones;
their existing outputs and downloads remain history. Active processing and
unfinished direct Run-source overlay drafts block publication.

The exact contract and verification requirements are in
[assistant-run-result.md](../../docs/assistant-run-result.md), tracked by
AICP-305.

## Confirmed audience and scope

One private operator, using the existing nginx dashboard login. On2026-10-27
the operator explicitly requested all current and future dashboard projects
after identifying a Run outside the initial two-project pilot. Enable that
reviewed scope through trusted deployment configuration:

```text
ASSISTANT_ALL_PROJECTS=true
```

This switch is shared by UI, API, source discovery and supervisor execution.
With the switch absent or false, only `ASSISTANT_PROJECT_IDS` is available;
an empty/malformed list denies access. A malformed nonempty list also denies
all mode. Empty list plus explicit true is valid. Retain the original valid
CSV as this deployment's restricted fallback:

```text
f880254c-bbfe-4a2d-83ec-324d7e90cbee,dca4f5da-dfe7-4822-855a-4468e9ad3433
```

These values are deployment configuration, not permissive code defaults.
Canonical IDs, actual ownership/existence, successful media and bounded queries
still apply in both modes. Sharing the dashboard password does not establish
per-user permissions; expanding the audience requires a new access design.
See [scope contract](../../docs/assistant-access-scope.md) and
AICP-309 for rollout evidence.

The operator confirmed **included subscription usage only; stop at the limit**
and will complete the separate corporate Claude login personally. Separately
billed usage and API fallback are not authorized.

## Prepare the Claude runner

The qualified installation is Claude Code **2.1.309**, using the operator-completed
separate app login and explicit model **claude-sonnet-5-5**. The reported plan is
Team; the operator confirmed extra usage disabled. No developer account cache
is copied or mounted into the application. Development/review remains **Sol-6.1**.

Provision an unprivileged OS identity dedicated to this runner, with an app-only
configuration directory and an empty working directory. Install a verified copy
of the selected official Claude binary in an administrator-owned executable
directory. The runner must not write its executable, service definition or
admission policy. Do not run it in either repository checkout or an existing
developer home directory. Do not grant it the Docker socket, production `.env`,
database/Redis credentials, Drive credentials, developer home directories or
the shared media tree. It receives authorized, bounded context over stdin.

The deployment layout to provision is:

```text
/opt/reskin-assistant/bin/claude       administrator-owned, selected version
/var/lib/reskin-assistant/config/     app-owned, mode 0700, persistent
/var/lib/reskin-assistant/work/       app-owned, mode 0700, empty
```

After that layout and the `reskin-assistant` OS identity exist, the named login
owner completes Anthropic's supported login directly in a VPS terminal:

```sh
cd /var/lib/reskin-assistant/work
runuser -u reskin-assistant -- env -i PATH=/usr/bin:/bin LANG=C.UTF-8 \
  CLAUDE_CONFIG_DIR=/var/lib/reskin-assistant/config \
  /opt/reskin-assistant/bin/claude auth login --claudeai
```

Any browser authorization code belongs only in that terminal's native login
prompt. Do not put it into chat, Linear, source files or logs. Record only
non-secret authentication mode, reported plan and CLI version. The login command
does not grant permission to exceed subscription limits or activate the feature.
Serialize runner requests that share this configuration directory. Preserve
updated authentication state across restarts; use this same supported login flow
for renewal, without restoring old credential snapshots.

[`provision-login.sh`](provision-login.sh) prepares this layout from the verified
installed 2.1.309 executable. It refuses replacement of different binary bytes,
does not copy credentials, and starts no service. Provisioning alone does not
qualify inference or satisfy the deployment isolation requirements below.

## Qualify the process boundary

Use the tool-free planner adapter with an explicit binary, model, config
directory and empty working directory. It must run under the dedicated identity,
with a minimal environment and bounded input, output, wall time and subprocess
cleanup. No shell, session resumption, inherited tools or paid fallback is part
of this initial adapter. Its structured proposal is untrusted: backend admission
must independently validate versions, artifacts, scope and supported operations.

The candidate uses `--safe-mode`, `--restricted`, an empty tool list, disabled
slash commands, empty MCP configuration and no session persistence. Verify these
flags and the effective managed policy on the exact installed version. In
2.1.309, `--bare` skips OAuth authentication and is unsuitable for this proposed
subscription-login path. A flag inventory is not proof of isolation.

Before live operation, test client preprocessing (`@file`, shell and slash
syntax), hooks/plugins/MCP, tool denial, explicit model, valid/error/refusal
output, quota failure, bounded output, cancellation and child-process cleanup.
Verify policy revocation again after planning and before job admission. Keep
raw CLI output and credentials out of public errors and progress events. Record
the image or host build, UID, binary hash, model, sanitized fixture and observed
result. Fake subprocess tests establish wrapper behavior only.

Set the account's extra-usage controls consistently with the operator's chosen
spending policy before the first live probe. A CLI dollar-budget flag is not
evidence that Team subscription overage is disabled. Do not set an API key or
switch to Console/API billing to recover from a subscription limit. Stop and
preserve work when the approved allowance is unavailable.

## Release and rollback

The separate runtime topology and build procedure live in
[`broker-runbook.md`](broker-runbook.md). The opt-in
[`compose.app.yml`](compose.app.yml) overlay adds the API socket mount and a
credential-free SQL/media supervisor; use it with the root compose file only
after qualification. It passes only the database URL and assistant settings to
the supervisor, without the general application `.env` or Drive secret mount.
This file's existence does not mean the overlay has been activated.

Keep activation disabled until API/chat/worker integrations, all four use cases,
media quality review, account qualification and production-environment checks
meet the linked acceptance criteria. The current library checkpoints alone do
not meet those criteria. SQL outbox records also need dispatch/reconciliation;
they do not make the existing Redis queue durable by themselves.

Before any production migration/deploy, complete the required independent CLI
review, run tests with the production Python/dependency lock, rehearse the
additive migration on an isolated PostgreSQL database, and obtain the required
Drive database backup via `scripts/backup_db.sh predeploy`. Apply migrations
before deploying code. Follow the existing migration copy/exec procedure in
`CLAUDE.md`; do not point the disposable test suites at production.

Deploy through the established procedure with `--scale worker=2`; restart nginx
after API recreation and check authenticated traffic through the proxy. A 301
from the HTTP page URL or a healthy internal API is insufficient. Verify chat
reopen, access denial outside both allowed projects, cancellation, an immutable
edit result and recovery after worker restart before admitting pilot traffic.

Rollback first disables admission and the planner, then drains or cancels work
using durable job state. Preserve successful versions, pinned artifacts and
unknown paid-call receipts. Roll back application images using the existing
runbook while retaining the additive history tables. Do not downgrade/drop
assistant history or garbage-collect pinned files as an automatic rollback step.
Record deployment revision, review evidence, proxy checks, pilot results and
remaining limitations in Linear.

## Mandatory media confinement (AICP-286)

The API and supervisor image now contains root-owned, non-setuid
`/usr/local/libexec/reskin-media-exec` and `reskin-media-policy.so`, built from
the same reviewed native source. Landlock ABI1 or later and seccomp are required;
an unavailable policy is an operation failure, never an unsandboxed fallback.
Keep the current unprivileged user, dropped capabilities, read-only root and
no-new-privileges configuration. No Docker socket or host launcher is needed.

Rebuild the ASR image from the reviewed application image so it contains the
matching helper/library, and regenerate its Python context with
`scripts/prepare_assistant_asr_build.py`. Update `ASSISTANT_ASR_IMAGE` only to the
qualified immutable image ID. Rebuild the Claude broker for changed action
schemas and pure-Python JPEG checks. Do not combine old ASR code with a new
supervisor and claim the entire media boundary is confined.

Before activation, run actual canary denials and media workflows under the
deployment UID/capability/seccomp settings, including real local Whisper and CJK
subtitle rendering. Record exact source and helper hashes. Drain current jobs
before replacing baked images. This change needs no schema migration. A rollback
must disable assistant admission until a confined image is restored; restoring
the previous direct media execution path is not an availability fallback.
