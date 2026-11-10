# Isolated planner broker

This stack is opt-in preparation. Building images does not authorize activation.
No daemon has been activated by these files. Native CLI compatibility was
measured separately; proxy authentication refresh and the complete deployed
boundary need the evidence below before calling the runtime qualified.

The backend connects through a Unix socket and authenticates through Linux
`SO_PEERCRED`, accepting only UID 10001 by default. Its socket directory mount
is read-only. This trusts the existing API/media workers, whose UID is 10001;
nginx and the browser receive no mount. The runner UID 999 has a separate PID
namespace, config, work directory and filesystem. No application `.env`, media,
database, Redis, provider/Drive secret, root home or Docker socket is mounted.
This assumes the current Docker deployment has no user-namespace remapping;
check kernel peer credentials again if that deployment setting changes.

`BrokerClient.submit(turn_id, context)`, `poll`, `cancel`, `acknowledge` and
`health` use bounded JSON over HTTP on that socket. Requests have a 64 KiB cap,
responses 256 KiB. One native process runs, with at most four queued contexts.
Queue and active deadlines are each 120 seconds. Eight client connections and
eight egress tunnels are permitted, with request/tunnel time and byte limits.
The model is pinned to `claude-sonnet-5-5`. No API billing fallback is configured.

The application SQL turn remains authoritative. Commit its lease before
submission. Compare the returned context SHA-256 to `context_fingerprint` of the
accepted context. Persist a terminal reply/failure before acknowledging it.
An ACK clears the broker reply while retaining the ID/hash tombstone. Context
is never written to the broker ledger. The ledger does retain unacknowledged
reply text for at most an hour; keep its private directory out of shared mounts
and backups/exports unless explicitly required. Missing, corrupt or full state
fails closed. SQLite is limited to 4096 pages (normally 16 MiB), plus its bounded
transaction journal. At 100,000 tombstones or 128 unacknowledged terminal turns,
admission stops. Do not delete/rotate a ledger to make an interrupted turn run
again. Archive/retention changes require review of the SQL idempotency history.

Same ID and canonical context returns the existing state; different context
returns conflict. Recovery fences an absent broker ID with a durable cancelled
tombstone bound to the SQL submission hash before clearing cleanup. A delayed
submission therefore returns the cancelled tombstone and never invokes native
inference. Missing-intent tombstones are already acknowledged (no execution or
payload exists), consume the bounded durable ID capacity, and survive restart.
On restart queued/running/cancel-requested entries become
interrupted, without inference replay. A client socket timeout does not cancel
a turn: poll the same ID, or cancel it. Unknown outcomes are not resubmitted
under new IDs automatically. Cancelled is published only after planner process
cleanup. Operator retries are new, explicit application turn requests.

Planner failures emit one bounded broker warning with only `invocation` (the
canonical step UUID), an allowlisted `code`, and an allowlisted `cause` category.
Use these fields to distinguish configuration/input/process failures, rejected
stream or terminal events, invalid action schemas, and scope/domain constraints.
Native limit codes use exact structured status values; they do not inspect an
account or establish its remaining quota. Exception messages, tracebacks, model
inputs/outputs, stderr and environment values must never be logged. Diagnostic
logging cannot block durable terminal persistence. Existing public error codes
and invocation counts remain unchanged; a failure never authorizes an automatic
retry or billing fallback.

## Network boundary

The credential-bearing runner uses Docker `network_mode: none`. Only loopback
exists. Its loopback proxy relay forwards bytes to a Unix socket served by a
separate UID 10002 gateway with no credentials. That gateway accepts only UID
999 and CONNECT to exact `api.anthropic.com`, `claude.ai` or
`platform.claude.com` on 443. DNS is resolved there; every returned address must
be public, and the checked address is pinned for connection. Private, metadata,
host, reserved/multicast and mixed DNS destinations fail closed. TLS stays
between the native client and Anthropic; the gateway does not decrypt it.

These hosts and `HTTPS_PROXY` are documented in the official
[Claude network configuration](https://code.claude.com/docs/en/network-config#network-access-requirements).
In particular `platform.claude.com` handles token refresh for claude.ai accounts.
The other documented download/plugin/connector/browser/telemetry hosts are not
allowed. Explicit environment disables nonessential traffic, automatic updates,
claude.ai MCP connectors and artifacts. Existing tool/MCP restrictions still
apply. Disabling variables and a hostname list alone are not qualification.
[Docker documents that network none creates only loopback](https://docs.docker.com/engine/network/drivers/none/).

## Preparation and checks

Do not run login provisioning again or copy an auth cache. Parent rollout work
owns all service changes. A reviewed activation will require four dedicated
directories: broker socket (999:998, 0755), state (999:998, 0700), gateway socket
(10002:10002, 0755), and the already provisioned app config (999:998, 0700).
Validate every path is canonical, not a symlink or an app-writable parent used
by privileged setup. Set only the explicit `ASSISTANT_*` variables required by
`compose.runner.yml`; never pass production `.env` as an env_file to this stack.
Bind mounts refuse missing sources. Images copy only reviewed runtime modules,
the pinned Python/Pydantic lock subset, and the checksum-pinned native binary.

Run fake-only unit tests and `offline-denials.py` in disposable empty-auth,
network-none containers before activation. Test:

- real peer UID allow/deny across the intended mounts;
- host/gateway, database/Redis, metadata, private and arbitrary public direct
  connections denied from the runner, for both supported IP families;
- exact CONNECT allowlist with synthetic public DNS and a harmless echo canary;
  nonallowlisted names/IP literals/wrong ports and private/mixed DNS denied;
- config/credential/production path canaries absent in the backend and runner
  as appropriate, using harmless fixture bytes, never real credential reads;
- timeout, cancel, orphan reaping, restart interruption, duplicate suppression,
  ledger corruption/full storage and client/response/body limits;
- reviewed native model/auth through the proxy, including refresh persistence,
  tool/connectors denial and subscription-limit behavior, as separate authorized
  live checks after code review. Extra usage stays disabled; no API fallback.

No ports are published. Parent integration must mount only the socket directory
into backend dispatchers and retain all existing access/lease validation.
Both opt-in services use `restart: unless-stopped` after explicit activation;
restarting the broker never replays incomplete native turns.
Rollback disables admission, cancels/drains broker work, then stops the isolated
stack. Keep the SQL history, local ledger and native refreshed config intact.
