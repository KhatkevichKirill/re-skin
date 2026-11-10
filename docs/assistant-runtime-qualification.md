# Assistant runtime qualification

Evidence date: 2026-10-23 UTC. Qualification task: AICP-263, under AICP-261. Accountable owner: the repository owner. Document executor: Codex session agent, GPT-6.1 Sol. This session agent is separate from any locally installed CLI inference runtime.

This is the historical qualification snapshot. Current deployment evidence is
in Linear and the [runtime runbook](../deploy/assistant/README.md). On2026-10-27
the operator expanded the original two-project scope to all private dashboard
projects through an explicit trusted setting; see
[access scope](assistant-access-scope.md) and
AICP-309. That change does not
alter the included-subscription-only or closed-tool/runtime constraints below.

**Verdict: UNQUALIFIED; no production adoption or launch approval.** A separate native Claude login has now been provisioned, but this document does not establish a running assistant or measured service capacity. Linear owns current task state and dependencies.

## Decision boundary

Codex CLI can run unattended tasks, but a developer's existing CLI login is insufficient evidence for an application runtime. The app needs independently provisioned, approved authentication in an app-only credential store and isolated configuration. A distinct non-human account is an option, not an established prerequisite. Do not copy, mount, borrow, refresh, share, or repurpose `/root/.codex/auth.json` or root's developer configuration for app execution.

The operator confirmed a private operator beta using the existing dashboard login and explicit project allowlists: `f880254c-bbfe-4a2d-83ec-324d7e90cbee` and `dca4f5da-dfe7-4822-855a-4468e9ad3433`. Apply these through deployment configuration, not permissive code defaults; absent/empty lists deny access. Named-user and shared public-service use are outside this beta. Until access enforcement and runtime qualification pass, keep assistant execution disabled.

## Corporate Claude candidate requested by the operator

After the initial Codex assessment, the operator asked to use an existing corporate Claude account instead. Claude Code/Agent SDK is the leading candidate to evaluate for the private operator beta. Development and review agents remain Sol-6.1; that choice does not select the application's runtime model.

Anthropic's October 7, 2026 plan guidance says the Agent SDK, `claude -p` and third-party apps can still use subscription limits; Max and Team plans also include monthly API credits. This establishes a supported headless-use candidate, not the account's actual available capacity or a guarantee of zero charges. [Claude plan guidance](https://support.claude.com/en/articles/15036540-use-the-claude-agent-sdk-with-your-claude-plan)

The product context matters. Agent SDK documentation directs developers offering their own products' Claude login/rate limits to use API authentication unless previously approved. Claude Code's guidance separately permits hosting the unmodified binary with end users authenticating through Anthropic's own flow. For this private beta, qualify the operator's own internal use explicitly; do not assume one account can later serve unrelated users, copy session tokens into the application, or offer a custom Claude login. If the chosen deployment needs API authentication, its budget and billing must be approved before use. [Agent SDK overview](https://code.claude.com/docs/en/agent-sdk/overview), [Authentication and product hosting guidance](https://code.claude.com/docs/en/legal-and-compliance)

Team and Enterprise plans differ. Team seats include Claude Code; some older Enterprise seat types need specific access. Usage-based Enterprise is billed at API rates without per-seat usage limits, while seat-based plans have different included usage/credit controls. Corporate account branding alone therefore establishes neither a fixed subscription cost nor this deployment's permissions. Synced organization skills/plugins must also be accounted for in the runner's tool inventory. [Team and Enterprise setup and billing](https://support.claude.com/en/articles/11845131-use-claude-code-with-your-team-or-enterprise-plan)

Safe local metadata subsequently verified Claude Code `2.1.309`, `loggedIn=true`, `authMethod=claude.ai`, `apiProvider=firstParty`, `subscriptionType=team`. Account identifiers and credential values were not printed. This identifies the existing CLI login's reported plan, not its live entitlement, corporate approval or app-runner authentication. The operator did not independently know the plan.

The operator confirmed included subscription usage only, stopping at the limit, with no paid API fallback, and confirmed Extra usage is disabled for the Team seat. The operator personally completed the separate supported login. Model entitlement, available limits, tool/plugin isolation, refresh/re-login, cancellation/recovery and measured capacity remain qualification work. Existing typed edit plans and local render adapters remain provider-independent.

The one-time provisioning script created `reskin-assistant` (UID 999, primary GID
998, no supplementary groups), private mode-0700 config/work directories, and
the administrator-owned mode-0755 Claude 2.1.309 executable. Its SHA-256 is
`4d704a8aecbe038affa3000f498ae17c1b4f753add8facb8764120affc99b658`.
The app-only native login reports `loggedIn=true`, `authMethod=claude.ai`,
`apiProvider=firstParty`, `subscriptionType=team`. No developer credentials were
copied. No daemon or application feature was activated. The script refuses
pre-existing identities/trees rather than modifying app-owned state as root.

### Offline adapter evidence

The disabled-by-default [planner adapter](../backend/app/assistant_planner.py)
now validates bounded, pinned context and clarify/propose/unavailable replies,
uses explicit tool-free CLI arguments and a minimal environment, and bounds
subprocess output, time and cancellation. **56 fake-subprocess tests passed**
on the host environment. These tests do not invoke Claude, establish account
entitlement, verify Anthropic's actual structured-output schema, or qualify the
runtime. The [rollout runbook](../deploy/assistant/README.md) records the separate
native login procedure, process isolation and release checks.

Installed 2.1.309 help and current official CLI documentation agree that
`--bare` does not read OAuth credentials. The subscription candidate therefore
uses safe/restricted modes and explicit empty tool/MCP configuration instead;
their effective behavior, client preprocessing and managed policies still need
live qualification. An external supervisor must reap orphan descendants and
enforce the process/network boundary; subprocess-group cleanup alone is not a
container or privilege boundary. [Programmatic execution](https://code.claude.com/docs/en/headless),
[CLI reference](https://code.claude.com/docs/en/cli-reference)

The Codex-specific observations below remain historical evidence for the original candidate. They do not qualify Claude, and the general runtime probes must be repeated for the selected provider and account.

One candidate is a fresh supported login under an operator-approved existing ChatGPT subscription into a dedicated app-only auth store, with a named owner, serialized jobs, persistent refreshed credentials and a renewal runbook. It must be provisioned independently, not seeded from root's cache. Headless login has documented device-code and SSH-callback options; availability and permissions must be checked for the chosen account. The advanced private-automation workflow establishes technical feasibility rather than a blanket conclusion about this app's audience or account eligibility. [Official OpenAI documentation: Authentication](https://learn.chatgpt.com/docs/auth), [Maintain account auth in CI/CD](https://learn.chatgpt.com/docs/auth/ci-cd-auth)

An alternative is a non-human ChatGPT workspace service account with a Codex-scoped access token. This route requires an eligible pay-as-you-go workspace, owner/admin provisioning, and CLI version at least `0.142.0`. Its plugins and permissions are assigned separately. Installed version compatibility does not establish workspace eligibility, model access or billing approval. This eligibility requirement does not automatically block the subscription-login candidate. [Service accounts](https://learn.chatgpt.com/docs/enterprise/service-accounts)

A personal Codex access token represents its human creator and may be another supported provisioning option where available and explicitly approved. It is not a non-human identity. Access-token creation and local Codex access are separate permissions. [Access tokens](https://learn.chatgpt.com/docs/enterprise/access-tokens)

The operator must identify the account/workspace, accountable credential owner and supported provisioning method. An explicitly funded Platform project service identity/API route is a separate option with separate access and billing. No automatic fallback, account upgrade or credential provisioning is authorized by this document. Do not infer plan or product-policy restrictions beyond the cited documentation; verify the selected account and intended trusted deployment.

## Evidence collected without inference

The following observations came from read-only local commands and selected non-secret metadata. No inference request, installation, account login, credential copy, service change, migration, or deployment was performed for this review.

| Observation | Verified result | What it does not prove |
|---|---|---|
| `/root/.local/bin/codex --version` | `codex-cli 0.163.0` | That the application image contains this binary or supports the proposed isolation |
| `codex login status` | `Logged in using ChatGPT` | Live token validity, service-account identity, model entitlement, or remaining quota |
| Installed `codex exec --help` | Supports `--json`, `--output-schema`, `--ephemeral`, `--image`, explicit model and sandbox, `--ignore-user-config`, and `--ignore-rules` | That any flag was exercised against inference or that defaults meet application policy |
| Local model cache | Modified `2026-10-23T02:41:03Z`; contains visible `gpt-6.1-sol` | Live access by the future service identity; cache is not an authorization check |
| Selected root config fields | Model `gpt-6-astra`; reasoning effort `high`; an MCP server named `linear` | An app-safe tool inventory; root config must not be inherited |
| Auth metadata only | Mode `chatgpt`; access/refresh-token presence; last refresh `2026-10-16T18:25:55Z`; file permissions `0600` | Validity, renewal success, safe sharing, or permission to reuse this credential |
| Environment presence checks only | No `OPENAI_API_KEY`, `CODEX_API_KEY`, `CODEX_ACCESS_TOKEN`, or custom `CODEX_HOME` present | Credential absence from files or other processes |
| Existing Compose configuration | API/media workers receive `.env` and mount `./secrets`; each media worker is limited to 6 GiB RAM and 2 CPUs | Measured free host capacity or a safe place for a credential-bearing agent |

No token values, account identifiers, credential contents, or raw environment dumps are needed in qualification evidence. Existing [Compose configuration](../docker-compose.yml) and [repository operational policy](../CLAUDE.md) establish why launching a CLI subprocess in the current API/media worker unchanged is unsuitable. The architecture wiki includes historical v1 information; current code and Compose take precedence for runtime planning.

The repository records sandbox experiments from September. Those are historical observations, not current denial-test results for the intended service identity, image, kernel, or configuration. No sandbox escape or credential-read probe was run here.

## What official documentation establishes

`codex exec` supports non-interactive operation, JSONL progress events and a JSON Schema for its final response. These interfaces help integration; they do not validate edit semantics or authorize side effects. API credentials remain the documented default for general automation. [Non-interactive mode](https://learn.chatgpt.com/docs/non-interactive-mode)

App Server provides model discovery, quota reads, events, and turn interruption; dynamic tools are experimental. `exec` versus App Server remains a qualification decision. Keep any App Server transport private and separately authenticated; its RPC endpoint must not become a browser-accessible general command or filesystem API. [App Server](https://learn.chatgpt.com/docs/app-server)

The model selected must be explicit and verified for the application identity. GPT-6.1 Sol supports text/image inputs and text outputs, rather than native video/audio inputs. The editor needs a bounded backend-generated frames/transcript contract for video understanding. API model documentation alone does not grant subscription access. [GPT-6.1 Sol](https://developers.openai.com/api/docs/models/gpt-6.1-sol), [Workspace model availability](https://learn.chatgpt.com/docs/enterprise/workspace-model-availability)

Quota consumption depends on workload and is shared with other covered activity. Public message estimates are not a throughput guarantee. This review has no verified service plan, remaining quota, applicable credit balance, reset schedule, overage eligibility, or per-job cost. Do not label jobs unlimited or free. [Pricing and usage limits](https://learn.chatgpt.com/docs/pricing)

Default read-only permission does not imply secret confidentiality. Command-network policy does not govern every other connection surface, including model/auth requests, hosted search, MCP and connectors. Each available surface needs its own controls. [Agent approvals and security](https://learn.chatgpt.com/docs/agent-approvals-security)

For ChatGPT OAuth, normal Codex runs automatically refresh stale credentials, write the updated cache and can refresh/retry after a 401. Persist that updated app-only cache; do not restore an old seed each run or share one copy across concurrent jobs/machines. Renewal can still fail and require an operator login. If a token-based route is selected instead, qualify its expiry, revocation and rotation separately. [Maintain account auth in CI/CD](https://learn.chatgpt.com/docs/auth/ci-cd-auth)

## Proposed isolated runner

This is a design to probe, not an implemented deployment configuration.

1. The trusted backend authorizes an assistant job against the authenticated operator and approved project scope. It creates an immutable snapshot revision and job ID. The browser supplies conversational intent, never a shell command, filesystem path, credential, runtime flag, arbitrary remote URL, or service identity selection.
2. A separate assistant supervisor claims the job with a lease. Initially allow one assistant job at a time until capacity and isolation evidence justify a reviewed increase. Keep this queue separate from long media jobs and from developer review sessions.
3. The supervisor launches a pinned, non-root runtime image with its own OS identity, clean application home/configuration, read-only root filesystem, bounded temporary storage, and a job-specific directory. Install only reviewed runtime dependencies during image construction, not during a task. Do not mount production code, root home, Docker socket, production `.env`, Drive/provider credentials, general media storage, or neighboring projects.
4. Give the model a sanitized snapshot and only the approved job's staged frames/transcript. Treat transcript, metadata, filenames and user-supplied content as data that can contain malicious instructions. Exclude access links, secrets and unrelated project material. Enforce byte/frame/duration limits before staging.
5. Authenticate the Codex control process using the selected approved method. For subscription OAuth, provision a fresh login directly into a dedicated protected app-only auth store and persist only its refreshed cache. For an approved access token, deliver it through a secret manager with defined expiry/rotation. The app's web/API processes do not need the Codex credential. Record identity/expiry metadata without token values. Neither route uses the developer auth cache.
6. Separate the credential-bearing control process from model-requested command/tool execution. Tool children must not receive the token, read an auth store, inspect the control process's environment through `/proc`, or reach credential helpers. Environment filtering alone is insufficient when files or process inspection remain available. Use enforced filesystem/process isolation and demonstrate denial; if the chosen CLI configuration cannot enforce this, the candidate fails.
7. Start with typed proposals and no side-effecting tools. If tools become necessary, expose only reviewed domain operations through an authenticated broker. Disable built-in arbitrary command execution unless it is independently required and qualified. Disable unneeded plugins, hooks, MCP servers, browser/computer use, search, automatic reviewers and subagents. A clean config is necessary; it is not proof that every default surface is disabled. Do not use `--ignore-rules` as a substitute for enforced application policy or bypass the sandbox.
8. Model/auth egress is restricted to the documented destinations needed by the chosen authentication/runtime. Domain-tool access, if enabled, reaches only the broker. No arbitrary public internet, private-network discovery, cloud metadata endpoints, shared Unix sockets, database, Redis, media-provider or Drive access is granted to the agent. Validate actual destinations; do not guess a complete hostname list from a single successful request.
9. The supervisor validates a complete typed response against the snapshot and application rules, persists a proposal, and reports progress through application-owned status. It does not infer success from exit code, transcript text, or a partial JSON object. Unknown operations, extra fields, invalid IDs, stale snapshots and out-of-range timings fail closed.
10. The existing trusted backend applies authorized operations and manages provider spending. Proposal acceptance does not grant the model a general API token. A broker capability, if needed, is bound to the job, principal, project, revision, permitted operation and expiry; it is revocable and useless for another job.

The official security cookbook describes restrictive profiles, environment reduction, lower-privilege identities and credential brokers. Those patterns inform the proposal, but only tests of the selected image/configuration establish the boundary here. A privileged control process sharing its token with an agent-readable environment is a failed design, even inside a container. [Security cookbook](https://learn.chatgpt.com/docs/enterprise/security-cookbook)

## Execution, recovery and spending contract

Persistent application states must distinguish queued, active, succeeded, failed, canceled and interrupted/recovering work. An active claim needs a lease and an attempt record. A worker crash or expired lease must not silently restart a possibly executed edit or paid media submission.

Use argument arrays and structured input; do not build shell command text from messages. Set explicit model, supported reasoning effort, permission policy, environment and tool inventory. Keep application conversation/job mapping explicit; never resume a global `--last` session. Store only reviewed events with retention and size limits. Redact credentials and signed URLs before logging, including tool failures and stderr.

The supervisor owns wall-clock deadline, process-group termination, CPU/memory/PID limits, disk quota, output-byte cap, queue length and concurrency. Numerical limits remain unset pending measured representative workloads and host-capacity evidence; unset limits block launch. A CLI context window or eventual token-usage event is not a hard per-job spending ceiling.

Cancellation first stops new tool/broker actions, revokes the job capability, and requests runtime interruption. It then terminates the whole process group after a bounded grace period and marks a durable terminal state. Disconnection from the browser does not implicitly cancel a job. An already accepted paid provider task needs its own cancellation/reconciliation policy; stopping the LLM cannot undo it.

Retry only bounded, classified transient failures, with jitter and an overall deadline. Quota exhaustion waits until verified capacity/reset or returns an actionable terminal result. Invalid/revoked credentials, unsupported model, policy rejection, schema failure and stale source revisions do not enter an unattended retry loop. Reconcile uncertain broker/provider outcomes using durable idempotency records before retrying any mutation.

Before paid media execution, the backend reserves an approved budget against the actual operation/provider parameters, records the durable request identity, and rejects work above the approved budget. No automatic model/speed change, API fallback, credit purchase, overage enablement or limit-reset redemption is part of this proposal. LLM account controls and downstream media-provider budgets are distinct gates.

## Qualification probes and required evidence

Run these only in an authorized isolated environment. The evidence below now includes separate native login provisioning, disposable container checks, offline failure tests and one successful synthetic response through the complete adapter. Fixtures use harmless synthetic identifiers; no user media or real credential-read challenge was used.

### Native Claude compatibility result — 2026-10-23

The complete adapter returned the expected validated `unavailable` reply for a
synthetic instruction requesting no edit. The exact selected and returned model
was **`claude-sonnet-5-5`**, using Claude Code **2.1.309** and the separately
authenticated Team configuration. No API key, paid fallback or media operation
was used. This establishes basic authentication and structured-response
compatibility, not launch readiness or edit quality.

The disposable runner used UID 999/GID 998 with a real Unix account/home entry,
an init process, read-only root filesystem, no Linux capabilities,
no-new-privileges, 1 GiB memory, 1 CPU, 128 PIDs, bounded temporary directories,
and the adapter's 60-second timeout. Only the app credential directory was
writable/persistent; no developer credentials, production environment, media or
Docker socket were mounted. Qualification image:
`sha256:8b56cb48b8cf8056f438ccfb567f686653f0a7b7ec556074cda0b57a18b17c49`.
Its default bridge networking was a probe-only allowance: destination/private
network denial remains unverified and is a launch gate.

The probe exposed and corrected three wire-compatibility issues: Claude's strict
schema validator rejects OpenAPI `discriminator` annotations; its output tool
requires an object root without a top-level union; and successful structured
completion uses `stop_reason=tool_use`. The transport now wraps the unchanged
reply union in one required `reply` field, strips only unsupported annotations,
and accepts that stop reason only inside the existing successful terminal guard.
Strict parsing, pinned identifiers, source validation and denial/error rejection
remain enforced. Schema-rejected requests reported zero model tokens. The final
native request passed end to end. **139 planner/contract tests passed** with the
production Python 3.12.15 / Pydantic 2.14.1 / pytest 9.1.1 environment; the earlier
full production-lock checkpoint passed **512 tests**.

| Gate | Probe and pass evidence | Present finding |
|---|---|---|
| Audience and scope | Approved users, project visibility, assets and operation permissions; direct API and UI enforce the same decisions | Private operator beta and two project UUIDs confirmed above; enforcement remains pending |
| Dedicated runtime auth | Operator identifies approved account/workspace and owner; supported independent app-only login, expiry/refresh, rotation and revocation; no copied/shared developer cache | Separate operator-completed Team login verified; refresh/revocation lifecycle remains pending |
| Model compatibility | Bounded request runs the explicitly selected app model; record returned model and supported inputs | Synthetic text/structured reply passed on claude-sonnet-5-5; Sol-6.1 remains the development/review model |
| Credential boundary | Canary service token/auth-store contents inaccessible through shell, tools, process environment, `/proc`, filesystem searches, logs and failure output | Not tested |
| Filesystem and tools | Approved snapshot readable; other jobs/projects, root home, secret canaries, sockets and production paths denied; requested escapes/escalations fail closed; repeat after restart | Not tested |
| Network and integrations | Required model/auth succeeds; forbidden host, metadata endpoint, private network, Unix socket, hosted search, plugin/MCP and developer Linear access denied | Model/auth connectivity passed; outbound destination/private-network denial remains pending |
| Output and operation validation | Complete schema passes; truncated/malformed/oversized/refusal output, invented IDs, forbidden operations, stale revision and invalid timings rejected | Native synthetic reply passed; strict negative tests and admission tests passed; full application integration pending |
| Lifecycle | Cancellation before/during a tool, timeout, runtime crash, supervisor restart, lost client, lease expiry and duplicate completion produce one durable result and no untracked process/action | Fake-process timeout/cancel/descendant and SQL race tests passed; live integrated recovery pending |
| Idempotency | Simulated timeout after accepted mutation/provider submission reconciles without a duplicate edit, upload or charge | SQL/outbox/paid-intent tests passed including PostgreSQL concurrency; real provider/dispatcher integration pending |
| Auth failure | Selected method's expired/revoked credentials and removed local/model permission fail safely; OAuth refresh persists or token replacement works as applicable; retired credentials cease working; logs remain clean | Not tested |
| Capacity | Approved representative workloads measure queue delay, p50/p95 completion, memory/CPU/disk peaks, usage and failures alongside existing media workers | No measurements or approved targets |
| Budgets | Verified account billing/limits and alert owner; configured per-job/app/downstream limits; exhausted quota or funds blocks admission/submission | Operator confirmed subscription-only usage and Extra usage disabled; no paid fallback; quota-failure integration and capacity measurement pending |
| Rollout | Disabled-by-default flag, bounded pilot scope, rollback, monitoring, operator runbook, required review and deployment checks | No runtime deployed |

Record each probe's date, tested commit, image digest, CLI version, service identity metadata, explicit model/effort, sanitized inputs, expected result, observed result and evidence location. Keep tokens and raw private media out of Linear, source control and review prompts.

## Go/no-go record

### Isolated broker qualification — 2026-10-23

The minimal runner and credential-free CONNECT gateway now pass direct-network
and forbidden-proxy denials. Cross-container checks verified allowed UID10001,
denied UID10002, a read-only backend socket mount, private config/state/process
canaries, cancellation, and restart interruption without inference replay.
The backend has no app auth or Docker socket mount.

One authorized synthetic native request through this topology completed with a
strict validated `unavailable` reply from **claude-sonnet-5-5** in **4.081s**.
The runner used the operator's separate app login, network mode none, and the
allowlisted gateway. Native logs were empty; only terminal metadata and token
counts were recorded. Included usage: 2 input / 128 output / 6518 cache-creation
tokens; no API key or separately billed fallback was used. Temporary containers
were stopped and app auth preserved. This is operational connectivity evidence,
not a claim that forced credential expiry or actual quota exhaustion was tested.

Independent Sol-6.1 review of b6e7c0d..bcfeaa7 returned seven correctness findings.
They are addressed in the follow-up checkpoint; final review and production
rollout verification remain mandatory. See the updated acceptance evidence.

**Current result: NO-GO for application launch; basic Claude compatibility passed.** Audience, project scope, subscription-only policy, app-only native login and a bounded structured reply are confirmed. Remaining gates include live access enforcement, credential/tool/network denial probes, all four editing workflows and UI, integrated recovery, quota/resource behavior, capacity and pilot quality acceptance. No assistant daemon or application feature is enabled.

The next authorized deliverable is a reviewed runtime decision and isolated qualification evidence under AICP-263. A completed documentation review must not mark runtime qualification or the overall editor launch complete. If a required control cannot be demonstrated, record the failed criterion and compare a separately authorized runtime rather than weakening the boundary. Merge, migration, service restart, deployment and launch follow the repository's existing operational approval and verification rules.
