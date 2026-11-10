# Use an assistant version as the main Run result

Design proposal, 2026-10-27, AICP-305. Baseline
`4fc4315ff91f2cb2def9bf445970900a4ac131d5`.

The operator selects a successful immutable assistant version and explicitly
publishes its full result to the Run. Selection and rendering alone do not
publish. This action is unavailable to Claude's internal API/tools. Existing
project permissions, subscription policy and immutable version storage apply.

## Operator API and UI

`POST /api/v2/assistant/runs/{run_id}/result` accepts `version_id`,
`expected_selection_epoch`, `expected_result_revision`, `idempotency_key`.
Use strict bounded contracts. The response is a durable promotion receipt.
No asynchronous worker or model call is needed: the verified file already
exists. UI displays busy/error state, prevents duplicate clicks and retains
the exact request identity after unknown network/5xx replies for explicit retry.
Never silently rebase a failed request to another selection/result.

Assistant context gains `run_result` with an opaque current revision,
verified matching assistant version ID if known, and ordered receipt history
with authenticated before/after media links. The UI shows the current-main
marker, Use as Run result next to the selected player, and access to prior
main results. After success, refresh only `#run-status-content` via its existing
HTMX endpoint; preserve chat/draft. Reopening obtains the durable receipt.

## Service and immutable history

Add a mutable recovery `AssistantRunResultIntent` and an append-only
`AssistantRunResultPromotion` receipt table with an additive migration.
Store scoped conversation/Run IDs, selected version/artifact binding,
actor/time, request digest and unique conversation/idempotency key, and exact
before/after path, SHA256, size, duration, Drive identity and pinned-copy identity.
Receipts refer to immutable artifacts; no source version is rewritten.

Resolve only the requested version's full video role (not preview), requiring
it is successful and belongs to this exact conversation/Run. Verify the full
artifact's recorded SHA/size and bounded media metadata. Probe outside DB locks.
Copy to a new exclusively-created per-promotion Run results directory, with
standard nginx-readable permissions; no hardlinks/chmod of assistant objects.
Verify copied bytes, fsync and publish atomically. Never overwrite an old file.

Acquire conversation lock then Run FOR UPDATE, matching current import lock
order. Reauthorize scope and require done Run, matching selected version ID AND
epoch, and matching expected result revision (path/Drive/stat/content identity).
Reverify staged/source/current identities under the final lock without starting
subprocesses. Reject active assistant edits and active legacy audio/range work,
plus active Run-source derived jobs whose queued source would otherwise change.
Snapshot current main bytes into immutable storage before changing any pointer;
retain the old file and expose the pinned snapshot through authorized history.

Before any external write, durably reserve one unresolved intent per Run with
exact request, source, immutable before snapshot and staged output identities.
Commit the intent, then reacquire conversation -> Run locks and revalidate.
Every competing writer must reject an unresolved intent. Selection may change
after reservation; explicit same-request retry resumes the already-authorized
exact version, as visibly stated in the UI, never a newly selected version.

Under the Run lock, update only an EXISTING Drive file's content, preserving
Drive ID, exported sheet link/stamp, Run identity, language and segment settings.
No existing Drive ID means local-only, explicitly disclosed; do not create a new
Drive file or fall back from update to create. Then atomically commit the new Run
result path, verified duration, immutable receipt and completed intent.
A duplicate matching key returns its old receipt
without reapplying/uploading, including after a later promotion. Changed payload
under the same key conflicts. Current identical content may return a no-op receipt.

Drive failure leaves local Run/receipt unchanged, but an ambiguous remote/DB
outcome retains its unresolved intent and writer exclusion until explicit
same-request reconciliation. No timeout or expiry clears the exclusion. Retry
revalidates current permissions and immutable bytes and repairs the exact bound
remote content before committing. Never claim distributed atomicity or silently
substitute a new key. Reauthorization precedes idempotent receipt lookup, which
precedes any stale selection/result checks.

Persist the Drive resumable session URI in the private intent BEFORE sending any
media bytes. It is sensitive runtime state: never expose it in context, receipts,
errors or logs. Explicit recovery queries/resumes that SAME session; only a
confirmed terminal200/201 response for the bound file/size/hash can release the
intent fence. An expired/unqueryable session remains unresolved for manual
reconciliation; never silently replace it with a new session after media may
have been sent. An initiation response lost before its URI is persisted has sent
no media and may safely reinitiate. This follows the provider's documented
[resumable protocol](https://developers.google.com/workspace/drive/api/guides/manage-uploads#resumable).

Use explicit resource limits: at most256MiB media;120s media duration;30s probe;
90s request preparation/delivery deadline, and5s database lock wait. Native media
and copying are outside DB locks. Check deadline/cancellation during bounded
hash/copy and final identity checks. Promotion uses dedicated per-client/per-call
Drive settings without changing legacy defaults: no automatic HTTP/chunk retries,
each transport timeout <=min(30s, remaining deadline), check remaining deadline
between chunks/requests. A possibly committed timed-out transfer is unknown.

## Existing pipeline integration

Use the shared Run lock for competing result mutations and relevant retries and
deletion. Add the missing lock to segment rerun before reads. Once a Run has any
successful assistant publication, conservatively block BOTH segment and range
reconstruction in place, even after later audio patches or selecting V1. The
durable receipt is the provenance fence; clone to a new Run for regeneration.
This explicit policy avoids pretending original segments can recreate chat edits.
Preserve existing completed derived outputs. Guard queued/running derived jobs
before promotion so their source cannot be silently reinterpreted. Reciprocal
admission uses the same Run lock and rejects unresolved intents for create,
retry, submit/render of direct Run-source tail/caption/overlay/PIP merge work.
PIP includes both parent and child Run IDs. Acquire multiple Runs in sorted ID
order. Treat direct Run-source overlay `created` drafts as blocking promotion,
requiring finishing/removing the draft first, avoiding an implicit timeline rebase.
Completed derived outputs and sources pointing to those outputs remain valid.
Only promotion acquires the conversation lock; legacy paths never acquire it
after locking a Run.

For the initial release, use a conservative explicit retained-job policy rather
than adding source-binding fields to every legacy model: after ANY successful
assistant publication on a Run, reject retry/re-render/re-submit of retained
direct-Run-source tail/caption/overlay/PIP jobs and require creating a fresh job
against the current result. This includes failed jobs created after publication;
their safe reuse can be added separately with persisted per-job source bindings.
Fresh job creation is allowed under the admission lock. First submit of a
`created` overlay is allowed: those drafts block promotion for their lifetime,
so a new publication cannot change their source between create and submit.
Do not change completed output bytes or disable downloads of retained jobs.

Replace cached Run duration on promotion. Fix old NULL-duration backfills with
path + NULL compare-and-swap after their probes. Cache-busting must incorporate
result identity/path, not only second-resolution mtime/size. Mutable Run/public
result routes must revalidate rather than cache forever; immutable version media
keeps its existing behavior. Existing clients may have already cached old
immutable public URLs; server headers cannot evict those entries. Newly rendered
links carry the new revision. New downstream jobs resolve the promoted file.

## Verification and rollout

Meaningful service/API tests: full versus preview, before/after hashes and
history, bad/missing source, unsuccessful/wrong Run/project version, disabled
scope, stale selection/result, duplicate/unknown reply replay, Drive failure,
duration, unchanged Run identity/settings/segments/Drive/export stamp. Real
PostgreSQL concurrent promotion and audio/range cases verify row-lock ordering.
Browser tests cover selected versus latest, progress/conflicts/exact retry,
current marker/history, and refreshed main player/download without lost draft.
Owned native media verifies shorter/longer results and retained downloads.

Independent Sol-6.1 CLI design and final code gates precede production writes.
Verify a Drive DB backup before migration/deploy; preserve two workers/assistant
overlays, restart nginx and verify authenticated proxy routes. Real validation
uses the already-rendered selected version on Run7c29c40f via the ordinary
operator route, no inference/generation. Capture before/after main/version bytes,
duration, Drive identity/content, receipt history and selected context. Deploy the
general capability to currently enabled Runs; do not publish other Runs in bulk.
