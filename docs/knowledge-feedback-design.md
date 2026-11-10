# Production knowledge base and feedback loop — design contract

AICP-294 (parent AICP-293), 2026-10-27. Status: **revision 8 — design APPROVED by the independent review (round 7)** — the
implementation contract for AICP-295..304 (see *Review record* at the end).

Source of truth at writing: GitHub `origin/main` =
`e6cd9de12329c1952833fbd58df81f0e9ad50d7b` (re-resolved 2026-10-27; unchanged
from the handoff baseline). Production `re-skin-api` app tree is byte-identical
to that revision (110 `.py/.html/.js` files hashed inside the container and
compared with the checkout); `re-skin-worker-1` matches on `app/models.py`.
Production Alembic head `j0e5f2a7b914`. `/root/re-skin` on `main` with one
untracked file (`tasks/run-local-segment-range-regeneration.md`) and 20 other
worktrees belonging to other sessions; this work lives in
`/root/re-skin-knowledge` on branch `feat/knowledge-feedback-design`.

## 1. Goal and non-goals

Capture production experience as **traceable evidence**, collect **operator
feedback** separately from automatic output and automatic checks, keep
**versioned, reviewable guidance**, and let the assistant/planner **cite** that
guidance and the cases behind it — reproducibly — when recommending a model,
segmentation, references or a repair.

Non-goals (unchanged from the issues): no autonomous production orchestrator; no
new generation mode (AICP-258's images-only rebuild stays unimplemented and is
never auto-selected); no paid generation or experiments; no paid-API fallback for
Claude (included subscription only); no widening of app or assistant access.
Knowledge never grants execution authority: every paid action keeps today's
approval, spend-cap (`spend.reserve`) and retry limits.

## 2. Evidence inventory (read-only, production, 2026-10-27)

All numbers are from `psql` against `re-skin-db` and `sha256sum`/`test -f`
inside `re-skin-api`; nothing was written.

### 2.1 What exists

| Source | Rows | What it can tell the knowledge base | Gaps |
|---|---|---|---|
| `video_projects` | 124 | source file, type, `mute_source`, `max_segment_sec`, transcript, `source_run_id` lineage | no record of WHO is on camera; name is free text |
| `runs` | 525 (by type/model below) | model key, prompt, run-level refs, resolution, audio mode, language, preprocess recipe, `segment_snapshot`, `stitch_plan`, `source_fallback_segments`, `final_creatives_exported_at` | `done` ≠ accepted; no verdict anywhere |
| `run_segments` | 929 (905 completed) | **current** attempt only: task id, result url/path, `cost_credits` (757/905), overrides (187 prompt, 119 refs, 2 GPT-Image) | re-runs **overwrite** task id/result/path in place (`api_v2.rerun_segment`, `pipeline_v2` submit reset ~L2105, in-poll resubmit ~L1520); earlier attempts, their prompts and refs are lost |
| `kie_spend_draws` (+ reservations) | 146 since 2026-10-17 | every paid kie call: `call_key` = kie taskId, estimated vs settled mcredits, run via reservation | nothing before 2026-10-17; no segment link; 11 of 51 video draws match **no** current RunSegment task id = superseded attempts whose recipe is gone |
| `run_segment_range_attempts` / `run_segment_image_jobs` / `run_audio_patches` | 3 / 2 / 3 | append-only correction history with cost, before/after snapshot | none (good model to follow) |
| `run_tail_edits` / `run_captions` / `overlay_jobs` | 126 / 69 / 86 | derived creatives (ending edits, captions, overlays), exported stamps | why the edit was made is not recorded |
| `assistant_*` (conversations 3 runs, 17 versions, promotions 1) | — | immutable version recipes, operator acceptance, Run-result promotion with before/after | only Runs 7c29c40f, faffd482, ed75f231 |
| `manual_creatives` | 14 | script-made creatives with `linear_issue` (AICP-238/246/250) | — |
| Linear Re-skin issues | AICP-230, 243, 256, 258, 259, 276/280/285, 252, 275 … | manual repairs, images-only rebuilds, measured costs, failure descriptions | prose; must be curated, not parsed |
| Repo wiki | `wiki/models/ai-models.md`, `kling-4-0.md`, `gpt-image-2-5.md`, `lessons/production-gotchas.md` | model constraints, measured truncation rate, cost ladders | operator recollection vs measurement often unmarked |

Run counts by project type × model × status (2026-06-22 … 2026-10-23):
face_swap — seedance 213 done/2 failed/2 review; seedance-2-5 135 done/2 failed/4
incomplete/3 review; seedance-fast 57; gemini-omni 8 done/1 failed/3 incomplete;
seedance-mini 1. localisation — 2-5 39(+1), fast 6, seedance 5. subtitle_removal —
gemini-omni 12, 2-5 12(+7), seedance 2. cover_change — seedance 5, gemini-omni 5.

**Media availability**: all 502 `runs.result_local_path`, all 905 completed
`run_segments.local_result_path` and all 124 project sources exist on disk
(77 GB under `data/`); all 500 `done` runs have a Drive id. Media of
**superseded** attempts does not exist (overwritten paths).

### 2.2 Representative cases (linked, not yet labeled)

| Case | Kind | Evidence |
|---|---|---|
| Run 3b1ea5c7-es (AICP-243) | failure → manual rebuild success | v2v Seedance 2.5 (9 refs) and Gemini Omni v2v kept the source's female hands; images-only Seedance 2.5 rebuild + spliced "Listenify" + CTA delivered in place |
| Run 0abca08e (AICP-256) | manual rebuild success + cost | images-only 1080p, 4740 credits/30 s; first 3 artifact frames trimmed |
| Run cdb8fe52-pt (AICP-230) | manual audio repair | brand word spliced from donor, extra word muted, CTA clip fixed |
| Run 7c29c40f (AICP-276/280/285) | assistant repair success + open defect | Version 4 removed two bad frames ≈7 s; original-person tail + missing CTA still open |
| Runs d740af67-es, 98146271-es | manual repair | swap ended before the scene cut → original person flashes; blacked out in place |
| Run 2d8011bc | failure mode + fix | reference photo pasted on a no-person shot at swap end; fix = end swap at the cut |
| Run c9ba8c0a-fr | manual repair | Seedance copied a leftover subtitle word; delogo + re-burn captions |
| Run faffd482 | manual repair | invented static phone on the wall; source region overlaid in place |
| Run da70b937 | manual repair | whole-track original-audio swap with atempo |
| Seedance 2.5 truncation | measured model behavior | 7/11 full-length; intermittent 14.72 s cut; short results are accepted and flagged, not retried (`pipeline_v2._download_and_complete_segment`) |
| Celebrity moderation | provider refusal | kie refuses a real named celebrity; retries never help |

### 2.3 The three operator observations (Operator, 2026-10-26, AICP-293)

| # | Statement | Classification | Resolved facts | Evidence gap |
|---|---|---|---|---|
| O1 | Avoid Gemini Omni for **text replacement** in the scenarios tested; identify where it works | operator constraint (scoped) + empirical claim | registry key `gemini-omni` (`kie_model_id` `gemini-omni-video`); "text replacement" maps to subtitle_removal / cover_change projects — 12 subtitle_removal + 5 cover_change Omni runs are `done` | which runs were the failing tests and which succeeded is unrecorded; Omni is still the subtitle_removal **default** in `project_types.py`, which this observation contradicts — flag for operator decision, do not change silently |
| O2 | Seedance 2 preferable and cheaper than 2.5 when a long continuous shot is unnecessary | empirical preference (operator-reported) | "Seedance 2" is ambiguous between `seedance` (2.0), `seedance-fast`, `seedance-mini`; 2.5 = `seedance-2-5`; settled spend draws exist only since 2026-10-17 | variant must be confirmed by Operator; comparable-quality labels absent; cost comparison must use settled draws, not the wiki's community estimates |
| O3 | 2–3 source characters keep their original face under Seedance despite different targets and ChatGPT frame-derived refs | empirical claim about specific sources | candidates by override intensity (NOT confirmed): project A (29 overridden segments over 11 runs), project B (10/11), project C (8/11), projects D and E | **assets not identified**; no label records face retention; first operator task in AICP-297/299 is to name them |

Each becomes a `kb_rule` (§4.6) with its own kind and an explicit
`evidence_gaps` list; none is promoted to a universal vendor fact.

### 2.4 Correction flows that exist today

Segment re-run (in place, history lost), retry run, range attempt
(append-only), segment GPT-Image regen (append-only), audio patch
(append-only, in-place Drive update), tail edit, captions, overlays, assistant
immutable versions + Use-as-Run-result promotion, manual scripts
(`manual_creatives`), and out-of-app manual repairs documented only in Linear.

## 3. Principles that shape the schema

1. **Separate tracks**: automatic output (attempt + artifact), automatic
   assessment, operator verdict, correction and accepted final are different
   rows with different `actor_kind`; nothing derives a verdict from a status.
2. **Immutable identity + append-only observations**: a thing that becomes
   known over time (an attempt's status, output, settled cost) is an immutable
   identity row plus insert-only observation rows — never an update. Feedback,
   rule versions, reviews, decisions and evaluation membership are insert-only.
   Service layer + a PostgreSQL trigger reject UPDATE/DELETE (precedent:
   `assistant_turns_immutable`, `h8c3d0e5f792`).
3. **Capture at the moment of truth, not at the moment of loss**: the provider
   request is recorded when it is admitted and the output when it is
   downloaded. Later in-place resets therefore destroy nothing the knowledge
   base still needs. (Revision 1 snapshotted at reset time; the review showed
   the override and the result file are already gone by then.)
4. **Unknown stays NULL** and is named in `missing_facts`; nothing infers a
   reason, verdict, time range or cost.
5. **Soft references** to existing tables (plain strings, no FK) plus a
   snapshot of identifying fields, because Runs and derived creatives can be
   hard-deleted and the knowledge record must outlive them.
6. **No new PostgreSQL ENUM types**: `varchar` + `CHECK`.
7. **Scope is resolved server-side** from the owning project of every id a
   request names; a body id never establishes scope.
8. **Exact versions**: model key **and** provider model id, canonical request
   payload + sha256, content hashes of every input, code revision.

## 4. Schema (one additive Alembic revision, AICP-295)

All tables prefixed `kb_`; UUID string ids; `timestamptz`; every
insert-only table also has a `seq bigint GENERATED ALWAYS AS IDENTITY` used as
the total order (timestamps are not tie-breakers).

### 4.1 `kb_cases` — one unit of production work

`id, case_key unique (run:<run_id> | segment:<run_id>:<segment_def_id> |
derived:<table>:<id> | manual:<manual_creative_id> |
external:<parent_case_id|linear-id>:<n>), scope
(run|segment|derived|manual|external), operation (face_swap|subtitle_removal|
cover_change|localisation|images_only_rebuild_manual|repair), project_id
(NULL only for external cases with no project), run_id, run_segment_id,
segment_def_id, parent_case_id (segment → run), source_artifact_id,
source_group_key (lineage key from the evaluation manifest), language,
time_start_sec, time_end_sec (from the frozen `segment_snapshot`), origin
(live|import), import_batch_id, snapshot jsonb, missing_facts jsonb, seq,
created_at`. Insert-only.

### 4.2 `kb_case_versions` — the case's current-output timeline

`case_id, version_no, unique(case_id, version_no), artifact_id, reason
(initial_output|restitch|range_attempt|audio_patch|assistant_promotion|
import), source_record, observed_at, seq`. Insert-only.

Which content a case's versions track is fixed by its scope:

| Case scope | Current-content source | Versions appended by |
|---|---|---|
| `run` | `runs.result_local_path` (the Run's own delivered final) | finalize capture after every stitch, the range/audio-patch/assistant promotion capture (§6.1), and the feedback path's sync check |
| `segment` | the segment's model output | output capture (§6.1) |
| `derived` (tail edit, caption, overlay) | that row's immutable `output_path` | importer, once, version 1 only |
| `external` (operator-linked repair) | the supplied file/link | the external-repair request, version 1 only |
| `manual` | `manual_creatives` row's file | importer, version 1 only |

So a tail edit, caption or overlay is **its own case** (`parent_case_id` = the
Run case), never a version of the Run case, and an external repair is its own
case unless the operator explicitly declares it replaced the Run's delivered
file (then the Run case sync check sees the new content only if
`runs.result_local_path` really changed; otherwise the declaration stays a
`link_external` event and the external case carries the verdict). Version
allocation always follows the lock hierarchy in §6.2.

### 4.3 `kb_artifacts` — immutable media reference

`id, kind (source|source_clip|reference_image|reference_audio|model_output|
stitched_final|delivered_final|assistant_version|tail_edit|caption|overlay|
external_repair), locator_kind (local_path|preserved_copy|drive_file|
assistant_artifact|kie_url|external_url), locator, sha256 (NULL = not hashed),
bytes, duration_sec, width, height, parent_artifact_id, time_start_sec,
time_end_sec, project_id, seq, created_at`; unique `(locator_kind, locator,
sha256)`. Insert-only. Availability is an observation
(`kb_artifact_checks(artifact_id, present bool, sha256_matches bool,
checked_at, seq)`), not a mutable column — a `local_path` artifact whose file
was later overwritten shows `sha256_matches=false`.

### 4.4 `kb_attempts` + `kb_attempt_observations`

`kb_attempts` (immutable identity, written **before** the provider call —
the intent — so a worker death between the call and any later write still
leaves a durable, reconcilable trace):
`id, case_id, idempotency_key unique, attempt_kind (generation|resubmit|
rerun|range_attempt|segment_image|preprocess_image|audio_patch|tail_edit|
caption|overlay|assistant_version|run_result_promotion|manual_script|
external_repair), origin (automatic|operator|assistant|external),
parent_attempt_id, provider (kie|byteplus|local|external), model_key,
provider_model_id, model_family (from `ai_models` at capture),
target_kind + target_id (the row the call is for, e.g. `run_segment`,
`range_attempt`, `segment_image_job`, `run_preprocess`), dispatch_key
(§6.1 A), predecessor_task_id (the task this dispatch replaces, if known),
request jsonb (the canonical
provider request actually sent: prompt, reference URLs in order, uploaded clip
URL, resolution, duration, generate_audio, voice refs, aspect, …),
request_sha256, input_hashes jsonb (ordered reference image sha256s, clip
sha256, voice sha256), reference_set_sha256 (sha256 of the ordered reference
content hashes; NULL if any hash is unknown), recipe_context jsonb (segment
range, pre/post roll, mute_source, audio_mode, max_segment_sec, preprocess
prompts), code_revision, seq, created_at`. No post-call value lives here.

`kb_attempt_submissions` (insert-only, the provider's acceptance):
`attempt_id unique, provider_task_id unique, spend_call_key unique (NULL
when no draw), submitted_at, seq`. This is the join to `kie_spend_draws` and
the only place a task id is stored. It is inserted **in the same transaction**
as the attempt's `status=submitted` observation, so an accepted attempt always
has both. A `redispatch` observation (payload: worker id, job id, reason) is
appended when a recovered job reuses an open intent (§6.1 A); attempts with one
or more are the "possibly charged more than once" set.

`kb_attempt_observations` (insert-only, ordered by `seq`): `attempt_id, kind
(status|output|cost_estimate|cost_settled|error|ambiguous|redispatch), status
(submitted|succeeded|failed|short_result|refused|superseded|
intent_without_submission|unknown),
output_artifact_id, mcredits, cost_source (spend_draw|run_segment_snapshot|
linear_report), error_class, payload jsonb, observed_at`. Current attempt state =
latest observation of each kind. `run_segment_snapshot` cost is labeled as
"current attempt at import time" and is never used for superseded attempts
(the column is overwritten by re-runs).

### 4.5 `kb_feedback_events` — append-only assessments, verdicts, corrections

`id, request_id (→ `kb_requests`), ordinal (position within the request),
unique(request_id, ordinal), case_id, case_version_no
(required for verdict/defect/assessment), artifact_id (exact content judged),
historical bool (true = deliberately about a non-current version; never
affects current state), event_type (auto_assessment|operator_verdict|defect|
correction|link_external|retraction|export_observed), actor_kind
(system_check|assistant|operator|importer), actor_label (**unverified display
annotation** — one shared dashboard login), verdict (accepted_as_is|
needs_repair|rejected), defect_code, severity (blocker|major|minor),
time_start_sec/time_end_sec (validated inside the artifact duration),
check_kind + claim + coverage jsonb + limitations (auto_assessment), method
(manual|automatic|assistant|external) + before_artifact_id + after_artifact_id
+ resolved (yes|no|unknown) + incremental_mcredits + operator_minutes +
cost_source (correction), note, supersedes_event_id, seq, created_at`.

**Verdict state machine** (per case, deterministic):
- Effective verdict of a case = the latest (`seq`) non-retracted,
  non-`historical` `operator_verdict` whose `case_version_no` equals the
  case's **latest** version. If the latest version has none → `unreviewed`
  (UI also shows "previous version N was accepted" when true).
- **Accepted final** = the artifact of the latest version iff its effective
  verdict is `accepted_as_is`. A new version always resets the case to
  `unreviewed`; an old acceptance never carries forward silently.
- Filing a `blocker`/`major` defect on a version whose effective verdict is
  `accepted_as_is` is refused unless the same request also carries a
  `needs_repair` or `rejected` verdict (the UI sends both atomically), so a
  contradicting defect cannot coexist with a live acceptance.
- `export_observed` (Final Creatives stamp) is a separate fact and never counts
  as acceptance.

**Requests / receipts**: `kb_requests(idempotency_key pk, request_sha256, kind,
case_id, child_event_ids text[], created_at)`. One HTTP submit = one request
row + all its child events in one transaction (e.g. a `needs_repair` verdict
and two defects). Replay with the same key and hash returns the stored
receipt; same key, different hash → 409. Events carry no key of their own.

**Retraction** (deterministic): a `retraction` event names exactly one
`supersedes_event_id` in the **same case**; a partial unique index allows at
most one retraction per target; retractions themselves cannot be retracted (to
restore a retracted judgment, submit it again as a new event); only an
operator may retract an operator event, and only the importer/system an
automatic one. Effective events = all non-retraction events with no
retraction pointing at them; this query is part of the contract and tested.

### 4.6 Guidance: rules, versions, reviews, evidence

- `kb_rules(id, slug unique, created_at)`.
- `kb_rule_versions(id, rule_id, version_no, unique(rule_id, version_no),
  kind (operator_constraint|empirical_guidance|machine_hypothesis),
  statement, rationale, applicability jsonb (§9.1 canonical form),
  recommendation jsonb {action: forbid|require|avoid|prefer|warn, target},
  confidence (operator_asserted|low|medium|high), shareable bool (reviewer
  asserts the text contains no project-specific content), scope_project_ids
  (NULL = global), model_registry_revision, effective_from, effective_to,
  author_kind (operator|claude|importer), author_label (unverified),
  provenance jsonb, evidence_gaps jsonb, supersedes_version_id, seq,
  created_at)`.
- `kb_rule_reviews(id, rule_version_id, status (proposed|approved|rejected|
  retired), reviewer_label (unverified), note, seq, created_at)`.
- `kb_rule_evidence(id, rule_version_id, case_id, event_id, stance
  (supports|contradicts|context), note, seq, created_at)`.

Effective version of a rule at time T = the **highest `version_no`** whose
latest review (by `seq`) is `approved` and whose window contains T. An approved
older version is therefore automatically excluded once a newer version is
approved; retiring the newest approved version does **not** resurrect an older
one (retire = the rule stops applying; re-approving an old text is a new
version). `machine_hypothesis` versions are never effective. Evidence citing a
holdout case is refused (§8).

### 4.7 Decisions

`kb_decisions(id, idempotency_key, mode (shadow|advisory), requested_by
(assistant_turn|new_run_form|offline_eval), context jsonb (structured query
§9.1), advisor_version, retrieval_params, consulted_rule_version_ids text[],
consulted_case_ids text[], consulted_sha256, recommendation jsonb (ordered
options with cited ids, conflicts, unknowns, hypotheses), seq, created_at)` and
`kb_decision_outcomes(id, decision_id, chosen jsonb, overrode bool,
override_note, attempt_id, seq, created_at)`. Written only by the trusted
advisor service, never from model-supplied provenance.

### 4.8 Import, invalidation and evaluation bookkeeping

- `kb_import_batches(id, source, code_revision, dry_run, started_at, seq)` and
  `kb_import_batch_observations(batch_id, kind (finished|failed|aborted),
  counts, report, observed_at, seq)`, both insert-only.
- `kb_import_members(batch_id, target_kind, target_id, unique(target_kind,
  target_id, batch_id))` — every row an import creates (cases, artifacts,
  attempts, observations, case versions, events, **rule versions**, rule
  evidence, eval membership) is listed here, so invalidation reaches all of
  them, not only cases.
- `kb_invalidations(id, target_kind (import_batch|case|attempt|event|
  rule_version|artifact), target_id, reason, actor_label, seq, created_at)`
  insert-only. Every read path excludes a row that is invalidated directly or
  through its batch, and — recursively — observations of an invalid attempt,
  versions/events whose artifact is invalid, rule evidence citing an invalid
  case/event, and rule versions from an invalid batch (which therefore stop
  being effective). **This replaces physical
  delete-by-batch** — later human evidence that referenced an invalidated row
  is kept and shown with a "references invalidated import" marker.
- `kb_import_cursors(source pk, cursor jsonb, lease_owner, lease_expires_at,
  updated_at)` — the only mutable table; bookkeeping, not evidence.
- `kb_capture_queue` / `kb_capture_done` — durable deliverable-capture work
  items (§6.1 C), both insert-only.
- `kb_eval_manifests(version pk, sha256, body jsonb, created_at)` — the
  committed manifest file, loaded verbatim and verified against its sha256.
- `kb_eval_membership(case_id, manifest_version, split, group_key, seq)`,
  `unique(case_id, manifest_version)`; insertion validates `group_key` against
  the manifest body and the case's project.

### 4.9 Constraints, indexes, triggers, downgrade

CHECK for every code list; `kb_attempt_submissions.provider_task_id` NOT NULL
UNIQUE and `spend_call_key` UNIQUE (NULLs allowed — calls without a draw); indexes on
`project_id`, `(case_id, seq)`, `(operation, model_key)`, `defect_code`,
`reference_set_sha256`; GIN `to_tsvector('simple', statement)`. Immutability
trigger on every insert-only table. **Downgrade refuses when any `kb_` table
has a row**, the same guard as `j0e5f2a7b914` (L81–88); with data present
rollback is "flags off, keep tables".

## 5. Defect taxonomy (v1)

| Code | Covers | Automatic check may only… |
|---|---|---|
| `segmentation_boundary` | swap ends before/after the scene cut, original person flashes, ref pasted on an empty shot | suggest (frame sample) |
| `source_face_retained` | target face not applied / original face remains | suggest (face similarity); never pass |
| `identity_drift` | persona changes across segments/seams | — |
| `body_hands_wrong` | wrong-gender hands, nails, body | — |
| `reference_quality` | blurry/wrong GPT-Image ref, wrong crop | — |
| `text_artifact` | leftover/hallucinated subtitle; failed text replacement | suggest (OCR) |
| `caption_error` | burned caption wrong text/timing/dropped line | suggest (caption diff) |
| `script_adherence` | missing/extra/paraphrased words | flag (ASR diff) |
| `pronunciation` | brand/word mispronounced | **operator only** |
| `lip_sync` | mouth ≠ audio | **operator only** |
| `audio_continuity` | drift, gap, wrong voice, level jump | flag (probe) |
| `scene_continuity` | invented objects, background change | — |
| `stitching` | seam glitch, frame dup/drop | flag (probe) |
| `truncation` | output shorter than the clip sent | determine (duration probe) |
| `cta_ending` | missing/wrong CTA or tail | — |
| `provider_refusal` | moderation (celebrity), provider internal error | determine (provider response) |
| `resolution_quality` | soft/low detail | suggest (PSNR) |
| `other` | requires a note | — |

Allowed `(check_kind → claim)` pairs are enforced: `asr →
transcript_match|transcript_mismatch`; `frame_sample →
defect_observed|no_defect_observed_in_sample` (sampled timestamps in
`coverage`); `face_similarity → suggest_source_face_retained`;
`duration_probe → short_result|full_length`. No automatic actor can write an
`operator_verdict`, nor any pass for `pronunciation`/`lip_sync`.

## 6. Capture, write paths and file boundaries

### 6.1 Where capture happens (AICP-299; CLAUDE.md §2 review again before code)

Capture is three kinds of insert, each in its **own short session** (never a
savepoint in the caller's transaction), never raising into the pipeline:

**A. Intent → submission** (every paid kie call). Immediately **before** the
provider call: insert the `kb_attempts` intent with the exact request about to
be sent and input hashes. Immediately after acceptance: insert the
`kb_attempt_submissions` row (task id, spend call key) together with the
`status=submitted` observation, in one transaction. Call sites (at
`e6cd9de`): `pipeline_v2` initial swap submit (`_create_swap_task` ~L1062)
and in-poll resubmit (~L1512); `_record_image_draw` caller (~L2778, run-level
GPT Image preprocessing); `pipeline_range_attempt` (~L810);
`pipeline_segment_image` (~L151).

*Dispatch identity.* `dispatch_key = sha256(kind, target_kind, target_id,
predecessor_task_id or "none", request_sha256)`. Under a transaction-scoped
advisory lock on `(target_kind, target_id)` the service **reuses an open
intent** (one with no submission and no terminal observation) that has the same
`dispatch_key`, appending a `redispatch` observation, instead of creating a
new one. So a worker that dies after inserting the intent and a recovered job
that sends the same recipe for the same row form **one** logical dispatch. An
intentional new attempt always differs in `predecessor_task_id` (in-poll
resubmit replaces a known task) or in `request_sha256` (rerun with a new
prompt/reference); a rerun with an identical recipe after a *completed* attempt
finds no open intent (the old one has a submission) and creates a new one.
Metadata only — pipeline behaviour is unchanged.

An intent with no submission after a timeout (2 × the job timeout) is marked
`intent_without_submission` by the reconciliation pass, and an intent with a
`redispatch` observation is reported as "possibly charged more than once":
the provider **may** have charged for the lost call. This makes the gap
visible; it does **not** prevent duplicate billing. Today's pipeline already has this window (recovery
re-submits a segment whose task id was never persisted; only range attempts
guard it with `submit_billed_at`). **Changing that retry behaviour is out of
scope here** — the knowledge layer does not alter pipeline semantics — but it is
now visible and counted; a separate Linear issue is proposed for a pipeline-side
pre-call guard on swap/preprocess/segment-image submits.

**B. Output and terminal failure** (every provider result). Immediately after
the result is downloaded and before the pipeline's DB write: hash, insert the
artifact, `output` + `status` observations. Sites: swap poll download
`_download_and_complete_segment` (~L1365) and submit-phase no-rebill resume
(~L1990); range-attempt download (`pipeline_range_attempt` ~L1392–1411);
segment-image output (`pipeline_segment_image` ~L261–288); run-level preprocess
image outputs (`pipeline_v2` ~L3174–3214). Terminal failures (`error`/`refused`)
where each of those paths marks its row failed. Preservation
(`KNOWLEDGE_PRESERVE_ATTEMPT_MEDIA`): swap/range clips are hardlinked to
`data/knowledge/attempts/<attempt_id>.<ext>` (`download_result` writes `.part`
+ `os.replace`, a new inode); image outputs are small and copied.

**C. Deliverable versions** (Run case), in two steps.

*C1 — preserve and record, no lock.* Runs **after** the writer's transaction
commits. For a stitch: in the shared `pipeline_v2._run_stitch_and_deliver` (~L2388; called by
both the production `tasks.run_finalize_run → finalize_run` path and the
compatibility `process_run`), right after the commit that sets
`result_local_path` to `final.mp4` (~L2687–2731) and **before the job releases
the per-run Redis lock** (`tasks.run_lock_key`), so no other stitch
of this Run can overwrite `final.mp4` meanwhile. `media.stitch` writes
`final.mp4` **in place** (`ffmpeg -y`, same inode), so a hardlink cannot
preserve it: hash it, and with preservation on **copy** it (≤ 100 MB by the
delivery cap) to `data/knowledge/finals/<sha256>.mp4`, re-hash the copy, then
insert the `stitched_final` artifact and a durable `kb_capture_queue` item — no
Run lock involved. `kb_capture_queue(id, capture_key unique, run_id,
artifact_id, artifact_sha256, reason, source_record, c1_seq bigint identity,
created_at)` with `capture_key = sha256(run_id, artifact_sha256, publication_id)`. The
**publication id** names one logical result publication, without adding a
column to `runs`:

| Publication | `publication_id` | Durable because |
|---|---|---|
| stitch in `_run_stitch_and_deliver` | `stitch:<run_id>:<invocation id>` — passed in as an explicit `publication_id` argument: `tasks.run_finalize_run` supplies the current RQ job id (`rq.get_current_job()`, RQ 1.15.1); the synchronous compatibility `process_run` mints a UUID once at entry and passes it through the whole call | fixed for one finalize invocation, so a C1 retry inside it reuses the key; every later invocation (including a recovered chain, which `recovery.py` starts afresh) gets a new id |
| range promotion | `range_attempt:<attempt id>` | append-only attempt row, one promotion each |
| audio patch promotion | `audio_patch:<patch id>` | append-only patch row |
| assistant Run-result promotion | `assistant_promotion:<promotion receipt id>` | append-only `assistant_run_result_promotions` row |
| reconciliation-found change | `reconcile:<run_id>:<observed hash>:<latest version_no>` | tied to the version it follows |

C1 stores the `publication_id` as the queue item's `source_record`, and C2
copies it unchanged into the appended case version's `source_record`, so the
timeline distinguishes an original finalize, a recovery, a reconciliation and
a same-byte re-publication. C1 inserts with `ON CONFLICT (capture_key) DO NOTHING` and the artifact is
unique on `(locator_kind, locator, sha256)`, so a retried C1 creates nothing
twice. A finalize re-run by recovery after a crash between C1 and the lock
release gets a new job id, but C2 consumes it as `duplicate` (same bytes as the
latest version, nothing published in between). A genuine A → B → A sequence
has three publication ids, and the third is appended because the latest
version is B. For
the promotions (`pipeline_range_attempt._promote` ~L1226,
`pipeline_audio_patch._promote` ~L462, `assistant_run_result` ~L390): they
re-point `result_local_path` to a **new** path and never overwrite the previous
final, so C1 runs after their commit with no time pressure.

*C2 — allocate the version.* The import tick drains `kb_capture_queue` **per
Run, in `c1_seq` order, while holding that Run's row lock** (§6.2 hierarchy —
the Run lock is also the claim: two drainers serialize on it). For each
pending item: if its `artifact_sha256` equals the latest version's content →
consumed as `duplicate`; if a version already exists that came from a queue
item with a higher `c1_seq` → consumed as `stale` (an older final never becomes
current after a newer one); otherwise append the version. Consumption is
`kb_capture_done(queue_id unique, outcome (appended|duplicate|stale),
case_version_no, seq)`, inserted in the same transaction as the version. A
lock wait delays the version, never loses or reorders a final.

If C1 itself fails (disk, hash mismatch because the file changed during the
copy), the artifact is recorded with `sha256` NULL and a `missing` check, and
the miss is counted.

Because capture happens at intent, output and finalize, the in-place resets in
`api_v2.rerun_segment` (override applied ~L4849 *before* the reset ~L4858), the
submit-phase reset (~L2094) and the in-poll `seedance_task_id` replacement
(~L1520) no longer lose anything; they need no hook. Short results are **not**
retried by the pipeline (`_download_and_complete_segment` docstring, ~L1355) —
revision 1 wrongly said they were.

Failure policy: capture never fails a paid generation. Every capture error
increments a counter and writes a structured log line. The **coverage
reconciliation** (importer, §6.3) checks both directions: every
`kie_spend_draws.call_key` without an attempt → `ambiguous` attempt from the
draw (cost known, request unknown); every intent without submission →
`intent_without_submission`; every Run whose `result_local_path` content hash
**differs from its latest Run case version** (not merely "absent from
history") and has no pending queue item → queue a `reconcile:` publication (or
`missing` if the file changed meanwhile).
Pilot gate (AICP-300): during the pilot window, 100 % of draws and finals are
captured or explicitly marked ambiguous/missing, and the ambiguous count is
reported; otherwise the pilot stops.

Rollout order: migrate with flags off → deploy (overlays, two workers, nginx)
→ wait until no run is in an active status and the reconcile tick has
converged (drain rule, memory/CLAUDE) → enable `KNOWLEDGE_CAPTURE_ENABLED`.
Runs admitted before enablement have no admission row; the reconciliation pass
covers them from draws.

### 6.2 Feedback write path and the single lock hierarchy (AICP-299)

**Lock hierarchy for every writer** (feedback, capture C, importer,
reconciliation, external repair): `runs` row (`FOR UPDATE`, when the case is
Run-bound) → `kb_cases` rows in ascending `id` order. Case creation happens
while holding the Run lock (`INSERT … ON CONFLICT (case_key) DO NOTHING`, then
`SELECT … FOR UPDATE`). No writer ever locks a case and then a Run. Capture A/B
inserts only attempts/observations/artifacts (no version allocation) and takes
no Run lock. This matches the existing result-mutating flows, which lock the Run
first (`api_v2._lock_run_for_result_mutation`,
`pipeline_range_attempt._promote`, `assistant_run_result` ~L175).

Feedback request, one transaction:
1. Path names the **case** (`/cases/{case_id}/…`); resolve case → project and
   check `KnowledgeScope`; every body id is re-resolved and must belong to the
   same case, else 404.
2. Take the locks above.
3. Run case only: hash the live `runs.result_local_path`; if it differs from the
   latest version, append a version (reason `restitch` or the most specific
   reason derivable, else `import`). Derived/external/manual cases have a fixed
   version 1 and skip this.
4. `expected_version_no` must equal the latest version → else 409 with the
   current version, unless `historical=true` (stored as historical; never
   affects state).
5. Insert the `kb_requests` receipt and all child events; replay semantics §4.5.

### 6.3 Importer (AICP-297) and incremental capture of append-only flows

Tail edits, captions, overlays, audio patches, range attempts, segment image
jobs, assistant versions/promotions and manual creatives are already
append-only, so a bounded incremental importer turns them into
attempts/case-versions/corrections. It runs as its **own** RQ job
(`knowledge_import_tick`, enabled by `KNOWLEDGE_CAPTURE_ENABLED`), not inside
`run_reconcile_tick` (which stays untouched): Redis lease + `kb_import_cursors`
lease columns, ≤ 200 source rows per tick, cursor per source on `(created_at,
id)`; draws are scanned by `settled_at IS NOT NULL` with no `cost_settled`
observation (draws have no `updated_at`; `mcredits` is replaced on
settlement). Historical import (dry-run, then isolated apply) uses the same
code with `origin=import`.

### 6.4 Files

New package `backend/app/knowledge/` (imported from `app/models.py` like
`assistant_models`):

| File | Issue | Responsibility |
|---|---|---|
| `models.py`, `taxonomy.py` | 295 | §4 models; code lists; claim rules |
| `access.py` | 295 | `KnowledgeScope` = `ProjectAllowlist.from_settings` ∩ `KNOWLEDGE_PROJECT_IDS`; `sql_filter`, `allows_project`, rule-visibility (§6.5) |
| `store.py` | 295 | typed idempotent writers; version/state resolution; invalidation-aware readers |
| `capture.py` | 299 | intent/submission (A), output/failure (B) and deliverable-version (C) capture at every site enumerated in §6.1 |
| `importer/` | 297 | dry-run/apply, sources, coverage reconciliation, seed YAML (`knowledge/seed/*.yaml`) |
| `retrieval.py`, `advisor.py` | 301 | §9 |
| `evaluation.py`, `scripts/knowledge_eval_report.py` | 302 | §8 |
| `api.py` + `templates/v2/partials/knowledge_feedback.html` | 295/299 | §6.6, §7 |
| `QueryKnowledgeAction` in `assistant_agent_contracts.py` + handler in `assistant_agent_steps.py` | 301 | closed, read-only action (§6.5) |

### 6.5 Authorization and leakage

- The dashboard has **one shared basic-auth login**. `actor_label` and
  `reviewer_label` are unverified annotations; the design does not claim to
  know *who* accepted or approved. Per-user identity is a separate auth change,
  out of scope; the release packets state this.
- Cases, artifacts, attempts and events are visible only within
  `KnowledgeScope`.
- A non-`shareable` rule version is returned only if **every** project in its
  `scope_project_ids` (when not NULL) **and** every project of its cited
  evidence is inside the caller's scope; otherwise it is suppressed entirely.
  `shareable` (a reviewer's explicit assertion that statement and rationale
  contain no project-specific content) is the only cross-scope exception, and
  then out-of-scope citations are returned as a count only.
- `QueryKnowledgeAction` accepts only structured filters (operation, model
  key, language, resolution, segment length, the current conversation's run);
  the handler re-derives project scope from the conversation (as
  `assistant_agent_loop` re-authorizes each tool use), re-checks scope on every
  returned id, caps the result (≤ 20 rules, ≤ 30 cases, ≤ 8 KB), and itself
  writes the `kb_decisions` row with the ids it returned. The assistant cannot
  write cases, events or rules.

### 6.6 HTTP API (`/api/v2/knowledge`, basic auth + CSRF, `KNOWLEDGE_ENABLED`)

```
GET  /runs/{rid}/timeline           Run case + its segment/derived/external child cases, versions, attempts+observations, effective events, accepted finals
POST /cases/{cid}/feedback          {idempotency_key, expected_version_no, historical?, verdict?, defects?[], note?}
POST /cases/{cid}/corrections       {idempotency_key, before_artifact_id, after_artifact_id?, method, resolved, mcredits?, minutes?, note?}
POST /runs/{rid}/external-repair    {idempotency_key, drive_link|upload, replaced_delivered?: bool, note?}  → new external case + link_external
POST /events/{eid}/retract          {idempotency_key, reason}
GET  /rules ; POST /rules ; POST /rules/{id}/versions ; POST /rule-versions/{id}/reviews
```

## 7. Feedback UX (AICP-299)

Run page card "Outcome" (flag `KNOWLEDGE_FEEDBACK_UI`, pilot projects only):

1. One block per case shown on the page — the Run's delivered final, and each
   tail edit / caption / overlay / external repair as its own case. Header:
   version N being judged, content hash prefix, when it became current;
   "previous version accepted" note when applicable.
2. **Accept as-is / Needs repair / Reject**; default **Unreviewed**, nothing
   preselected.
3. "Add defect": code (§5), severity, start/end from the player ("Set
   start"/"Set end"), optional note. A blocker/major defect on an accepted
   version asks for Needs repair/Reject in the same submit.
4. Corrections list, auto-filled from captured flows with method and known
   cost; operator marks resolved yes/no and minutes.
5. "Link external repair": Drive link or upload + optional reason (blank
   stays unknown).
6. Timeline with actor badges (Auto / Assistant / Operator / Imported) and
   "invalidated import" markers.

HTMX forms carry `idempotency_key` + `expected_version_no`; a 409 re-renders the
card with the current version.

## 8. Evaluation design (fixed before any rule development)

### 8.1 Split

Committed: `scripts/knowledge_holdout_split.py`,
`docs/knowledge/lineage-families-v1.txt`,
`docs/knowledge/holdout-manifest-v1.json` (sha256
`ea5f5d52ce1d5b5d8485fb5d29116090d9cf337113ca35405f843bf194d2766f`).
Unit = source lineage group (identical normalized source sha256 ∪
`source_run_id` parent ∪ curated name family): 124 projects → 54 groups; 14
already-discussed projects pinned to development; 25 % per stratum by salted
hash → face_swap 10/39, localisation 2/5, subtitle_removal 3/10 groups, 21
projects (~115 runs). Amendments only move holdout → development (new manifest
version); removed groups are reported as excluded.

Known leakage gaps, printed in every report: unnamed projects and unlisted name
variants; the **same person across different creatives** (no identity
grouping — the same creator name appears on both sides).

### 8.2 Enforcement (not only at citation)

- Retrieval and the advisor exclude holdout cases (and any rule version whose
  evidence includes one — refused at write time anyway).
- Rule-evidence writes citing holdout are refused.
- Knowledge browse views and exports hide holdout operator labels; the
  holdout labeling view shows one case at a time with no advisor output and no
  other labels (blinded).
- Residual risk stated honestly: with one shared login, a person can still
  open a holdout Run page and later write a rule from memory. Mitigation is
  procedural (labelers do not author rules during the evaluation window) and
  the report records which rule versions were created after holdout labeling
  started.

### 8.3 What offline evaluation can and cannot show

Historical data has outcomes only for the option the operator actually chose,
so offline evaluation **cannot** show that the advisor raises acceptance or
lowers cost. AICP-302 reports only:
- QA-vs-operator agreement on labeled holdout cases (confusion matrices per
  check kind, false passes such as similarity "ok" on `source_face_retained`);
- advisor-vs-historical-choice agreement and, where the historical choice was
  labeled bad, whether the advisor would have warned (retrospective
  warning recall) — explicitly labeled retrospective;
- descriptive production metrics (below) for the baseline period.

Improvement claims require the **prospective** decision-logged pilot
(AICP-303 advisory mode, AICP-304): every advised decision and its override
outcome are logged; comparison is advised-and-followed vs overridden vs
pre-pilot baseline, with sample sizes and confidence intervals, or reported
inconclusive. Randomization is an operator choice, not assumed.

Descriptive metrics (denominators always shown): accepted-without-correction
rate (accepted_as_is with zero correction events / reviewed cases), reviewed
defect rate per code, corrections and operator minutes, attempts per accepted
creative (superseded, failed and `ambiguous` attempts included), settled cost
per accepted creative (pre-2026-10-17 cost flagged as `run_segment_snapshot`
or unknown).

### 8.4 Labels

Holdout has **no operator labels** today. A blinded labeling pass (target ≥ 40
holdout runs across strata, labeler ≠ rule author during the window) is a
prerequisite for any AICP-302 quality metric; without it the report says
"unreviewed — inconclusive".

## 9. Retrieval and advisor semantics (AICP-301)

### 9.1 Canonical applicability and matching

Applicability is a conjunction over a fixed dimension list, each either absent
(wildcard) or a non-empty sorted set: `operation`, `model_key`,
`model_family`, `subject_id`, `project_id`, `language`, `resolution`,
`segment_sec_band` (half-open: `[0,10)`, `[10,15)`, `[15,30]`, `>30`,
`unknown`), `mute_source`,
`generate_audio`. A version matches a query iff every specified dimension
contains the query's value; if the query lacks a value for a specified
dimension the version is reported as "not evaluated: missing <dim>" and is not
applied. Specificity = the tuple of specified dimensions weighted in the order
`subject_id > project_id > model_key > model_family > operation > others`.

### 9.2 Precedence (total order) and conflicts

Among effective versions (§4.6) that match:
1. `operator_constraint` (forbid/require) over `empirical_guidance`;
   `machine_hypothesis` is listed as hypotheses only.
2. Higher specificity tuple first; then later `effective_from`; then higher
   `seq`.
3. If two versions at the top precedence level recommend incompatible actions
   on the same target (forbid vs require/prefer), the advisor returns
   `conflict` with both cited and **no recommendation** for that target.
Contradicting evidence and evidence gaps of every applied version are always
returned.

### 9.3 Difficult source (O3)

Subject-driven advice is **off until an operator has named subjects** (O3's
assets are unknown today). `kb_subject_links(subject_id, project_id,
segment_def_id NULL, time_start_sec, time_end_sec, seq)` rejects overlapping
links to different subjects for the same project/segment range; a segment
covered by several subjects or none is `subject=unknown/multiple` and gets no
subject advice.

Trigger: ≥ 2 non-retracted operator `source_face_retained` defects for the
same subject, on attempts of the same `model_family` (from `ai_models` at the
recorded `code_revision`) with **distinct non-NULL `reference_set_sha256`**.
Then the advisor returns `avoid` for an unchanged v2v attempt in that family
and lists alternatives that have operator-accepted cases in development data;
the images-only rebuild appears only as "manual, not available in the app
(AICP-258), changes timing and gestures" and is never auto-selected. Shadow
mode records it; advisory mode shows it with an override control whose choice
lands in `kb_decision_outcomes`. Retries are not blocked: knowledge has no
execution authority.

No embeddings: the corpus is hundreds of cases and a few dozen rules;
metadata filtering + PostgreSQL full-text keeps every decision explainable.

## 10. Feature flags (all default off)

`KNOWLEDGE_ENABLED`, `KNOWLEDGE_PROJECT_IDS`, `KNOWLEDGE_CAPTURE_ENABLED`,
`KNOWLEDGE_PRESERVE_ATTEMPT_MEDIA`, `KNOWLEDGE_FEEDBACK_UI`,
`KNOWLEDGE_ADVISOR_MODE` (`off|shadow|advisory`).

## 11. Migration, backfill, rollback

- **295/296**: one additive revision on the head current at implementation
  (`j0e5f2a7b914` today; other sessions add migrations — re-check). Only
  CREATE TABLE/INDEX/CHECK/TRIGGER; no change to existing tables; downgrade
  refuses with data. Rehearse upgrade → downgrade → upgrade on isolated
  PostgreSQL 16 restored from a production schema dump, then upgrade on a
  restored full dump. Release: Drive backup, migrate first, deploy with
  assistant overlays + `--scale worker=2`, restart nginx, verify through
  :8847/https, flags off.
- **297/298**: dry-run against production read-only → isolated apply on a
  restored dump → release packet → authorized apply with an
  `kb_import_batches` row. Rerun = no-op. Correction of a bad import =
  `kb_invalidations` on the batch (append-only), then a corrected batch.
  Physical deletion is not part of the plan once any human event exists.
- **299/300**: capture + feedback UI per pilot project, rollout order §6.1.
- **301/303**: advisor `shadow` first; `advisory` only after explicit
  activation.

## 12. Child contract reconciliation

| Issue | Contract from this design |
|---|---|
| 295 | §4 schema incl. observations, case versions, invalidations, eval manifest tables; §5 validation; `store/access/taxonomy`, rule endpoints and read timeline; idempotency, triggers, downgrade guard; PG tests (`RESKIN_TEST_PG_URL`) for idempotency, concurrency, triggers, scope |
| 296 | §11 release packet; flags off |
| 297 | importer + coverage reconciliation + seed YAML (O1–O3, §2.2 cases); manifest load + membership; dry-run report |
| 298 | production dry-run + authorized apply; invalidation as rollback |
| 299 | §6.1 capture A/B/C at every listed site (new §2 design review of the pipeline hooks), §6.2 lock hierarchy + write path, §7 UI |
| 300 | per-project enablement, rollout order, 100 % capture-coverage pilot gate |
| 301 | §9 + `QueryKnowledgeAction`; subject advice off until subjects are named |
| 302 | §8.3 metrics only; blinded labeling pass prerequisite; no improvement claim offline |
| 303 | shadow → advisory; decision-logged prospective comparison |
| 304 | integrated pilot verification |

## 13. Operator decisions

Answered by Operator on 2026-10-27:

1. O2: "Seedance 2" means base **`seedance`** (2.0), not Fast or Mini. O2
   compares `seedance` with `seedance-2-5`.
2. O3: the 2–3 stubborn source characters will be named by Operator later.
   Until then the §2 candidates stay unconfirmed and subject advice stays
   off (§9.3).
3. O1: recorded as an **operator constraint the advisor warns on**
   (Gemini Omni for text replacement). The subtitle_removal default is not
   changed by this work.
4. An unverified `actor_label` is **acceptable for the pilot**; per-user
   identity is not a prerequisite for rule approval.

Still open:

5. Who does the blinded holdout labeling for AICP-302, and when.
6. O3 subject names (see 2).

## Review record

**Round 1 — GPT-5.6 Terra (Codex CLI 0.163.0, `model_reasoning_effort=high`,
read-only sandbox), session `01a17bd4-8972-71f2-ae69-9d88c926c360`, on
`e6cd9de..43f831c`.** Verdict "not ready to implement": 3 BLOCKER, 6 MAJOR,
2 MINOR. Every cited code fact was re-checked and holds. Resolution in
revision 2:

| Finding | Resolution |
|---|---|
| B1 reset-time capture records the new override / loses overwritten output | capture moved to admission (`_record_swap_draw`, `_record_image_draw` and the range/segment-image draw sites) and both output-download points (§6.1); resets need no hook; capture gaps observable via draw reconciliation and a pilot gate |
| B2 holdout isolated only at citation; offline eval cannot show improvement | §8.2 enforcement in retrieval/rules/browse + blinded labeling; manifest stored/verified in DB, unique membership; §8.3 limits offline claims, prospective pilot for improvement |
| B3 downgrade/delete-by-batch contradict immutability | downgrade refuses with data (§4.9); `kb_invalidations` replaces physical delete (§4.8, §11) |
| M1 accepted final undefined | `kb_case_versions` + deterministic verdict state machine (§4.2, §4.5) |
| M2 concurrency/idempotency | Run→case `FOR UPDATE`, `expected_version_no`, request hash (§6.2) |
| M3 authorization / shared login / rule-text leakage / assistant boundary | §6.5 |
| M4 immutable attempts vs progressive cost | identity + observations; settled-draw scan (§4.4, §6.3) |
| M5 recovery/two workers/rollout/tick | provider-task-id idempotent capture, separate leased import job, rollout order (§6.1, §6.3) |
| M6 precedence/difficult-source nondeterministic | §9.1–9.3 canonical applicability, total order, conflict result, reference-set hash, registry family |
| m1 reset inventory wrong (short results not retried) | corrected (§6.1) |
| m2 subject links | overlap rejection; subject advice off until named (§9.3) |

**Round 2 — same Terra session, on `43f831c..defc604`.** Round-1 B2, B3, M2,
M3, M4, M6 and both MINORs judged RESOLVED; B1, M1, M5 not yet. New: 2
BLOCKER, 5 MAJOR, 2 MINOR. Code facts re-checked: they hold, and one more was
found while checking — `media.stitch` writes `final.mp4` in place with
`ffmpeg -y`, so an old final can only be preserved by copying. Resolution in
revision 3:

| Finding | Resolution |
|---|---|
| B: derived version cannot be accepted via the Run endpoint | derived/external/manual outputs are their own cases; feedback targets `/cases/{cid}`; per-scope current-content table (§4.2, §6.2, §6.6) |
| B: paid submission can vanish (death after provider accept) | capture A writes the attempt **intent before** the call; `intent_without_submission` reconciliation; changing the pipeline's own resubmit behaviour is out of scope and proposed as a separate issue (§6.1 A) |
| M: range/segment-image/preprocess outputs not captured | capture B enumerates all five download sites and their terminal failures (§6.1 B) |
| M: initial stitched final captured only lazily | capture C at every stitch and every result re-point; copy-based preservation (§6.1 C) |
| M: compound submit vs per-event idempotency | `kb_requests` receipt with ordered child events (§4.5) |
| M: invalidation misses rules and non-case rows | `kb_import_members` + recursive invalid-read rule (§4.8) |
| M: lock order unspecified for importer/capture | single hierarchy Run → cases by id for every writer (§6.2) |
| M: retraction graph undefined | same-case, one retraction per target, no retraction of retractions, effective-event query (§4.5) |
| m: band boundaries | half-open bands + `>30`/`unknown` (§9.1) |
| m: import batch completion not modeled | `kb_import_batch_observations` (§4.8) |

**Round 3 — same Terra session, on `defc604..e708181`.** The scope decision
(knowledge layer records the post-accept ambiguity, does not change pipeline
recovery; must not be described as preventing duplicate billing) judged
acceptable. 9 of 10 round-2 findings RESOLVED; deliverable capture not yet.
New: 1 BLOCKER, 3 MAJOR. Promotion paths re-checked: all three re-point to a
new path; only the stitch overwrites in place. Resolution in revision 4:

| Finding | Resolution |
|---|---|
| B: immutable pre-call attempt cannot hold the post-call task id | `kb_attempt_submissions` (unique task id / call key), nothing post-call on `kb_attempts` (§4.4) |
| M: capture C lock timeout can lose an in-place final | C1 preserve+record after commit and before the per-run lock is released, no Run lock; C2 version allocation from a durable queue (§6.1 C) |
| M: intent nonce has no durable source | `dispatch_key` over target row + predecessor task + request hash; open-intent reuse under an advisory lock (§6.1 A) |
| M: non-shareable rule text can leak via scope intersection | non-shareable requires full scope containment of scope projects and evidence (§6.5) |

**Round 4 — same Terra session, on `e708181..d646819`.** All round-3 findings
and the round-2 deliverable-capture finding RESOLVED. New: 1 BLOCKER, 2 MAJOR,
1 MINOR. Resolution in revision 5:

| Finding | Resolution |
|---|---|
| B: capture queue neither idempotent nor ordered | `capture_key` unique + `ON CONFLICT DO NOTHING`; per-Run drain in `c1_seq` order under the Run lock; `kb_capture_done(queue_id unique, outcome)` with `duplicate`/`stale` (§6.1 C) |
| M: `redispatched` not a valid observation | typed `redispatch` observation kind with payload (§4.4) |
| M: accepted submission lacks `submitted` status | submission row + `status=submitted` in one transaction (§4.4, §6.1 A) |
| m: C1 named `process_run` | hook in shared `_run_stitch_and_deliver` (finalize_run and process_run) (§6.1 C) |

**Round 5 — same Terra session, on `d646819..5b37432`** (ADR `49f301b` checked:
no contradiction). Round-4 redispatch, submitted-status and entry-point findings
RESOLVED; queue identity NOT RESOLVED. Resolution in revision 6:

| Finding | Resolution |
|---|---|
| M: capture key cannot tell a C1 retry from a re-publication of identical bytes | per-site durable `publication_id` (finalize job id, range attempt / audio patch / assistant promotion row ids, reconcile key); reconciliation compares with the latest version, not history (§6.1 C) |
| m: stale provider-task index wording | §4.9 names `kb_attempt_submissions.provider_task_id` |

**Round 6 — same Terra session, on `5b37432..f5944a0`.** RQ job identity
confirmed sound for `run_finalize_run` (RQ 1.15.1 `get_current_job()`; recovery
starts a new chain). Index wording RESOLVED. Remaining MAJOR: the synchronous
`process_run` has no RQ job. Resolution in revision 7: `publication_id` is an
explicit argument of `_run_stitch_and_deliver`, the RQ job id in production and
a per-invocation UUID minted at `process_run` entry (§6.1 C).

**Round 7 — same Terra session, on `f5944a0..3240214`. DESIGN APPROVED.**
Round-6 MAJOR RESOLVED; no new BLOCKER or MAJOR. One MINOR (wire
`publication_id` into `source_record` of the queue item and the case version)
folded into revision 8 (§6.1 C).

Summary of the gate: 7 rounds, GPT-5.6 Terra (Codex CLI 0.163.0,
`model_reasoning_effort=high` confirmed in every transcript header,
`--sandbox read-only`), one continuous session
`01a17bd4-8972-71f2-ae69-9d88c926c360`; 7 BLOCKER, 18 MAJOR and 7 MINOR
findings in total, all resolved or accepted as documented scope limits
(post-accept duplicate billing is made visible, not prevented; shared login
means unverified actor labels). No other reviewer model was substituted.

