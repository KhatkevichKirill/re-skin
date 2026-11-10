# The "by Assistant" operator experience (AICP-367)

Design only: mockups + interaction notes, implemented in follow-up issues (§9). Builds on the
AICP-369 split (pre-run work on the project page, the run's work on the Run page —
`docs/autopilot-run-view.md`) and does not change any flow decision.

Operator decisions (Operator, 2026-11-02):

* **Cover both flows now.** Subtitle-removal screens are grounded in the two real projects
  (514067f2, 81d0afc0). Localisation has not run yet; its screens follow
  `docs/autopilot-localisation-flow.md` / `-finishing.md` and are revisited after its first real run.
* **Notification channel: Slack.** The page stays the only place to answer.
  *Changed 2026-11-03 (Operator): **Telegram bot, called directly** (no n8n in between) — see §2.
  While testing, messages go to Operator's personal chat with the bot.*

Mockups: the "by-Assistant operator UX" design canvas (claude.ai artifact, private to Operator; link
in Linear). Second design opinion: GPT-6.1 Terra (`codex exec`, effort high), 2026-11-02 — its
findings are folded in below (notification lifecycle, reject semantics by origin, inbox rule, contract).

## 0. What the real projects showed

Snapshot: prod `project_autopilots` / `autopilot_approvals` / `autopilot_journal`, queried
2026-11-02 ~20:15 UTC (times below UTC). 4 attempts, 11 approvals, all subtitle removal.

| Observation | Evidence | Design answer |
|---|---|---|
| Nobody knows something waits | `blocked:` VOID ask on 514067f2: created → answered **5 h 55 min**; `redeliver_failed` on 81d0afc0 pending since 20:55 | Telegram notifications (§2) + a "Waiting for you" inbox (§3) |
| Questions are engineering sentences | e.g. "Segment 3 needs the operator: caption leftovers, but no fill can be placed: the final is not verified on the source timeline (alignment): leftovers ma…" | Short title + per-option consequence; full text under *Details* (§4) |
| Saying no means different things | by code (§4.3): a subtitle flow ask → the attempt **fails**; a core budget/regeneration/overrun ask → **stopped**; localisation listening → it asks for a corrective regeneration (flow-derived, not yet seen in prod) | The control names its effect per origin (§4.3) |
| A manual run paused the assistant mid-review | 81d0afc0: 18:46 "manual paid action: new run … Resume to continue", resumed 19:24 | Pause banner names the cause and the one action that clears it (§5) |
| The journal is mostly noise | 81d0afc0: 33 lines, 13 of them `decision: waiting …` / `observed … admitted` | Timeline of what matters, raw log behind a toggle (§6) |
| Localisation parameters are raw JSON | `dashboard.html` and `run_status_content.html` (project-from-run): `{"languages": [...], "tails": {...}}` | Language chips + per-language tail picker (§1) |

## 1. Choosing the mode (new project, project-from-run)

Both forms today: a `Mode` select + a JSON text field. Replace with:

* **Two option cards**, Manual / by Assistant: *Manual — you start every run.* / *by Assistant — it
  runs the {type} flow. It asks you to approve a budget before its first paid step, and again for
  each regeneration, an overrun, and anything it cannot decide.* (Not "before anything paid": a
  launch inside the approved budget proceeds without a question, `docs/autopilot-design.md`.)
  by Assistant is disabled with the reason when the type has no flow or `AUTOPILOT_ENABLED` is off
  (same rule as now: `autopilot_project_types()`).
* **Localisation parameters as fields**: an ordered language chip list (order = generation order;
  the first one is made first and gives the probe) and, per language, a CTA tail picker filtered to
  that language's library clips. The fields serialise to the same `autopilot_params` JSON, so the
  API (`_validate_project_mode`) is unchanged. Whole-video scope is pre-selected for by Assistant +
  localisation.
* **What happens next** under the cards: "It analyses the video, then asks you to approve a budget
  before the first paid step." Plus, only when notifications are configured: "Questions also go
  to Telegram."
* **Project-from-run** keeps the AICP-363 default (parent's mode when the target type has a flow)
  and shows the same cards.

### 1a. Implementation notes (AICP-378, 2026-11-04)

* One partial for both forms: `templates/v2/partials/autopilot_mode.html` (`mode_picker` +
  `mode_script`, JS `apMode`). The New project form posts the radios as `mode` and one hidden
  `autopilot_params` (disabled, so not posted, when there is nothing to send); the project-from-run
  form reads `apMode.value()` into its FormData. `_validate_project_mode` is unchanged.
* The run form's cards start on the API's own default (the parent's mode when the target type has a
  flow) until the operator picks a card.
* Localisation: languages in order (↑ to move one earlier, × to remove; the first is marked
  "probe"); a tail per language from `GET /api/v2/library-clips`, fetched the first time the fields
  show, only clips tagged with that language (untagged clips are not offered, although the API
  accepts them); "No CTA tail" = null. by Assistant + localisation sets "What to localise" to Whole
  video and disables Hook only (the flow refuses hook scope); back to Manual restores the previous
  choice.
* "Advanced parameters (JSON)" keeps the other optional keys (localisation `model` / `resolution`;
  subtitle `resolution` / `language` / `prompt`); the language fields win over the same keys.
* "Questions also go to Telegram." only when `notify.config_problem()` is None.

## 2. Notifications (Telegram)

Channel: a **Telegram bot called directly by the notifier** (Operator, 2026-11-03). Slack was the
first choice; the workspace app lacked history scopes and Operator expected friction there. n8n in
between was considered and rejected: editing a message needs its `message_id` back in our table,
so n8n would be a pass-through that adds a hop, a second retry layer and a second failure point.
Its one gain — re-routing without a deploy — is not needed now.

Answers are **not** taken in Telegram (v1): they need the media, the bound-version re-check and the
budget line. (Inline buttons would not need a public endpoint — the bot could long-poll
`getUpdates` — but the first two reasons stand.) Revisit only for one-click kinds if waits stay
long.

### 2.1 What a message shows (desired state, not events)

The notifier does **not** react to journal events: several approval transitions write no journal
line (step reopen/skip supersede `core.py:799,1771`; consumption `:741,1460`; reconcile can put a
consumed approval back to approved `:1489`). It **reconciles**: on each pass it computes, from the
current rows, the desired presentation of each notifiable thing and brings the chat to it.

| Thing | Desired presentation (from current rows) |
|---|---|
| Approval of the **latest attempt** of its project, `pending`, attempt live | *Waiting for you* — title, project, where (run label or project), cost, **Open** |
| … `pending` but the attempt is paused | same, plus "the assistant is paused — answer, then Resume" |
| … answered, not yet consumed | *Answered by X: approve (option) — the assistant checks it next* |
| … consumed | *Done: approve (option)* — the call to action struck through |
| … superseded after an answer | *Your answer was not used: the run changed — a new question follows* |
| … superseded before an answer / attempt ended / attempt replaced | *No longer needed* (struck through) |
| Attempt **system**-paused (status_reason not "paused by …") | *Paused — {reason}* + **Open** |
| Attempt failed, or stopped by the system (rejected-budget / regeneration consumption, reject grace) | *Ended — {reason}* + **Open** |
| Attempt completed | *Done — {deliverables}* |
| Reminder | a reply to a still-*Waiting* message after 2 h, then every 24 h — computed at send time from the current row, never queued |

Operator Pause/Resume/Stop send nothing new; they only change the presentation of open messages
(e.g. *No longer needed*). The cause of a stop is read from `status_reason` ("stopped by …" =
operator), not from the journal kind, since a rejected budget becomes `stopped` later, in reconcile.

### 2.2 Mechanics

* **Bot API**, HTTPS to `api.telegram.org`: `sendMessage` (new message, and reminders as a reply via
  `reply_parameters`), `editMessageText` (every later state, by the stored `message_id`).
  `parse_mode=HTML`, link previews off. Config `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`; either unset
  → off, and the UI does not promise notifications. The token lives only in `.env` — never in logs,
  Linear or prompts; error logs print the method and the Telegram `description`, not the URL.
* **Channel adapter.** The reconciler talks to a three-call interface (`post`, `edit`, `reply`)
  returning a message reference; Telegram is the one implementation. Slack (or a second chat) can
  be added later without touching the reconciler.
* **One table, `autopilot_notifications`** (the one migration, `u7c2e4a6b8d1`): key (`approval:<id>`,
  `pause:<journal line id>` — a second pause after a Resume is a new message — or `end:<attempt id>`;
  `notifier:since` marks the first start: open questions and system pauses at that moment are sent,
  older ends are not), chat id, `message_id`, rendered-state hash, last reminder, claim owner
  + claim expiry, attempts, last error. Delivery is **at-least-once**, not exactly-once:
  * a sender claims a row (`UPDATE … WHERE claim expired`, own expiry) before any call, and writes
    the returned `message_id` in its own short transaction right after the call;
  * **no duplicate repair.** A bot cannot read chat history, so the Slack design's "search the
    channel for the metadata key" step does not exist. If the process dies between Telegram's
    answer and that write, the claim expires and the row is posted again: one duplicate message,
    left showing its last state. The window is milliseconds; accepted (Operator, 2026-11-03);
  * updates are idempotent (render the current desired state; skip when the hash is equal), so a
    late retry can never write an older state over a newer one. Telegram's `400 message is not
    modified` counts as success;
  * `400 message to edit not found` (someone deleted it): a still-*Waiting* row posts a new
    message; any other state just records that it is gone.
* **Not in the autopilot loop.** The ticker is one synchronous loop gated by `AUTOPILOT_ENABLED`
  (`worker/autopilot_ticker.py:38,76`). The notifier runs as its own thread with its own DB session
  in that container: request timeout 10 s, at most N calls per pass and about one message per
  second per chat (Telegram's limit), `429` → honour `parameters.retry_after`, other errors →
  exponential backoff with jitter per row, a row failing 10 times is shown as *notification
  failing* on the inbox page. A group upgraded to a supergroup answers `400` with
  `migrate_to_chat_id`: the notifier logs it and stops until the config is changed (no silent
  re-routing). It keeps running when `AUTOPILOT_ENABLED` is false or the deploy hold is on (it only
  reads autopilot state); a kill mid-request is recovered by the claim expiry above.
* **Links** come from one function shared with the inbox (`app/autopilot/links.py`, AICP-373,
  deployed 2026-11-03): placement from `run_view.approval_runs` (project for budget / pre-run /
  ambiguous), the latest-attempt rule, and a project fallback when the run is gone; base from
  `PUBLIC_BASE_URL` (required when notifications are on, else the notifier refuses to start with a
  clear log line). Targets carry `#ap-<approval id>`; the card scrolls to the fragment after its
  HTMX load. An old link whose approval is gone lands on the card, which says "this question is no
  longer open".
* **What leaves the company.** Telegram is outside the corporate workspace. Messages carry no
  media, no Drive links, no secrets — project names, titles, costs and links only; the links are
  behind basic auth. Titles are plain text, HTML-escaped (`&`, `<`, `>`); a message is cut to stay
  under Telegram's 4096-character limit.

## 3. "Waiting for you" inbox

Header badge on every v2 page: **Waiting for you · 3** = answerable approvals only: `pending`
approvals of each project's **latest** attempt while it is live (active or paused) — the same rule
the cards use (`run_view.py:106,263`). A small separate marker shows system-paused attempts
("1 paused"); they are not added to the number. `/v2/assistant`:

* **To answer** — one row per answerable approval, oldest first: waited, project, where (run label
  or project), title, cost, **Answer** (the shared link function). A paused attempt's rows say
  "paused — answer, then Resume".
* **Paused** — system-paused latest attempts with reason and **Open** (Resume is on the card).
  Operator-paused ones are listed greyed ("you paused it").
* **Ended in the last 7 days** — latest attempts per project that failed or were stopped by the
  system, by `finished_at`.
* **Notification failing** — rows the Telegram sender gave up on (§2.2).

## 4. The approval card

Same data, new layout, on the page where AICP-369 places it.

### 4.1 Anatomy

1. **Header** — origin/kind in plain words: *Budget* (core budget), *Costs credits* (a core
   regeneration/overrun, or a flow ask whose options carry a cost), *Check this* (handoff),
   *Listen* (listening_probe/final). Title, waited time, run label.
2. **Media first.** Two videos side by side with one shared scrub bar when an approval has a result
   and a source (eye checks already ship a `result | source` composite — shown as is); images in a
   lightbox; `preview_url` rendered. Listening: the final large, per-segment sheets as a strip.
3. **Options as buttons**, each with its consequence and its cost. Costs and "after this: X of B"
   are labelled **estimated when asked** (`cost_mcredits` and committed-at-ask are snapshots; the
   launch re-checks spend and bounds) — never presented as guaranteed.
4. **Saying no** (§4.3) as a separate, quieter control.
5. **Comment** — optional in the contract; the UI asks for one where it becomes a reason or a
   correction (§4.3) but does not block.
6. **Details** (collapsed): full question, proposal fields, bound versions, step + generation.

### 4.2 Titles and consequences — the contract change

Presentation only; never part of handler params, targets, bound versions or `params_sha256`.

* `contract.Ask.title` (≤ 80 chars, plain text) and `contract.Option.consequence` (≤ 120 chars);
  the same two fields on the subtitle planner's own `Ask`/option types
  (`subtitle_removal_plan.py:164`) and its `to_contract` converter.
* `_apply_ask` builds each stored option dict field by field (`core.py:1043`): add
  `consequence` there, and store the title as `proposal["_title"]` (reserved key; flows may not set
  it themselves).
* **Core-generated** budget / regeneration / overrun approvals get their title from core templates
  (`Approve the budget: N credits`, `Regenerate: {handler} (~e credits)`, `Over budget: …`).
* **Fallback** for rows created before the change and for missing fields: the first sentence of
  `question` (cut at 80) and the option label. Existing approvals are never superseded just to
  refresh wording.

Examples (subtitle): `segmentation:` → *Check the keep segments and the caption band before launch*;
`blocked:` → *Segment 1 was refused by the provider (safety filter)* with void → "fills the caption
area from the source", retry → "resubmits the unfinished segments; may be refused again";
`redeliver_failed:` → *Run d46bb1b8 could not be stitched (3 tries)*.

### 4.3 Saying no — by origin, as the code does it

| Origin | What reject does now | Control |
|---|---|---|
| core **budget** (`#budget`) | consumed in reconcile → attempt **stopped** (`core.py:1439`) | **Stop the assistant…** (confirm) |
| core **regeneration / overrun** (`#regeneration`, `#overrun`) | consumed when the launch path advances → **stopped** (`core.py:1218`) | **Don't spend this — stop the assistant…** (confirm) |
| subtitle flow asks (any kind, incl. flow `regeneration` like `blocked:`, `retry:`) | `_rejection` → `Fail` → attempt **failed**, comment in the reason (`subtitle_removal_plan.py:279`) | **End this attempt…** (confirm; comment asked for, not required) |
| localisation listening probe | records the cue, then **asks** a separate `fix:` regeneration (empty comment = same prompt) (`localisation.py:667`) | **Request changes** — "you approve the corrected regeneration and its cost next" |
| localisation listening final | records the cue, asks `fix:`; **fails** if no segment can be regenerated (`localisation_finish.py:455`) | **Request changes** — same note, plus "if nothing can be regenerated, the attempt ends" |
| localisation `regen:` / `fix:` / `retry:` asks | reject **or** the `stop` option → `Fail` (`localisation_finish.py:411`, `localisation.py:684`) | the `stop` option posts `decision=approve, option=stop` and is rendered as **End this attempt…**; reject is the same end |
| any | a reject left unconsumed while the step waits → **stopped** after `AUTOPILOT_REJECT_GRACE_SEC` (`core.py:945`) | (fallback; the card shows "ending…" until the status changes) |

No behaviour change; only labels, confirms and the comment prompt. The card decides by the
approval's origin (`#` in the name = core) and flow, not by `kind` alone.

### 4.3a Implementation notes (AICP-376, 2026-11-04)

* `app/autopilot/card.py` turns an `approval_dict` into the card view (`present`) and picks the
  "no" control by origin (`no_control`); the localisation `stop` option is moved out of the
  choices into that control. Every button posts the same `decision` / `option` the v1 form did
  (`hx-vals`), inside one `<fieldset>` that is disabled while an answer is in flight.
* Titles: core templates for budget / regeneration / overrun; the subtitle flow fills titles and
  option consequences from one table by approval-name prefix at conversion
  (`flows/subtitle_removal.py`); the localisation asks set them inline.
* Deviation: the full question stays **visible** under the title (not only under *Details*): some
  questions carry warnings the options do not repeat (e.g. a re-stitch dropping an audio patch).

### 4.4 Budget approval

Lines (`proposal.lines`) as a table, total in credits and USD, tolerance, *already committed*.
**Approve N credits** / **Stop the assistant…**. Tolerance stays editable on the project card.

## 5. Status, Pause / Resume / Stop

* **Status line** from the active step's reason: *Cleaning — waiting for run d46bb1b8
  (stitching)* / *Waiting for you: 1 question* / *Paused: a manual run was started (run 3b79ba7d,
  ~4,200 credits)*.
* **Stepper**: subtitle Segmentation → Launch → Clean → Handoff; localisation Intake → Transcribe →
  Segments, then one row per language: Translate → Prompts → Probe → Rest → QA → Captions → Tail →
  Cleanup.
* **Budget bar**: committed / approved, tolerance band shaded, consumed vs settled in the tooltip.
* **Paused banner** with the cause and its fix: manual spend → *Resume* ("it re-checks the runs
  first"); refused 8× / step errors → *Resume* + the journal lines; flow not deployed → no Resume.
* **Pause** = temporary, no confirm. **Stop** = final, confirm: "Stops this attempt. Work already
  running (renders, fills) continues and is still billed. *Start a new autopilot* later creates a
  new attempt with the same parameters; the project's spend so far still counts, and it decides
  again from the project's current state — answers given to this attempt are not carried over."

## 6. Journal

* **Timeline**, newest first, default filter *Important*: `approval_requested`,
  `approval_answered`, `launch` (with cost), `budget`, `paused`/`resumed`/`stopped`/`failed`,
  `completed`, `reopened`, `refused`, `warning`, `comment`. Hidden by default: `decision` "waiting:"
  (the latest one is the status line), `observed … admitted`, `step_started`.
* Filtered **server-side with pagination** (cursor on `autopilot_journal.id`), not by filtering
  the current 20/30-line card window, so older important lines are reachable.
* On the run card, step-only lines are attributed by time window (journal rows have no generation,
  `docs/autopilot-run-view.md`) — the timeline marks them "during this run", not as exact.
* Toggle **All lines** = today's raw list; "Full JSON" stays.

### 6a. Implementation notes (AICP-377, 2026-11-04)

* `app/autopilot/status.py` (presentation only): the status line (ended → paused → "Waiting for
  you: N" → the active steps' reasons, up to 3), the stepper (the flow's declared step order; the
  steps without a language in one row, one row per language; a step with a pending question shows
  "?"; without the flow in the build, the rows' own order), the budget bar (scale = max(committed,
  approved × (1 + tolerance)); the tolerance band shaded; consumed/settled in the tooltip) and the
  paused banner (cause + fix by the pause reason; refusals / step errors / reopen loops list the
  paused step's last 3 `refused`/`warning` lines; a flow that is not in the build gets no Resume,
  "Stop it, then Start a new autopilot").
* Stop asks with the §5 copy; Pause asks nothing.
* `app/autopilot/timeline.py` + `GET /v2/projects/{pid}/autopilot-journal` /
  `GET /v2/runs/{rid}/autopilot-journal` (`all`, `before`): 30 lines a page, newest first, cursor
  on `autopilot_journal.id`. Important = `created`, `approval_requested`, `approval_answered`,
  `launch`, `budget`, `paused`, `resumed`, `stopped`, `failed`, `completed`, `reopened`, `refused`,
  `warning`, `comment`. The project journal is pinned to one attempt (`autopilot_id`).
* The Journal box loads when it is opened and is `hx-preserve`d (one id per attempt / run), so the
  card's 10 s refresh keeps the loaded pages and the filter; closing and reopening it reloads the
  newest page.
* On the Run page, a line that belongs to the run only by its step and time window (no effect, no
  approval) is marked *during this run*; the scope is `run_view.run_journal_scope`, the same rule
  the old card window used.

## 7. Localisation specifics (from the docs; revisit after the first real run)

* Project card: one row per language with its steps, its probe and final runs (`probe ES`,
  `final ES`) and its open questions.
* **Listening probe**: probe video large, "probe vs source" sheet, prompt collapsible;
  **Sounds right** / **Request changes** (§4.3).
* **Listening final**: final large, per-segment sheets as a strip; same controls; `regen:` asks
  highlight the flagged segments on the strip.

## 8. What stays as is

The core, the flows' decisions, the answer semantics (`core.answer`, re-validation on
consumption), AICP-369 placement. Desktop dashboard only.

## 9. Implementation slices (follow-up issues)

0. **Shared pieces first** (small): the link/placement function (§2.2), `#ap-<id>` anchors +
   fragment scroll, the title fallback (§4.2), the "answerable approval" query (§3).
1. **Telegram notifier**: migration, reconciler thread, claims, reminders, inbox "notification
   failing", behind a small channel adapter. Bot and chat are ready (`TELEGRAM_BOT_TOKEN`,
   `TELEGRAM_CHAT_ID` in `.env`, 2026-11-03).
2. **Inbox**: badge + `/v2/assistant`.
3. **Approval card v2**: anatomy, §4.3 controls, budget layout, synced players; contract
   `title`/`consequence` (contract + planner + core templates) and filling them in both flows.
4. **Status / stepper / paginated journal timeline / paused banner.**
5. **Mode cards + localisation parameter fields** (both creation forms).

1 and 2 depend only on 0; 1 has the most value (the 5 h 55 min wait).
