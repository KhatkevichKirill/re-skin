# The assistant's work inside its Run (AICP-369)

By-Assistant projects (AICP-362/363). Before this change the whole autopilot (budget, approvals,
listening checks, journal, Pause/Resume/Stop) lived in one card on the **project** page, while the
result is produced by a **Run**, and nothing in the UI said how a hand-started Run relates to the
assistant. This change is display only: the core, the flows and their decisions are unchanged and
there is **no migration**.

Operator decisions (Operator, 2026-11-02):

* **Pre-run steps stay on the project page.** There is no draft Run. Analysis, segmentation
  review, transcription, translation, prompts and the budget approval happen in the project card;
  once the assistant has created a Run, the work on that Run is shown on the Run's page.
  (A draft Run status would touch every run list, the reaper, the drain check, the spend scan and
  the exports.)
* **Manual runs are ignored, and labelled.** The assistant never adopts a Run started by hand
  (verified in the flows: subtitle reads its run from the `launch` step output, localisation from
  its own creation effect, never "the latest run"). Shared project state (segments, spend) still
  affects the assistant, so the wording is "does not *automatically* use this run".

Second design opinion: GPT-6.1 Terra (`codex exec`, reasoning effort high), 2026-11-02. It agreed
with no-draft and derivation, and reshaped the approval-placement rule (below).

## 1. Which runs are the assistant's (`app/autopilot/run_view.py: project_runs`)

Derived, over **all** autopilot attempts of the project: a run is the assistant's when a
run-creating action effect (`create_run`, `localisation_create_run`) lists it in
`created_refs["runs"]`. `CreateRun.execute` writes that ref in the same commit that creates the
run, so it is the admission fact itself. The record carries:

| field | from |
|---|---|
| role | subtitle `launch` → `main`; localisation `probe:L` → `probe`, `rest:L` → `final`; else the step id |
| language | the `:L` suffix |
| steps working on it | `main`: launch, clean, handoff; `probe:L`: probe:L; `final`: rest/qa/captions/tail/cleanup:L |
| current | the effect's attempt is the project's latest AND its generation is the step's current one AND no newer run of the same role + language exists in that attempt |

A deleted run or one of another project is skipped. A run with no record is **manual** — unless a
run-creating effect of the project is still `launching` or ended `unknown` without refs, in which
case runs created at or after that effect are **unconfirmed** (never guessed manual). The
launch protocol writes the effect row before the run exists, so no grace window is needed.

## 2. Where a pending approval is shown (`approval_runs`)

1. `budget` → the project (spend is project-wide).
2. Candidate runs = `run:<id>` keys of `bound_versions` ∪ `producer: run:<id>` artifacts, limited
   to this attempt's assistant runs. Exactly one → that run.
3. Several candidates → the step's owner run if it is one of them, else the project queue (with
   links to the candidates).
4. No candidate → the step's owner run: the single **current** assistant run whose steps include
   the approval's step (a reopened step whose new run does not exist yet has none) → else the
   project queue.

The project card lists **every** pending approval: unresolved ones as full answer forms, run ones
as "waiting on run X — Answer on the run". Nothing is reachable only from a run page.

## 3. Pages and endpoints

| | what |
|---|---|
| project card (`partials/autopilot_card.html`, `GET/POST /v2/projects/{pid}/autopilot-card…`) | status, budget, **the assistant's runs** (role, language, status, earlier-attempt marker), project-queue approvals in full, run approvals as links, Pause/Resume/Stop; steps and journal collapse once a run exists |
| run card (`partials/autopilot_run_card.html`, `GET /v2/runs/{rid}/autopilot-card`) | loaded by `run_detail.html` on assistant projects only, **outside** the 3 s status fragment; role badge, replaced marker, budget, related runs (probe ↔ final), this run's approvals, its steps, its journal, Pause/Resume/Stop. A manual run shows "Manual run — the assistant does not automatically use this run"; an unconfirmed one says so |
| `POST /v2/runs/{rid}/autopilot-card/answer` · `/switch` | the same `core.answer` / `core.set_switch`, with `expected_id` = the attempt that created the run: a manual run is refused ("the assistant did not create this run"), an earlier attempt's run is refused ("no longer the project's current one"); no controls are rendered for either. The core raises 409; like the project card, the HTMX endpoint returns the card (HTTP 200) with the error shown, because HTMX does not swap a non-2xx reply. Answer forms are shown while the run's attempt is the latest (any status — a stopped attempt's pending approvals stay answerable, as before); Pause/Resume/Stop only while it is live |
| runs table (project page) | `by Assistant · <role>` / `manual` / `unconfirmed` badge under the run name |
| New Run form (assistant project) | a notice plus an `hx-confirm`: a manual run, not used automatically, pauses the assistant once billed |

`core.answer` / `core.set_switch` gained an optional `expected_id` (the project card sends the
attempt it rendered too), checked under the autopilot row lock. Operator actions now take the project row lock first (the
module's lock order: project, then autopilot), which serializes them with `start_new` (it creates a
successor under that lock), so an answer can never land on an attempt being replaced. `core.view` was split into
`budget_view` / `approval_dict` / `step_dict` / `journal_dict` so both cards serialize alike; the
API JSON only gains `autopilot_id` and `step_id` on each approval.

**Run journal**: entries whose `effect_id` is one of this run's effects (created / targeted /
`params.run_id`), whose `approval_id` resolves to this run, plus step-only lines of its steps from
the run's creating effect until a newer run of the same role (journal rows have no generation);
the full project journal is linked.

**Refresh**: both cards poll every 10 s with `hx-trigger="every 10s [apIdle(this)]"` — skipped
while a textarea has text, a radio is picked, an input is focused or a request of the card is in
flight. Forms use `hx-sync="closest .ap-card:replace"` (a submit aborts an in-flight poll) and
`hx-disabled-elt="find button"` (no double submit), and each POST returns the card it came from.
VOID-fill approvals now render `proposal.preview_url` (an `/api/v2/…` jpg) as an image.

## 4. What it does not change

The core's answer semantics: an answer is recorded, then re-validated against the run's version
when consumed; a stale one is superseded and asked again (the card says so). Manual spend still
pauses through the reconcile pass (best effort, not a synchronous barrier). Notifications stay in
AICP-367.
