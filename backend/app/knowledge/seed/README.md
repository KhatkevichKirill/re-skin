# Knowledge seed guidance

`aicp-293-observations.yaml` is the curated seed set for the knowledge base
(AICP-293 / AICP-297): the three operator observations O1-O3 and a few
empirical rules backed by the repo wiki or `docs/knowledge-feedback-design.md`.
It is data for the importer; it is not read at runtime.

## Format

Top-level `rules:` list. Each rule has `slug`, `kind`, `statement`, `rationale`,
`applicability` (only the section 9.1 dimensions), `recommendation`
(`action` + `target`), `confidence`, `shareable`, `effective_from`,
`author_label`, `provenance`, `evidence_gaps`, and an optional `evidence` list
(`run_id`, `stance`, `note`). The seed file currently has no `evidence` links:
the repo records only 8-character run ids, and a link needs a case row for the
full run id. Short ids stay in `provenance.runs_short_ids` and in the text.

## How each kind is treated

- `operator_constraint`: an operator's stated constraint (O1). Only an
  operator or importer may author it. Applied first by the advisor, but O1 is
  a warn, not a forbid, and changes no defaults.
- `empirical_guidance`: a claim backed by measurement or cases, with an honest
  confidence and explicit `evidence_gaps`. Operator-reported preferences (O2)
  are recorded as reported, not as vendor facts.
- `machine_hypothesis`: shown as a hypothesis, never effective (O3). O3 has no
  subject scoping until Operator names the subjects; candidate projects appear
  only in `evidence_gaps` text.

Every rule here is `shareable: false`. A shareable rule may not use
`project_id` or `subject_id` and needs a structured `target` over generic
dimensions.

## Evidence and holdout

The store refuses to cite a holdout case as rule evidence, and rejects any
evidence before the evaluation manifest is loaded. Do not add evidence for
cases in the frozen holdout groups.

## Adding or superseding a rule

History is never edited. To change a rule, add an entry with the same `slug`:
the store records it as the next version and marks the previous one superseded.
A new version still needs a review (approval) before it is effective; the
highest approved version wins. To stop a rule, retire it through a review;
do not delete or rewrite its YAML entry. New slugs use the pattern
`^[a-z0-9][a-z0-9-]{1,79}$`.
