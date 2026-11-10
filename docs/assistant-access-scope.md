# Private operator assistant access scope

AICP-309, 2026-10-27. Starting revision
`ffc5cb2daee9ce8be4ff0b667475e927f3f75a0f`.

The operator requested all Runs and identified Run da70b937 in a third project
as missing the feature. This explicitly expands this shared-login private
deployment beyond the initial two-project beta. It is not a per-user ACL model.

## Explicit deployment switch

Add `ASSISTANT_ALL_PROJECTS`, default false. Only trusted configuration parsed
to the literal boolean True grants all-project scope. Requests, planner output,
empty/invalid project lists and wildcard strings cannot enable this mode.
Keep `ASSISTANT_ENABLED` as the overall gate and all subscription/tool/media
restrictions unchanged. All-project mode covers existing and future actual
dashboard projects; canonical IDs and persisted ownership/existence still apply.

Keep the immutable `ProjectAllowlist` policy with a shared
`from_settings(settings)` factory. Add strict `all_projects`, `has_scope`,
`allows_project(id)` and `sql_filter(project_id_column)`. The SQL predicate is
true only for explicit all-project mode; otherwise it remains the existing
normalized-ID `IN` filter. Unknown/nonboolean mode values must not grant all.
Absent/off mode preserves default-deny, malformed-list denial and allowlisting.
Malformed nonempty CSV denies the entire policy, even with literal True; empty
CSV plus explicit True is valid. Continue rejecting malformed references in
every mode. Only None means no optional project filter; empty/malformed/wildcard
filters fail rather than dropping their equality predicate. Explicit valid
project filters stay in place in all mode. A valid nonexistent project may
produce an empty catalog page; owner-addressed reads still deny generically.

Replace all direct policy membership/SQL filters consistently: catalog scope,
service conversation, require_run, donor/import enrichment, internal read-only
project listing, source imports, web availability, and supervisor construction
and empty-scope startup check. Preserve every Run/Project/conversation/version
join, exact asset owner binding, success state, file integrity and pagination/
resource cap. UI visibility additionally requires the actual project to exist,
matching the API ownership join. This changes the authorized project set, not
tool privileges.

Pass the new setting to both API and assistant supervisor in compose.app.yml.
The old CSV may be empty in explicit all mode; empty+false remains deny-all.
Retain the old valid CSV in this deployment as the rollback-to-restricted scope,
and add only the new flag to the private runtime configuration without copying
or printing its credential-bearing contents. Broker/ASR need no new authority.

## Verification and rollout

Meaningful tests cover strict default-deny/allowlist behavior, explicit all mode
including projects created after policy construction, malformed IDs, disabled
assistant, wrong/deleted owner bindings and API/UI/supervisor consistency.
Exercise target access, source search/import/catalog and internal project
listing on a project outside the former list; no provider/model calls.

Independent Sol-6.1 CLI design and exact-code review precede deployment. No
migration. Verify a Drive DB backup; preserve overlays/two workers and restart
nginx. Read-only rollout checks compare API and supervisor flags and source
hashes. On Run da70b937, verify the normal assistant card/open-chat/imported
selected version and Use as Run result control. Do not publish or change its
main media to test visibility. Verify Run7c29c40f's main/version/history remains
unchanged. All-project rollout must not render or publish other Runs in bulk.
