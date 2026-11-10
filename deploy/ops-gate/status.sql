-- Gate state (AICP-350). Read-only:
--   docker exec -i re-skin-db psql -U reskin -d reskin -At -F' ' < deploy/ops-gate/status.sql
-- One line: paused|admitting <generation> <minutes paused or -> <owner or -> <reason or ->
-- scripts/deploy.sh re-checks this before backup, migration and `up -d`: the gate must
-- still be paused under ITS owner token and generation, otherwise it aborts.
SELECT CASE WHEN paused THEN 'paused' ELSE 'admitting' END,
       generation,
       CASE WHEN paused AND paused_at IS NOT NULL
            THEN floor(extract(epoch FROM (now() - paused_at)) / 60)::bigint::text ELSE '-' END,
       coalesce(owner, '-'),
       coalesce(replace(reason, E'\n', ' '), '-')
  FROM ops_admission_gate WHERE id = 1;
