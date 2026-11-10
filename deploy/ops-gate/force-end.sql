-- OPERATOR RECOVERY ONLY (AICP-350, design §2/§6 step 3): reopen admissions left
-- paused by a deploy that died without running its exit trap (SIGKILL, OOM, reboot,
-- lost SSH session). FIRST confirm that deploy is gone: no `docker compose` / build
-- process on the host (`ps aux | grep -E 'docker( |-)compose|buildx'`), its session
-- ended. Compare-and-set on the generation shown by status.sql / the dashboard banner:
--   docker exec -i re-skin-db psql -U reskin -d reskin -At -v ON_ERROR_STOP=1 \
--     -v gen=<N> < deploy/ops-gate/force-end.sql
-- Prints `force-ended <N>` or `no-match` (not paused, or a newer begin happened).
\set ON_ERROR_STOP on
WITH u AS (
  UPDATE ops_admission_gate
     SET paused = false, ended_at = now(), ended_by = 'force-end'
   WHERE id = 1 AND paused AND generation = CAST(:'gen' AS bigint)
  RETURNING generation
)
SELECT CASE WHEN count(*) = 1 THEN 'force-ended ' || max(generation) ELSE 'no-match' END FROM u;
