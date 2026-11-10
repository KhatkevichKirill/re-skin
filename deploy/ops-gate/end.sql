-- Reopen admissions (AICP-350): compare-and-set on the owner token `begin` was run
-- with, and on its generation when known. scripts/deploy.sh runs this from its exit
-- trap on every exit path. By hand:
--   docker exec -i re-skin-db psql -U reskin -d reskin -At -v ON_ERROR_STOP=1 \
--     -v owner="<the token you began with>" -v gen=<generation or empty> < deploy/ops-gate/end.sql
-- Prints `ended <generation>`, or `not-owner` when the row is not paused under this
-- owner (+ generation): nothing changed (someone else's gate stays paused). An empty
-- gen matches any generation of this owner token (the token is unique per begin, so
-- this also cleans up a begin whose output was lost).
\set ON_ERROR_STOP on
WITH u AS (
  UPDATE ops_admission_gate
     SET paused = false, ended_at = now(), ended_by = :'owner'
   WHERE id = 1 AND paused AND owner = :'owner'
     AND (NULLIF(:'gen', '') IS NULL OR generation = NULLIF(:'gen', '')::bigint)
  RETURNING generation
)
SELECT CASE WHEN count(*) = 1 THEN 'ended ' || max(generation) ELSE 'not-owner' END FROM u;
