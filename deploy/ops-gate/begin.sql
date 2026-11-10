-- Pause all NEW admissions (AICP-350; docs/assistant-drain-check-design.md §2, §6).
-- Admitted work keeps running; wait for `active` = 0 (deploy/drain-check.sql) before
-- touching the stack. Normally run by scripts/deploy.sh, which ends the gate from an
-- exit trap. By hand (pick a UNIQUE owner token and keep it; `end` needs it):
--   docker exec -i re-skin-db psql -U reskin -d reskin -At -v ON_ERROR_STOP=1 \
--     -v owner="manual:$USER:$(date +%s):$RANDOM" -v reason="<why>" < deploy/ops-gate/begin.sql
-- Prints `begun <generation>`, or `refused <generation> <owner>` when the gate is
-- already paused (nothing changed: never steal another deployer's gate).
-- The UPDATE waits for every admission in flight (each holds the row FOR SHARE until its
-- commit), so once this returns no admission is half-done and none can start.
\set ON_ERROR_STOP on
\set QUIET on
INSERT INTO ops_admission_gate (id, paused, generation) VALUES (1, false, 0)
  ON CONFLICT (id) DO NOTHING;
WITH u AS (
  UPDATE ops_admission_gate
     SET paused = true, owner = :'owner', reason = NULLIF(:'reason', ''),
         generation = generation + 1, paused_at = now(), ended_at = NULL, ended_by = NULL
   WHERE id = 1 AND NOT paused
  RETURNING generation
)
SELECT CASE WHEN (SELECT count(*) FROM u) = 1
            THEN 'begun ' || (SELECT generation FROM u)
            ELSE 'refused ' || g.generation || ' ' || coalesce(g.owner, '?')
       END
  FROM ops_admission_gate g WHERE g.id = 1;
