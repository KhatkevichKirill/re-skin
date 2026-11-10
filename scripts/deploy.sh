#!/usr/bin/env bash
#
# Gated production deploy (AICP-350; docs/assistant-drain-check-design.md §6).
#
#   1. install the exit trap FIRST, then pause new admissions (deploy/ops-gate/begin.sql)
#      under a unique owner token;
#   2. wait until deploy/drain-check.sql reports active = 0 (parked rows are listed only);
#   3. before the backup, before the migration and before `up -d`, re-check that the gate
#      is still paused under OUR owner token + generation — otherwise abort;
#   4. backup -> [migrate] -> up -d --build --scale worker=2 (+ assistant overlays, + fal
#      profile) -> recreate assistant-supervisor -> restart nginx -> curl through the proxy;
#   5. reopen admissions (deploy/ops-gate/end.sql, a CAS on our owner token + generation)
#      from the EXIT trap, on EVERY exit path: success, a failed step, Ctrl-C, TERM, HUP.
#      A failed deploy therefore reopens admission; if repair needs the pause, run begin.sql
#      again by hand before touching anything.
#
# The gate never expires. It survives only if this process is killed outright (SIGKILL,
# OOM, reboot): the dashboard banner turns red after OPS_GATE_WARN_AFTER_MIN (45) minutes
# and shows the deploy/ops-gate/force-end.sql command — run it only after confirming this
# deploy is gone (no docker compose / build process left on the host).
#
# FIRST ROLLOUT: the image that ships the gate is deployed with the old MANUAL drain
# (CLAUDE.md) — the running old image does not enforce the gate. This script refuses to
# run until the running api image has app/ops_gate.py.
#
# Usage (from the prod checkout, /root/re-skin):
#   scripts/deploy.sh --reason "AICP-123 deploy" [--migrate] [--fal] [--no-assistant]
#                     [--drain-timeout MIN] [--poll SEC] [--dry-run]
#
#   --migrate        copy backend/alembic into re-skin-api and run `alembic upgrade head`
#                    (after the backup, before `up -d`)
#   --fal            add `--profile fal` (required on every deploy while Fal is enabled or
#                    has unfinished tasks — CLAUDE.md, AICP-330)
#   --no-assistant   drop the AICP-261 assistant overlays (default: keep them)
#   --drain-timeout  give up (and reopen admission) after MIN minutes undrained (default 40)
#   --dry-run        pause, drain and re-check for real, but only PRINT backup/migrate/up/
#                    nginx/curl; the trap still reopens admission
#
# Env: DB_CONTAINER (re-skin-db), API_CONTAINER (re-skin-api), PROXY_URL
#      (http://localhost:8847/api/v2/ops/admission).
#
set -Eeuo pipefail

cd "$(dirname "$0")/.."

DB_CONTAINER="${DB_CONTAINER:-re-skin-db}"
API_CONTAINER="${API_CONTAINER:-re-skin-api}"
PROXY_URL="${PROXY_URL:-http://localhost:8847/api/v2/ops/admission}"
ASSISTANT_ENV="/var/lib/reskin-assistant-runtime/assistant.env"

REASON=""
MIGRATE=0
FAL=0
ASSISTANT=1
DRY_RUN=0
DRAIN_TIMEOUT_MIN=40
POLL_SEC=20

usage() { sed -n '2,/^set -Eeuo/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'; }

while [ $# -gt 0 ]; do
  case "$1" in
    --reason) REASON="${2:-}"; shift 2 ;;
    --migrate) MIGRATE=1; shift ;;
    --fal) FAL=1; shift ;;
    --no-assistant) ASSISTANT=0; shift ;;
    --drain-timeout) DRAIN_TIMEOUT_MIN="${2:-}"; shift 2 ;;
    --poll) POLL_SEC="${2:-}"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done
[ -n "$REASON" ] || { echo "--reason is required (shown on the dashboard banner)" >&2; exit 2; }
case "$DRAIN_TIMEOUT_MIN$POLL_SEC" in *[!0-9]*) echo "--drain-timeout/--poll take integers" >&2; exit 2 ;; esac

LOG="/tmp/reskin-deploy-$(date -u +%Y%m%dT%H%M%SZ)-$$.log"
NONCE="$(od -An -N6 -tx1 /dev/urandom | tr -d ' \n')"
OWNER="deploy:${USER:-$(id -un)}@$(hostname -s):$$:${NONCE}"
GEN=""
END_STATE="admissions NOT reopened — see above"

log() { printf '%s %s\n' "$(date -u +%H:%M:%S)" "$*" | tee -a "$LOG" 2>/dev/null || true; }
psql_db() { docker exec -i "$DB_CONTAINER" psql -U reskin -d reskin -At -F' ' -v ON_ERROR_STOP=1 "$@"; }
run() {
  if [ "$DRY_RUN" = 1 ]; then log "DRY-RUN: $*"; else log "+ $*"; "$@"; fi
}

end_gate() {
  # CAS on our owner token (+ generation once known): does nothing when begin was refused,
  # never ran, or someone force-ended and re-began meanwhile.
  local out
  if out="$(psql_db -v owner="$OWNER" -v gen="$GEN" < deploy/ops-gate/end.sql 2>&1)"; then
    log "gate: ${out}"
    case "$out" in "ended "*) END_STATE="admissions reopened" ;; *) END_STATE="gate left as it is (not ours)" ;; esac
  else
    log "!! could not reopen admissions automatically: ${out}"
    log "!! run: docker exec -i $DB_CONTAINER psql -U reskin -d reskin -At -v ON_ERROR_STOP=1 -v owner='$OWNER' -v gen='$GEN' < deploy/ops-gate/end.sql"
    return 1
  fi
}

on_exit() {
  local rc=$?
  set +e
  trap '' PIPE HUP INT TERM   # finish the cleanup even if the terminal is gone
  if ! end_gate && [ "$rc" = 0 ]; then
    rc=1   # never report success while admissions may still be paused
  fi
  if [ "$rc" = 0 ]; then log "deploy finished; log: $LOG"; else log "deploy FAILED (rc=$rc); $END_STATE; log: $LOG"; fi
  exit "$rc"
}

# 1. The trap first: from here on every exit path runs end_gate.
trap on_exit EXIT
trap 'log "!! interrupted (INT)"; exit 130' INT
trap 'log "!! terminated (TERM)"; exit 143' TERM
trap 'log "!! hangup (HUP)"; exit 129' HUP
trap '' PIPE

log "owner token: $OWNER"
if ! docker exec "$API_CONTAINER" python -c "import app.ops_gate" >/dev/null 2>&1; then
  log "!! the running $API_CONTAINER does not enforce the admission gate (no app/ops_gate.py):"
  log "!! this is the first rollout — deploy with the manual drain from CLAUDE.md instead."
  exit 1
fi

log "pausing admissions (begin waits for admissions in flight and for running Drive re-link /"
log "caption-placement scan jobs, which hold the gate while they run)…"
out="$(psql_db -v owner="$OWNER" -v reason="$REASON" < deploy/ops-gate/begin.sql)"
case "$out" in
  "begun "*) GEN="${out#begun }"; log "gate paused: generation $GEN ($REASON)" ;;
  "refused "*) log "!! the gate is already paused by someone else: ${out#refused } — not deploying"; exit 1 ;;
  *) log "!! unexpected begin output: $out"; exit 1 ;;
esac

check_gen() {
  local st state gen _min owner _rest
  st="$(psql_db < deploy/ops-gate/status.sql)"
  read -r state gen _min owner _rest <<<"$st"
  if [ "$state" != "paused" ] || [ "$gen" != "$GEN" ] || [ "$owner" != "$OWNER" ]; then
    log "!! before $1: the gate is no longer ours ($st) — aborting"
    exit 1
  fi
}

# 2. Wait for active = 0.
deadline=$(( $(date +%s) + DRAIN_TIMEOUT_MIN * 60 ))
while :; do
  drain="$(psql_db < deploy/drain-check.sql)"
  active="$(awk '$1=="active"{s+=$3} END{print s+0}' <<<"$drain")"
  rows="$(awk '$1=="active"' <<<"$drain" | wc -l)"
  busy="$(awk '$1=="active" && $3>0 {printf "%s=%s ", $2, $3}' <<<"$drain")"
  parked="$(awk '$1=="parked" && $3>0 {printf "%s=%s ", $2, $3}' <<<"$drain")"
  [ "$rows" -gt 0 ] || { log "!! drain-check returned no active rows: $drain"; exit 1; }
  if [ "$active" = "0" ]; then
    log "drained: active=0${parked:+ (parked, not waited on: $parked)}"
    break
  fi
  if [ "$(date +%s)" -ge "$deadline" ]; then
    log "!! still active after ${DRAIN_TIMEOUT_MIN} min: $busy— giving up"
    exit 1
  fi
  log "waiting: active=$active ($busy)${parked:+ parked: $parked}"
  sleep "$POLL_SEC"
done

COMPOSE=(docker compose)
if [ "$ASSISTANT" = 1 ]; then
  COMPOSE+=(--env-file .env --env-file "$ASSISTANT_ENV" -f docker-compose.yml
            -f deploy/assistant/compose.app.yml -f deploy/assistant/compose.asr.yml)
fi
if [ "$FAL" = 1 ]; then COMPOSE+=(--profile fal); fi

# 3 + 4. Backup -> migrate -> up, each behind a generation re-check.
check_gen "the backup"
run scripts/backup_db.sh predeploy

if [ "$MIGRATE" = 1 ]; then
  check_gen "the migration"
  run docker cp backend/alembic "$API_CONTAINER":/app/alembic
  run docker cp backend/alembic.ini "$API_CONTAINER":/app/alembic.ini
  run docker exec -w /app "$API_CONTAINER" python3 -m alembic upgrade head
fi

check_gen "up -d"
run "${COMPOSE[@]}" up -d --build --scale worker=2
if [ "$ASSISTANT" = 1 ]; then
  # Not recreated by `up -d --build` (image only, no build) — CLAUDE.md, 2026-10-30.
  run "${COMPOSE[@]}" up -d --no-deps --force-recreate assistant-supervisor
fi
run "${COMPOSE[@]}" restart nginx

if [ "$DRY_RUN" = 1 ]; then
  log "DRY-RUN: curl $PROXY_URL"
else
  user="$(grep -E '^BASIC_AUTH_USER=' .env | tail -1 | cut -d= -f2- || true)"
  pass="$(grep -E '^BASIC_AUTH_PASS=' .env | tail -1 | cut -d= -f2- || true)"
  code=000
  for _ in $(seq 30); do
    code="$(curl -s -o /dev/null -w '%{http_code}' -u "${user:-reskin}:${pass}" "$PROXY_URL" || true)"
    [ "$code" = "200" ] && break
    sleep 5
  done
  unset pass
  [ "$code" = "200" ] || { log "!! $PROXY_URL answered $code through the proxy"; exit 1; }
  log "proxy check: $PROXY_URL -> 200"
  if [ "$ASSISTANT" = 1 ]; then
    want="$(docker image inspect re-skin-api:latest --format '{{.Id}}')"
    have="$(docker inspect re-skin-assistant-supervisor-1 --format '{{.Image}}' 2>/dev/null || true)"
    [ "$want" = "$have" ] || log "!! assistant-supervisor runs $have, not re-skin-api:latest ($want)"
  fi
fi
# 5. end_gate runs from the EXIT trap.
