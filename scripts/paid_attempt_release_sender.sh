#!/usr/bin/env bash
# Host attestation for a paid submit attempt whose sender is not proven stopped
# (AICP-351, docs/assistant-drain-check-design.md §5 / §9 item 2; the
# scripts/rebuild_release_sender.sh pattern).
#
# A `parked` (video/image) or `started` / `superseded` (LLM) attempt has its
# sender released automatically only on proof: its own write after the POST,
# its container restarted (same hostname, new boot id) or its RQ job ended by
# its own worker. A RECREATED container (deploy, `up -d`, scale) has a new
# hostname, so nothing inside a container can prove the old one is gone. This
# script runs on the VPS host, checks with `docker ps -a --no-trunc` that no
# container whose id starts with the attempt's `sender_host` is running (absent,
# or `exited`), and only then asks the app to release it (audited in
# paid_submit_attempt_actions). Until then the attempt counts as ACTIVE in
# deploy/drain-check.sql and the operator cannot resubmit/abandon it.
#
# Usage (from /root/re-skin, on the host):
#   scripts/paid_attempt_release_sender.sh <attempt id>
#
# Read-only except for that one CAS. It never sends anything to a provider.
set -euo pipefail

ROW="${1:-}"
if ! [[ "$ROW" =~ ^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$ ]]; then
  echo "usage: $0 <attempt id (uuid)>" >&2
  exit 2
fi

DB_CONTAINER="${DB_CONTAINER:-re-skin-db}"
API_CONTAINER="${API_CONTAINER:-re-skin-api}"

HOST="$(docker exec "$DB_CONTAINER" psql -U reskin -d reskin -At -c "
  SELECT sender_host FROM paid_submit_attempts
  WHERE id = '${ROW}' AND sender_released_at IS NULL
    AND status IN ('started', 'parked', 'superseded')")"

if [[ -z "$HOST" ]]; then
  echo "refused: no attempt ${ROW} with an unreleased sender (or it has no sender_host)" >&2
  exit 3
fi
if ! [[ "$HOST" =~ ^[0-9a-f]{12,64}$ ]]; then
  echo "refused: sender_host '${HOST}' is not a container id" >&2
  exit 3
fi

STATES="$(docker ps -a --no-trunc --format '{{.ID}} {{.State}}' \
  | awk -v h="$HOST" 'index($1, h) == 1 { print $2 }')"
COUNT="$(printf '%s' "$STATES" | grep -c . || true)"

if [[ "$COUNT" -eq 0 ]]; then
  STATE="absent"
elif [[ "$COUNT" -eq 1 && "$STATES" == "exited" ]]; then
  STATE="exited"
else
  echo "refused: container ${HOST} is still present as: ${STATES//$'\n'/, }" >&2
  echo "(a frozen or running sender may still send; wait for it or stop it first)" >&2
  exit 4
fi

echo "attesting: sender container ${HOST} is ${STATE}"
docker exec "$API_CONTAINER" python -m app.paid_attempts release-sender "$ROW" \
  --attest "${HOST}:${STATE}"
