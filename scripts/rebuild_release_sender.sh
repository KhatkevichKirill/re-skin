#!/usr/bin/env bash
# Host attestation for a rebuild paid-call row stuck in `sending`
# (docs/rebuild-from-reference-design.md §7.2).
#
# A `sending` row becomes `unknown` automatically only when its sender is
# PROVEN gone: its container restarted (same hostname, new boot id) or its RQ
# job was ended by its own worker. A RECREATED container (deploy, `up -d`,
# scale) has a new hostname, so nothing inside a container can prove the old
# one is gone. This script runs on the VPS host, checks with
# `docker ps -a --no-trunc` that no container whose id starts with the row's
# `sender_host` is running (absent, or `exited`), and only then asks the app to
# release the row (`sending -> unknown`, attestation logged on the row).
#
# Usage (from /root/re-skin, on the host):
#   scripts/rebuild_release_sender.sh <row id>
#
# Read-only except for that one CAS. It never sends anything to a provider;
# `unknown` means "the provider may have billed this attempt".
set -euo pipefail

ROW="${1:-}"
if ! [[ "$ROW" =~ ^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$ ]]; then
  echo "usage: $0 <row id (uuid)>" >&2
  exit 2
fi

DB_CONTAINER="${DB_CONTAINER:-re-skin-db}"
API_CONTAINER="${API_CONTAINER:-re-skin-api}"

HOST="$(docker exec "$DB_CONTAINER" psql -U reskin -d reskin -At -c "
  SELECT sender_host FROM (
    SELECT id, status, sender_host FROM rebuild_keyframes
    UNION ALL SELECT id, status, sender_host FROM rebuild_chunk_attempts
    UNION ALL SELECT id, status, sender_host FROM rebuild_llm_calls
  ) t WHERE id = '${ROW}' AND status = 'sending'")"

if [[ -z "$HOST" ]]; then
  echo "refused: no rebuild row ${ROW} in 'sending' (or it has no sender_host)" >&2
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
docker exec "$API_CONTAINER" python -m app.rebuild_paid_call release-sender "$ROW" \
  --attest "${HOST}:${STATE}"
