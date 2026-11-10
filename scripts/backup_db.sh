#!/usr/bin/env bash
#
# Back up the production Postgres DB and upload the dump to Google Drive.
#
# Dumps the re-skin-db container (custom pg_dump format) and uploads the file
# via the re-skin-api container, which already has the Google service account
# mounted and the client libraries installed. The destination Drive folder must
# be shared with the service-account email.
#
# Usage:
#   scripts/backup_db.sh [label]
#     label   optional suffix in the file name, e.g. "predeploy"
#
# Env overrides:
#   GDRIVE_BACKUP_FOLDER_ID   target Drive folder (required)
#   DB_CONTAINER              default: re-skin-db
#   API_CONTAINER             default: re-skin-api
#
set -euo pipefail

DB_CONTAINER="${DB_CONTAINER:-re-skin-db}"
API_CONTAINER="${API_CONTAINER:-re-skin-api}"
FOLDER_ID="${GDRIVE_BACKUP_FOLDER_ID:?set GDRIVE_BACKUP_FOLDER_ID}"

ts="$(date -u +%Y%m%d-%H%M%S)"
label="${1:-}"
name="reskin-${ts}${label:+-$label}.dump"
tmp="/tmp/${name}"

echo "→ dumping ${DB_CONTAINER} → ${name}"
docker exec "$DB_CONTAINER" sh -c \
  "pg_dump -Fc -U \"\$POSTGRES_USER\" -d \"\$POSTGRES_DB\" -f \"$tmp\""

echo "→ staging dump into ${API_CONTAINER}"
docker cp "${DB_CONTAINER}:${tmp}" "$tmp"
docker cp "$tmp" "${API_CONTAINER}:${tmp}"

echo "→ uploading to Google Drive folder ${FOLDER_ID}"
docker exec -i "$API_CONTAINER" python3 - "$tmp" "$FOLDER_ID" "$name" <<'PY'
import hashlib
import os
import sys
from app.gdrive_client import GDriveClient

path, folder, name = sys.argv[1], sys.argv[2], sys.argv[3]
client = GDriveClient()
res = client.upload_file(
    path, folder_id=folder, name=name, mime_type="application/octet-stream"
)
if not res.get("id"):
    raise RuntimeError("Drive upload returned no file ID")
remote = client.service.files().get(
    fileId=res["id"], fields="id,size,md5Checksum", supportsAllDrives=True
).execute()
digest = hashlib.md5()
with open(path, "rb") as dump:
    for chunk in iter(lambda: dump.read(1024 * 1024), b""):
        digest.update(chunk)
if int(remote.get("size", -1)) != os.path.getsize(path) or remote.get("md5Checksum") != digest.hexdigest():
    raise RuntimeError("Uploaded backup size/checksum verification failed; temporary dump retained")
print("uploaded id={} link={}".format(res.get("id"), res.get("webViewLink")))
print("verified remote size and checksum")
PY

echo "→ cleaning up temp files"
docker exec "$DB_CONTAINER" rm -f "$tmp" || true
# -u 0: `docker cp` stages the dump owned by root, and the api container runs
# as the unprivileged reskin user (AICP-158), which cannot unlink a root-owned
# file in the sticky /tmp — without it the DB dump would linger in the
# container until the next recreate.
docker exec -u 0 "$API_CONTAINER" rm -f "$tmp" || true
rm -f "$tmp" || true

echo "✓ backup complete: ${name}"
