#!/bin/sh
# Entrypoint for nginx container.
# Generates htpasswd file from BASIC_AUTH_USER and BASIC_AUTH_PASS environment variables.

set -e

# Get credentials from environment
BASIC_AUTH_USER="${BASIC_AUTH_USER:-reskin}"
BASIC_AUTH_PASS="${BASIC_AUTH_PASS:-}"

# Check if both variables are set
if [ -z "$BASIC_AUTH_USER" ] || [ -z "$BASIC_AUTH_PASS" ]; then
    echo "Error: BASIC_AUTH_USER and BASIC_AUTH_PASS must be set" >&2
    exit 1
fi

# bcrypt htpasswd (AICP-158; was openssl apr1/MD5). The password goes in on
# stdin (-i), never on argv, so it does not show up in the process list.
# Cost is kept moderate on purpose: nginx re-verifies the hash with crypt() on
# EVERY request, synchronously in the worker, and the UI polls HTMX fragments
# and streams media — a high cost would stall the event loop. Override with
# BASIC_AUTH_BCRYPT_COST (4-17) if needed.
BCRYPT_COST="${BASIC_AUTH_BCRYPT_COST:-6}"
umask 077
printf '%s' "$BASIC_AUTH_PASS" | htpasswd -niB -C "$BCRYPT_COST" "$BASIC_AUTH_USER" > /etc/nginx/.htpasswd
# Readable by the nginx worker user only (master is root, workers are nginx).
chown root:nginx /etc/nginx/.htpasswd
chmod 640 /etc/nginx/.htpasswd
unset BASIC_AUTH_PASS
umask 022

echo "htpasswd file generated for user: $BASIC_AUTH_USER"

# Wait for API to be ready (simple retry loop)
echo "Waiting for API to be ready..."
for i in $(seq 1 30); do
    if nc -z api 8000 2>/dev/null; then
        echo "API is ready"
        break
    fi
    echo "Waiting for API... (attempt $i/30)"
    sleep 1
done

# Start nginx
exec nginx -g 'daemon off;'
