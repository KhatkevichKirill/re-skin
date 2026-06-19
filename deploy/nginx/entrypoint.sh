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

# Create htpasswd file using openssl (busybox compatible)
# Format: username:password_hash
# Using 'openssl passwd -apr1' to generate Apache apr1 hash
HASH=$(openssl passwd -apr1 "$BASIC_AUTH_PASS")
echo "${BASIC_AUTH_USER}:${HASH}" > /etc/nginx/.htpasswd
chmod 600 /etc/nginx/.htpasswd

echo "htpasswd file generated for user: $BASIC_AUTH_USER"

# Start nginx
exec nginx -g 'daemon off;'
