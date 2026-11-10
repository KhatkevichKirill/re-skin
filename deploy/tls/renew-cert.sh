#!/bin/sh
# Renew the dashboard's Let's Encrypt certificate, reload nginx if it changed,
# then fail if the certificate nginx actually SERVES is close to expiry
# (AICP-150). Run by reskin-cert-renew.timer; see deploy/tls/README.md.
#
# Let's Encrypt no longer emails expiry warnings, so a failed run of this
# script (systemd unit state "failed") is the expiry signal on the box.
#
# Usage: renew-cert.sh [extra certbot args]   e.g. renew-cert.sh --dry-run
set -eu

NAME=203-0-113-10.sslip.io
# Pinned by digest, same policy as the other images (AICP-158).
CERTBOT_IMAGE=certbot/certbot:v5.8.0@sha256:f70ad0adbb7e117f0fe42a63c553f28ea451edabc0148757b6efcd9735acaa20
NGINX=re-skin-nginx
LIVE=/etc/letsencrypt/live/$NAME/fullchain.pem
# 14 days: renewal normally happens with ~30 days left (90-day certs), so this
# only trips after ~2 weeks of failed renewals and still leaves 2 to fix it.
WARN_DAYS=${WARN_DAYS:-14}

before=$(sha256sum "$LIVE" 2>/dev/null || true)

# Same paths inside and outside the container, so the renewal config certbot
# writes (webroot_path etc.) means the same thing to a host-side certbot.
docker run --rm --name reskin-certbot-renew \
    -v /etc/letsencrypt:/etc/letsencrypt \
    -v /var/lib/letsencrypt:/var/lib/letsencrypt \
    -v /var/log/letsencrypt:/var/log/letsencrypt \
    -v /var/www/re-skin-acme:/var/www/re-skin-acme \
    "$CERTBOT_IMAGE" renew --non-interactive --no-random-sleep-on-renew "$@"

after=$(sha256sum "$LIVE")
if [ "$before" != "$after" ]; then
    # Graceful: new workers pick up the cert, old ones finish their requests.
    docker exec "$NGINX" nginx -t
    docker exec "$NGINX" nginx -s reload
    echo "certificate changed; nginx reloaded"
    sleep 5
fi

# Check what clients get, not the file: catches a missed reload too.
served=$(openssl s_client -connect 127.0.0.1:443 -servername "$NAME" </dev/null 2>/dev/null \
         | openssl x509 -noout -enddate -checkend $((WARN_DAYS * 86400))) || {
    echo "ERROR: certificate served on :443 expires within $WARN_DAYS days (or none served): ${served:-no certificate}" >&2
    exit 1
}
echo "served certificate ok: $served"
