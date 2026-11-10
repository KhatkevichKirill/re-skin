# Pinned by digest (AICP-158) so a rebuild is reproducible. nginx 1.31.2 on
# Alpine; to move it, pull the new tag and take the digest from
# `docker image inspect --format '{{json .RepoDigests}}' nginx:<tag>`.
FROM nginx:1.31.2-alpine@sha256:54f2a904c251d5a34adf545a72d32515a15e08418dae0266e23be2e18c66fefa

# apache2-utils: `htpasswd -B` (bcrypt) for the basic-auth file; nginx verifies
# it through musl's crypt(), which supports $2y$.
# netcat-openbsd: the wait-for-api loop in the entrypoint.
RUN apk add --no-cache apache2-utils netcat-openbsd

# Copy nginx configuration
COPY nginx.conf /etc/nginx/nginx.conf

# Copy entrypoint script
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Use the entrypoint script
ENTRYPOINT ["/entrypoint.sh"]
