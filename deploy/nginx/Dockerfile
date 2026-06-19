FROM nginx:alpine

# Install openssl for password hashing and netcat for health checks (alpine has a minimal base)
RUN apk add --no-cache openssl netcat-openbsd

# Copy nginx configuration
COPY nginx.conf /etc/nginx/nginx.conf

# Copy entrypoint script
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Use the entrypoint script
ENTRYPOINT ["/entrypoint.sh"]
