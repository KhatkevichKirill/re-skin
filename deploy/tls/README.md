# HTTPS for the dashboard (AICP-150) — design + owner-run runbook

**Deployed.** First issuance and the nginx :443/:80 listeners went live
2026-10-16 (main 88a6fed); HSTS raised to one year 2026-10-17 (main e96a016);
renewal runs from the host `reskin-cert-renew.timer`. The runbook below is
kept as the record and for re-issuance/rollback; any change still needs owner
approval.

## Design

| Decision | Choice | Why |
|---|---|---|
| Name | `203-0-113-10.sslip.io` | No domain is owned. sslip.io answers any `a-b-c-d.sslip.io` with `a.b.c.d` (verified 2026-10-16 via 8.8.8.8 and 1.1.1.1: A `203.0.113.10`, no AAAA, no CAA). |
| CA | Let's Encrypt, 90-day ECDSA cert for exactly that name | Free, automatable, trusted everywhere. |
| ACME challenge | **HTTP-01, webroot served by the nginx container** on :80 | Renewals need no downtime and no port juggling: certbot writes a file into `/var/www/re-skin-acme`, nginx serves it. TLS-ALPN-01 would need the ACME client to own :443 (certbot has no TLS-ALPN at all; lego's standalone would have to stop nginx) or an nginx stream/ALPN split — more moving parts for no gain. DNS-01 is impossible: we do not control sslip.io's zone. |
| First issuance | `certbot --standalone` on :80, **before** nginx publishes :80 | Breaks the chicken-and-egg (nginx refuses to start without the cert files it references). :80 is free today. Afterwards `certbot reconfigure` moves the lineage to webroot. |
| Where certbot runs | **On the host, as the pinned `certbot/certbot:v5.8.0` image** driven by a systemd timer (`reskin-cert-renew.timer` → `deploy/tls/renew-cert.sh`) | No package install (Ubuntu 22.04's apt certbot is 1.21: no `reconfigure`, no ARI, no `--ip-address` fallback), current certbot, pinned like the other images. Not a compose service: the reload needs `docker exec` on the host (a container would need the Docker socket), and a compose service would put a cron-style job into the stack that `up -d --scale worker=2` manages. |
| Reload | `nginx -t` then `nginx -s reload` via `docker exec`, **only when the cert file changed** | Graceful: no dropped connections, no full restart. `docker compose restart nginx` (the existing post-deploy rule) also picks up a new cert. |
| :443 | TLS 1.2 + 1.3 (Mozilla intermediate), HTTP/2, session cache, no tickets, no OCSP stapling (LE dropped OCSP) | Standard. |
| HSTS | `max-age=31536000` (1 year) on :443 only, **no** `includeSubDomains`, **no** `preload` | Started at `max-age=300` (2026-10-16) so an early rollback cost browsers at most 5 minutes; raised to 1 year 2026-10-17 once https, renewal and `PUBLIC_BASE_URL` were all live (below). The name is borrowed (sslip.io is shared), so never wider than this one host. |
| :80 | ACME webroot + `308` to `https://<name>` for everything else | 308 keeps POST a POST. |
| :8847 (legacy http) | Kept. **Browser page loads (GET/HEAD outside `/api/`) → 301 to https, before basic auth runs.** `/public/*`, `/api/*` and every non-GET are still proxied over http exactly as today. | Existing `http://203.0.113.10:8847/public/...` links in the Final Creatives sheet, webhook payloads and ad tooling keep working byte-for-byte. Operators are moved to https without another password prompt over http. Scripts that call `/api/` over http (curl does not follow redirects, and drops `-u` across hosts when it does) keep working. |
| Basic auth | Unchanged, same htpasswd on both listeners | — |
| CSRF (AICP-152) | No setting needed | On :443 nginx forwards `Host: 203-0-113-10.sslip.io` (no port) and `X-Forwarded-Proto: https`, so `app/csrf.py` derives port 443 and trusts `Origin: https://203-0-113-10.sslip.io` by the Host rule. Pinned by tests in `backend/tests/test_csrf.py`. `http://<name>` (port 80) is correctly a different origin. After `PUBLIC_BASE_URL` flips, the legacy `:8847` pages still pass via their own Host. |

nginx.conf has the name literally in three places (`$tls_host` map and the
two `ssl_certificate*` paths): nginx only loads certificate files at startup
when the path has no variables.

### Firewall (AICP-149, not applied yet)

Published Docker ports (`80`, `443`, `8847`, and the existing `5432`) are
DNAT'ed in `PREROUTING` and filtered in **`FORWARD` → `DOCKER-USER`**, not in
`INPUT`. A ufw/`INPUT` rule neither opens nor closes them. The AICP-149 rules
must therefore:

- in `DOCKER-USER`: allow `tcp/80` and `tcp/443` from **anywhere** (Let's
  Encrypt validates from several unpublished vantage points, so :80 cannot be
  IP-allow-listed), and `tcp/8847` for the transition. Match on the original
  port like the existing 5432 pair does
  (`-m conntrack --ctorigdstport 443 --ctdir ORIGINAL -j ACCEPT`), placed
  before any default `DROP`;
- keep outbound `tcp/443` to the ACME API (`acme-v02.api.letsencrypt.org`) for
  renewals;
- the one-off standalone issuance below uses `-p 80:80`, so it also goes
  through `DOCKER-USER`.

Any provider-level (cloud panel) firewall must also allow 80/443 inbound.

## Runbook (owner-run, from `/root/re-skin`, after the merge to `main`)

`NAME=203-0-113-10.sslip.io` and
`IMG=certbot/certbot:v5.8.0@sha256:f70ad0adbb7e117f0fe42a63c553f28ea451edabc0148757b6efcd9735acaa20`
in the shell for the commands below.

0. **Pre-checks**
   ```
   getent hosts $NAME                      # 203.0.113.10
   ss -tlnp | grep -E ':(80|443) ' || echo "80/443 free"
   ```
   Open 80/443 inbound at the provider firewall if there is one.

1. **Dry run against Let's Encrypt staging** (nothing stored as a real cert;
   binds host :80 for ~10 s):
   ```
   mkdir -p /etc/letsencrypt /var/lib/letsencrypt /var/log/letsencrypt /var/www/re-skin-acme
   docker run --rm --name reskin-certbot-issue -p 80:80 \
     -v /etc/letsencrypt:/etc/letsencrypt -v /var/lib/letsencrypt:/var/lib/letsencrypt \
     -v /var/log/letsencrypt:/var/log/letsencrypt \
     $IMG certonly --standalone --dry-run -d $NAME \
     --non-interactive --agree-tos --register-unsafely-without-email
   ```
   (Use `-m <address>` instead of `--register-unsafely-without-email` if you
   want LE's policy/incident mails; LE no longer sends expiry mails either way.)

2. **Real issuance** — the same command without `--dry-run`. Check:
   ```
   ls -l /etc/letsencrypt/live/$NAME/      # fullchain.pem privkey.pem ...
   openssl x509 -in /etc/letsencrypt/live/$NAME/fullchain.pem -noout -subject -enddate
   ```

3. **Deploy nginx** (only nginx changes; no drain needed, api/worker untouched;
   in-flight uploads/media streams through nginx are cut once):
   ```
   docker compose up -d --no-deps nginx
   docker logs --since 1m re-skin-nginx      # no [emerg]
   ```
   If AICP-152 / 153-158 ship in the same window, follow their runbooks
   instead and let their `up -d --scale worker=2` + `restart nginx` carry this
   change; step 2 must still come first.

4. **Move renewals to webroot** (does a staging dry run through nginx :80 and
   saves the lineage's authenticator):
   ```
   docker run --rm --name reskin-certbot-reconf \
     -v /etc/letsencrypt:/etc/letsencrypt -v /var/lib/letsencrypt:/var/lib/letsencrypt \
     -v /var/log/letsencrypt:/var/log/letsencrypt -v /var/www/re-skin-acme:/var/www/re-skin-acme \
     $IMG reconfigure --cert-name $NAME --webroot -w /var/www/re-skin-acme
   grep -E 'authenticator|webroot' /etc/letsencrypt/renewal/$NAME.conf   # webroot
   ```

5. **Install and test the renewal timer**:
   ```
   cp deploy/tls/reskin-cert-renew.service deploy/tls/reskin-cert-renew.timer /etc/systemd/system/
   systemctl daemon-reload
   deploy/tls/renew-cert.sh --dry-run        # staging renewal via webroot + served-cert check
   systemctl enable --now reskin-cert-renew.timer
   systemctl list-timers reskin-cert-renew.timer
   systemctl start reskin-cert-renew.service && systemctl status --no-pager reskin-cert-renew.service
   ```

6. **Verify**
   ```
   curl -sI https://$NAME/v2/ | grep -iE '^HTTP|strict-transport'          # 401 + max-age=31536000
   curl -s -o /dev/null -w '%{http_code}\n' -u "<user>:<pass>" https://$NAME/v2/   # 200
   curl -sI http://$NAME/v2/ | grep -iE '^HTTP|location'                   # 308 -> https
   curl -sI http://203.0.113.10:8847/v2/ | grep -iE '^HTTP|location'     # 301 -> https, no 401
   curl -s -o /dev/null -w '%{http_code}\n' -u "<user>:<pass>" http://203.0.113.10:8847/api/v2/projects  # 200 (still http)
   # an existing /public/ link from the sheet, over http -> 200 video/mp4:
   curl -s -o /dev/null -w '%{http_code} %{content_type}\n' -r 0-0 'http://203.0.113.10:8847/public/runs/<id>/result?token=<t>'
   # same link over https -> 206/200:
   curl -s -o /dev/null -w '%{http_code}\n' -r 0-0 'https://'$NAME'/public/runs/<id>/result?token=<t>'
   echo | openssl s_client -connect $NAME:443 -servername $NAME 2>/dev/null | openssl x509 -noout -issuer -enddate
   ```
   In a browser on `https://$NAME/v2/`: log in, open a project, do one HTMX
   POST (e.g. rename something harmless) — it must not 403 (CSRF), and play
   a media file (X-Accel).

7. **`PUBLIC_BASE_URL`** — see the switch plan below. Not part of steps 1-6.

### Expiry: how it gets noticed

Let's Encrypt stopped sending expiry emails in 2025, so the box has to notice:

- `renew-cert.sh` exits non-zero when certbot fails **or** when the cert
  served on :443 has under 14 days left (it checks the live listener, so a
  missed reload counts too). The unit then shows as failed:
  `systemctl --failed`, `journalctl -u reskin-cert-renew`.
- Nobody reads that unprompted. **Recommended: an external TLS-expiry monitor**
  on `https://203-0-113-10.sslip.io/` (any uptime service with an SSL-expiry
  check; a 401 page is fine for it) that alerts the owner — owner's choice of
  service.

### HSTS ramp

Done: `max-age=300` from 2026-10-16, `max-age=31536000` from 2026-10-17
(owner-approved; the intermediate `86400` step was skipped). The value lives in
the `$hsts_header` map in `deploy/nginx/nginx.conf`, which only sets it for
`$scheme = https`, so :8847 and :80 never send it. To change it: edit the map,
`docker compose exec nginx nginx -t`, then `docker compose restart nginx` (or
`nginx -s reload`; the file is bind-mounted). Never add
`includeSubDomains`/`preload` — sslip.io is not ours.

HSTS is keyed on the host name only: it pins `203-0-113-10.sslip.io` to
https in browsers that have seen it, and does **not** affect
`http://203.0.113.10:8847` (an IP literal, never HSTS'd).

## `PUBLIC_BASE_URL` switch plan (+ AICP-151)

`PUBLIC_BASE_URL` builds the webhook `run_url`/`result_url`
(`app/automation.py`) and is a trusted CSRF origin.

- Flip it to `https://203-0-113-10.sslip.io` **after** step 6 has passed and
  the timer has done one real (non-dry) successful run or at least a week has
  passed — newly issued https links are only as durable as the cert pipeline.
- It is read from `.env` by api **and** worker (the webhook is built in the
  worker): `.env` change + `docker compose up -d --scale worker=2` (restart
  does not re-read `.env`) + `docker compose restart nginx`, under the normal
  drain rule.
- Old `http://203.0.113.10:8847/public/...` links keep working: :8847 keeps
  proxying `/public/` over http.
- **AICP-151** (dedicated `PUBLIC_LINK_SECRET`, no `BASIC_AUTH_PASS` fallback; see CLAUDE.md) invalidates every existing
  token link anyway. Ship the flip in the **same window** as 151 so external
  consumers re-fetch links once: new links are https + new secret, old ones
  break exactly once. Only after that is it worth retiring :8847 (drop the
  port or redirect everything) — a separate decision. Not implemented here.

## Rollback

1. `git revert` the AICP-150 merge (or restore `deploy/nginx/nginx.conf` and
   `docker-compose.yml` from the previous commit), then
   `docker compose up -d --no-deps nginx` — back to :8847-only, as today.
2. `systemctl disable --now reskin-cert-renew.timer` (the cert files under
   `/etc/letsencrypt` can stay; they are harmless).
3. If `PUBLIC_BASE_URL` was already flipped: set it back to
   `http://203.0.113.10:8847`, `up -d --scale worker=2`, `restart nginx`.
   https links issued meanwhile stop working until https is back.
4. HSTS is `max-age=31536000`: browsers that saw it keep forcing
   `https://203-0-113-10.sslip.io` for up to a year, so a rollback to http
   cannot use that name — use `http://203.0.113.10:8847` (IP, not HSTS'd)
   meanwhile. Lowering the header to `max-age=0` only reaches browsers that
   load an https page again.

If nginx fails to start after step 3 (`[emerg] cannot load certificate`), the
cert is missing or unreadable: fix step 2, or roll back with step 1 above.

## sslip.io / rate-limit notes

- **Dependency:** the dashboard name now depends on sslip.io's DNS
  (`ns-ovh.sslip.io`, `ns-0[01].nip.io`). If it is down, the name stops
  resolving (dashboard by name and renewals fail); `http://203.0.113.10:8847`
  still works. sslip.io is not on the Public Suffix List, so for Let's Encrypt
  every `*.sslip.io` user shares one "registered domain".
- **Rate limits:** LE's "50 new certificates per registered domain per week"
  would be shared by all sslip.io users, but sslip.io says LE raised its limit
  to **250,000/week** and refused a raise to 500,000 — so it is a busy, shared
  bucket that can occasionally fill (sslip.io's own page tells rate-limited
  users to retry later or use an IP certificate). **Renewals of the same name
  are exempt** from that limit, so the practical risk is the **first
  issuance** only: if step 2 is refused with `too many certificates already
  issued for "sslip.io"`, wait and retry; nothing is broken in the meantime
  (nginx is not touched until step 3). Per-name limits that apply to us alone:
  5 duplicate certs/week and 5 failed validations/hour — use `--dry-run`
  (staging) for experiments, as above.
- **Fallback if issuance stays blocked:** a Let's Encrypt **IP-address**
  certificate for `203.0.113.10` (certbot ≥ 5 `--ip-address`, `shortlived`
  profile, ~6-day lifetime — the timer then has to renew every few days and
  the 14-day warning threshold must drop). nginx would then need the IP as the
  TLS name. Not set up here.
- No CAA record on sslip.io or the name, so LE may issue.
