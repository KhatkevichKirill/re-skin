# Container hardening + dependency upgrade — production window runbook (AICP-153 / AICP-158)

Branch `fix/aicp-153-158-deps-containers`. One rebuild covers both issues.
**DEPLOYED to production 2026-10-17 ~17:14 UTC** (≈25 s hard downtime; rollback
images `re-skin-{api,worker,nginx}:rollback-e96a016`). Kept as the record and
for rollback. Before the deploy every step was rehearsed on an
isolated compose project (`-p reskin153test`, no shared containers, volumes,
ports or data) on 2026-10-17 — see *Rehearsal evidence* at the end.

## What changes at deploy time

| Area | Before (prod at b2a8297) | After |
|---|---|---|
| api/worker base | `python:3.10-slim` (floating, 3.10.20) | `python:3.12.15-slim-trixie@sha256:a6e34c59…` (Debian 13 → same ffmpeg **7.1.5**) |
| api/worker deps | `requirements.txt`, unpinned transitives, pytest/respx in image | `requirements.lock` (pip-compile, every artifact hashed, `--require-hashes`): fastapi 0.143.2, starlette 1.8.0, python-multipart 0.0.32, jinja2 3.1.6, uvicorn 0.54.0, h11 0.16.0, sqlalchemy 2.0.54 (httpx stays 0.27.2 / httpcore 1.0.9). Tests in `requirements-dev.txt`/`.lock`, not in the image |
| api/worker build | single stage with `build-essential` | builder stage (toolchain) → runtime stage (no compilers) |
| api/worker user | uid 0, all default caps | `reskin` uid/gid **10001**, `cap_drop: [ALL]`, `no-new-privileges` |
| InsightFace volume | mounted at `/root/.insightface` | same named volume `re-skin_insightface-models`, mounted at `/home/reskin/.insightface` |
| nginx | whole `.env` via `env_file`, image built from floating `nginx:alpine`, apr1 htpasswd | only `BASIC_AUTH_USER`/`BASIC_AUTH_PASS`, `nginx:1.31.2-alpine@sha256:54f2a904…` (the digest prod's base already is), bcrypt (`htpasswd -B`, cost 6), `cap_drop: [ALL]` + `CHOWN/SETUID/SETGID`, `no-new-privileges` |
| db / redis | `postgres:16-alpine`, `redis:7-alpine` floating | pinned by digest to the images prod runs **today** (16.14 / 7.4.9 — **no version change, no data migration**), `no-new-privileges` |

Not changed: Postgres data volume, Alembic head (`b8d0f2a4c6e9`, no migration
in this change), `.env` contents, TLS cert/renewal setup, nginx.conf routes.

### How the AICP-150 TLS pieces work under the hardening (verified)

- **Ports 80/443 with `cap_drop: ALL`** (no `NET_BIND_SERVICE`): the nginx
  service sets `sysctls: net.ipv4.ip_unprivileged_port_start: 0` (already on
  main), so the master binds 80/443/8847 without the capability.
- **Certificate + key**: host `/etc/letsencrypt/{live,archive}` are `0700
  root:root`, `privkey*.pem` is `0600 root:root`. Only the nginx **master**
  (uid 0) opens them, at start and on reload; it is the owner, so owner bits
  suffice even without `DAC_OVERRIDE`/`DAC_READ_SEARCH`. No permission change
  on the host is needed or wanted. Workers (uid 101) never touch the files.
- **ACME webroot** `/var/www/re-skin-acme` (0755, files certbot writes 0644)
  is mounted read-only and served by the workers — world-readable is enough.
- **Renewal** (`deploy/tls/renew-cert.sh`, `reskin-cert-renew.timer`):
  certbot runs in its own root container that is untouched by this change;
  the script's `docker exec re-skin-nginx nginx -t` and `nginx -s reload`
  run as root inside the hardened container and work (rehearsed: config test
  passes, reload respawns the workers, which re-reads the cert as root-owner).

### What the new app code from main needed (verified)

- CSRF (`app/csrf.py`) is a pure ASGI middleware — unaffected by starlette 1.x.
- Webhook IP pinning uses `httpx.Client.post(..., extensions={"sni_hostname":…})`;
  httpx stays 0.27.2 / httpcore 1.0.9, so the extension behaves as on prod
  (rehearsed: a pinned-IP HTTPS request with SNI verifies; the validator
  refuses http, loopback and non-allow-listed hosts).
- Spend ledger, manual_creatives, input hardening, `PUBLIC_LINK_SECRET`: no
  dependency or filesystem interaction beyond `./data`; their env vars reach
  api/worker through `env_file: .env` exactly as today. nginx needs none of
  them (its config has no env templating).
- **Fixed on this branch**: the stitch cut pool (`pipeline_v2`
  `_run_stitch_and_deliver`) read ORM rows through the one shared Session from
  `STITCH_CUT_CONCURRENCY` threads. Under 3.12 + SQLAlchemy 2.0.54 that failed
  three stitch tests on every run (`ObjectDeletedError`); the threads now get a
  plain snapshot. The race is latent on prod's 3.10 image too.

## Production facts (measured read-only 2026-10-17 ~02:30 UTC)

| Item | Value |
|---|---|
| `/root/re-skin/data` | **72 GB, 7 414 inodes**, every one `root:root` (dirs 755, files 644). `chown -R` is metadata-only → **≈1–2 s** (a 17-inode copy took 0.01 s; 7.4k inodes on ext4 is seconds at most) |
| `/root/re-skin/secrets` | dir `755 root:root`; `gdrive-sa.json` 2.3 KB, **`644 root:root` (world-readable today)**; `.gitkeep` |
| `/root/re-skin/.env` | `600 root:root` already (since AICP-151) |
| `re-skin_insightface-models` | 600.7 MB, 9 entries, `root:root` 755/644 (readable by 10001 even before the chown) |
| `re-skin_postgres-data` | untouched; `re-skin-db` = `postgres@sha256:e013e867…` = **16.14** (branch pins the same digest) |
| `re-skin-redis` | `redis@sha256:6ab0b6e7…` = **7.4.9** (branch pins the same digest); queue data only, drained |
| api / worker / nginx images | `re-skin-api` `52adaa05cbfd`, `re-skin-worker` `37a73a8479af` (built 01:49 UTC from b2a8297), `re-skin-nginx` `78a18b9bbc00` (3 months old, base `nginx@sha256:54f2a904…`) |
| `/etc/letsencrypt` | live/archive `0700 root`, `privkey1.pem 0600 root`; cert valid to 2027-01-14 |
| Disk | 31 GB free on `/` (one new api+worker pair ≈ 2.6 GB each, shared layers) |

## Window runbook

All commands as host root from `/root/re-skin`. `$U`/`$P` below stand for
the basic-auth user and password from `.env` — export them in your own shell
(`set -a; . ./.env; set +a` prints nothing) and never echo them.
`NAME=203-0-113-10.sslip.io`.

### 0. Preconditions (T-1 day / T-30 min)

- Owner has approved the window and answered the open questions (Linear
  AICP-153 / AICP-158).
- Branch rebased on the **current** `origin/main` (Codex sessions move main;
  last rebased onto e96a016, the AICP-150 one-year HSTS), suite green, Terra gate
  re-run on `main...HEAD` if main moved since the last gate.
- Nobody else is deploying: `git -C /root/re-skin log -1 --oneline`,
  `git fetch && git status -sb`, `docker ps` (no other deploy's containers
  restarting), and tell the Codex-session owner.
- Operators warned: dashboard down ~5 min; nothing new may be launched from
  T-15 min.
- `df -h /` ≥ 10 GB free.

### 1. Rollback tags + merge (prod keeps serving)

```
PREV=$(git -C /root/re-skin rev-parse --short HEAD)          # e.g. b2a8297
for s in api worker nginx; do docker tag re-skin-$s:latest re-skin-$s:rollback-$PREV; done
docker images --format '{{.Repository}}:{{.Tag}} {{.ID}}' | grep "rollback-$PREV"   # 3 lines
git -C /root/re-skin merge --no-ff fix/aicp-153-158-deps-containers \
    -m "Merge fix/aicp-153-158-deps-containers: py3.12 + patched deps, non-root hardened containers (AICP-153, AICP-158)"
```

The tags MUST exist before step 2: the builds re-point `:latest`.

### 2. Build, one image at a time (prod keeps serving)

```
docker compose build worker
docker compose build api
docker compose build nginx
docker run --rm --entrypoint sh re-skin-api -c 'id -u; python -V; ffmpeg -version | head -1'
#   10001 / Python 3.12.15 / ffmpeg version 7.1.5-0+deb13u1
docker run --rm --entrypoint sh re-skin-worker -c 'id -u'                                # 10001
```

Concurrent `api`+`worker` builds fail with pip exit 2 on this box — keep them
sequential. Building does not touch the running containers.

### 3. Drain (everything — db and redis are recreated too)

Stop launching work in the UI, then wait until **every** row reads 0:

```
docker exec -i re-skin-db psql -U reskin -d reskin -At -F' ' < deploy/drain-check.sql
docker exec re-skin-worker-1 sh -c 'rq info -u "$REDIS_URL"'   # queues 0, both workers "idle"
```

`runs` includes `stitching`/`delivering`/`preprocessing`, not just
`processing` (2026-08-11 lesson). Tail edits are not re-queued after an
interruption, so they must be at 0 too. The automation tick (in the workers,
every 5 min) can launch a run at any moment: go straight on to step 4.

### 4. Backup + stop (downtime starts at `stop`)

```
scripts/backup_db.sh predeploy                    # old (root) containers — fine
docker exec -i re-skin-db psql -U reskin -d reskin -At -F' ' < deploy/drain-check.sql   # still all 0?
docker compose stop worker api                    # workers first; nginx keeps answering (502)
docker exec -i re-skin-db psql -U reskin -d reskin -At -F' ' < deploy/drain-check.sql   # re-check
```

If a run went active between the check and the stop (automation tick), it was
interrupted mid-job; the new workers' reconcile tick re-enqueues it within
~3 min (no rebill for completed segments). Note its id and watch it in step 8.

### 5. Ownership + permission migration (one time, ~5 s)

```
chown -R 10001:10001 data
chown -R 10001:10001 secrets
chmod 700 secrets
chmod 600 secrets/gdrive-sa.json
docker run --rm --network none -v re-skin_insightface-models:/v alpine:3.19 chown -R 10001:10001 /v
# verify
find data secrets ! -uid 10001 | head           # empty
stat -c '%U:%G %a %n' secrets secrets/gdrive-sa.json .env   # 10001 700 / 10001 600 / root 600
```

- Do **not** chmod `data/` restrictively: nginx workers (uid 101) serve
  X-Accel media from it and need world-read/x; the app's umask 022 keeps new
  files 644/755 (rehearsed: worker output is `10001 644`).
- `chmod 700/600` without the chown would make the service account
  unreadable inside the new containers (`PermissionError`).
- `/etc/letsencrypt` and `/var/www/re-skin-acme`: no change (see above).
- Anything later created under `data/` as host root (manual fixes, `docker
  cp`) that the app must rewrite needs this `chown -R` again.

### 6. Up (recreates db, redis, api, workers, nginx)

```
docker compose up -d --scale worker=2
docker compose restart nginx                      # stale-upstream rule, always
docker compose ps                                 # 6 containers; db/redis (healthy)
```

`--build` is unnecessary after step 2 (harmless). Downtime ends when
`https://$NAME/v2/` answers 200 in step 7.

### 7. Verify (≈10 min; all through nginx, never only the container)

```
# identity / privileges
for c in re-skin-api re-skin-worker-1 re-skin-worker-2; do docker exec $c sh -c 'id -u; grep -E "^CapEff|^NoNewPrivs" /proc/1/status'; done
#   10001 / CapEff: 0000000000000000 / NoNewPrivs: 1   (x3)
docker exec re-skin-api sh -c 'python -V; ffmpeg -version | head -1; ffprobe -version | head -1'
docker exec re-skin-api python -c 'import fastapi,starlette,multipart,jinja2;print(fastapi.__version__,starlette.__version__,multipart.__version__,jinja2.__version__)'
#   0.143.2 1.8.0 0.0.32 3.1.6
docker exec re-skin-nginx sh -c 'env | grep -c -E "KIE|POSTGRES|GDRIVE|REDIS|PUBLIC_LINK|DATABASE|ARK"'   # 0
docker exec re-skin-nginx cut -d: -f2 /etc/nginx/.htpasswd | cut -c1-4                                   # $2y$
docker exec re-skin-db psql -U reskin -d reskin -At -c 'select version_num from alembic_version'         # b8d0f2a4c6e9

# HTTPS front door, :80, legacy :8847
curl -sI https://$NAME/v2/ | grep -iE '^HTTP|strict-transport'                         # 401 + max-age=31536000
curl -s -o /dev/null -w '%{http_code}\n' -u "$U:$P" https://$NAME/v2/                   # 200
curl -s -o /dev/null -w '%{http_code} %{redirect_url}\n' http://$NAME/v2/               # 308 https://…
curl -s -o /dev/null -w '%{http_code} %{redirect_url}\n' http://localhost:8847/v2/      # 301 https://…
curl -s -o /dev/null -w '%{http_code}\n' -u "$U:$P" http://localhost:8847/api/v2/projects  # 200

# CSRF (middleware runs before routing: 403 = rejected, 404 = allowed)
curl -s -o /dev/null -w '%{http_code}\n' -u "$U:$P" -X POST -H 'Origin: https://evil.example' https://$NAME/api/v2/csrf-probe   # 403
curl -s -o /dev/null -w '%{http_code}\n' -u "$U:$P" -X POST -H "Origin: https://$NAME" https://$NAME/api/v2/csrf-probe          # 404

# spend header (AICP-155)
curl -s -u "$U:$P" https://$NAME/v2/credits-fragment | grep -o 'today: <strong[^>]*>\$[0-9.,]*'

# public link via X-Accel from a 10001-owned file (token never printed)
RID=$(docker exec re-skin-db psql -U reskin -d reskin -At -c "select id from runs where status='done' and result_local_path is not null order by updated_at desc limit 1")
TOK=$(docker exec re-skin-api python -c "from app.public import make_result_token; print(make_result_token('$RID'))")
curl -s -r 0-1023 -o /dev/null -w '%{http_code} %{size_download}\n' "https://$NAME/public/runs/$RID/result?token=$TOK"   # 206 1024
curl -s -r 0-1023 -o /dev/null -w '%{http_code}\n' "http://localhost:8847/public/runs/$RID/result?token=$TOK"          # 206
unset TOK

# worker writes + InsightFace + webhook validator (no POST is sent)
docker exec re-skin-worker-1 sh -c 'f=/app/data/.aicp158-probe; echo x > $f && rm $f && echo write-ok'
docker exec re-skin-worker-2 python -c 'from app.face import _build_detector; _build_detector(); print("insightface ok")' 2>&1 | tail -1
docker exec re-skin-worker-1 python -c 'from app import webhook_safety as w; from app.config import settings; print(len(w.pinned_targets(settings.AUTOMATION_WEBHOOK_URL)), "vetted target(s)")'

# Drive service account + docker exec -u 0 cleanup on the NEW containers
scripts/backup_db.sh postdeploy

# TLS renewal path
deploy/tls/renew-cert.sh --dry-run        # certbot staging dry run + "served certificate ok"
docker exec re-skin-nginx nginx -t && docker exec re-skin-nginx nginx -s reload

# logs
docker compose logs --since 10m api worker nginx 2>&1 | grep -iE 'permission|denied|PermissionError|Traceback|emerg' || echo clean
```

Then one real ffmpeg job through the UI (no kie.ai spend): on a done run,
**Change ending** (tail edit) at any cut → it must reach `done` (worker
ffmpeg + Drive upload as 10001); play it on the run page (X-Accel). It leaves
a Drive file and a tail-edit row: keep it, or delete the row with the UI
button (the Drive copy stays). Optionally upload a small source to a scratch
project (ingest + face detect) and delete it.

If step 4 interrupted a run, watch it reach a terminal status.

### 8. After

- `git -C /root/re-skin push origin main`; Linear AICP-153/158 Current state.
- Keep the `rollback-$PREV` images until the next deploy has been stable.

## Operational changes after this deploy

- `docker exec re-skin-api …` now runs as `reskin`. Root inside the container
  (`docker exec -u 0`) has **no** capabilities, so it can only touch
  root-owned files — enough to remove files `docker cp` staged (always
  root-owned), nothing more.
- The Alembic recipe in CLAUDE.md still works (rehearsed): `docker cp` puts
  root-owned, world-readable files under `/app`, `alembic upgrade` only reads.
- `docker cp` a file INTO `/app/data` creates it root-owned: the app can read
  it (if the mode allows) and unlink it (the dir is 10001's) but not rewrite
  it — `chown 10001:10001` it on the host.
- `requirements.txt` is the human-edited list; after changing it, regenerate
  `requirements.lock` with the command in its header (Python **3.12**).
  Tests: `pip install --require-hashes -r requirements.lock -r
  requirements-dev.lock` (mixing the hashed lock with the unhashed
  `requirements-dev.txt` is refused).

## Rollback (rehearsed — ~2 min)

Ownership does **not** need reverting: the old images run as root with the
default capability set (`DAC_OVERRIDE`), so they read and write 10001-owned
`data/`, read the `700/600` service account and load InsightFace from the
same volume at `/root/.insightface` (all rehearsed). The compose file must go
back too — the old image with the new compose would look for InsightFace
under `/root/.insightface` while the volume is mounted at `/home/reskin/…`.

```
cd /root/re-skin
# 2026-10-17 deploy was a FAST-FORWARD (e96a016 -> ae8ec58), not a merge:
git checkout e96a016 -- docker-compose.yml deploy/nginx backend/Dockerfile   # then commit as a revert
# (for a --no-ff merge deploy instead: git revert -m 1 --no-edit HEAD)
for s in api worker nginx; do docker tag re-skin-$s:rollback-$PREV re-skin-$s:latest; done
docker compose up -d --scale worker=2            # NO --build: uses the retagged images
docker compose restart nginx
docker exec re-skin-api id -u                    # 0
curl -s -o /dev/null -w '%{http_code}\n' -u "$U:$P" https://$NAME/v2/   # 200
```

db/redis are recreated once more (back to the tag references) — drain first
if anything is running. Files the root images write during a rollback are
root-owned: before rolling forward again, repeat step 5's `chown -R`.
If the DB itself is damaged (not expected — no migration, same Postgres
image), restore the step-4 `predeploy` dump from Drive per CLAUDE.md.

## Downtime estimate

- **Hard downtime** (api stopped → `https://$NAME/v2/` 200 again): ≈ 1–3 min
  — stop ≤ 20 s, chown/chmod ≈ 5 s, `up -d` with db/redis recreate ≈ 15–30 s
  (db healthcheck), api start ≈ 5 s, nginx restart + API wait ≈ 5–10 s.
- **No new paid work** from the drain start until step 7 passes: drain time
  (unbounded — whatever is in flight) + ≈ 15 min.
- Rollback, if needed: ≈ 2 min hard downtime.

## Rehearsal evidence (2026-10-17, isolated project `reskin153test`)

Generated from this branch's `docker-compose.yml` with only names, ports
(127.0.0.1:18080/18443/18847), env (dummy secrets) and host paths swapped; a
copy of one prod project (root-owned, then chowned as in step 5), a copy of
the InsightFace volume, and a self-signed cert laid out exactly like
`/etc/letsencrypt` (0700 dirs, 0600 key, live→archive symlinks).

- api + worker `uid=10001`, `CapEff 0`, `NoNewPrivs 1`; Python 3.12.15, ffmpeg/ffprobe 7.1.5; fastapi 0.143.2 / starlette 1.8.0 / multipart 0.0.32 / jinja2 3.1.6 / uvicorn 0.54.0 / httpx 0.27.2.
- Alembic `upgrade head` via the CLAUDE.md `docker cp` recipe on an empty DB → `b8d0f2a4c6e9`.
- InsightFace buffalo_l loads from `/home/reskin/.insightface`; `cut_clip` and an RQ `process`-queue ffmpeg job write `10001 644` files; delete OK.
- nginx: master root `CapEff 0xc1` (CHOWN/SETUID/SETGID), binds 80/443/8847; zero app secrets in env; htpasswd `$2y$06`, `root:nginx 640`; `nginx -t` + `-s reload` via `docker exec` OK.
- https 401/200 + HSTS (`max-age=300` with the pre-e96a016 conf; the rebased one-year conf passes `nginx -t` under the same caps); :80 ACME file 200, else 308; :8847 page 301, `/api/` 200; CSRF evil Origin 403 / own Origin 404 / none 404; public link 206 (X-Accel), bad token 403; spend header `$0.00 / $250`; nginx limit_req 429 after the burst.
- Webhook validator: configured URL vetted, http / loopback / foreign host refused; SNI-pinned HTTPS GET 200.
- Rollback: old prod images (`re-skin-api:latest`/`re-skin-worker:latest`/`re-skin-nginx:latest` at b2a8297) with the b2a8297 compose on the chowned data: uid 0, https 200, public 206, worker write, InsightFace from `/root/.insightface`.
- Not rehearsable off-prod: a real certbot renewal, Drive upload with the real service account, a real kie.ai run — covered by step 7.
