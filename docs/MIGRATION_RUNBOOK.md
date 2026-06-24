# re-skin — Migration Runbook (self-instructions for Claude)

Target host: **4 vCPU / 16 GiB RAM / 160 GiB disk**. Goal: move the running
`re-skin` stack from the old VPS to this new one with **no data loss** and
minimal downtime, then re-tune resource limits for the bigger box.

> **Where to run me:** on the **NEW** VPS. All setup (Docker, compose, bring-up,
> verification) happens here. The old box only holds 3 things to copy.

---

## 0. What the repo gives us vs. what must come from the OLD box

`git clone` brings all code + `docker-compose.yml` + migrations. **Three things
are gitignored and MUST be copied from the old machine** — without them the move
is not seamless:

| Artifact | Why it's critical |
|---|---|
| `.env` | Secrets: `KIE_API_KEY`, `GDRIVE_DEFAULT_FOLDER_ID`, `BASIC_AUTH_USER/PASS`, `REDIS_PASSWORD`, `APP_EXTERNAL_PORT`, `PUBLIC_LINK_SECRET`. Copying verbatim keeps auth, Drive delivery, and existing public-link tokens identical. |
| `secrets/gdrive-sa.json` | Google Drive service-account key (delivery). |
| `data/` | SQLite `app.db` **+ all project/run media**. Preserves every project, run, and the schema (the live DB already carries the manually-applied `runs.model` / `video_projects.name` columns). |

---

## 1. Prerequisites (NEW box)

- Install Docker Engine + the Compose plugin, and `rsync`.
- Open the app port (default **8847**, value of `APP_EXTERNAL_PORT`) in the firewall/security group.
- Ensure git can clone the repo (it may be private): `gh auth login`, a PAT, or an SSH deploy key.

```bash
docker --version && docker compose version && rsync --version   # sanity
```

## 2. Clone the repo

```bash
git clone https://github.com/KhatkevichKirill/re-skin.git /root/re-skin
cd /root/re-skin && git checkout main
```

## 3. Bring over the 3 artifacts from the OLD box

Either the operator copies them in, **or** (if this box has SSH access to the old
one — operator provides `OLD=user@old-ip`) I rsync them:

```bash
OLD=root@OLD_IP          # set this
rsync -avz  "$OLD":/root/re-skin/.env            ./.env
rsync -avz  "$OLD":/root/re-skin/secrets/        ./secrets/
rsync -avz --progress "$OLD":/root/re-skin/data/ ./data/   # first pass, old still running
```

## 4. Final consistent data sync (minimize downtime)

SQLite is being written while the old stack runs; copy it **after** quiescing to
avoid a torn WAL. Do the bulk copy in step 3 while old is live, then:

```bash
# on the OLD box:
cd /root/re-skin && docker compose down

# back on the NEW box — fast delta, now the DB is consistent
rsync -avz --progress "$OLD":/root/re-skin/data/ ./data/
```

(`data/` includes `app.db`, `app.db-wal`, `app.db-shm`, and `projects/…` media —
rsync the whole dir so WAL state is consistent.)

## 5. Re-tune `docker-compose.yml` for 4 vCPU / 16 GiB

The old caps (worker 2.5 G / no-swap / 1.5 CPU) were sized for a 3.8 GiB box to
prevent an OOM freeze. On 16 GiB we can be generous. I will edit:

- **worker**: `cpus: "3"`, `mem_limit: 8g`, `memswap_limit: 10g` (keep
  `restart: unless-stopped`). Optionally add a **second worker** for throughput
  (their volume ≈100 gen/day; jobs are mostly cloud-wait so 2 workers parallelize
  well) — requires dropping the fixed `container_name` on worker and
  `docker compose up -d --scale worker=2`, or a `worker2` service.
- **api**: `mem_limit: 2g`.
- **env**: `FFMPEG_THREADS=4` (more cores now), keep `FFMPEG_MAX_FPS=30`.
- Keep `restart: unless-stopped` on all services.

Decision on 1 vs 2 workers: start with **1 generous worker**, watch a few real
jobs (`docker stats`), then add a 2nd if the queue backs up.

## 6. Database

The copied `app.db` already has the full schema (it was `create_all`-built and
hand-ALTERed). **Do not** run `alembic upgrade head` against it (it's unstamped →
alembic would try to recreate existing tables). Instead, stamp it so future
migrations are clean:

```bash
cd /root/re-skin/backend
DATABASE_URL="sqlite:////root/re-skin/data/app.db" alembic stamp head
```

(If ever starting from an EMPTY db instead of copying: `alembic upgrade head`
creates everything from scratch — that path is fine too.)

## 7. Build & start

```bash
cd /root/re-skin
docker compose up -d --build
docker compose ps          # redis, api, worker, nginx all "Up"
```

## 8. Verify (must all pass before cutover)

```bash
# nginx → api alive (basic-auth gate): expect 401, NOT 000/502
curl -s -o /dev/null -w 'app: %{http_code}\n' http://127.0.0.1:8847/v2/

# new code present in the image
docker exec re-skin-api grep -c "runs/{rid}/copy" app/app.py 2>/dev/null || \
docker exec re-skin-api python3 -c "import app.public, app.api_v2; print('routers ok')"

# data migrated: project/run counts match the old box
docker exec re-skin-api python3 -c "
import sqlite3; c=sqlite3.connect('/app/data/app.db')
print('projects:', c.execute('SELECT count(*) FROM video_projects').fetchone()[0])
print('runs:', c.execute('SELECT count(*) FROM runs').fetchone()[0])
print('done runs:', c.execute(\"SELECT count(*) FROM runs WHERE status='done'\").fetchone()[0])
"
```

Then in a browser: log in (basic auth), confirm the projects list shows the
migrated projects, open a done run, test **Download** and a **public token link**
(`/public/runs/<id>/result?token=…`), and tail `docker compose logs -f worker`
while starting one cheap 480p run end-to-end (generate → stitch → deliver to Drive).

Optionally run the test suite once: `cd backend && pip install -r requirements.txt && pytest -q`.

## 9. Cutover & cleanup

- Point usage at the **new IP:8847**. Public token links are rebuilt from the
  page origin at copy-time, so new links use the new IP automatically. **Links
  already shared with the old IP will break** once the old box is gone —
  regenerate via the Copy buttons. (Tokens themselves stay valid because
  `PUBLIC_LINK_SECRET` was copied with `.env`.)
- Keep the old box stopped (not deleted) for a few days as a fallback.
- After confidence: decommission the old VPS.

## Gotchas

- **InsightFace model (~300 MB)** re-downloads on the first analyze (the
  `insightface-models` named volume is per-machine, not in `data/`). One-time.
- **Don't start the new stack until `.env` + `secrets/` + `data/` are in place**,
  or it'll come up empty / unauthenticated-broken.
- If `curl` to `:8847` returns **502**, restart nginx (it caches the api upstream
  IP): `docker compose restart nginx`.
- Disk: media accumulates (~tens of GB over time). 160 GiB is comfortable; set up
  a periodic cleanup of old `data/projects/<id>` later if needed.
