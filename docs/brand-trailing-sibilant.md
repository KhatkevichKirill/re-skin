# Brand trailing-sibilant check and free fix (AICP-389)

Seedance 2.5 sometimes says a brand name with an extra sibilant ending in localised
speech: "Speechifies" instead of "Listenify". Regenerating the segment costs credits and
often repeats the defect. The ASR check of the QA step cannot see it, because the Gemini
caption ASR normalises the word back to "Listenify".

This feature closes the gap for one shape of the defect. In every QA round of the
localisation autopilot (`qa:L`), a free check looks for a **separate trailing sibilant
after each scripted brand occurrence, followed by a pause**. When it finds one, the
autopilot offers a ready fix with before/after previews. If the operator approves it, the
fix is applied in place: same Job ID, same Drive file id, no regeneration and no credits.

The detector only proposes. Operator listening stays the judge.

Code:

- `backend/app/brand_check.py`: detector, alignment, renders.
- `backend/app/autopilot/flows/localisation_brandfix.py`: the compute, the ask, the
  handler.
- `backend/app/pipeline_audio_patch.py`: patch kind `same_run_lowpass`.
- `worker/brand_asr_worker.py`: the container.

## 1. Where it runs in `qa:L`

```
segqa:<tag>  ->  best:<rerun>  ->  brand:<tag>  ->  brandfix:<sha12>:<occ> ...  ->  asr:<tag>  ->  regen:<tag>  ->  listen:<tag>
 (free)          (restore)         (free, NEW)      (ask, NEW)                      (paid)         (ask)            (ask)
```

The check runs **after** segment QA and the best-take decision, and **before** the paid
ASR caption. It needs only the stitched final and the script on the final's timeline
(`localisation_qa.final_script`, from the segment-QA piece lengths). That way the round
that fixes the word never pays for a caption of the broken audio. On a fresh autopilot,
the first caption is still the budgeted one.

Names inside `qa:L`:

| Name | What it is |
|---|---|
| `brand:<tag>`, `brand:<tag>:r2`, `brand:<tag>:r3` | The compute (bounded retries). |
| `brandcheck:<tag>` | Handoff: the check stayed unavailable. Options: "go on without it", stop. |
| `brandfix:<base sha12>:<occ>` | The fix ask, one occurrence at a time. Options: "apply this fix (free)", "keep the current audio". |
| `brandapply:<base sha12>:<occ>` (+ `:rK`) | The launch of `localisation_apply_brand_audio_patch`. |
| `brandretry:<patch8>` | Handoff: the patch failed. Options: retry (same approved bytes), keep, stop. |
| `brandcap:<base sha12>` | Handoff: `MAX_BRAND_PATCHES` (3) filter patches were already done on this run. |

Rules:

- **Ordering against existing asks.** The fix ask *replaces* the round's `regen:`/`listen:`
  asks while they are still pending (`Ask.replaces`). If either one is already answered,
  the round skips the brand check, so an existing answer is always honoured. This matters
  for rounds that were in flight at deploy time.
- **The ask name carries the base SHA-256 prefix.** After "keep", the same occurrence on
  the same bytes is never asked again. A new final (rerun, restore, patch) is checked
  afresh.
- **The ask is bound** to:
  - the run version;
  - the transcript version;
  - the compute effect;
  - through the option's `params_sha256`, the base and candidate SHA-256.

  A changed base (any rewrite of the final, even under the same path) changes
  `core.run_version`, so the pending ask is superseded.
- **After an apply**, the changed result supersedes the round's open asks, and the next
  round runs on the patched final:
  - every window patched by a `done` filter patch is passed to the check as
    `skip_windows`, and a finding there is reported as `patched`, never offered again;
  - patches per run are bounded (`brandcap:`).
- **Patch count is separate from `MAX_QA_ROUNDS`.** The round number counts only reruns
  and restores.
- **Reruns on a patched final (§7 of the issue).** The `regen:` and `fix:` asks offered
  while the current result is an audio patch (any kind) say "drops the brand fix at
  X–Y s". Only that operator choice carries `discard_audio_patch=true`. The best-take
  restore is skipped while the result is a patch: it is never acknowledged automatically.
- **The flag `AUTOPILOT_BRAND_FIX_ENABLED`** is off by default (`.env`, needs `up -d`).
  - While it is off, no check runs.
  - Open `brand*` asks are retired.
  - The handler refuses new admissions (409).

## 2. The detector (`app.brand_check`)

Detection runs in five steps, all on 16 kHz mono audio of the final.

1. **Occurrences.** Every scripted brand occurrence gets its own id (`o<k>`, in order).
   - A multi-word brand ("App Store"), or any brand in `ja`/`zh`, is `not_checked`. It is
     never counted as passed.
   - So is a scripted inflection ("Listenify's"): that ending is the script's own.
2. **Word ASR.** `assistant_asr.LocalASR` runs faster-whisper small, pinned, offline, in a
   seccomp'd child with no network.
   - It works over 30 s chunks with a 5 s overlap. Chunk ends are floored to the audio
     stream's duration.
   - Overlapping words merge to the copy farther from its chunk edge, so a word cut by a
     boundary comes from the other chunk.
3. **Alignment by neighbours.** Whisper's spelling of a brand is unreliable: "Speech
   Fice", "pitchfile", "Speed Fight", "Speechfy". So the whole foreground script is
   aligned to the whole ASR word sequence (difflib over tokens, apostrophes split).
   - The brand is the ASR words between the nearest aligned script token before it and
     the one after it, across line boundaries.
   - Spelling decides only when one side has no anchor (similarity ≥ 0.6).
   - An occurrence that cannot be located is `unclear`, with that reason.
4. **The rule** (10 ms hop, 20 ms Hann frames, levels relative to the whole final's 95th
   percentile). A fix is found only when all of these hold:
   - the voiced ending of the brand is found;
   - a fricative run (3-frame smoothed share of >3.5 kHz energy ≥ 0.35, level ≥ −42 dB)
     starts within 150 ms with no voicing in between;
   - it lasts 80–400 ms, with a fill ≥ 0.6 and a mean high-band share ≥ 0.5;
   - it is followed by a measured pause of ≥ 100 ms.

   The window is `[voiced end, fricative end + 10 ms]`.
5. **No fix, with a reason**:
   - unknown alignment or ending;
   - no measured pause;
   - noise that does not look like one sibilant;
   - another speaker's script line overlapping the brand line by more than 0.2 s;
   - a next word that starts with a sibilant and a pause under 150 ms (the next word's
     "s" is then indistinguishable).

### 2.1 Calibration (2026-11-05)

The labelled finals were run with the production chunking. The fixtures are excerpts
plus words in `backend/tests/fixtures/brand_check/`.

| Final | Language | Brand verdict | Measured |
|---|---|---|---|
| dea38297 original ("Speechifies") | fr | **found** [35.34, 35.54] | fricative 110 ms, high-band share 0.66, −27.8 dB, pause 230 ms |
| dea38297 after the manual fix | fr | clean | — |
| 4c5bfc21 (project 2476176f) | es | clean | — |
| 14e4b10f (project 2476176f) | pt | unclear (no fix) | 80 ms, high-band share 0.49, pause 70 ms |
| 46656649, 8b7eb981 | fr | clean | — |
| e0b5a939, 56f91aa5 | pt | clean / unclear | — |
| 7c471ab3, b1458698, 7c90f778 (4 occurrences), 7aaca878 | es | clean / unclear | — |
| ed75f231 (brand + "cz-" word) | pl | clean | — |

The brand near a chunk boundary was checked with the real ASR in the container on the
dea38297 original, with the first chunk ending at 35.2 s (inside the word) and at 34.9 s
(just before it). Both gave **found [35.34, 35.54]**, aligned by anchors.

The hand-measured window on 2026-11-04 was 35.34–35.54. The pt final of the same project
is the closest negative, and its pause and high-band share are below the thresholds. Only
one real positive exists so far, so the thresholds lean to precision. A false positive
costs one ask that the operator answers "keep".

## 3. The fix render

All files are written inside the compute attempt directory from the exact base file.

- **Full candidate.** Inside the window, the audio is replaced by a 2× low-pass 2.2 kHz
  copy.
  - The 25 ms complementary ramps are sample-accurate (`aeval`, outside the window).
  - Mixing uses `amix normalize=0`, the video is `-c:v copy`, and the audio is AAC 192k.
  - This is exactly what the operator chose by ear on dea38297-fr ("variant A").
- **Self-checks.** A candidate is offered only if all three hold:
  - video packets are equal (`ffprobe -count_packets`);
  - duration is within 50 ms;
  - outside the window ±(ramp + 100 ms), the level is within **0.5 dB** of the base (the
    audio-patch tolerance of `tests/test_media.py`).

  The sample residual is recorded too. It measures the AAC re-encode (about −34 to −45 dB
  on speech), not the fix.
- **Two context previews**, original and fixed, covering the window ±2 s at 720p.
- **Result record.** The result keeps the base path, base SHA-256 and signature, the run
  and transcript versions, and the detector/render versions. For each candidate it also
  keeps the window, occurrence id, measurements, render parameters, file hashes and
  checks.
- **Stability.** The base is hashed before and after. A final that changed during the
  check is unusable, and the check is retried.

## 4. Approval → patch → promotion

- **The handler** `localisation_apply_brand_audio_patch` (free, autopilot ticker):
  - takes the run-row lock (`api_v2._lock_run_for_result_mutation`);
  - refuses with 409 when another audio patch or a range attempt is active;
  - then calls `pipeline_audio_patch.create_lowpass_patch`, which re-checks under that
    lock:
    - the run is `done`;
    - the result path is still the base;
    - the base bytes hash to the approved SHA;
    - the candidate hashes to the approved SHA.

  Anything stale is a 409 ("stale brand fix"). It commits one `RunAudioPatch` with
  `kind='same_run_lowpass'`, the window, `filter_params`, `base_sha256`,
  `candidate_sha256` and `candidate_path`, then enqueues `run_process_audio_patch`.
  - A failed enqueue is logged. The queued row is reaped to `failed` by the recovery
    reaper, and the step offers a retry, which is a new row. It is never a duplicate of
    the same admission.
- **`process_audio_patch`** for this kind never renders. It copies the approved candidate
  into `storage.audio_patch_dir` (staging name, atomic rename), verifies its SHA, then
  calls `_promote`. Under the run-row lock and **before** the Drive update, `_promote`
  re-checks:
  - the status;
  - the base path;
  - the base SHA;
  - the delivered file's SHA;
  - that no range attempt is active (another audio patch is excluded by the partial
    unique index).

  It then replaces the Drive content under the same id (`files().update`), re-points
  `result_local_path` and commits `done`. There is no size enforcement on this path:
  re-encoding would change the approved bytes.
- **The flow waits for the patch's real terminal status.** A `failed` patch becomes the
  `brandretry:` handoff, never "fixed".

## 5. The brand-asr container

`re-skin-brand-asr-worker` (`python -m worker.brand_asr_worker`):

- **Image.** `backend/Dockerfile` target `brand-asr`, which is the app stage plus `/opt/asr`
  (`backend/requirements-asr-cp312.lock`, every wheel hashed, `--only-binary`) and the
  model manifest. Every other service builds the default last stage `runtime`, which is
  the app stage unchanged.
- **Queue.** It is the only consumer of the `brand_asr` queue: one RQ worker, so one ASR
  job at a time. There is no scheduler, InsightFace or run recovery.
- **Limits.** `mem_limit 1536m` / `cpus 1.0` / `pids_limit 256`, `cap_drop ALL`,
  `no-new-privileges`, `restart: unless-stopped`.
  - The ASR child's own sandbox requires a cgroup bound of ≤ 2 GiB and ≤ 2 CPUs.
- **Mounts.** `./data` (reads the final, writes the attempt directory) and the asr-small
  model, read-only, with `create_host_path: false`
  (`${BRAND_ASR_MODEL_HOST_DIR:-/var/lib/reskin-assistant-runtime/asr-small}`). No secrets
  mount. The model loads offline only, and its manifest (revision + SHA-256 of every
  file) is checked on every call.
- **Unchanged.** The run workers and the assistant ASR sidecar are not changed.
- **Routing.** Compute effects route by their registered queue
  (`registry.register_compute(..., queue=...)`, `registry.compute_queue`). The core's
  liveness check and re-enqueue use it.
- **Without the container,** the check stays queued, is re-enqueued, then fails as lost.
  After 3 attempts it becomes the `brandcheck:` handoff.
- **Queue delay** (`requested_at` → start) and wall time are recorded in the result.

### 5.1 Measured

On 2026-11-05 the check ran in the built image with the compose limits (1.5 GiB, 1 CPU,
no network) on the dea38297-fr original final (37.8 s, 2 chunks), with the run workers
idle:

| Measurement | Value |
|---|---|
| ASR | 79 s |
| Whole check incl. render and previews | 100 s |
| Peak RSS, whisper child | 823 MB |
| Peak RSS, parent | 167 MB |
| Result | found [35.34, 35.54]; video packets 1133 = 1133; level outside the window +0.02 dB |

**Still to record before enabling** (acceptance): the same numbers on a real final while
the run workers are busy.

## 6. Deploy

1. Back up: `scripts/backup_db.sh predeploy`.
2. Apply migration `v3a9c5e7b1d4` (on `u7c2e4a6b8d1`) **before** the image. It is
   additive: `run_audio_patches.kind` (default `donor`), `filter_params`, `base_sha256`,
   `candidate_sha256`, `candidate_path`, nullable donor fields, plus the check constraint
   `ck_run_audio_patches_kind_fields`. It was rehearsed up/down/up on a copy of the
   production schema.
3. Run the normal autopilot hold/drain and the standard deploy command. The new service
   is in the default profile.
4. Measure §5.1 under load. Then set `AUTOPILOT_BRAND_FIX_ENABLED=true` in `.env` and run
   `up -d`.

## 7. Rollback

1. Set `AUTOPILOT_BRAND_FIX_ENABLED=false` (`up -d`). New checks stop, open asks are
   retired, and new admissions are refused.
2. Hold and drain (`deploy/drain-check.sql` already counts active `run_audio_patches`).
3. Resolve every active filter patch and its effects before reverting the image.
4. Every read path of the old image must tolerate filter rows:
   - the old API's patch list would fail on `donor_run_id` NULL;
   - only the run page's patch list and the patch API read those columns.

   So before rolling the image back, mark filter rows as hidden or keep the new image.
5. Keep the schema and the patch files.
