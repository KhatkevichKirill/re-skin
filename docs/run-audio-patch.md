# Run audio patch — "Fix audio from another run"

**Status:** shipped 2026-09-07 · **Code:** `app/pipeline_audio_patch.py`,
`app/media.py::splice_audio_window`, `app/models.py::RunAudioPatch`,
migration `e4b7c1a9d260`

## 1. The problem

In a `localisation` project Seedance writes the speech itself (`mute_source`
on, `audio_mode="seedance"`), from a script embedded in the prompt. It
occasionally mispronounces a brand name — "Listenify" said as a Spanish word —
in a take that is otherwise perfect, while a sibling run of the same project,
speaking the same script over the same driving footage, says it correctly.

Regenerating is the expensive answer: at Seedance 2.5's 38 credits/s a 30 s
clip is a real bill, and the mispronunciation is random, so the retry may come
back wrong again. The cheap answer is to borrow the word from the sibling.

### 1.1 Why a window, and not the whole track

Measured on project `77e0e370`, whose eight runs share one 29.67 s swap
segment and one library outro:

| | run `3eb88752` | run `2714a7c3` |
|---|---|---|
| Seedance clip | 30.042 s | 29.375 s |
| final | 37.098 s | 36.385 s |
| end of sentence 1 | 3.4214 s | 3.4182 s (**Δ 3 ms**) |
| start of sentence 2 | 3.7960 s | 3.7598 s (**Δ 36 ms**) |
| last pause before the outro | 27.894 s | 26.915 s (**Δ ~1 s**) |

Sentence boundaries are anchored by the driving footage, so two runs start
almost perfectly aligned — and then drift, because the model quantises its
output length differently per generation. Swapping the whole track would put
the outro ~0.67 s out of sync, leave the tail of the video silent, and desync
lip-sync by up to a second by the end.

A **window** avoids all of it: the word being fixed is early enough (≈3.8–5.5 s
here) that the two runs are still within ~50 ms of each other, and every sample
outside the window is left exactly where it was.

## 2. Why this is not a derived creative

Tail edits, caption burns and overlays are all *new creatives* built from a
run: each gets its own Drive file, its own Job ID and its own Final Creatives
row. An audio patch is not one of those. A take with the brand mispronounced is
a **defect**, not a deliverable worth tracking beside the good one — nobody
wants both rows in the sheet.

So the patch repairs the run's own result in place, and that decision is what
makes the feature small:

* no new `Pipeline Type` token (nothing to teach the downstream renamer in
  `ai-fb-audit`),
* no new Job ID rule, no sibling-collision problem,
* no sheet export task, no `Sent to Final` value,
* no new chaining rules against the three creatives that already chain,
* the existing "Add to Final Creatives" button, the run's Drive link, its
  `Idea`, its language suffix and every automation copy keep working untouched.

The shape it *does* borrow is `RunSegmentRangeAttempt`'s — this codebase's
other "change a done run in place, with history": append-only rows, candidate
bytes written somewhere the active result is not, and a promotion that only
happens once everything else has succeeded.

## 3. The splice

`media.splice_audio_window(base, donor, dst, base_start, base_end,
donor_start, crossfade_sec)`.

```
base gain  1 ────╲__________╱──── 1
donor gain 0 ____╱‾‾‾‾‾‾‾‾‾‾╲____ 0
            t0  t0+xf   t1-xf  t1
```

Both tracks run the full base timeline under complementary gain envelopes and
are summed (`amix … normalize=0`). The envelopes sum to exactly 1 at every
instant, so the seam cannot dip or clip. The picture is `-c:v copy` — not one
frame re-encoded.

Three things are deliberate:

* **No `acrossfade`/`concat`.** `acrossfade` consumes its fade length from both
  sides, which would pull every sample after the window earlier by twice the
  fade — reintroducing the very desync the window exists to avoid.
* **`asetnsamples=n=128` before `volume`.** `eval=frame` re-evaluates the gain
  once per audio frame; a decoder's native ~1024-sample frame would render a
  60 ms ramp as three audible steps.
* **Neither `-t` nor `-shortest`.** The copied video stream must reach its own
  last frame; the audio branch is already bounded by `amix duration=first`.

The donor always supplies exactly `base_end - base_start` seconds. There is no
donor end to set: any other length would have to time-stretch the borrowed
audio or move what follows it.

## 4. Lifecycle

`queued → processing → delivering → done` (`… → failed`). `done` and `failed`
are **terminal** — a further correction is a NEW row, which patches whatever
the current result is by then, so successive patches compose.

`_promote()` is the whole safety argument:

1. `session.refresh(run, with_for_update=True)` — locks the Run row.
2. Refuse if `run.status != done`, or if `run.result_local_path !=
   patch.base_result_path` (something else promoted while this queued).
3. **Replace the Drive file's content, inside the lock** —
   `files().update(fileId=…)`, so the id and therefore every existing link
   survives. A run with no Drive result yet gets a normal upload instead.
4. Re-point `run.result_local_path` at the candidate, stamp `promoted_at`,
   mark the patch `done` — one commit.

The Drive call is inside the lock on purpose: doing it first would mean a patch
that then loses the staleness race has already replaced a creative it is about
to disown. Holding a row lock across a ~20 MB upload is affordable because the
partial unique index makes contention on that row near-impossible — and because
prod's `idle_in_transaction_session_timeout`, `statement_timeout` and
`lock_timeout` are all 0 (verified 2026-09-07). Setting any of them means
revisiting this ordering.

**Accepted residual risk.** A worker killed between Drive acknowledging the
update and the commit landing leaves Drive holding the patched video while the
Run still points at the old local file. The reaper marks the patch `failed`;
re-running the same window fixes it, because replacing the same content twice
is idempotent. The alternative — promote first, upload after — trades this for
the strictly worse failure of the sheet's link serving the broken audio while
the app claims it is fixed.

Nothing is ever overwritten: the candidate lives in
`storage.audio_patch_dir`, so the pre-patch delivery stays on disk (it is what
`base_result_path` names) and every earlier version is still downloadable from
`GET /api/v2/audio-patches/{id}/result`.

## 5. Concurrency

* `uq_run_audio_patches_active_per_run` — a partial unique index over
  `('queued','processing','delivering')`. The application check
  (`api_v2._active_audio_patch`) is a non-locking SELECT that two concurrent
  creates can both pass; this index is what actually enforces one at a time,
  and `create_run_audio_patch` turns the loser's `IntegrityError` into a 409.
  The WHERE clause is spelled out by hand in three places — the model, the
  migration and `state_machine.AUDIO_PATCH_ACTIVE_STATUSES` — and they must be
  kept in step.
* `version_id` (SQLAlchemy `version_id_col`) — stops two worker sessions
  processing the SAME patch from both promoting.
* **Mutual exclusion with range regeneration**, both directions: both features
  finish by re-pointing `Run.result_local_path`, so each 409s while the other
  is in flight. Both create paths take a **Run-row lock first**
  (`api_v2._lock_run_for_result_mutation`), because the two partial unique
  indexes are per table and cannot see each other: without the lock one of
  each could be created concurrently, and since
  `pipeline_range_attempt._promote` validates only `segment_snapshot`, the
  range attempt would then overwrite the patched result *and* re-point the run
  at a new Drive file — leaving an already-exported sheet row on the old id.
  (Found by the §1 review gate after this shipped.) Since AICP-349 a range
  attempt replaces the content behind the same Drive id instead, so it also
  needs `discard_audio_patch=true` on a patched result, like a rerun.
* **The candidate is rendered to a unique dot-prefixed staging file and
  published with `os.replace`.** The candidate path is derived from the patch id alone, so two
  workers on the same patch would otherwise both aim ffmpeg's `-y` at one file
  — one truncating it while the other uploads, putting half a video behind a
  delivered creative's Drive id. `version_id` only catches that race at commit
  time, which is after the upload. The staging name keeps its `.mp4` suffix —
  ffmpeg picks the container from the extension, and a `.part` name dies with
  "Unable to find a suitable output format". A SIGKILL (which is what a
  redeploy does to an in-flight job here) skips the cleanup, so each render
  first sweeps stale staging files from the run's whole `audio_patches/` tree —
  **not** just its own directory, because `failed` is terminal, so the
  operator's retry is a new patch with a new directory and the killed one's
  debris would otherwise sit somewhere nothing ever opens again. The age
  threshold is floored past the job timeout, which is what keeps the sweep safe
  when two workers are rendering under that tree at once.

Both create paths take the Run lock **before** they read status or snapshot,
not after: validating first and locking second leaves exactly the window the
lock exists to close. Nothing that shells out runs inside the lock either —
`create_run_audio_patch` warms both runs' cached `result_duration_sec` first,
because `media._run` passes no timeout to `subprocess.run` and a hung ffprobe
inside the locked window would hold the row lock indefinitely (prod runs with
`idle_in_transaction_session_timeout=0`).
* The recovery reaper carries a `_DerivedCreativeSpec` for it. Unreaped rows
  matter more here than for a tail edit: an active row holds the unique index
  and would block the operator from ever retrying the correction.

## 6. Operator flow

Run page → **Fix audio from another run** (done runs only). The block is
collapsed by default — two players are the tallest thing on the page and most
visits are not about fixing audio — and the donor video is not fetched until it
is opened:

1. Scrub this run's player to the bad word; **Set** the window start and end.
   **▶ Window** plays exactly that range back.
2. Pick a donor from the project's other done runs — a sibling, because a donor
   is only useful if it speaks the same script over the same footage — scrub to
   the same word, **Set** its start. **▶ Window** plays the same length from
   there.
3. **Fix audio.** On success the run's own video, its Drive file's content and
   the "Audio-fix history" card below all update; the Job ID and the Drive URL
   do not change.

## 7. What this does not do

* It does not touch derived creatives already built from the OLD result. A tail
  edit, caption burn or overlay rendered before the patch keeps its own file
  and its own sheet row, still carrying the mispronunciation — rebuild it from
  the patched run if that matters. A derived creative that is *in flight* when
  a patch promotes is not blocked either: it resolves `result_local_path` when
  its worker gets to it, so it reads whichever version is current then. Both
  are valid videos and nothing breaks; if it matters which, wait for the patch
  before starting the tail edit.
* It does not reset human QC verdicts or a `creative_asset_name` the downstream
  renamer already assigned to an exported row. Fixing the file does not fix
  those columns; they are the operator's call.
* It does not time-stretch, and it offers no whole-track mode — see §1.1.
* It does not accept a donor from another project. The picker only offers
  siblings and the API enforces it: a donor is only meaningful when it speaks
  the same script over the same footage.

## Same-run low-pass kind (AICP-389)

Since migration `v3a9c5e7b1d4` a patch has a `kind`. Every row before it, and every
patch made from the run page, is `donor`, the feature described above. The localisation
autopilot's brand fix adds `same_run_lowpass`: the window of the run's own audio is
low-passed to remove a trailing "-s" on a brand word. Such a row:

- has no donor (`donor_run_id`, `donor_start_sec` NULL, enforced by
  `ck_run_audio_patches_kind_fields`);
- carries `filter_params`, `base_sha256`, `candidate_sha256` and `candidate_path`;
- never renders in the worker: the operator-approved candidate is copied, verified by
  SHA-256 and promoted as is;
- is promoted by `_promote` only after it re-checks, under the run-row lock and before
  the Drive update, the base and candidate hashes and that no range attempt is active.

Everything else is shared: the same partial unique index, the same reaper, the same
in-place Drive update and the same `discard_audio_patch` acknowledgement on reruns. See
`docs/brand-trailing-sibilant.md`.
