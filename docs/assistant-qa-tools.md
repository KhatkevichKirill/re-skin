# Assistant QA tools (AICP-345)

Tracking: AICP-345. The free local QA
tools of AICP-343 (`backend/app/qa_tools.py`)
are callable by the run-scoped assistant's agent loop (workflow revision 2) through its closed
tool boundary. They are typed actions like `inspect_frames`, return bounded JSON evidence, and
never accept or return a filesystem path.

## Actions

| Action | Inputs (IDs only) | Evidence (`kind: qa`) |
|---|---|---|
| `detect_captions` | `target` (source pair or candidate), `calibration` (must be the imported **Original project source video**), optional windows | verdict, hits/frames, up to 24 spans, calibration result, detector config digest, digest of every per-frame score |
| `check_alignment` | `source` (source pair), `result`, `mode` `timeline` or `window` (≤ 30 s, result ≤ 20 s) | timeline: points, window medians, ≤ 24 outliers, `in_sync`; window: ≤ 40 drift samples; digest of the full detail |
| `compare_frames` | `source`, `result`, ≤ 8 windows, fractions or `every_sec` (≤ 48 pairs) | per-pair mean grayscale difference (no image; use `inspect_frames` to look) |
| `script_captions` | optional `library_donors` (stitch-plan index → RunSegment id) | lines, pieces, lineage, eligible caption jobs |

At most four QA calls per turn, 6 KiB per observation (12 KiB for `script_captions`) and 24 KiB of
QA evidence per turn, inside the existing 64 KiB aggregate. The detector config is the project's
`video_projects.caption_detector`, never model-supplied. `clean_uncalibrated` proves nothing.

## Confinement

Every decode runs through the mandatory native launcher. `assistant_media._stream` is the
binary-stdout sibling of `_run`: the same `prepare_media_command` grants (exact inputs, no
outputs), fixed-size records read from a pipe with budget and cancellation checks, a record cap,
and the process group killed and reaped on any early exit. One continuous ffmpeg `fps=` filter
means the confined tools sample exactly the frames the CLI samples. Single-frame grabs decode to
raw BGR, so no compressed image is decoded in the supervisor. Inputs must pass the strict editing
profile (`probe_trusted_media`) and the 120 s / 1920 px limits. `qa_tools.use_media()` selects the
backend; the CLI keeps its direct ffmpeg backend unchanged.

Parity on production files (run 657e1f7f, a supervisor-equivalent container): 788/788 detector
scores identical to the CLI, final 0 hits, source calibration 665/788, the same 156 alignment
points, 26/26 comparison differences identical. One confined calibrated scan of the 79 s final
takes ~180 s of the 600 s per-call limit.

## Captions from the script

`app/assistant_script_captions.py` resolves the pinned context's run (never a model-named one):

* the run must be `done`, stitched, `audio_mode=seedance`, with a frozen `segment_snapshot`;
* every stitch piece must be a completed, generated swap clip (keep and source-fallback pieces
  are refused for now);
* a library entry needs a donor RunSegment of the same project whose window is the replaced swap
  segment of this run; while the library row exists its bytes must equal the donor's;
* the summed container length of the pieces must match the result (± 0.1 s);
* the assistant's selected version must be byte-identical to the run's current result.

Lines come only from stored prompts (`RunSegment.prompt_override` or `Run.prompt`). The resolver
cannot prove that a prompt was not edited after its clip was generated, nor that the words were
spoken: the lines are script-derived and need review.

Acceptance (read-only on production data): run 6c4a9818 (pl) with the probe run's segment 0 as the
library donor reproduces all 22 lines of the burned caption f7124be8 exactly. The library clip was
deleted, so the lineage is `library_donor_attestation_required`.

That run's final was assembled by hand and is not strict CFR (5 irregular frame gaps), so the
assistant cannot import it as a version today. A live turn on it is refused before any tool runs.

## Applying the lines (operator-confirmed write)

The agent may finish with reply kind `caption_lines` (agent-only; the legacy planner schema never
offers it), naming one eligible caption and the exact `lines_sha256` of a `script_captions`
observation of the same turn. The UI shows the lines and the consequences. **Apply captions**
(`POST /api/v2/assistant/runs/{run}/turns/{turn}/apply-captions`) does exactly what the operator's
manual `PATCH /captions/{id}/lines` + Retry does, in one transaction:

* the lines are re-read from the turn's durable evidence, never from the request;
* the run row and then the caption row are locked; the caption must still be `done`/`failed`,
  `source_type=run`, same language and lines as at proposal time; the run result must still hash to
  the evidence's `final_sha256`;
* a library-donor lineage needs `attest_library_donor: true` (the UI's Apply sends it after showing
  the donor);
* lines, `lines_source=manual`, `queued`, the same released-unused spend reservation as Retry, and
  an audit message keyed `caption-apply:<turn>` commit together; the burn is enqueued afterwards.
  A repeat returns the recorded application without queueing again. If the enqueue fails the
  caption stays `queued` and the recovery reaper fails it after 300 s (Retry-able).

Manual lines skip transcription, so no paid call is made. Re-burning uploads a **new** Drive file.
Tail edits made from the caption keep their old files. Like operator edits, the lines may be reused
by later captions of the same audio.

Known limits shared with the manual path (follow-up work): the caption worker has no attempt
identity (a duplicate or delayed RQ job can burn again), it resolves `run.result_local_path` at burn
time, and a Drive upload that succeeds remotely but fails locally can be repeated. Proving
composition generally needs a stitch manifest written at stitch time.

## Deploy

Contracts and planner are baked into the broker runner image (`deploy/assistant/Dockerfile.runner`):
rebuild and qualify it together with the app image, drain live assistant turns, force-recreate
`assistant-supervisor` and check its image. No migration.
