# Burned captions — design spec / build contract

Status: in progress (branch `feat/burned-captions`).

This file is the **interface contract** every part of the feature is built
against. If an implementation disagrees with this document, the document wins —
raise the conflict rather than silently diverging. (Same convention as
`docs/localisation.md`.)

## 1. What the feature does

An operator opens a **done** Run and presses **Add Captions**. We re-transcribe
the *delivered* video, render the spoken lines as a subtitle file, burn them in
with ffmpeg, and deliver the captioned copy to Google Drive as a new derived
creative — alongside the existing "Add to Final Creatives" button, which is
**unchanged**. (At the time this was written there was also an "Add to Caption
Demo" button; that export was removed on 2026-08-28 — see
`wiki/components/sheet-exports.md` § Removed.)

### 1.1 Why re-transcribe instead of reusing the translation

For a `localisation` run we already have the translated hook text, but it is not
burnable as-is:

- The translated lines carry **source** timings. The delivered hook is Seedance
  speech at its own pacing, so those timestamps do not line up with the audio.
- The delivered creative is hook + a **pre-uploaded library tail** whose speech
  was never transcribed. Captioning from the translation would leave the
  majority of the runtime uncaptioned.
- `POST /projects/{pid}/localisation-prompt` **writes nothing to the DB**
  (`api_v2.py`), so the translated lines do not survive run creation anyway.

Transcribing the delivered file solves hook and tail in one call, with timings
that match the actual audio — and works for **every project type**, not just
localisation, which is what made this a candidate to replace the external
Caption Demo pipeline rather than just supplement it. (It has since done so:
the Caption Demo export was removed on 2026-08-28.)

Cost: one `gemini-2.5-pro` call per captioned run (the same route
`localisation.transcribe_video` already uses) plus one re-encode. When
2.5-pro is unavailable it falls back to `gemini-3.1-pro`, then
`gemini-3-5-flash-openai` (see
`docs/localisation.md` §8).

### 1.2 Style

**One universal style for every caption, every language** (operator decision,
2026-08-09). No per-run styling, no attempt to match any external captioning
tool's look. The only thing that varies by language is the font, because Latin
fonts carry no CJK glyphs.

#### 1.2.1 `captions/v2` — the retune (2026-08-14)

v1 shipped captions that ran off the left and right edges of every delivered
video. The cause was two decisions that were each defensible alone and broken
together: rows were wrapped by **counting characters** (28 Latin / 14 CJK) and
`WrapStyle: 2` told libass never to re-wrap. On the 1080x1920 frame this
pipeline delivers, 28 characters of the v1 style measured ~1300px inside a
908px safe area — so a full row overhung the *frame* by ~100px per side, with
nothing left to catch it. Measured on delivered files, confirmed on extracted
frames.

v2 (`CAPTION_STYLE_VERSION = "captions/v2"`) changes five things:

| | v1 | v2 | Why |
|---|---|---|---|
| Wrap | 28/14 characters | measured pixels (`measure_text`) | A character is not a width |
| `WrapStyle` | 2 (never wrap) | 0 (smart wrap) | libass re-wraps what our measurement got wrong instead of letting it overhang |
| Rows over the cap | dropped, with a warning | shown as the next caption in the same time window | Losing the end of a sentence is worse than showing it a beat later |
| `MAX_ROWS` | 3 | 2, balanced | Three rows eat the lower frame; balancing kills the orphan-word row |
| Background | outline only | semi-transparent box (`BorderStyle 3`, `BOX_ALPHA`) | Contrast over bright, busy footage |

plus the ratios that follow from them: `FONT_SIZE_RATIO` 0.045 → 0.036 (a
legible size that lets two rows hold a whole sentence), `MARGIN_H_RATIO` 0.08 →
0.06 (the margin is now enforced, so it can be tighter), `MARGIN_V_RATIO` 0.12 →
0.16 (clears the platform caption/CTA bar), `SHADOW_RATIO` → 0.

`SHADOW_RATIO = 0` is not cosmetic and must not be "restored": under
`BorderStyle 3` libass paints the shadow as a second, **opaque** `BackColour`
box behind the translucent one, so any non-zero shadow fills the transparency
back in and the band renders as a solid black bar.

Still one universal style — v2 retunes the single look, it does not add a
styling API (§4).

## 2. Data model

### 2.1 `run_captions` (new table)

Mirrors `run_tail_edits` — the existing precedent for "a derived creative built
from a done Run's result, delivered to Drive, exportable to the sheet".

| Column | Type | Notes |
|---|---|---|
| `id` | String(36) PK | uuid, `default=_new_uuid` |
| `run_id` | String(36) FK → `runs.id` ON DELETE CASCADE | `nullable=False, index=True` |
| `created_at` | DateTime | `default=_utcnow, server_default=func.now(), index=True` |
| `updated_at` | DateTime | `default=_utcnow, onupdate=_utcnow, server_default=func.now()` |
| `language` | String(8) NULL | Caption language. NULL → resolved from the run at process time |
| `lines` | JSON NULL | The timed lines actually burned (see §3.1). Written by the transcribe step; also the operator-edit target |
| `lines_source` | String(16) NULL | `"transcribe"` or `"manual"`. `"manual"` means an operator supplied/edited `lines` and the transcribe step MUST be skipped |
| `style_version` | String(32) NULL | `captions.CAPTION_STYLE_VERSION` at burn time |
| `status` | Enum `CaptionStatus`, pg name `caption_status_enum` | `nullable=False, default=CaptionStatus.queued` |
| `output_path` | Text NULL | Local path of the burned file |
| `gdrive_file_id` | String(255) NULL | |
| `error_message` | Text NULL | Truncated to 2000 chars |
| `final_creatives_exported_at` | DateTime NULL | Mirrors `RunTailEdit` |

Relationship: `run: Mapped["Run"] = relationship("Run", viewonly=True)`.

**No `caption_demo_status` column.** `RunTailEdit` and `OverlayJob` carry one
(now a legacy column — see `wiki/components/sheet-exports.md` § Removed);
`RunCaption` never had it, since a caption job's whole output IS an
already-captioned video.

One Alembic migration creates the table and the enum type. Follow the existing
migration conventions in `backend/alembic/versions/` (down_revision = current
head; a real `downgrade()` that drops the table then the enum).

### 2.2 `CaptionStatus` (`app/state_machine.py`)

```python
class CaptionStatus(str, enum.Enum):
    queued = "queued"
    transcribing = "transcribing"
    burning = "burning"
    delivering = "delivering"
    done = "done"
    failed = "failed"

CAPTION_TRANSITIONS = {
    queued:       {transcribing, burning, failed},   # burning: manual lines skip transcribe
    transcribing: {burning, failed},
    burning:      {delivering, failed},
    delivering:   {done, failed},
    done:         {queued},                          # allow re-run
    failed:       {queued},                          # allow retry
}
```

`can_transition` / `transition` must accept `CaptionStatus` exactly like the
other five status enums (extend the `Union[...]` annotations and the
`isinstance` chain).

### 2.3 Storage (`app/storage.py`)

```python
def caption_dir(caption_id: str, project_id: str) -> str
```

→ `<DATA_DIR>/projects/<project_id>/captions/<caption_id>/`, containing:

- `captions.ass` — the rendered subtitle file (kept: it is the evidence for a
  "why does this caption say that" bug report)
- `{run_id}_captioned.mp4` — the delivered output

Document the layout in the module's existing comment-block style.

## 3. Module contracts

### 3.1 `app/captions.py` (new, pure — no DB, no ffmpeg, no network)

```python
CAPTION_STYLE_VERSION = "captions/v2"

def font_for_language(language: str | None, *, sample_text: str = "") -> str
def font_size_px(height: int) -> int
def usable_width_px(width: int, height: int) -> float
def measure_text(text: str, *, font: str, size_px: int) -> float
def prepare_lines(lines: Iterable[dict], *, duration_sec: float | None = None) -> list[dict]
def wrap_rows(text, *, cjk=None, width=NOMINAL_WIDTH, height=NOMINAL_HEIGHT,
              font: str | None = None) -> list[str]
def wrap_text(text, *, cjk=None, width=..., height=..., font=None) -> str
def split_line(rows: list[str], *, max_rows: int = MAX_ROWS) -> list[list[str]]
def layout_text(text, *, cjk=None, width=..., height=..., font=None,
                max_rows: int = MAX_ROWS) -> list[list[str]]
def escape_text(text: str) -> str
def format_timestamp(seconds: float) -> str
def build_ass(lines, *, width: int, height: int,
              language: str | None = None,
              duration_sec: float | None = None) -> str
```

`measure_text` reads the real font file through Pillow (pinned in
`requirements.txt` for this reason; it was already present as an insightface
transitive) and falls back to a per-script em estimate when Pillow or the font
file is missing, so the module stays importable and correct on a bare box.
Wrapping is width-driven throughout: `wrap_rows` fits rows to
`usable_width_px` (frame less both margins less the box padding) and then
*balances* them by bisecting for the narrowest limit that needs the same row
count; `split_line` groups the rows into `MAX_ROWS`-tall captions; `layout_text`
is the two composed, re-balancing each caption's own rows after the split.

`wrap_text` no longer caps or drops rows — that was v1's failure mode. A line
too long for one caption becomes several, dividing the line's own `[start,
end]` window in proportion to each caption's character count. A caption that
cannot be given `MIN_LINE_DURATION_SEC` inside that window is dropped from the
tail (logged). That, and an isolated line too close to the clip end (see
`prepare_lines` below), are the only paths by which v2 discards transcript text.

Line shape in and out: `{"start": float, "end": float, "text": str}` — the same
`start/end/text` keys the §4.1 localisation transcript uses, so a transcript's
`lines` can be handed over untouched (extra keys ignored).

`prepare_lines` normalises untrustworthy model output: drop blank text and
unparseable timestamps, clamp into `[0, duration_sec]`, sort by start,
de-overlap by **truncating the earlier line** (never shifting the later one —
a caption's start is when the words are said), enforce a minimum on-screen
time, drop what cannot be repaired. It must never raise on bad input.

**Quick replies are merged, not dropped (AICP-247).** When truncating the
earlier line would leave it under `MIN_LINE_DURATION_SEC`, the two lines become
one caption: the earlier line's start, the later of the two ends, both texts.
A chain of quick lines folds into one caption until a line starts late enough
for the truncation to keep the floor. A merged-in line's words therefore
appear less than `MIN_LINE_DURATION_SEC` before they are spoken; nothing else
moves. A line too close to the clip end for the floor joins the caption before
it when that caption is still up or ended less than one floor earlier;
otherwise (an isolated flash at the cut) it is still dropped. When the merged
lines have different known `speaker`s and `SPEAKER_CHANGE_STARTS_NEW_ROW` is on
(the default, pending Operator's approval), each speaker starts a new row of the
caption, but only if all the parts fit within the row cap; otherwise the text is
space-joined and wrapped as usual. `build_ass` makes the decision layout-aware:
a caption needs `MIN_LINE_DURATION_SEC` per rendered caption its text wraps into,
so a truncation or merge never leaves a long caption too little time and has
`_split_windows` drop the tail where a merged-in reply sits. A merged caption is
held up to that time, but never past the clip end. Residual: text that needs more
floor-time than remains before the clip end still loses its last split
captions (logged by `_split_windows`); shrinking captions under the floor was
not chosen. Split windows give every
caption the floor first and share only the remainder by length. Example (`c9ba8c0a-fr`): "Quoi ?" 1.1–1.5,
"Mets ça." 1.5–1.8, "Ok." 1.9 → "Quoi ? / Mets ça." 1.1–1.9, then "Ok." from
1.9.

`build_ass` returns a complete ASS file (`[Script Info]` / `[V4+ Styles]` /
`[Events]`) with `PlayResX/PlayResY` set to the passed frame size so all sizes
are ratios of the real video. Empty input is a valid file with no events, not
an error.

House style (all ratios of **height**, except the horizontal margin), as of
`captions/v2` — see §1.2.1 for what changed from v1 and why: bold white text
on a semi-transparent black box, no shadow, bottom-centred (`Alignment: 2`),
`BorderStyle: 3`. Font size ≈ 3.6% of height, box padding ≈ 0.45%, shadow = 0,
bottom margin ≈ 16% of height (clears platform UI), left/right margin ≈ 6% of
**width**. At most 2 balanced rows per caption.

ASS gotchas the implementation must get right (assert them in tests):
- Colours are `&HAABBGGRR` — alpha first, blue before red. `00` alpha = opaque.
- Bold is `-1`, not `1`.
- Timestamps are `H:MM:SS.cc` (centiseconds, one hour digit).
- `{` opens an override block and `\` starts an escape, so dialogue text
  containing either must be neutralised; real newlines become `\N`.
- Under `BorderStyle: 3` the box is filled with **`OutlineColour`** (not
  `BackColour`, which stays the shadow) and **`Outline` becomes padding**, not
  a border width. Both silently render wrong, never error.

Fonts: `DEFAULT_FONT = "DejaVu Sans"`, `CJK_FONT = "Noto Sans CJK JP"`.
`font_for_language` returns the CJK font for `ja`/`zh`/`ko` and, when no
language is known, sniffs *sample_text* for CJK codepoints. A font name that
resolves to nothing renders as empty boxes **without failing ffmpeg**, which is
why §5 (Dockerfile) is part of this feature and not an optimisation.

### 3.2 `app/media.py` — one new function

```python
def burn_subtitles(src: str, ass_path: str, dst: str, *,
                   duration_sec: float | None = None,
                   has_audio: bool | None = None) -> None
```

Single ffmpeg pass: `-vf subtitles=<escaped ass_path>` over *src* → *dst*.

- Video flags come from `delivery_video_opts(duration_sec, audio_bitrate_bps=...)`
  so the burn respects the 100 MB delivery cap like every other output.
- Audio is **stream-copied** (`-c:a copy`) when the source has audio, `-an`
  otherwise — this is a video-only filter, so re-encoding audio would lose
  quality for nothing. Charge `_STITCH_AUDIO_BPS` against the size budget.
- `-movflags +faststart`, `-threads _FFMPEG_THREADS`, `-y` — match the
  conventions of the neighbouring functions.
- The `subtitles=` filter argument needs its own escaping layer (`\`, `:`, `'`
  are all special inside a filter argument). Write a small helper and test it
  against a path containing each; do not rely on our paths being uuid-clean.
- Raises `MediaError` on failure via the module's `_run`.

### 3.3 `app/pipeline_captions.py` (new)

```python
def process_caption(caption_id: str, *, gdrive: GDriveClient | None = None) -> None
```

Mirror `pipeline_tail.process_tail_edit` closely — orphan-resume on a non-queued
start status, `_truncate_error`, `_deliver_with_retry` with
`CAPTION_DELIVER_ATTEMPTS` / `CAPTION_DELIVER_BACKOFF_SEC` env knobs, failure
sets `error_message` and re-raises.

Phases:

1. `queued → transcribing` (skip straight to `burning` when
   `lines_source == "manual"` and `lines` is non-empty). Validate the parent Run
   has `result_local_path` on disk first; probe it once for width/height/
   duration/has_audio.
   Transcribe with `localisation.transcribe_video(run.result_local_path)` —
   the *delivered* file, not the source. Store the returned `lines` on the row,
   set `lines_source="transcribe"`, and resolve the caption language as
   `caption.language or run.language or transcript["source_language"]`.
   A transcript with **no lines is a success, not a failure** (a wordless
   creative) — it produces a caption-free re-encode; log it and continue.
   A `LocalisationError` fails the job with its message.
2. `transcribing → burning`: `captions.build_ass(...)` → write `captions.ass` →
   `media.burn_subtitles(...)` → `media.enforce_size_limit(output)` (wrapped in
   try/except and logged, never fatal — same as `pipeline_tail`). Record
   `style_version` and `output_path`.
3. `burning → delivering`: upload to `run.gdrive_folder_id or
   settings.GDRIVE_DEFAULT_FOLDER_ID` as `{run_id}_captioned.mp4`, store
   `gdrive_file_id`.
4. `→ done`, clearing any stale `error_message`.

### 3.4 `app/tasks.py` — four additions

```python
def run_process_caption(caption_id: str) -> None            # RQ target, builds a real GDriveClient
def enqueue_process_caption(caption_id: str) -> None        # QUEUE_PROCESS
def run_export_caption_to_sheet(caption_id: str) -> None    # RQ target
def enqueue_export_caption_to_sheet(caption_id: str) -> None # QUEUE_OVERLAY, _EXPORT_JOB_TIMEOUT
```

The sheet export mirrors `run_export_tail_edit_to_sheet` exactly, including the
job-id dedupe scan, with these field values:

| Field | Value |
|---|---|
| `Date` | `date.today().isoformat()` |
| `Job ID` | `languages.job_id_for(caption.id, run.language)` — the caption's **own** 8-char prefix (the run's would collide with the run's own row), parent run's language |
| `Input Link` | parent run's Drive link, else `caption.run_id` |
| `Creative Link` | `https://drive.google.com/file/d/{caption.gdrive_file_id}/view` |
| `Pipeline Type` | `"re-skin-captions"` |
| `Run on Meta` | `"need_to_rename"` |
| `Idea` | `build_idea(project.name, run.name, run.model, project_id=project.id, run_id=caption.run_id)` — same as the tail-edit export: a captioned cut is the same creative idea |

### 3.5 `app/api_v2.py` + `app/schemas_v2.py`

Schemas: `CaptionCreateResponse {caption_id, status}`, `CaptionResponse`,
`CaptionListItem` (id, status, language, lines_source, created_at,
error_message, gdrive_file_id, final_creatives_exported_at, line_count).

| Method | Route | Behaviour |
|---|---|---|
| POST | `/runs/{rid}/captions` | 409 unless the run is `done` with a `result_local_path` on disk. Optional form field `language` (validated against `languages.is_valid`). Creates the row and calls `enqueue_process_caption`. Several caption jobs per run are allowed (same as tail edits) |
| GET | `/runs/{rid}/captions` | newest first |
| POST | `/captions/{cid}/retry` | only from `done`/`failed`; transition to `queued`, clear `error_message`, re-enqueue |
| PATCH | `/captions/{cid}/lines` | JSON body `{"lines": [{start,end,text}, …]}`. 409 while the job is in flight (`transcribing`/`burning`/`delivering`). Sets `lines_source="manual"`. Does **not** re-enqueue — the operator presses retry |
| GET | `/captions/{cid}/result` | `FileResponse` of `output_path`, 404 when absent — mirror `download_tail_edit_result` |
| DELETE | `/captions/{cid}` | 204, hard delete + `shutil.rmtree(caption_dir(...), ignore_errors=True)` — mirror `delete_tail_edit` (no dependents to check) |
| POST | `/captions/{cid}/export-to-sheet` | idempotent via `final_creatives_exported_at`; mirror `export_tail_edit_to_sheet` |

### 3.6 `app/web_v2.py` + templates

- `GET /v2/runs/{rid}/captions-fragment` → renders
  `partials/run_captions_list.html`, mirroring the tail-edits fragment route.
- Context helpers mirroring `_get_tail_edits` / `_tail_edits_still_active` /
  `_tail_edit_context`: keys `captions`, `captions_active`, `run_id`.
  Active = status in `{queued, transcribing, burning, delivering}`.
  Built only for a `done` run (the card is gated on `sv == "done"`).
- `partials/run_captions_list.html` — a "Captions" card mirroring
  `run_tail_edits_list.html`: HTMX `every 3s` poll while active, one row per
  caption job with status badge, error tail, Download / Drive / Creatives
  buttons, and a delete button.
- `partials/run_status_content.html` — a second button in the existing
  `.action-row`, after "Add to Final Creatives": `Add Captions`, wired to
  `POST /api/v2/runs/{rid}/captions` with the same fetch-and-report JS shape as
  `exportToSheet`. The existing button is **not** touched. (At the time this
  was written there was a third button, "Add to Caption Demo", between the
  two; it was removed on 2026-08-28.)

## 3.7 Chaining derived creatives (v2, 2026-08-10)

v1 hung both derived creatives off the Run's result and nothing else, so the
two could never be combined: you could caption the result **or** re-cut its
ending, never both. v2 lets each one take the other as its input.

Both directions are built, because they produce genuinely different videos and
the operator picks per job:

| Order | Result |
|---|---|
| caption → tail edit | Captioned body, **uncaptioned** new tail. Right when the library tail already carries its own burned-in text. |
| tail edit → caption | Captions across the **whole** re-cut video, tail included. Right when the new tail is bare footage. |

Neither is a default; a job with no source override still reads the Run's
result exactly as in v1.

### 3.7.1 Model changes

Both follow the `OverlayJob` precedent (`source_type` + soft reference, no FK —
`app/overlay_pipeline.py::_resolve_source` is the shape to copy).

`run_tail_edits` gains:
- `source_type` — Enum `("run", "caption")`, name `tail_edit_source_type_enum`,
  `nullable=False`, server default `'run'` so every existing row keeps its
  meaning.
- `source_caption_id` — String(36) NULL, soft reference to `run_captions.id`.

`run_captions` gains:
- `source_type` — Enum `("run", "tail_edit")`, name
  `caption_source_type_enum`, `nullable=False`, server default `'run'`.
- `source_tail_edit_id` — String(36) NULL, soft reference to
  `run_tail_edits.id`.

`run_id` stays `NOT NULL` on both tables in every case. It is the **parent run**
— the provenance used for the project id, the language suffix, the sheet `Idea`
and the output filename — not the video being read. Only the source fields say
which file is the input. One migration for both tables.

### 3.7.2 Pipeline changes

Each pipeline gains a `_resolve_source(...)` helper returning the input path,
mirroring `overlay_pipeline._resolve_source`:

- `pipeline_tail`: `source_type == "caption"` → that caption's `output_path`;
  otherwise `run.result_local_path`. Fail the job with a clear message when the
  referenced caption is missing, not `done`, or has no file on disk — the same
  shape as the existing "run has no result video on disk" failure.
- `pipeline_captions`: `source_type == "tail_edit"` → that tail edit's
  `output_path`; otherwise `run.result_local_path`.

**Probe the resolved source, never the Run's result.** A tail edit is shorter
than the run it came from, so captioning one while probing the run would size
the ASS to the wrong duration and clamp captions against a length the video
does not have.

Output filenames stay `{run_id}_…` (the parent run is still the creative) but
must not collide across chained jobs:
- tail edit from a caption → `{run_id}_captioned_{clip_name}.mp4`
- caption from a tail edit → `{run_id}_{clip_name}_captioned.mp4`

### 3.7.3 API changes

- `POST /runs/{rid}/tail-edits` gains optional form field `source_caption_id`.
- `POST /runs/{rid}/captions` gains optional form field `source_tail_edit_id`.

Both validate: the referenced entity exists, **belongs to this run**, is `done`,
and has an output file on disk — 400 for "not found / not this run's", 409 for
"not done / no file". Omitted → `source_type="run"`, byte-for-byte the v1
behaviour.

The tail-edit `cut_sec` upper bound is checked against the **resolved
source's** duration — as of fix/tail-edit-cut-bounds (2026-08-28), that check
runs synchronously in `create_tail_edit` itself (400, naming the exact bound),
not only in the worker; see `wiki/components/tail-edit.md` for the full bound
contract. The worker keeps its own copy of the check as a last line of
defence, not because the endpoint's check is incomplete.

### 3.7.4 UI changes

- The "Change ending" form gains a source `<select>`: "Run result (no
  captions)" plus one option per `done` caption job. Default = the run result.
- The captions card's create control gains the mirror-image choice: caption the
  run result, or a `done` tail edit.
- Both lists show which source a job used, so a row is self-explanatory once
  several chained jobs exist.
- Deleting an entity that another job used as its source must not orphan the
  UI: the lists resolve a missing source id to a plain "(deleted)" label rather
  than erroring, exactly as `_tail_clip_names` already tolerates a deleted
  library clip.

## 3.8 Transcript reuse across same-audio captions (2026-09-28)

Every `face_swap` run of a project under `audio_mode="original"` delivers the
**same soundtrack** — only the picture differs. Measured on project 91fb06ff:
all six finals decode to identical PCM. Captioning N runs used to cost N Gemini
transcriptions of one audio track, and the N transcripts disagreed by
0.3–0.5 s on line timings (model jitter, not the audio).

- `media.audio_fingerprint(path)` — sha256 over a version tag, the
  audio-minus-video start offset (ms) and the decoded mono 16 kHz s16le PCM of
  the **resolved** source (a chained tail edit has a different tail, so a
  different fingerprint). `None` on no audio / any failure → transcribe as
  before. The offset is in the key because captions are timed against the
  picture.
- `run_captions.audio_sha256` (indexed) + `run_captions.lines_reused_from`
  (soft ref), migration `f1c3e5a7b902`. `lines_source` gains `"reused"`.
- `process_caption` fingerprints the source and **commits** before any wait or
  Gemini call, then looks for a donor: another caption with the same
  fingerprint, non-empty `lines`, a non-NULL `lines_source`, and (when known)
  the same language. `manual` beats `transcribe` (an operator's fix reaches the
  project's next creatives — operator decision), then newest `updated_at`.
  A manual caption records its fingerprint too, so it can be a donor.
- If an **older** caption with the same audio is `transcribing` right now, the
  job waits for it (`CAPTION_DONOR_WAIT_SEC`, default 300, polled every
  `CAPTION_DONOR_POLL_SEC`) instead of paying for the same call on the other
  worker. Older only, so two jobs never wait on each other.
- `PATCH .../lines` clears `lines_reused_from` (the text is the operator's now).

Not done: a "fresh transcript" retry that bypasses reuse (Retry is only offered
on failed captions; to fix a bad transcript, edit the lines — the edit then
propagates). Rows older than the migration have no fingerprint and never act
as donors; a backfill must not fingerprint a run whose result was later
replaced (audio patch, range promotion), since the lines were read from the
old file.

## 3.9 Placement over the original caption area (AICP-368, 2026-10-31)

Subtitle removal leaves patches where the old captions were (VOID fills of
leftover segments, Omni's reconstruction of the band). House captions sit
below that band, so the patches stay visible. A **placement** centres every
caption block on the original band instead. Module: `app/caption_placement.py`
(its docstring is the full contract).

- **Band = union** of the source run's delivered VOID fill boxes
  (`run_segment_fill_jobs.mask_box`, `applied`, `delivery_state == delivered`
  — in the copied final; frozen in `source.fill_band` once read) and the
  detector's **median** band on the subtitle-removal parent's source
  (`[x0_p5, y0_p50, x1_p95, y1_p50]` from `qa_tools.calibrate_captions` with the
  parent's `caption_detector`, streamed at 2 fps). Not `y0_p5`/`y1_p95`:
  two-line captions drag them up. Real case 328341d4 / 2845eef3 (measured on
  the real source 2026-10-31, capped container): detector median 0.605–0.643,
  fills 0.581–0.663 → `center_y 0.6219`; a two-row caption box then spans
  0.581–0.663, i.e. exactly the fills.
- `video_projects.caption_placement` + `run_captions.placement` (JSON NULL,
  migration `s3c8e1a6d204`). NULL = house style, byte-for-byte (`build_ass`
  output compared against the pre-change module).
- **Several removal passes** (a subtitle-removal project made from another
  one's run): the evidence is the whole chain (`removal_chain`) — every pass's
  fills and a scan of every pass's own source, plus each pass project's own
  stored band — never just the nearest pass.
- **Which fills count**: only those in the copied picture —
  `delivery_state == delivered`, or `republished` when the final is still a
  platform stitch (a promoted range attempt, or the pipeline final behind
  promoted audio patches). Not an assistant publication, not a fill waiting
  for its re-stitch. Once read, the evidence is frozen (`source.fill_band`,
  `evidence_band`) and reused by the scan, by children and by re-derives; the
  promote holds the Run row lock across copy + lookup.
- **Project-from-run**: a settled parent placement is copied (`inherited`);
  otherwise (no value, `scan: pending|failed`, or the parent is itself a
  removal pass) a project with a removal chain gets a provisional value from
  the fills (`derived`, `scan: pending`) and an `analyze`-queue job
  (`run_caption_placement_scan`) replaces it with the union. The job writes
  under the project row lock only while `source.kind == derived` and
  `source.rev` is the one it was enqueued for; any failed pass makes it
  `scan: failed` (retryable); a deleted lineage settles it as failed on the
  frozen evidence. A deploy that kills it leaves the provisional value; re-run
  with `POST /api/v2/projects/{pid}/caption-placement/derive` or
  `python -m app.caption_placement derive <pid>` (inline, streamed). A
  placement problem never fails the promote.
- **Operator**: `PATCH /projects/{pid}` `caption_placement` (`kind: manual`);
  `null` stores the sticky `{"mode": "bottom"}`. The run page's caption control
  shows a position picker (default "Over original captions") when the project
  has a band placement; `POST /runs/{rid}/captions` takes
  `placement=project|bottom` and freezes the choice on the caption row.
- **Burn**: `captions.build_ass(placement=Placement(...))` → Alignment 5 + one
  `{\pos(W/2, center_y·H)}` per event (libass centres the block, box included,
  for 1–3 rows — verified on rendered frames), wrap to `max_width`, up to 3
  rows per caption (house: 2), overflow still split in time. A short caption
  that cannot cover the band still covers its centre. A resolved source whose
  aspect differs from the measured frame (>2 %) burns at the house position. The scan reads the parent's STORED band: a
  subtitle-removal project whose autopilot saved its located caption band
  (docs/autopilot-subtitle-flow.md §1.5) gives the right band; one without a
  band of its own is scanned with the default band and can come back
  `uncalibrated` (then the fill boxes alone decide) — save the band on the
  parent (`python -m app.qa_tools locate` prints it) and re-derive.

Not done: a band-sized opaque background that would cover the patches between
captions or around a one-word caption (translucent box, text-sized — the ticket
accepts "centre on the band"). An inherited placement is never refreshed from
its parent. A project made before AICP-368 cannot recover a fill that was
superseded after its copy (no supersession history). A caption created while
the scan is pending freezes the provisional value.

## 4. Non-goals

- Matching any external captioning tool's look.
- Overlay jobs sourcing from a caption (the overlay `source_type` enum is
  untouched; extend
  later if it earns its place).
- Karaoke/word-level highlighting — line-level timing only.
- Reusing the localisation translation as caption text. A later revision may
  overwrite the *hook* text with the stored translation while keeping the
  re-transcribed *timings*; that needs the translated lines persisted at run
  creation, which today they are not (§1.1).

## 5. Deploy notes

- `backend/Dockerfile` must add `fonts-dejavu-core` and `fonts-noto-cjk` to the
  `apt-get install` list. The image currently ships DejaVu only by accident of
  another package's dependencies, and **no CJK font at all** — Japanese, the
  flagship localisation target, would burn as tofu boxes. `fonts-noto-cjk` is
  large (~100 MB); that is the price of JA captions.
- ffmpeg in the image is 7.1.5 with `--enable-libass --enable-libfontconfig
  --enable-libfreetype --enable-libfribidi --enable-libharfbuzz`; the
  `subtitles`, `ass` and `drawtext` filters are all present. No ffmpeg change
  is needed.
- Migration runs **before** the rebuild, per CLAUDE.md.
