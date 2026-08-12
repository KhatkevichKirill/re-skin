# Public Localisation ↔ Private Parity Report

## 1. Repository / branch identity

| Item | Value |
|---|---|
| Public repository URL | `git@github.com:KhatkevichKirill/re-skin.git` |
| Worktree / clone path | `/root/re-skin-public-localisation-parity` |
| Branch | `codex/public-localisation-private-parity` |
| Public base commit (`origin/main` at start) | `b97ac31ce2de1596632a6be224795008505ac39d` |
| Implementation commit | `c619bfea2d1f4c6a4aa1f3d11afe6ede5c5561b7` |
| Branch tip | Determined via `git rev-parse HEAD` in this worktree (authoritative; not pinned in this file). |
| Origin remote | `git@github.com:KhatkevichKirill/re-skin.git` (public only) |
| Private behavioural reference | `/root/re-skin-localisation-release-integration` @ `be9407302217342a50acbd05fab76e993f587462` |

**Origin confirmation:** `git remote -v` shows only the public `re-skin` remote. No private remote was added; no private commits were merged or cherry-picked.

## 2. Outcome summary

**Status: success.**

All three Localisation improvements from the private release were ported semantically into the public repository:

1. **Transcript schema v2** with `video_summary` / `scene_context`, propagated into translation.
2. **Seedance 2.0 Fast (`seedance-fast`)** as the Localisation New Run default (2.5 remains selectable).
3. **Adaptive target-language lip sync / facial reactions** in both Localisation prompt modes.

Focused localisation tests and the full public backend suite passed (`963 passed`). No push, PR, merge, deploy, or live/paid model call was performed.

## 3. Public / private gap analysis (pre-edit)

### Public base

- Commit: `b97ac31` — `Merge pull request #11 from KhatkevichKirill/feat/localisation`
- Public already contained a usable Localisation pipeline (`localisation.py`, project type, API, UI, docs, tests). Task continued (no wholesale subsystem dump).

### Behavior status at base

| Behavior | Public at `b97ac31` | Action |
|---|---|---|
| A. `video_summary` / `scene_context` | **Absent** (schema v1, `transcribe/v1`, `translate/v1`) | Port |
| B. Fast default | **Absent** (`default_model="seedance-2-5"`) | Port |
| C. Adaptive lips prompts | **Absent** (legacy “original motion and lip movements” + `speak(from` spacing defect) | Port |

### Structural differences requiring adaptation

- Public `api_v2.py` is much smaller than private (~2.5k vs ~5.3k lines). Only the transcript PATCH validation and `localisation-prompt` translate call sites were touched.
- Private `localisation.py` includes `_upload_proxy` / `media.make_llm_proxy` — **not** present in public `media.py`. **Not copied.** Public continues to upload the source path directly via `KieClient.upload_file`.
- Private `project_types.localisation_mode_of` / `run_copies_are_useful` and automation/Copy-run gating — public has no automation module wiring for these. **Not copied** (out of scope for the three behaviors).
- Private docs mention clip-library / Copy-run / automation details that the public §0 already flags as out of repo. Public docs were updated only for the three behaviors, without importing private-only product surfaces.

### Private-only material deliberately excluded

- `make_llm_proxy` upload path and related media helpers/tests
- `localisation_mode_of` / `run_copies_are_useful` and Copy-run / Automation 409 gates
- Private production history, commit hashes, VPS/host details, Drive IDs, operator run IDs
- Unrelated private `api_v2` features present in the larger private tree

### Planned / final public file set

| File | Role |
|---|---|
| `backend/app/localisation.py` | Schema/prompt v2, context normalisation, translate kwargs |
| `backend/app/project_types.py` | Adaptive prompts + `seedance-fast` default |
| `backend/app/api_v2.py` | PATCH validation + context forward into translate |
| `docs/localisation.md` | Public contract for v2 / Fast / adaptive lips |
| `backend/tests/test_localisation.py` | Unit coverage for A + C |
| `backend/tests/test_project_types.py` | Registry + prompt coverage for B + C |
| `backend/tests/test_api_v2.py` | PATCH/translate/integration coverage |
| `backend/tests/test_web_v2.py` | New Run form preselects Fast |
| `tasks/reports/public_localisation_private_parity.md` | This report |

## 4. Files changed and why

1. **`backend/app/localisation.py`** — bump schema/prompt to v2; ask Gemini for both context fields; store trimmed/empty-safe strings; inject both into `TRANSLATE_PROMPT`; accept optional kwargs on `translate_lines` (default `""` for v1 compatibility).
2. **`backend/app/project_types.py`** — replace freeze-lips templates with adaptive lips/reactions templates; set Localisation `default_model` to `seedance-fast`; leave other project-type defaults unchanged.
3. **`backend/app/api_v2.py`** — `_optional_transcript_context` (400 on non-string); PATCH preserves/strips context; `localisation-prompt` forwards cached context into `translate_lines`.
4. **`docs/localisation.md`** — document schema v2 distinction, Fast default + 15s ceiling, 2.5 alternative, mute_source, adaptive lips, probabilistic quality caveat, v1 backward compatibility.
5. **Tests** — adapt/add coverage for all acceptance items; stub network via existing `respx` / `fake_translate`.
6. **This report** — review artifact.

## 5. Final transcript JSON and translation contracts

### Transcript (schema v2)

New transcriptions store:

```json
{
  "schema_version": 2,
  "model": "<resolved>",
  "prompt_version": "transcribe/v2",
  "created_at": "<iso>",
  "source_language": "<iso639-1>",
  "video_summary": "<trimmed string or \"\">",
  "scene_context": "<trimmed string or \"\">",
  "lines": [ ... ],
  "on_screen_text": [ ... ]
}
```

- `video_summary`: high-level purpose/narrative/progression.
- `scene_context`: grounded situational evidence for ambiguous dialogue.
- Missing/null/non-string model values → `""` at store time (`_normalise_context_text`).
- No DB migration: still JSON on `video_projects.transcript`.

### `translate_lines`

```python
translate_lines(
    lines,
    *,
    source_language,
    target_language,
    model=None,
    video_summary="",
    scene_context="",
)
```

Prompt sections:

```
VIDEO SUMMARY:
{video_summary or "(none provided)"}

SCENE CONTEXT:
{scene_context or "(none provided)"}
```

1:1 id/order/speaker/timing contract unchanged.

### API PATCH

- Omitted/null context → stored as `""`.
- Explicit non-string → HTTP 400 `transcript.video_summary|scene_context must be a string`.
- Unknown envelope keys still round-trip; UI already preserves keys it does not display.

### API `localisation-prompt`

Reads `transcript.video_summary` / `scene_context` (empty defaults for v1) and passes them into `translate_lines`.

## 6. Final model-default and prompt behavior

- `spec_for(LOCALISATION).default_model == "seedance-fast"`.
- New Run form preselects via `_new_run_defaults` / registry (verified in tests).
- `seedance-2-5` remains in `VALID_MODELS`, produces audio, 30s ceiling, selectable.
- Gemini filtered by `requires_audio_model` (`produces_audio=False`).
- Localisation retains `default_audio_mode="seedance"` and `default_mute_source=True`.
- Fast 15s ceiling already enforced by `ai_models.max_clip_sec`, segmentation / form model filtering, and `kie_client` → `validate_duration` at submit time — no extra backstop required.
- Both `_LOCALISATION_SWAP_PROMPT` and `_LOCALISATION_KEEP_PROMPT` require re-speaking assigned dialogue, natural lip sync, facial reactions, speaker grounding, and non-speech continuity; neither freezes original lips; neither has `speak(from`.
- Keep vs swap identity rules remain distinct.

## 7. Backward-compatibility decisions

- Old schema-v1 transcripts without context fields remain translatable (empty-string defaults).
- PATCH of a v1-shaped body (no context keys) stores `""` for both fields rather than rejecting.
- Prompt templates changed; historical “proven run” literal wording is no longer the contract (documented as superseded because it conflicted with localisation).
- Private upload-proxy and Copy-run mode helpers were **not** introduced, so public callers keep their existing upload and copy UX.

## 8. Tests added or changed

- `test_localisation.py`: context in TRANSCRIPT_JSON; store/trim/null tests; schema asks for fields; translate prompt includes context; omitted kwargs still work; live-template adaptive lips render; versions → v2.
- `test_project_types.py`: Fast default + 15s ceiling; 2.5 still optional; other types unchanged; adaptive lips / continuity / no freeze / no `speak(from`.
- `test_api_v2.py`: `fake_translate` accepts context kwargs; PATCH preserve/reject; translate receives context; v1 still works; `TestLocalisationReleaseIntegration` covers combined workflow items 1–4 (+ multi-segment).
- `test_web_v2.py`: form selects `seedance-fast`.

## 9. Validation commands and results

Environment: `KIE_API_KEY=test-dummy-not-live` (placeholder for client construction only). External calls stubbed via `respx` / monkeypatched `translate_lines`.

| Command | Result |
|---|---|
| `pytest tests/test_localisation.py tests/test_project_types.py -q` | **139 passed** in ~5.5s |
| `pytest tests/test_api_v2.py -q -k 'Localisation or Transcript or PatchTranscript or localisation or transcript'` | **102 passed**, 178 deselected |
| `pytest tests/test_web_v2.py -q -k 'localisation or Localisation or mute_source or audio_less or NewRun or form_'` | **21 passed**, 57 deselected |
| `pytest tests/ -q` (full backend) | **963 passed**, 385 warnings in ~244s |

No baseline failures: full suite green on this branch; no same-environment re-run of untouched base was required for classification.

## 10. Baseline-failure evidence

None claimed. Full suite passed on the implementation branch.

## 11. Publication-safety audit

| Check | Result |
|---|---|
| `origin` is public `re-skin` | Yes |
| No private commit merge/cherry-pick | Yes (semantic port only) |
| `git diff --check <base>..HEAD` | Clean (after EOF fix) |
| No `.env`, secrets, SA JSON, DB dumps, media, caches tracked | Yes |
| No private hostname / VPS / Drive ID / production run IDs in diff | Yes (grep clean for private markers) |
| Docs do not guarantee perfect lip sync | Yes — explicitly probabilistic / not guaranteed |
| No push / PR / merge / deploy | Confirmed |
| No live paid model calls | Confirmed (stubs only) |

## 12. Confirmation of non-actions

- No live API / paid model calls.
- No `git push`.
- No pull request created.
- No merge to `main`.
- No deploy, Docker rebuild, or service restart.
- Production checkout `/root/re-skin` was not modified.

## 13. Risks, unresolved questions, recommended manual checks

1. **Upload size:** public still uploads the raw source for transcription (no `make_llm_proxy`). Large sources may hit the kie upload ceiling — separate from this parity task; consider a follow-up if operators hit 100 MB failures.
2. **Prompt quality is probabilistic:** operators should visually QA lip sync on a few language pairs after selecting Fast vs 2.5.
3. **UI does not edit `video_summary` / `scene_context`:** round-trip preserves them via “keep unknown keys”; if operators need to edit context in-browser, that is a future UI task.
4. **Manual New Run check:** create a localisation project, confirm model dropdown preselects Seedance 2.0 Fast, Translate fills adaptive-lips prompt, and mute-source remains on.
5. **Long hooks (>15s):** confirm UI/capability logic moves the operator to 2.5 or splits segments (existing public machinery; smoke-test once after publish).

## 14. Compact diff / stat summary (vs public base)

```
 backend/app/api_v2.py               |  41 +++++++
 backend/app/localisation.py         | 110 +++++++++++++----
 backend/app/project_types.py        |  98 ++++++++--------
 backend/tests/test_api_v2.py        | 228 +++++++++++++++++++++++++++++++++++-
 backend/tests/test_localisation.py  | 150 +++++++++++++++++++++++-
 backend/tests/test_project_types.py | 146 ++++++++-------
 backend/tests/test_web_v2.py        |   5 +-
 docs/localisation.md                |  57 +++++++--
 tasks/reports/public_localisation_private_parity.md | (this report)
```

~(8 implementation files + report); semantic port of three Localisation improvements only.
