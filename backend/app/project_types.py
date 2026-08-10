"""
Project types — what an operator is trying to do to a source video.

A VideoProject's ``project_type`` decides:

1. **How ``analyze_project`` segments the video** (``ProjectTypeSpec
   .segmentation``). Only the face-swap flow runs InsightFace detection;
   subtitle removal cuts on scene changes; localisation splits the hook off the
   front; everything else probes the video and lays down ONE full-length
   ``keep`` segment — a blank slate the operator splits by hand in the Segment
   Editor.
2. **What the New Run form is pre-filled with** — model, prompt, audio mode —
   and which models it offers at all (``requires_audio_model``).
3. **What the Create Project form pre-checks** — ``default_mute_source``.

The form fields are only defaults: every one stays editable per run, and the
run itself stores no project-type of its own (the model/prompt it was created
with is the record of what it did). ``requires_audio_model`` is the one that is
more than a default — a model that emits no audio cannot localise anything (see
docs/localisation.md §2), so the form filters rather than pre-selects.

Nothing else in the pipeline branches on the type: once segments exist, a run
on a subtitle-removal project is submitted exactly like a face-swap run.

Keep this module free of app imports — api_v2, web_v2, pipeline_v2 and
localisation all import it, and it is the single source of truth for the enum's
valid values (mirrored by the ``project_type_enum`` Postgres type, migrations
a2c4e6f8b135 and a4c8e2b6d139).
"""

from __future__ import annotations

from dataclasses import dataclass

# Enum values as stored in video_projects.project_type.
FACE_SWAP = "face_swap"
SUBTITLE_REMOVAL = "subtitle_removal"
COVER_CHANGE = "cover_change"
LOCALISATION = "localisation"

# Segmentation strategies (ProjectTypeSpec.segmentation) — the branch
# pipeline_v2.analyze_project dispatches on. ONE field rather than a boolean per
# strategy: the strategies are mutually exclusive, and a fourth boolean would
# have made "which one wins" a question the reader has to answer from the order
# of the if/elif chain (docs/localisation.md §5).
SEG_DETECT_FACES = "detect_faces"   # InsightFace swap/keep partition.
SEG_SCENE_DETECT = "scene_detect"   # PySceneDetect content cuts, all "swap".
SEG_HOOK_SPLIT = "hook_split"       # swap[0, hook] + keep[hook, end].
SEG_BLANK = "blank"                 # one full-length "keep" segment.


@dataclass(frozen=True)
class ProjectTypeSpec:
    """Per-type behaviour + New Run / Create Project form defaults."""

    key: str
    label: str
    # Segmentation strategy analyze_project uses for this type — one of the
    # SEG_* constants above.
    segmentation: str
    default_model: str
    default_prompt: str
    # Run.audio_mode the New Run form pre-selects. "original" overlays the
    # source soundtrack onto the stitched result; "seedance" keeps each
    # generated clip's own audio. Localisation MUST default to "seedance":
    # overlaying the untranslated source track would silently undo the whole
    # localisation (docs/localisation.md §2).
    default_audio_mode: str
    # VideoProject.mute_source the Create Project form pre-checks. Muting the
    # clip we upload is what stops the model reproducing the language it can
    # hear — it beats prompting, which is why localisation defaults it on.
    default_mute_source: bool
    # When True the New Run form offers ONLY models with
    # ai_models.spec_for(m).produces_audio. This is a hard filter, not a
    # default: Gemini Omni never emits audio and create_run force-downgrades
    # such runs to audio_mode="original", which is unusable for localisation.
    requires_audio_model: bool
    # Whether reference images are part of the normal workflow. Purely a UI
    # hint — the API never requires them (Seedance and Gemini both accept an
    # empty image list).
    uses_references: bool
    # One-line explanation shown next to the type in the UI.
    hint: str


_FACE_SWAP_PROMPT = (
    "Replace the main person in the reference video with the person shown in the "
    "reference image. Keep their face and identity consistent with the reference image "
    "throughout. Change only the character — keep everything else exactly the same: "
    "the phone or tablet screen and its contents, all on-screen text and captions, "
    "the background, lighting, framing, and the original motion and lip movements."
)

# Operator-supplied defaults — kept verbatim (including wording quirks) so the
# text that reaches the model is exactly what was signed off on.
_SUBTITLE_REMOVAL_PROMPT = (
    "Remove all subtitles, captions and overlay text from video. Do not touch "
    "character, do not change anything on phone screen. keep everything else exactly "
    "the same: the character, character's lops, the background, lighting, framing, "
    "and the original motion and lip movements. Do not add any audio or voices. Do "
    "not add any extra things and movements."
)

_COVER_CHANGE_PROMPT = """The handheld book is the target object.

At the beginning of the video, the man is holding the target book.
During the transfer, the same target book moves naturally from his hands to the woman's hands.
After the transfer, the woman continues holding the same replacement book.

The identity of the replacement book must remain perfectly consistent before, during, and after the handoff."""

# --- Localisation (docs/localisation.md §4.3) --------------------------------
#
# Same convention as _SUBTITLE_REMOVAL_PROMPT above: **verbatim, quirks and
# all**. The text below is the prompt of run 1736426f — the only run that has
# ever proved this feature works end to end (seedance-2-5, audio_mode=seedance,
# mute_source, one 10s swap segment, EN→JA) — and what reaches the model is what
# was signed off on, so it is not tidied up. That includes the missing space in
# "speak(from", which stays.
#
# Exactly two departures from the run text, both forced by turning one run into
# a template:
#
#   1. The three slots. "(from english to japanese)" becomes
#      "(from {source_language} to {target_language})" — localisation.build_prompt
#      fills them with lowercase English language NAMES ("english", "japanese"),
#      so the rendered sentence is byte-identical to the proven one for an EN→JA
#      run. The dialogue block (localisation.format_dialogue, "**Speaker:**" +
#      five spaces) is appended through {dialogue} on the line after the
#      paragraph, exactly as in the run. The run's stored text separates them
#      with CRLF only because it was typed into a browser textarea; "\n" is the
#      same prompt.
#   2. "the woman ... the woman shown in the reference image" is restored to the
#      registry's own neutral phrasing from _FACE_SWAP_PROMPT ("the main person
#      ... the person shown in the reference image"). The operator typed that
#      run's gender in by hand on top of the face-swap default; a template
#      cannot know it, and guessing it wrong is worse than not stating it.
#
# NOTE for future edits: these strings are consumed with str.format(), so the
# only braces they may contain are the three slots.
_LOCALISATION_SWAP_PROMPT = (
    "Replace the main person in the reference video with the person shown in the "
    "reference image. Keep their face and identity consistent with the reference image "
    "throughout. Change only the character and language which they speak"
    "(from {source_language} to {target_language}) — keep everything else exactly the "
    "same: the phone or tablet screen and its contents, all on-screen text and "
    "captions, the background, lighting, framing, and the original motion and lip "
    "movements.\n{dialogue}"
)

# "Speech only" — the mode with no reference image. Structurally the SWAP
# prompt with the character replacement inverted into an explicit instruction to
# keep the person: everything after the language clause is word-for-word the
# proven text, because that half is what protects the screen contents, the
# captions and the framing, and it has been signed off.
#
# The trailing "and the original motion and lip movements" is deliberately kept
# even though the lips must in fact change to speak the new language. It reads
# as a contradiction, it is in the proven prompt, and Seedance re-speaks the
# line anyway — so it is left exactly where it was rather than "fixed" on a
# guess in the variant that has never been run.
_LOCALISATION_KEEP_PROMPT = (
    "Keep the main person in the reference video exactly as they are. Do not replace "
    "the character: their face, identity, hair, clothing and appearance must stay the "
    "same person throughout, unchanged from the reference video. Change only the "
    "language which they speak"
    "(from {source_language} to {target_language}) — keep everything else exactly the "
    "same: the phone or tablet screen and its contents, all on-screen text and "
    "captions, the background, lighting, framing, and the original motion and lip "
    "movements.\n{dialogue}"
)


PROJECT_TYPES: dict[str, ProjectTypeSpec] = {
    FACE_SWAP: ProjectTypeSpec(
        key=FACE_SWAP,
        label="Face changing",
        segmentation=SEG_DETECT_FACES,
        default_model="seedance",
        default_prompt=_FACE_SWAP_PROMPT,
        default_audio_mode="original",
        default_mute_source=False,
        requires_audio_model=False,
        uses_references=True,
        hint=(
            "Detects faces and proposes swap/keep segments automatically. "
            "Upload reference images of the new character."
        ),
    ),
    SUBTITLE_REMOVAL: ProjectTypeSpec(
        key=SUBTITLE_REMOVAL,
        label="Removing subtitles",
        segmentation=SEG_SCENE_DETECT,
        default_model="gemini-omni",
        default_prompt=_SUBTITLE_REMOVAL_PROMPT,
        default_audio_mode="original",
        default_mute_source=False,
        requires_audio_model=False,
        uses_references=False,
        hint=(
            "No face detection — the video is auto-cut on scene changes and every "
            "scene is marked swap, ready to run. No reference images needed."
        ),
    ),
    COVER_CHANGE: ProjectTypeSpec(
        key=COVER_CHANGE,
        label="Change cover",
        segmentation=SEG_BLANK,
        default_model="seedance-fast",
        default_prompt=_COVER_CHANGE_PROMPT,
        default_audio_mode="original",
        default_mute_source=False,
        requires_audio_model=False,
        uses_references=True,
        hint=(
            "No face detection — the Segment Editor starts with one full-length "
            "segment you split yourself. Add a reference image of the new cover."
        ),
    ),
    LOCALISATION: ProjectTypeSpec(
        key=LOCALISATION,
        label="Localisation",
        segmentation=SEG_HOOK_SPLIT,
        # Seedance 2.5 is the only model this type can sensibly default to: it
        # generates up to 30s per clip (a hook is one segment, not three) and
        # takes an explicit audio switch. Gemini Omni is filtered out entirely
        # by requires_audio_model.
        default_model="seedance-2-5",
        default_prompt=_LOCALISATION_SWAP_PROMPT,
        default_audio_mode="seedance",
        default_mute_source=True,
        requires_audio_model=True,
        uses_references=True,
        hint=(
            "No face detection — the hook is cut off the front as one swap segment "
            "and the rest is left as keep, to be replaced by a language-matched tail. "
            "The source is transcribed automatically so the hook can be re-spoken in "
            "another language. Add a reference image only to change the character too."
        ),
    ),
}

DEFAULT_PROJECT_TYPE = FACE_SWAP
VALID_PROJECT_TYPES = frozenset(PROJECT_TYPES)


def spec_for(project_type: str | None) -> ProjectTypeSpec:
    """Return the spec for *project_type*, falling back to the face-swap flow.

    The fallback keeps every caller safe against a NULL column on a row that
    predates the project_type migration.
    """
    return PROJECT_TYPES.get(project_type or "", PROJECT_TYPES[DEFAULT_PROJECT_TYPE])


def label_for(project_type: str | None) -> str:
    """Human-readable label for a project type (used by templates)."""
    return spec_for(project_type).label
