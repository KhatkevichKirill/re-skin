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
from typing import Optional

# Enum values as stored in video_projects.project_type.
FACE_SWAP = "face_swap"
SUBTITLE_REMOVAL = "subtitle_removal"
COVER_CHANGE = "cover_change"
LOCALISATION = "localisation"
REFRAME = "reframe"
# Rebuild from reference (docs/rebuild-from-reference-design.md): the video is
# re-generated shot by shot from clean keyframes (images-only runs), keeping the
# source soundtrack. Behind settings.REBUILD_ENABLED.
REBUILD = "rebuild"

# Segmentation strategies (ProjectTypeSpec.segmentation) — the branch
# pipeline_v2.analyze_project dispatches on. ONE field rather than a boolean per
# strategy: the strategies are mutually exclusive, and a fourth boolean would
# have made "which one wins" a question the reader has to answer from the order
# of the if/elif chain (docs/localisation.md §5).
SEG_DETECT_FACES = "detect_faces"   # InsightFace swap/keep partition.
SEG_SCENE_DETECT = "scene_detect"   # PySceneDetect content cuts, all "swap".
SEG_HOOK_SPLIT = "hook_split"       # swap[0, hook] + keep[hook, end].
SEG_BLANK = "blank"                 # one full-length "keep" segment.
# Rebuild: fine shot detection -> short shots folded into effective shots ->
# chunks of whole shots (<= 14.72s / <= 8 shots), all "swap"; the shots are
# persisted in rebuild_shots (app/rebuild.py, design §5.1).
SEG_SHOT_CHUNKS = "shot_chunks"

# How a type's runs generate video (ProjectTypeSpec.generation_mode, mirrored
# by runs.generation_mode). "v2v" = a source clip is edited by the model (every
# type before rebuild); "images_only" = the model generates each chunk from
# still keyframes, no source clip (rebuild).
GENERATION_V2V = "v2v"
GENERATION_IMAGES_ONLY = "images_only"
# A subtitle-removal run whose swap segments are VOID fills from the source
# (docs/segment-fill-batch-design.md §7): no model submit except the segments
# the operator sent to the model as a fallback (run_segments.model_fallback).
GENERATION_VOID_SOURCE = "void_source"
GENERATION_MODES = (GENERATION_V2V, GENERATION_IMAGES_ONLY, GENERATION_VOID_SOURCE)
CAPTION_REMOVAL_VOID_SOURCE = "void_source"
CAPTION_REMOVAL_OMNI = "omni"
CAPTION_REMOVAL_METHODS = (CAPTION_REMOVAL_VOID_SOURCE, CAPTION_REMOVAL_OMNI)


def caption_removal_method(project) -> str:
    """The project's caption removal method; NULL (a project from before the
    mode existed, or another type) reads as "omni"."""
    v = getattr(project, "caption_removal_method", None)
    return v if v in CAPTION_REMOVAL_METHODS else CAPTION_REMOVAL_OMNI


def default_caption_removal_method(project_type: str) -> Optional[str]:
    """What a NEW project of *project_type* starts with: VOID from the source
    for subtitle removal (operator decision 2026-11-03), nothing otherwise."""
    return CAPTION_REMOVAL_VOID_SOURCE if project_type == SUBTITLE_REMOVAL else None


def is_void_source_run(run) -> bool:
    return getattr(run, "generation_mode", GENERATION_V2V) == GENERATION_VOID_SOURCE

# How much of a ``localisation`` project is re-generated
# (VideoProject.localisation_scope, AICP-340). Like hook_sec and
# max_segment_sec it is consumed at ANALYZE time, and it only ever changes
# together with a re-analysis (POST /projects/{pid}/analyze?localisation_scope=
# records the request; analyze_project applies it in the same commit as the
# new segments),
# so the stored value always describes the segmentation on the page:
#   hook  — SEG_HOOK_SPLIT exactly as before: swap[0, hook] + keep[hook, end],
#           the keep tail replaced at stitch time by a library clip.
#   whole — the whole video is re-spoken: analysis cuts ALL-swap segments at
#           scene changes, each within the localisation clip target
#           (pipeline_v2.whole_video_target_sec), and no hook clamp applies.
# Meaningless for every other project type, which never reads it.
LOCALISATION_SCOPE_HOOK = "hook"
LOCALISATION_SCOPE_WHOLE = "whole"
LOCALISATION_SCOPES = (LOCALISATION_SCOPE_HOOK, LOCALISATION_SCOPE_WHOLE)


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
    # False for a type whose page has no runs at all (``reframe``): the New
    # Run form is hidden and ``create_run`` answers 409 for it.
    supports_runs: bool = True
    # How runs on this type generate video (GENERATION_* above). The New Run
    # form and create_run read it; a run stores its own copy
    # (runs.generation_mode), set at creation and never changed.
    generation_mode: str = GENERATION_V2V
    # Whether a project DERIVED from a run (Move to new project) carries the
    # parent lineage's mute_source over (see derived_mute_source). False for
    # face swap: its stitch overlays the ORIGINAL soundtrack and the prompt
    # asserts that track, so a "Remove original audio" inherited from a
    # localisation parent was always wrong there and had to be unticked by
    # hand (project c0cf56c2 from run 7c90f778, 2026-11-10).
    inherits_mute_source: bool = True


# Operator-supplied defaults — kept verbatim (including wording quirks) so the
# text that reaches the model is exactly what was signed off on.
#
# Face swap: the original one-paragraph version ended on "the original motion
# and lip movements", which pinned the SOURCE person's mouth shapes onto the new
# face. This version splits performance from face — head, body, gesture and gaze
# timing are preserved, while expressions and lips are re-derived from the new
# character's own anatomy, the same distinction the localisation templates below
# had to make for translated speech.
#
# The audio paragraph asserts the ORIGINAL track rather than describing a
# generated voice, because on this type's defaults no generated voice reaches
# the viewer. audio_mode="original" is what does it: the stitch overlays the
# source soundtrack and throws the generated one away. The default model
# (Seedance 2.0) has no audio switch and writes a track regardless — it is
# simply discarded; on 2.5 the same condition sets generate_audio=False in
# _build_submit_state, so it is never produced at all. mute_source is a
# separate lever and not part of that derivation: it decides what the model
# HEARS, not what it writes.
#
# An audio instruction that fights what the model is fed does not fail quietly
# — it confuses the model about the video too (wiki/models/ai-models.md, "Input
# audio"). And because the source soundtrack is what the viewer hears, the new
# mouth has to match the ORIGINAL words and timing, which is the one thing the
# operator text did not say.
#
# The exception is a run that does NOT stitch (stitch_plan == []): the
# per-segment clips are then the deliverable and generate_audio is True
# whatever audio_mode says, so the generated voice does ship. An operator doing
# that on 2.5 should edit this paragraph out — which is why it lives in an
# editable default and not in code.
_FACE_SWAP_PROMPT = """Replace the main person in the reference video with the generated by seedream person shown in the reference image.

Keep the new character’s identity, facial features, skin tone, hairstyle, and overall appearance consistent with the reference image throughout the entire video.

Preserve the original performance and timing of the reference video: keep the same head movements, body movements, gestures, hand and arm movements, shoulder movements, poses, gaze direction, and interactions with objects.

Adapt the facial expressions, mouth shapes, and lip movements naturally to the new character’s own facial features and anatomy. Do not copy or warp the original person’s facial features or exact mouth shapes onto the new character. The new character should perform the same expressions and speech with natural, identity-consistent facial motion.

Keep the original audio exactly as it is: the original voice, background sounds, music, and ambient audio. Do not generate a new voice or new speech. The new character’s lip movements must match the original spoken words and their timing.

Change only the character. Keep everything else exactly the same: the phone or tablet and its screen contents, all on-screen text and captions, the background, lighting, framing, camera movement, objects, and scene composition."""

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
# Adapted for translated speech. Early localisation used a face-swap-derived
# paragraph (historical run 1736426f) that asked Seedance to keep "the original
# motion and lip movements" — that freezes mouth motion and fights re-speaking
# the target language. These templates instead require natural lip sync and
# speech-related facial reactions for the assigned translated dialogue, while
# still protecting non-speech scene continuity and on-screen text.
#
# Model output is probabilistic: the prompt instructs adaptive lips/reactions,
# but the app cannot guarantee perfect sync on every clip or language pair.
#
# Slots filled by localisation.build_prompt with lowercase English language
# NAMES ("english", "japanese") and a dialogue block from
# localisation.format_timed_dialogue (AICP-335; the untimed format_dialogue
# "**Speaker:**" block when no clip window is given). Consumed with
# str.format() — the only braces allowed are the three slots below.
_LOCALISATION_SWAP_PROMPT = (
    "Replace the main person in the reference video with the person shown in the "
    "reference image. Keep their face and identity consistent with the reference image "
    "throughout. Re-speak the assigned dialogue in {target_language} "
    "(from {source_language} to {target_language}). Naturally adapt mouth shapes, lip "
    "movements and timing to lip-sync the translated speech; also adapt facial "
    "expressions, emotional reactions and speaker behavior where the translated "
    "delivery requires it. Preserve framing, camera movement, background, lighting, "
    "props, phone or tablet screens and their contents, captions and other on-screen "
    "text. Preserve overall action, choreography and body movement except for natural "
    "speech-related adjustments — do not freeze the original face or mouth motion. "
    "Each visually grounded speaker must say only the dialogue assigned to that "
    "speaker; do not swap lines between on-screen and off-screen speakers.\n"
    "{dialogue}"
)

# "Speech only" — no reference image. Same speech / lip / continuity rules as
# SWAP, but identity must stay the person already on screen.
_LOCALISATION_KEEP_PROMPT = (
    "Keep the main person in the reference video exactly as they are. Do not replace "
    "the character: their face, identity, hair, clothing and appearance must stay the "
    "same person throughout, unchanged from the reference video. Re-speak the assigned "
    "dialogue in {target_language} "
    "(from {source_language} to {target_language}). Naturally adapt mouth shapes, lip "
    "movements and timing to lip-sync the translated speech; also adapt facial "
    "expressions, emotional reactions and speaker behavior where the translated "
    "delivery requires it. Preserve framing, camera movement, background, lighting, "
    "props, phone or tablet screens and their contents, captions and other on-screen "
    "text. Preserve overall action, choreography and body movement except for natural "
    "speech-related adjustments — do not freeze the original face or mouth motion. "
    "Each visually grounded speaker must say only the dialogue assigned to that "
    "speaker; do not swap lines between on-screen and off-screen speakers.\n"
    "{dialogue}"
)

# The two values of the New Run form's localisation radio (docs/localisation.md
# §7). The choice is not stored on the run (§3.3) — the prompt it produced is
# the record, which is what `localisation_mode_of` below reads back.
LOCALISATION_MODE_KEEP = "keep"   # speech only
LOCALISATION_MODE_SWAP = "swap"   # speech + character

# The sentence that tells the two templates above apart. It is
# _LOCALISATION_KEEP_PROMPT's second clause and appears in no other prompt in
# this registry, and the templates stay English through translation (only the
# appended dialogue block changes language), so it survives every prompt
# localisation.build_prompt produces. Lowercased for a case-insensitive match.
_LOCALISATION_KEEP_MARKER = "do not replace the character"


PROJECT_TYPES: dict[str, ProjectTypeSpec] = {
    FACE_SWAP: ProjectTypeSpec(
        key=FACE_SWAP,
        label="Face changing",
        segmentation=SEG_DETECT_FACES,
        default_model="seedance",
        default_prompt=_FACE_SWAP_PROMPT,
        default_audio_mode="original",
        default_mute_source=False,
        inherits_mute_source=False,
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
        # Seedance 2.0 Fast is the operational default: it produces audio
        # (required here), is faster/cheaper for routine localisation, and
        # covers the default 10s hook under the universal segmentation cap.
        # Its hard ceiling is 15s per clip — longer hooks need the project
        # segmented into valid pieces, or the operator picks Seedance 2.5
        # (still registered and selectable) for its 30s ceiling / explicit
        # generate_audio switch. Gemini Omni is filtered out entirely by
        # requires_audio_model.
        default_model="seedance-fast",
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
    REFRAME: ProjectTypeSpec(
        key=REFRAME,
        label="Reframe (9:16 / 16:9)",
        segmentation=SEG_BLANK,
        default_model="seedance",
        default_prompt="",
        default_audio_mode="original",
        default_mute_source=False,
        requires_audio_model=False,
        uses_references=False,
        supports_runs=False,
        hint=(
            "AI outpainting to the other orientation (MiniMax H3 Max Reframe). "
            "The chosen window of this video is SENT TO FAL, a third-party "
            "provider, and each generation is paid. No runs and no segments: "
            "generate, then pick the ending and deliver."
        ),
    ),
    REBUILD: ProjectTypeSpec(
        key=REBUILD,
        label="Rebuild from reference",
        segmentation=SEG_SHOT_CHUNKS,
        # Kling 4.0 at 720p is the decided default (design §15, Q1); Seedance
        # 2.5 images-only is the alternative. Prompts are generated per chunk
        # from the storyboard, so there is no default prompt.
        default_model="kling-4-0",
        default_prompt="",
        default_audio_mode="original",
        default_mute_source=False,
        requires_audio_model=False,
        uses_references=False,
        generation_mode=GENERATION_IMAGES_ONLY,
        hint=(
            "Recreate the video shot by shot from clean keyframes; original "
            "soundtrack kept."
        ),
    ),
}

# Default run resolution of an images-only (rebuild) run (design §15, Q1).
REBUILD_DEFAULT_RESOLUTION = "720p"

DEFAULT_PROJECT_TYPE = FACE_SWAP
VALID_PROJECT_TYPES = frozenset(PROJECT_TYPES)


def spec_for(project_type: str | None) -> ProjectTypeSpec:
    """Return the spec for *project_type*, falling back to the face-swap flow.

    The fallback keeps every caller safe against a NULL column on a row that
    predates the project_type migration.
    """
    return PROJECT_TYPES.get(project_type or "", PROJECT_TYPES[DEFAULT_PROJECT_TYPE])


def derived_mute_source(project_type: str | None, parent_mute_source: bool | None) -> bool:
    """``mute_source`` for a project DERIVED from another one (a run moved into
    a new project, AICP-339).

    The type's ``default_mute_source`` wins when it is on; otherwise the
    lineage's choice is carried over. A derived project has no Create Project
    form to pre-check the box, so without this a localisation project made
    from a face-swap run inherited ``False`` and Seedance heard (and mixed in)
    the original language. Types whose default is off keep inheriting exactly
    as before, so a muted lineage is never silently un-muted — except face
    swap (``inherits_mute_source=False``), which always starts with it off:
    a face-swap run keeps the original soundtrack, so a muted (typically
    localisation) parent must not leak into it. The operator can still change
    it afterwards (PATCH /projects/{pid}).
    """
    spec = spec_for(project_type)
    if spec.default_mute_source:
        return True
    return spec.inherits_mute_source and bool(parent_mute_source)


def is_rebuild(project_type: str | None) -> bool:
    """True for a ``rebuild`` project (exact match: no face-swap fallback)."""
    return project_type == REBUILD


def selectable_project_types(*, rebuild_enabled: bool) -> list[ProjectTypeSpec]:
    """The types a Create Project / Move-to-project form may offer.

    ``rebuild`` is listed only while ``settings.REBUILD_ENABLED`` is on (the
    caller passes it: this module stays free of app imports)."""
    return [
        spec for key, spec in PROJECT_TYPES.items()
        if key != REBUILD or rebuild_enabled
    ]


def label_for(project_type: str | None) -> str:
    """Human-readable label for a project type (used by templates)."""
    return spec_for(project_type).label


def localisation_mode_of(prompt: str | None) -> str:
    """Which localisation mode a run's *prompt* was built from.

    ``LOCALISATION_MODE_KEEP`` only for a prompt carrying the speech-only
    template's marker sentence; everything else — the swap template, and any
    prompt an operator rewrote past recognition — reads as
    ``LOCALISATION_MODE_SWAP``. The asymmetry is deliberate: the caller uses
    this to *withdraw* Copy run (copying a speech-only run re-submits an
    identical generation, and the reference photo a copy adds is inert against
    a prompt that says to keep the person on screen), and an unrecognised
    prompt should keep the feature rather than lose it on a guess.

    Reference images are deliberately not consulted: a photo attached to a
    speech-only prompt does nothing, while a swap prompt submitted *without*
    one is exactly the case where copying with a photo is the fix.

    Only meaningful for runs on a ``localisation`` project — callers gate on
    the project type first (the other types' prompts are not modes of this
    one).
    """
    return (
        LOCALISATION_MODE_KEEP
        if _LOCALISATION_KEEP_MARKER in (prompt or "").lower()
        else LOCALISATION_MODE_SWAP
    )


def run_copies_are_useful(project_type: str | None, prompt: str | None) -> bool:
    """Whether re-running this run with a fresh reference photo can produce
    anything new — the shared premise of **Copy run** and **Automation**.

    False for exactly one case: a speech-only localisation run. Its prompt
    instructs the model to keep the person on screen, so the photo either
    feature attaches is inert and the copy is the same generation again — once
    by hand, or every day on a quota.

    Both features gate on this, in the UI and server-side, so they can never
    disagree about which runs are worth copying.
    """
    if spec_for(project_type).key != LOCALISATION:
        return True
    return localisation_mode_of(prompt) != LOCALISATION_MODE_KEEP
