"""Closed code lists for the knowledge layer (design §4, §5).

Each tuple below is the single source for the matching CHECK constraint in
``knowledge.models`` and the additive migration, and for service validation.
Changing a list is a schema change: add a migration that replaces the CHECK.
"""
from __future__ import annotations

CASE_SCOPES = ("run", "segment", "derived", "manual", "external")
CASE_OPERATIONS = ("face_swap", "subtitle_removal", "cover_change", "localisation",
                   "images_only_rebuild_manual", "repair")
ORIGINS = ("live", "import")
VERSION_REASONS = ("initial_output", "restitch", "range_attempt", "audio_patch",
                   "assistant_promotion", "import", "regeneration", "external_repair")

ARTIFACT_KINDS = ("source", "source_clip", "reference_image", "reference_audio", "model_output",
                  "stitched_final", "delivered_final", "assistant_version", "tail_edit", "caption",
                  "overlay", "external_repair")
LOCATOR_KINDS = ("local_path", "preserved_copy", "drive_file", "assistant_artifact", "kie_url",
                 "external_url")

ATTEMPT_KINDS = ("generation", "resubmit", "rerun", "range_attempt", "segment_image",
                 "preprocess_image", "audio_patch", "tail_edit", "caption", "overlay",
                 "assistant_version", "run_result_promotion", "manual_script", "external_repair")
ATTEMPT_ORIGINS = ("automatic", "operator", "assistant", "external")
PROVIDERS = ("kie", "byteplus", "local", "external")
OBSERVATION_KINDS = ("status", "output", "cost_estimate", "cost_settled", "error", "ambiguous",
                     "redispatch", "input_content")
ATTEMPT_STATUSES = ("submitted", "succeeded", "failed", "short_result", "refused", "superseded",
                    "intent_without_submission", "unknown")
COST_SOURCES = ("spend_draw", "run_segment_snapshot", "linear_report")

EVENT_TYPES = ("auto_assessment", "operator_verdict", "defect", "correction", "link_external",
               "retraction", "export_observed")
ACTOR_KINDS = ("system_check", "assistant", "operator", "importer")
VERDICTS = ("accepted_as_is", "needs_repair", "rejected")
SEVERITIES = ("blocker", "major", "minor")
CORRECTION_METHODS = ("manual", "automatic", "assistant", "external")
RESOLVED = ("yes", "no", "unknown")

# §5 defect taxonomy v1, code -> what an automatic actor may do with it.
DEFECT_CODES = {
    "segmentation_boundary": "suggest",
    "source_face_retained": "suggest",
    "identity_drift": None,
    "body_hands_wrong": None,
    "reference_quality": None,
    "text_artifact": "suggest",
    "caption_error": "suggest",
    "script_adherence": "flag",
    "pronunciation": "operator_only",
    "lip_sync": "operator_only",
    "audio_continuity": "flag",
    "scene_continuity": None,
    "stitching": "flag",
    "truncation": "determine",
    "cta_ending": None,
    "provider_refusal": "determine",
    "resolution_quality": "suggest",
    "other": None,
}
OPERATOR_ONLY_DEFECTS = frozenset(c for c, rule in DEFECT_CODES.items() if rule == "operator_only")

# Allowed automatic (check_kind -> claims). No other pair may be written.
CHECK_CLAIMS = {
    "asr": ("transcript_match", "transcript_mismatch"),
    "frame_sample": ("defect_observed", "no_defect_observed_in_sample"),
    "face_similarity": ("suggest_source_face_retained",),
    "duration_probe": ("short_result", "full_length"),
}

RULE_KINDS = ("operator_constraint", "empirical_guidance", "machine_hypothesis")
RULE_ACTIONS = ("forbid", "require", "avoid", "prefer", "warn")
RULE_CONFIDENCE = ("operator_asserted", "low", "medium", "high")
AUTHOR_KINDS = ("operator", "claude", "importer")
REVIEW_STATUSES = ("proposed", "approved", "rejected", "retired")
EVIDENCE_STANCES = ("supports", "contradicts", "context")

DECISION_MODES = ("shadow", "advisory")
DECISION_REQUESTERS = ("assistant_turn", "new_run_form", "offline_eval")

IMPORT_OUTCOMES = ("finished", "failed", "aborted")
# Every row an import creates is listed under one of these kinds, so invalidating
# the batch reaches it (InvalidSet reads exactly these names).
IMPORT_MEMBER_KINDS = ("case", "case_version", "artifact", "attempt", "submission", "observation", "event",
                       "rule_version", "rule_evidence", "eval_membership")
INVALIDATION_TARGETS = ("import_batch", "case", "attempt", "event", "rule_version", "artifact")
LINK_KINDS = ("stitch", "range", "audio", "assistant")  # publication links (AICP-299 §3.2)
CAPTURE_OUTCOMES = ("appended", "historical", "duplicate", "stale", "withdrawn")
EVAL_SPLITS = ("holdout", "development")

# §9.1 canonical applicability dimensions, highest specificity first; the
# remaining dimensions share the lowest weight.
APPLICABILITY_RANKED = ("subject_id", "project_id", "model_key", "model_family", "operation")
APPLICABILITY_OTHER = ("language", "resolution", "segment_sec_band", "mute_source", "generate_audio")
APPLICABILITY_DIMENSIONS = APPLICABILITY_RANKED + APPLICABILITY_OTHER
SEGMENT_SEC_BANDS = ("[0,10)", "[10,15)", "[15,30]", ">30", "unknown")


def check_sql(column: str, values) -> str:
    return f"{column} IN ({', '.join(repr(v) for v in values)})"


def segment_sec_band(seconds) -> str:
    if seconds is None:
        return "unknown"
    if seconds < 10:
        return "[0,10)"
    if seconds < 15:
        return "[10,15)"
    if seconds <= 30:
        return "[15,30]"
    return ">30"


def canonical_applicability(raw) -> dict:
    """Normalize to ``{dim: sorted non-empty list}``; absent = wildcard.

    Raises ValueError for unknown dimensions or empty sets so two spellings of
    one scope cannot exist side by side (design §9.1).
    """
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError("applicability must be an object")
    out = {}
    for dim, value in raw.items():
        if dim not in APPLICABILITY_DIMENSIONS:
            raise ValueError(f"unknown applicability dimension: {dim}")
        values = value if isinstance(value, (list, tuple, set, frozenset)) else [value]
        if not values:
            raise ValueError(f"applicability {dim} must not be empty")
        for item in values:
            if isinstance(item, bool):
                if dim not in ("mute_source", "generate_audio"):
                    raise ValueError(f"applicability {dim} must be text")
            elif not isinstance(item, str) or not item:
                raise ValueError(f"applicability {dim} must be non-empty text")
        if dim == "segment_sec_band" and any(v not in SEGMENT_SEC_BANDS for v in values):
            raise ValueError("unknown segment_sec_band")
        out[dim] = sorted(set(values), key=lambda v: (str(type(v)), v))
    return dict(sorted(out.items()))


def specificity(applicability: dict) -> tuple:
    """Comparable tuple, larger = more specific (design §9.1)."""
    return tuple(int(d in applicability) for d in APPLICABILITY_RANKED) + (
        sum(d in applicability for d in APPLICABILITY_OTHER),)


def validate_auto_assessment(check_kind, claim, defect_code=None) -> None:
    if check_kind not in CHECK_CLAIMS or claim not in CHECK_CLAIMS[check_kind]:
        raise ValueError(f"automatic check {check_kind!r} may not claim {claim!r}")
    if defect_code in OPERATOR_ONLY_DEFECTS:
        raise ValueError(f"{defect_code} can only be judged by an operator")
