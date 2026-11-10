"""Knowledge code lists, applicability canon and scope — no database needed (AICP-295)."""
from __future__ import annotations

import json
import os
import sys
import uuid

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

MANIFEST = os.path.join(os.path.dirname(__file__), "..", "..", "docs", "knowledge", "holdout-manifest-v1.json")


def test_applicability_is_canonical():
    from app.knowledge import taxonomy as t

    assert t.canonical_applicability({"model_key": ["seedance-2-5", "seedance"], "operation": "face_swap"}) == {
        "model_key": ["seedance", "seedance-2-5"], "operation": ["face_swap"]}
    assert t.canonical_applicability(None) == {}
    for bad in ({"colour": "red"}, {"model_key": []}, {"segment_sec_band": "[0,5)"}, {"model_key": [""]},
                {"language": [True]}):
        with pytest.raises(ValueError):
            t.canonical_applicability(bad)
    assert t.canonical_applicability({"mute_source": True}) == {"mute_source": [True]}


def test_specificity_orders_subject_over_project_over_model():
    from app.knowledge import taxonomy as t

    subject = t.specificity({"subject_id": ["s"]})
    project = t.specificity({"project_id": ["p"], "model_key": ["m"], "language": ["es"]})
    model = t.specificity({"model_key": ["m"], "operation": ["face_swap"], "language": ["es"]})
    assert subject > project > model > t.specificity({})


def test_segment_bands_are_half_open():
    from app.knowledge.taxonomy import segment_sec_band

    assert [segment_sec_band(x) for x in (None, 0, 9.99, 10, 14.99, 15, 30, 30.01)] == [
        "unknown", "[0,10)", "[0,10)", "[10,15)", "[10,15)", "[15,30]", "[15,30]", ">30"]


def test_automatic_claims_are_closed_and_never_judge_pronunciation_or_lip_sync():
    from app.knowledge.taxonomy import validate_auto_assessment

    validate_auto_assessment("duration_probe", "short_result", "truncation")
    for args in (("asr", "no_defect_observed_in_sample", None), ("frame_sample", "defect_observed", "lip_sync"),
                 ("asr", "transcript_match", "pronunciation"), ("ocr", "defect_observed", None)):
        with pytest.raises(ValueError):
            validate_auto_assessment(*args)


def test_committed_manifest_hash_matches_the_loader():
    from app.knowledge.store import manifest_sha256

    body = json.load(open(MANIFEST))
    assert manifest_sha256(body) == body["manifest_sha256"] == (
        "ea5f5d52ce1d5b5d8485fb5d29116090d9cf337113ca35405f843bf194d2766f")


def test_scope_intersection_and_wildcard():
    from app.assistant_access import ProjectAllowlist
    from app.knowledge.access import KnowledgeScope

    a, b = str(uuid.uuid4()), str(uuid.uuid4())
    assert KnowledgeScope.build(ProjectAllowlist(all_projects=True), "*").everything
    narrow = KnowledgeScope.build(ProjectAllowlist(f"{a},{b}"), "*")
    assert narrow.allows_project(b) and not narrow.allows_project(str(uuid.uuid4())) and not narrow.everything
    assert not KnowledgeScope.build(ProjectAllowlist(f"{a}"), "").allows_project(a)
    vis = KnowledgeScope.build(ProjectAllowlist(all_projects=True), a)
    assert vis.rule_visibility(shareable=False, scope_project_ids=None, evidence_project_ids=[a]) == "full"
    assert vis.rule_visibility(shareable=False, scope_project_ids=None, evidence_project_ids=[a, b]) is None
    assert vis.rule_visibility(shareable=True, scope_project_ids=[b], evidence_project_ids=[]) == "shareable"


def test_settings_default_off():
    from app.config import Settings

    assert Settings.KNOWLEDGE_ENABLED is False or os.getenv("KNOWLEDGE_ENABLED")
    assert Settings.KNOWLEDGE_ADVISOR_MODE in ("off",) or os.getenv("KNOWLEDGE_ADVISOR_MODE")


def test_holdout_manifest_ships_in_the_image():
    """The importer's frozen manifest must reach /app/docs in the image (prod tick failed without it)."""
    import os

    from app.knowledge.importer import context

    root = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
    assert os.path.isfile(context.MANIFEST_PATH)
    dockerfile = open(os.path.join(root, "backend", "Dockerfile")).read()
    assert "COPY docs/knowledge/holdout-manifest-v1.json ./docs/knowledge/" in dockerfile
    ignore = open(os.path.join(root, ".dockerignore")).read().splitlines()
    assert "!docs/knowledge/holdout-manifest-v1.json" in ignore
    assert ignore.index("!docs/knowledge/holdout-manifest-v1.json") > ignore.index("docs/")
    # The image layout (/app/app/knowledge/importer -> /app/docs) is one of the candidates.
    image_here = "/app/app/knowledge/importer"
    assert os.path.normpath(os.path.join(image_here, "..", "..", "..", context._MANIFEST_REL)) == \
        "/app/docs/knowledge/holdout-manifest-v1.json"
