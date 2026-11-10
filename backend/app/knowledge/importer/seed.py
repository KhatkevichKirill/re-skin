"""Seed guidance loader: ``knowledge/seed/*.yaml`` → unreviewed rule versions (design §2.3, §4.6).

Every version is written with ``author_kind="importer"``; the YAML's
``author_label`` stays the human attribution. Versions are created without a
review: approval is a separate operator action. Optional ``evidence`` entries
name real runs; a run with no current case is skipped with a report line, and a
holdout citation is refused by the store (the whole version is then reported,
never written without it).
"""
from __future__ import annotations

import glob
import os
from datetime import datetime
from typing import Optional

import yaml

from .. import store
from .context import ImportContext

SEED_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "seed"))
RULE_FIELDS = {"slug", "kind", "statement", "rationale", "applicability", "recommendation", "confidence", "shareable",
               "effective_from", "author_label", "provenance", "evidence_gaps", "evidence"}
REQUIRED = ("slug", "kind", "statement", "recommendation", "confidence", "effective_from")


def _when(value) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    else:
        raise ValueError("effective_from must be an ISO-8601 timestamp")
    if parsed.tzinfo is None:
        raise ValueError("effective_from must carry a timezone (e.g. a trailing Z)")
    return parsed


def load_seed_files(seed_dir: Optional[str] = None) -> list[tuple[str, dict]]:
    """Every rule in ``seed_dir``; a missing or empty directory is an error, never
    "no rules" (a mistyped --seed-dir must not look like a successful seed)."""
    directory = seed_dir or SEED_DIR
    paths = sorted(glob.glob(os.path.join(directory, "*.yaml")))
    if not os.path.isdir(directory) or not paths:
        raise ValueError(f"seed directory {directory!r} is missing or has no *.yaml files")
    out = []
    for path in paths:
        with open(path, encoding="utf-8") as fh:
            doc = yaml.safe_load(fh) or {}
        rules = doc.get("rules") if isinstance(doc, dict) else None
        if not isinstance(rules, list):
            raise ValueError(f"{os.path.basename(path)}: top level must be a mapping with a 'rules' list")
        out.extend((os.path.basename(path), r) for r in rules)
    return out


def import_seed(ctx: ImportContext, seed_dir: Optional[str] = None) -> None:
    stats = ctx.report.source("seed")
    try:
        rules = load_seed_files(seed_dir)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        ctx.report.note("failures", source="seed", error=f"{type(exc).__name__}: {exc}")
        stats["failed"] += 1
        return
    for filename, rule in rules:
        stats["rows"] += 1
        slug = rule.get("slug") if isinstance(rule, dict) else None
        try:
            if not isinstance(rule, dict) or set(rule) - RULE_FIELDS or any(rule.get(k) in (None, "") for k in REQUIRED):
                raise ValueError(f"malformed rule: unknown {sorted(set(rule) - RULE_FIELDS) if isinstance(rule, dict) else ''}"
                                 f" or missing one of {REQUIRED}")
            effective_from = _when(rule["effective_from"])
            if not isinstance(rule.get("shareable", False), bool):
                raise ValueError("shareable must be a YAML boolean (true/false), not a string")
        except ValueError as exc:
            stats["failed"] += 1
            ctx.report.note("failures", source="seed", file=filename, slug=slug, error=str(exc))
            continue
        evidence = []
        for item in rule.get("evidence") or []:
            run_id = item.get("run_id") if isinstance(item, dict) else None
            case = store.current_case(ctx.session, store.run_case_key(run_id)) if run_id else None
            if case is None:
                ctx.report.note("skipped", what="seed evidence names a run with no current case", slug=slug,
                                run_id=run_id)
                continue
            evidence.append({"case_id": case.id, "stance": item.get("stance"), "note": item.get("note")})
        body = {k: rule.get(k) for k in sorted(RULE_FIELDS - {"evidence"})}
        key, done = ctx.request_key("seed:" + slug[:60] + ":" + store.canonical_sha256([body, evidence])[:40])
        if done:
            ctx.report.note("duplicates", what="seed rule version unchanged", slug=slug)
            continue
        batch_id = ctx.ensure_batch()
        sp = ctx.session.begin_nested()
        try:
            version = store.add_rule_version(
                ctx.session, ctx.scope, slug=slug, idempotency_key=key, kind=rule["kind"],
                statement=rule["statement"], recommendation=rule["recommendation"], confidence=rule["confidence"],
                author_kind="importer", effective_from=effective_from, rationale=rule.get("rationale"),
                applicability=rule.get("applicability"), shareable=rule.get("shareable", False),
                author_label=rule.get("author_label"),
                provenance={**(rule.get("provenance") or {}), "seed_file": filename},
                evidence_gaps=rule.get("evidence_gaps") or [], evidence=evidence,
                import_batch_id=batch_id)
            ctx.register_rule_version(version)
            sp.commit()
        except store.BatchWithdrawn:
            raise  # the batch was withdrawn mid-import: abort the pass
        except store.KnowledgeError as exc:
            sp.rollback()
            ctx.report.note("refused", source="seed", slug=slug, error=str(exc))
            continue
        if not ctx.dry_run:
            ctx.heartbeat("seed")
            ctx.session.commit()
