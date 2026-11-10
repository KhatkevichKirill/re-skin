#!/usr/bin/env python3
"""Deterministic historical holdout split for the production knowledge base (AICP-294).

Splits by SOURCE LINEAGE GROUP, not by Run: sibling Runs of one source (and
projects derived from another project's Run via source_run_id, or re-uploads of
the byte-identical normalized source) share one group, so a holdout case can
never be "explained" by a development rule built from its own sibling.

Input (pipe-separated, one project per line, produced read-only from prod):
  projects.txt : id|project_type|parent_project_id_via_source_run|source_path|created|name
  src_sha.txt  : sha256sum output "<sha>  <source_path>"
Pinned-development projects are those already discussed in Linear/wiki (they
have already influenced guidance, so they cannot be unseen evaluation data).

families.txt (docs/knowledge/lineage-families-v1.txt) adds curated name families:
re-uploads, language variants and re-encodes have different bytes, so sha256 and
source_run_id alone leave sibling creatives on both sides of the split.

Usage: knowledge_holdout_split.py projects.txt src_sha.txt pinned.txt families.txt > manifest.json
"""
import hashlib
import re
import json
import math
import sys
from collections import defaultdict

SALT = "aicp-294-holdout-v1"
HOLDOUT_FRACTION = 0.25


def _norm(name):
    return re.sub(r"[\s_\-.]+", "", name.lower())


def main(projects_path, sha_path, pinned_path, families_path):
    sha_by_path = {}
    for line in open(sha_path):
        sha, path = line.rstrip("\n").split(None, 1)
        sha_by_path[path.strip()] = sha
    projects = {}
    for line in open(projects_path):
        pid, ptype, parent, path, created, name = line.rstrip("\n").split("|", 5)
        projects[pid] = dict(type=ptype, parent=parent or None, sha=sha_by_path[path], created=created, name=name)
    pinned_raw = [l.split("#")[0].strip() for l in open(pinned_path)]
    pinned = {p for p in pinned_raw if p}
    families = []
    for line in open(families_path):
        line = line.split("#")[0].strip()
        if line:
            families.append([_norm(x) for x in line.split("|") if x.strip()])

    parent = {p: p for p in projects}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    by_sha = defaultdict(list)
    for pid, p in projects.items():
        by_sha[p["sha"]].append(pid)
        if p["parent"] and p["parent"] in projects:
            union(pid, p["parent"])
    for prefixes in families:
        members = [pid for pid, p in projects.items()
                   if p["name"] and any(_norm(p["name"]).startswith(px) for px in prefixes)]
        for other in members[1:]:
            union(members[0], other)
    for pids in by_sha.values():
        for other in pids[1:]:
            union(pids[0], other)

    groups = defaultdict(list)
    for pid in projects:
        groups[find(pid)].append(pid)

    def stratum(members):
        earliest = min(members, key=lambda m: (projects[m]["created"], m))
        return projects[earliest]["type"]

    strata = defaultdict(list)
    for gkey, members in groups.items():
        strata[stratum(members)].append(gkey)

    out_groups = []
    for st, gkeys in sorted(strata.items()):
        eligible = [g for g in gkeys if not (set(groups[g]) & pinned)]
        n_hold = math.ceil(HOLDOUT_FRACTION * len(gkeys)) if eligible else 0
        ranked = sorted(eligible, key=lambda g: hashlib.sha256(f"{SALT}:{g}".encode()).hexdigest())
        hold = set(ranked[:n_hold])
        for g in sorted(gkeys):
            members = sorted(groups[g])
            if g in hold:
                split, reason = "holdout", "hash_rank"
            elif set(members) & pinned:
                split, reason = "development", "pinned_already_discussed"
            else:
                split, reason = "development", "hash_rank"
            out_groups.append(dict(group_key=g, stratum=st, split=split, reason=reason, project_ids=members))

    body = dict(
        version=1, salt=SALT, holdout_fraction=HOLDOUT_FRACTION, unit="source_lineage_group",
        lineage_rules=["identical normalized source sha256", "VideoProject.source_run_id -> parent project",
                       "curated name family (lineage-families-v1.txt)"],
        known_leakage_gaps=["unnamed projects and unlisted name variants are not linked beyond sha256/source_run_id",
                            "the same on-camera person across DIFFERENT creatives is not linked (no identity grouping)"],
        groups=out_groups,
    )
    canon = json.dumps(body, sort_keys=True, separators=(",", ":"))
    body["manifest_sha256"] = hashlib.sha256(canon.encode()).hexdigest()
    json.dump(body, sys.stdout, indent=1, sort_keys=True)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main(*sys.argv[1:5])
