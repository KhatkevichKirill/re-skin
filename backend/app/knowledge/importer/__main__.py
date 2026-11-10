"""``python -m app.knowledge.importer --dry-run|--apply [--source NAME ...]`` → JSON report on stdout."""
from __future__ import annotations

import argparse
import json
import sys

from .runner import STEPS, run_import


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m app.knowledge.importer")
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="compute the full report, then roll back")
    mode.add_argument("--apply", action="store_true", help="write one import batch")
    p.add_argument("--source", action="append", choices=STEPS, help="limit to these steps (repeatable)")
    p.add_argument("--limit", type=int, default=None, help="max source rows this pass (default: all)")
    p.add_argument("--from-start", action="store_true", help="ignore cursors (rerun after an invalidation)")
    p.add_argument("--hash", action="store_true", help="sha256 local media the process can read")
    p.add_argument("--no-seed", action="store_true", help="skip knowledge/seed/*.yaml")
    p.add_argument("--code-revision", default=None)
    args = p.parse_args(argv)

    from ...db import SessionLocal
    from ..store import sha256_file

    report = run_import(SessionLocal, apply=args.apply, sources=args.source, limit=args.limit,
                        from_start=args.from_start, hasher=sha256_file if args.hash else None,
                        revision=args.code_revision, seed_dir=False if args.no_seed else None)
    json.dump(report, sys.stdout, indent=2, sort_keys=True, default=str)
    sys.stdout.write("\n")
    return 0 if report.get("outcome") == "finished" else 1


if __name__ == "__main__":
    raise SystemExit(main())
