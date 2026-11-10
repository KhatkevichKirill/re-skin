"""``python -m app.autopilot hold on|off|status``, ``tick`` (one pass) and
``notify status|pass|test`` (AICP-374 chat notifications).

Run inside the re-skin-api container. ``hold on`` returns only once no
autopilot admission is in progress (the gate barrier, design §9).
"""

from __future__ import annotations

import argparse
import json
import sys


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m app.autopilot")
    sub = ap.add_subparsers(dest="cmd", required=True)
    h = sub.add_parser("hold")
    h.add_argument("action", choices=("on", "off", "status"))
    h.add_argument("--note", default=None)
    sub.add_parser("tick", help="run one autopilot pass now (needs AUTOPILOT_ENABLED)")
    nt = sub.add_parser("notify", help="chat notifications: status, one pass, or a test message")
    nt.add_argument("action", choices=("status", "pass", "test"))
    args = ap.parse_args(argv)

    from ..db import SessionLocal
    from . import gate

    if args.cmd == "hold":
        db = SessionLocal()
        try:
            if args.action == "on":
                gate.hold_on(db, by="cli", note=args.note)
                print("held — safe to drain")
            elif args.action == "off":
                gate.hold_off(db)
                print("released")
            print(json.dumps(gate.status(db)))
        finally:
            db.close()
        return 0
    if args.cmd == "notify":
        from . import notify

        if args.action == "status":
            db = SessionLocal()
            try:
                print(json.dumps(notify.status(db), indent=1))
            finally:
                db.close()
            return 0
        problem = notify.config_problem()
        if problem:
            print(f"notifications {problem}")
            return 1
        n = notify.make_notifier()
        if args.action == "test":
            mid = n.channel.post(n.chat_id, "re-skin: test message from the autopilot notifier")
            print(json.dumps({"sent": True, "message_id": mid}))
            return 0
        s = n.run_pass()
        print(json.dumps(s.__dict__))
        return 0
    if args.cmd == "tick":
        from . import actions  # noqa: F401 — registers the handlers
        from .core import run_pass

        print(json.dumps({"ticked": run_pass()}))
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
