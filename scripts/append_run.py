#!/usr/bin/env python3
"""Append one row to results/runs.jsonl under an exclusive lock.

Usage: append_run.py '<json object>'   (or pipe JSON objects, one per line, on stdin)
Required keys: run_id, agent, candidate, metric, value, split. Missing optional keys
(time, commit, pod, CI, n) are filled with defaults.
"""
import fcntl, json, os, subprocess, sys
from datetime import datetime, timezone

PATH = os.path.expanduser("~/Code/inkling-abl/results/runs.jsonl")
REQUIRED = ("run_id", "agent", "candidate", "metric", "value", "split")


def commit() -> str:
    r = subprocess.run(["git", "-C", os.path.dirname(os.path.dirname(PATH)), "rev-parse", "--short", "HEAD"],
                       capture_output=True, text=True)
    return r.stdout.strip() or "none"


def normalise(row: dict) -> dict:
    missing = [k for k in REQUIRED if k not in row]
    if missing:
        raise SystemExit(f"missing keys: {missing}")
    row.setdefault("time", datetime.now(timezone.utc).isoformat(timespec="seconds"))
    row.setdefault("commit", commit())
    for k in ("pod", "CI", "n"):
        row.setdefault(k, None)
    return row


def main() -> None:
    lines = sys.argv[1:] or [l for l in sys.stdin if l.strip()]
    rows = [normalise(json.loads(l)) for l in lines]
    with open(PATH, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        for r in rows:
            f.write(json.dumps(r) + "\n")
        f.flush()
        fcntl.flock(f, fcntl.LOCK_UN)
    print(f"appended {len(rows)}")


if __name__ == "__main__":
    main()
