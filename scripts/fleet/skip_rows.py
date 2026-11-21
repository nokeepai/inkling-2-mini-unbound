"""Mark queue jobs as skipped so eval workers never claim them (no runs.jsonl rows): appends one
{"done": cand_id, "kind": "eval", "key", "worker": "skip", "skipped": true, "reason"} row per open job under flock.
  python3 skip_rows.py --reason "..." --priority 5 <cand_id> [<cand_id> ...]
Only jobs whose priority equals --priority (if given) and that are not already done or currently claimed."""
import argparse
import fcntl
import json
import os
import sys

sys.path.insert(0, os.path.expanduser("~/Code/inkling-abl/harness"))
import eval_worker as W  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("cands", nargs="+")
ap.add_argument("--reason", required=True)
ap.add_argument("--priority", type=float, default=None)
a = ap.parse_args()
want = set(a.cands)
jobs = [json.loads(l) for l in open(W.QUEUE) if l.strip()]
jobs = [j for j in jobs if j.get("cand_id") in want and (a.priority is None or float(j.get("priority", 0)) == a.priority)]
with open(W.CLAIMS, "a+") as f:
    fcntl.flock(f, fcntl.LOCK_EX)
    f.seek(0)
    seen: dict[str, str] = {}
    for l in f:
        try:
            c = json.loads(l)
        except json.JSONDecodeError:
            continue
        if c.get("kind") == "eval" and "key" in c:
            seen[c["key"]] = "done" if "done" in c else ("claimed" if "claim" in c else "open")
    for j in jobs:
        k = W.Worker.job_key(j)
        if seen.get(k) in ("done", "claimed"):
            print(f"{k}: already {seen[k]}")
            continue
        f.write(json.dumps({"done": j["cand_id"], "kind": "eval", "key": k, "worker": "skip", "skipped": True,
                            "reason": a.reason, "time": W.now()}) + "\n")
        seen[k] = "done"
        print(f"{k}: skipped")
    fcntl.flock(f, fcntl.LOCK_UN)
