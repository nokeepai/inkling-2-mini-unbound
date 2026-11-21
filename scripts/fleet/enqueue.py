"""Append one row to R/state/queue.jsonl under flock.
  python3 enqueue.py <cand_id> <path rel. to R> [--stage screen|full] [--finalist] [--priority N] [--family F] [--note T]
"""
import argparse
import fcntl
import json
import os
from datetime import datetime, timezone

R = os.path.expanduser("~/Code/inkling-abl")
ap = argparse.ArgumentParser()
ap.add_argument("cand_id")
ap.add_argument("path")
ap.add_argument("--stage", choices=("screen", "full"), default="full")
ap.add_argument("--finalist", action="store_true")
ap.add_argument("--priority", type=float, default=0)
ap.add_argument("--family", default=None)
ap.add_argument("--note", default="")
ap.add_argument("--by", default="fleet")
ap.add_argument("--node", default=None, help='pod class: "fleet" (default when absent) or e.g. "eval-2"')
ap.add_argument("--think-cap", type=int, default=1536, help="thinking cap (1536 inner loop; 4096 hygiene reruns)")
a = ap.parse_args()
row = {"cand_id": a.cand_id, "priority": a.priority, "stage": a.stage,
       "time": datetime.now(timezone.utc).isoformat(timespec="seconds"), "by": a.by}
if a.cand_id == "orig":  # baseline: no tensors, the server already serves it; the KL worker must skip it
    row.update({"family": "baselines", "kind": "eval_only"})
else:
    assert os.path.isdir(os.path.join(R, a.path)), f"no such candidate dir: {a.path}"
    row.update({"family": a.family or a.path.split("/")[1], "path": a.path})
if a.node:
    row["node"] = a.node
if a.think_cap != 1536:
    row["think_cap"] = a.think_cap
if a.finalist:
    row["finalist"] = True
if a.note:
    row["note"] = a.note
with open(f"{R}/state/queue.jsonl", "a") as f:
    fcntl.flock(f, fcntl.LOCK_EX)
    f.write(json.dumps(row) + "\n")
    fcntl.flock(f, fcntl.LOCK_UN)
print(json.dumps(row))
