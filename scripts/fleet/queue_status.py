"""Eval-queue status: per stage, how many jobs are done / running (claimed) / waiting (ready) / blocked (not ready).
  python3 queue_status.py [--list]"""
import json
import os
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.expanduser("~/Code/inkling-abl/harness"))
import eval_worker as W  # noqa: E402

jobs: dict[str, dict] = {}
for l in open(W.QUEUE):
    if l.strip():
        j = json.loads(l)
        if "cand_id" in j and ("path" in j or j["cand_id"] == "orig"):
            jobs[W.Worker.job_key(j)] = j  # last row wins
state: dict[str, str] = {}
for l in open(W.CLAIMS):
    try:
        c = json.loads(l)
    except json.JSONDecodeError:
        continue
    if c.get("kind") != "eval" or "key" not in c:
        continue
    if "done" in c:
        state[c["key"]] = "skipped" if c.get("skipped") else "done"
    elif "release" in c:
        state[c["key"]] = "open"
    elif "claim" in c and time.time() - datetime.fromisoformat(c["time"]).timestamp() < W.STALE_SEC:
        state[c["key"]] = "running:" + c["worker"].split(":")[-1]
counts: dict[tuple[str, str], int] = {}
rows = []
for k, j in jobs.items():
    s = state.get(k, "open")
    if s == "open":
        s = "waiting" if W.ready(j) else "blocked"
    stage = j.get("stage", "full") + ("+finalist" if j.get("finalist") else "")
    bucket = s.split(":")[0]
    counts[(stage, bucket)] = counts.get((stage, bucket), 0) + 1
    rows.append((stage, s, j.get("priority"), j.get("node", "fleet"), j["cand_id"]))
for (stage, bucket), n in sorted(counts.items()):
    print(f"{stage:16s} {bucket:9s} {n}")
if "--list" in sys.argv:
    for r in sorted(rows):
        if r[1] not in ("done", "skipped"):
            print(*r)
