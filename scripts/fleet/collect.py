"""Finish a job whose Mac worker died while the pod job ran on: fetch outputs, append rows, mark the claim done.
  python3 collect.py <pod> <cand_id> <tag> <stage> [--gpu i] [--finalist]
Refuses unless /srv/evals/<cand>_<tag>/fleet_done.json exists on the pod."""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.expanduser("~/Code/inkling-abl/harness"))
import eval_worker as W  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("pod")
ap.add_argument("cand_id")
ap.add_argument("tag")
ap.add_argument("stage", choices=("screen", "full"))
ap.add_argument("--gpu", type=int, default=None)
ap.add_argument("--finalist", action="store_true")
ap.add_argument("--think-cap", type=int, default=1536)
a = ap.parse_args()
w = W.Worker(a.pod, 0, a.gpu)
done = W.pod_exec(a.pod, f"cat /srv/evals/{a.cand_id}_{a.tag}/fleet_done.json 2>/dev/null || echo MISSING")
if "MISSING" in done:
    raise SystemExit("pod job has not finished (no fleet_done.json)")
fam = [json.loads(l).get("family") for l in open(W.QUEUE) if f'"cand_id": "{a.cand_id}"' in l]
job = {"cand_id": a.cand_id, "stage": a.stage, "finalist": a.finalist, "family": fam[-1] if fam else None,
       "think_cap": a.think_cap}
dest = w.fetch_outputs(a.cand_id, a.tag)
summary = json.load(open(f"{dest}/metrics.json"))
sha = (summary.get("prep") or {}).get("sha256") or ""
n = w.append_rows(job, a.tag, summary, sha)
w.mark("done", job, tag=a.tag, sec=summary.get("total_sec"), rows=n, note="collected after worker stop")
print(f"collected {n} rows into {dest}; pod total_sec {summary.get('total_sec')}; timing {summary.get('timing')}")
