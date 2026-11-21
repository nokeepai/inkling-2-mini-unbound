"""Turn downloaded fidelity result files into runs.jsonl rows via append_run.py (stdin, one locked append).
Usage: python3 push_runs.py <kl_results.jsonl> <cap_results.jsonl|-> [--only-tags a,b]"""
import json
import os
import subprocess
import sys
from datetime import datetime, timezone

APPEND = "/Users/nikshepsvn/Code/inkling-abl/scripts/append_run.py"
POD = os.environ.get("KL_POD", "ink-research-2")


def rid(*parts):
    return "fid-" + "-".join(str(p) for p in parts) + "-" + datetime.now(timezone.utc).strftime("%H%M%S")


def kl_rows(path):
    rows = []
    for l in open(path):
        r = json.loads(l)
        base = {"agent": "fidelity", "candidate": r["cand"], "split": "val", "pod": POD, "n": r["n"],
                "engine": "hf_w4a16" if not r["act_quant"] else "hf_w4a4emu", "tag": r["tag"], "bs": r["bs"], "order": r["order"], "act_quant": r["act_quant"],
                "n_pos": r["n_pos"], "sec": r["total_sec"]}
        rows.append({**base, "run_id": rid(r["cand"], r["tag"], "kl"), "metric": "kl_256", "value": r["kl"],
                     "CI": r["kl_ci"]})
        rows.append({**base, "run_id": rid(r["cand"], r["tag"], "top1"), "metric": "top1_256", "value": r["top1"],
                     "CI": r["top1_ci"]})
    return rows


def cap_rows(path):
    rows = []
    for l in open(path):
        r = json.loads(l)
        extra = {k: r[k] for k in ("finish_rate", "think_cap_hits", "mean_think_tokens") if k in r}
        rows.append({"run_id": rid(r["cand"], r["metric"]), "agent": "fidelity", "candidate": r["cand"],
                     "metric": r["metric"], "value": r["value"], "CI": r["CI"], "n": r["n"],
                     "split": "val", "pod": POD, "engine": r.get("engine", "hf"), "act_quant": r["act_quant"], **extra})
    return rows


rows = kl_rows(sys.argv[1]) if sys.argv[1] != "-" else []
if len(sys.argv) > 2 and sys.argv[2] != "-":
    rows += cap_rows(sys.argv[2])
subprocess.run([sys.executable, APPEND], input="\n".join(json.dumps(r) for r in rows), text=True, check=True)
