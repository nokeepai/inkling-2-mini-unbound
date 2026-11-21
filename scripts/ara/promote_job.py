"""Turn an already-fitted pod job (e.g. a proxy-sweep job) into a c3 candidate and enqueue it for screen.
  promote_job.py <pod_job_id> <cand_id> [--fit]   (--fit: run fit_job first, using params already in the job dir)
Not an Optuna trial: scored like any candidate, reported in PARETO.md by hand."""
import json
import os
import sys

args = sys.argv[1:]
sys.argv = [sys.argv[0], "1", "1"]
sys.path.insert(0, os.path.dirname(__file__))
import search as S  # noqa: E402

job, cid = args[0], args[1]
d = f"{S.R}/methods/ara/cands/{cid}"
os.makedirs(d, exist_ok=True)
if "--fit" in args:
    S.sh(S.LIUM + ["exec", S.POD, f"bash /root/ara/fit_job.sh {job}"], timeout=1200)
for f in ("lowrank.safetensors", "fit.json", "verify.json", "params.json"):
    S.sh(S.LIUM + ["scp", S.POD, "--download", f"/root/ara/jobs/{job}/{f}", f"{d}/{f}"])
ver = json.load(open(f"{d}/verify.json"))
assert ver["exact_cpu"] == ver["n"], ver
p = json.load(open(f"{d}/params.json"))
os.remove(f"{d}/params.json")
L = json.load(open(f"{d}/fit.json"))["layers"]
diag = {"rel_dW_mean": round(sum(x["rel_dW"] for x in L) / len(L), 4),
        "mean_gap_ratio": round(sum(x["mean_gap_ratio"] for x in L) / len(L), 4),
        "preserve_mean": round(sum(x["preserve"] for x in L) / len(L), 5),
        "rownorm_ratio": [round(ver["rownorm_ratio_min"], 4), round(ver["rownorm_ratio_max"], 4)],
        "build_exact": f"{ver['exact_cpu']}/{ver['n']}", "tensor_sha256": ver["tensor_sha256"], "pod_job": job}
spec = {"cand_id": cid, "family": "c3", "parent": None, "created": S.datetime.now(S.timezone.utc).isoformat(timespec="seconds"),
        "agent": "ara", "params": p, "kind": "weights", "diag": diag}
json.dump(spec, open(f"{d}/spec.json", "w"), indent=1)
S.enqueue(cid, "screen", 7.0)
print("enqueued", cid, "from", job, diag["rel_dW_mean"])
