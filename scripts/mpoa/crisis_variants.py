"""Crisis-protected variants of c1t013 and c1h_svd1_s2: ridge-orthogonalise Q against the crisis atom only (lam 0.3, 1).
Everything else identical. Enqueued FULL at priority 9.6 with think_cap 4096."""
import fcntl
import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(__file__))
import mpoa_lib as M

R = M.R
PARENTS = ["c1t013", "c1h_svd1_s2"]
rows = []
for parent in PARENTS:
    p0 = json.load(open(f"{R}/methods/mpoa/cands/{parent}/spec.json"))["params"]
    for lam, tag in ((0.3, "03"), (1.0, "1")):
        p = dict(p0)
        p["protect"] = {"crisis": lam}
        p["protect_k"] = 2
        cid = f"{parent}_cr{tag}"
        out = M.write_candidate(cid, "c1", p, parent=parent)
        b = out["diag"]["basis"]
        Q0, _ = M.basis(p0)
        Q1, _ = M.basis(p)
        import torch
        pc = [round(float(x), 3) for x in torch.linalg.svdvals(Q0.T @ Q1)]
        print(cid, "crisis overlap", b.get("protect_overlap_before"), "->", b.get("protect_overlap_after"), "principal cos vs parent", pc, flush=True)
        rows.append({"cand_id": cid, "family": "c1", "path": f"methods/mpoa/cands/{cid}", "priority": 9.6, "stage": "full",
                     "node": "fleet", "think_cap": 4096, "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                     "by": "mpoa", "note": "crisis-protected variant (lead 15:1xZ)"})
with open(f"{R}/state/queue.jsonl", "a") as f:
    fcntl.flock(f, fcntl.LOCK_EX)
    for r in rows:
        f.write(json.dumps(r) + "\n")
    fcntl.flock(f, fcntl.LOCK_UN)
print("enqueued", [r["cand_id"] for r in rows])
