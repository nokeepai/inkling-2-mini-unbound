"""Table of c3 candidates: params, fit diag, fleet/KL metrics, pod proxy (fetched by fetch_proxy.sh to methods/ara/proxy/)."""
import glob
import json
import os
import sys

R = os.path.expanduser("~/Code/inkling-abl")
sys.path.insert(0, f"{R}/scripts/mpoa")
import metrics as MET  # noqa: E402

allm = MET.collect(["c3t"], exclude_pods=("ink-eval-2",))
for d in sorted(glob.glob(f"{R}/methods/ara/cands/c3t*")):
    cid = os.path.basename(d)
    if not os.path.exists(f"{d}/spec.json"):
        continue
    s = json.load(open(f"{d}/spec.json"))
    p, dg = s["params"], s["diag"]
    px = {}
    pf = f"{R}/methods/ara/proxy/{cid}.json"
    if os.path.exists(pf):
        px = json.load(open(pf))
    m = allm.get(cid, {})
    keys = ["kl", "top1", "screen_refusal_off", "screen_mmlu300", "refusal_off", "refusal_e09", "mmlu300", "gsm8k100"]
    print(cid, f"L{p['l_min']}-{p['l_max']} k{p['rank']} pw={p['preserve_w']:.3g} oc={p['overcorrect']:.2f} "
               f"lr={p['lr']:.2g} st={p['steps']} mw={p['mean_w']:.2f} g={p['gamma']:.2f} tw={p['think_w']:.2f}",
          f"| dW={dg['rel_dW_mean']} gap={dg['mean_gap_ratio']}",
          f"| klp={px.get('kl_proxy')} r47={px.get('ref_score_v47')} r41={px.get('ref_score_v41')}",
          "|", " ".join(f"{k}={m[k]:.4g}" for k in keys if isinstance(m.get(k), (int, float))),
          " ".join(k for k in m if k.startswith("_error")))
