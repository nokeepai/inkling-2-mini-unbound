"""Proxy-only sweep for C3 (train split; diagnostic pre-search while the fleet/KL workers are down).

  proxy_search.py <n_trials> [--stop HH:MM]

ask -> fit on ink-research-1 (job id p3t###, never enqueued) -> wait for the pod proxy daemon's proxy.json ->
tell (ref_score_v47, kl_proxy). Study: state/optuna/c3proxy.db. Configs on its Pareto front are copied by hand
(inject.jsonl) into the real c3 study, which is the only one scored on val.
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone

import optuna

sys.path.insert(0, os.path.dirname(__file__))
import search as S  # noqa: E402  (reuses params_for, sh, LIUM, POD; its argv parsing needs 2 ints)

N = int(sys.argv[1])
R = S.R
LOG = f"{R}/methods/ara/PROXY_LOG.md"


def run(cid: str, p: dict) -> dict:
    d = f"{R}/methods/ara/proxyjobs"
    os.makedirs(d, exist_ok=True)
    pj = f"{d}/{cid}.json"
    json.dump(p, open(pj, "w"))
    S.sh(S.LIUM + ["exec", S.POD, f"mkdir -p /root/ara/jobs/{cid}"])
    S.sh(S.LIUM + ["scp", S.POD, pj, f"/root/ara/jobs/{cid}/params.json"])
    S.sh(S.LIUM + ["exec", S.POD, f"bash /root/ara/fit_job.sh {cid}"], timeout=1500)
    for _ in range(60):
        out = S.sh(S.LIUM + ["exec", S.POD, f"cat /root/ara/jobs/{cid}/proxy.json 2>/dev/null || echo NONE"])
        line = [l for l in out.splitlines() if l.startswith("{") or l.startswith("NONE")]
        if line and line[-1].startswith("{"):
            return json.loads(line[-1])
        time.sleep(10)
    raise RuntimeError("proxy timeout")


def main() -> None:
    st = optuna.create_study(study_name="c3proxy3", storage=f"sqlite:///{R}/state/optuna/c3proxy.db",
                             directions=["minimize", "minimize"], load_if_exists=True,
                             sampler=optuna.samplers.TPESampler(n_startup_trials=8, multivariate=True, seed=11))
    if not st.trials:
        for ov in ({"preserve_w": 10.0, "pres2_w": 300.0, "mean_w": 1.0, "gamma": 0.5},
                   {"l_min": 4, "preserve_w": 10.0, "pres2_w": 30.0, "mean_w": 1.0, "gamma": 0.5},
                   {"preserve_w": 10.0, "pres2_w": 30.0, "mean_w": 2.0, "gamma": 1.0, "lr": 5e-4}):
            st.enqueue_trial({**S.BASE, **ov})
    for _ in range(N):
        if S.past_stop():
            break
        t = st.ask()
        p = S.params_for(t)
        cid = f"r3t{t.number:03d}"
        try:
            px = run(cid, p)
        except Exception as e:
            st.tell(t, state=optuna.trial.TrialState.FAIL)
            print("fail", cid, e, flush=True)
            continue
        fit = json.loads(S.sh(S.LIUM + ["exec", S.POD, f"python3 -c \"import json;L=json.load(open('/root/ara/jobs/{cid}/fit.json'))['layers'];print(json.dumps({{'dW':round(sum(x['rel_dW'] for x in L)/len(L),4),'gap':round(sum(x['mean_gap_ratio'] for x in L)/len(L),4)}}))\""]).splitlines()[-1])
        t.set_user_attr("params_json", json.dumps(p))
        st.tell(t, [max(px["ref_score_v47"], 0.0), px["kl_proxy"]])   # overshoot below the harmless level earns nothing
        short = {k: (round(v, 5) if isinstance(v, float) else v) for k, v in p.items() if k not in ("mp", "seed")}
        with open(LOG, "a") as f:
            f.write(f"- {datetime.now(timezone.utc).strftime('%H:%M:%SZ')} {cid} {json.dumps(short, separators=(',', ':'))} "
                    f"fit{json.dumps(fit)} -> {json.dumps(px)}\n")


if __name__ == "__main__":
    main()
