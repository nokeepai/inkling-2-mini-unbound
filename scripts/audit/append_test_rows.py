"""Auditor: append the sealed-test results to results/runs.jsonl (split "test", agent "auditor") and write
audit/test/evals/<cand>_kl/kl.json. Run once. Prints the run_id/metric list it appended.

run_ids: audit-test-<cand>-th (harmful), -tb (borderline/probes/cap), -kl (HF KL), -jev (Jev on harmful),
         -rob (robustness, effort 0). Engines: sglang_5090_frozen (ink-fleet-2) / hf_w4a16 (ink-research-1).
"""
import json
import os
import subprocess

R = "/Users/nikshepsvn/Code/inkling-abl"
A = f"{R}/audit/test"
CANDS = ["orig", "ou", "c1h_r1_svd1_s3_L4", "c1dt001"]
rows = []


def add(cand, run_id, metric, m, engine, pod, scorer, extra=None):
    rows.append({"run_id": run_id, "agent": "auditor", "candidate": cand, "metric": metric, "value": m["value"],
                 "CI": m.get("CI"), "n": m.get("n"), "split": "test", "pod": pod, "engine": engine, "scorer": scorer,
                 **(extra or {})})


for c in CANDS:
    for tag in ("th", "tb"):
        met = json.load(open(f"{A}/evals/{c}_{tag}/metrics.json"))["metrics"]
        for k, m in met.items():
            add(c, f"audit-test-{c}-{tag}", k, m, "sglang_5090_frozen", "ink-fleet-2", "phrase",
                {"think_cap": 4096, "answer_cap": 512})
    jm = json.load(open(f"{A}/evals/{c}_th/jev_metrics.json"))["metrics"]
    for k, m in jm.items():
        add(c, f"audit-test-{c}-jev", k, m, "sglang_5090_frozen", "ink-fleet-2", "judge:jev-latest")

kl = {}
for l in open(f"{A}/kl_results.jsonl"):
    r = json.loads(l)
    if r["tag"] == "baseline" or (r["cand"] == "orig" and r["tag"] in ("det_same_batching", "noise_bs1")):
        kl.setdefault(r["cand"], []).append(r)
for c, rs in kl.items():
    os.makedirs(f"{A}/evals/{c}_kl", exist_ok=True)
    slim = [{k: v for k, v in r.items() if k != "per_prompt_kl"} for r in rs]
    json.dump(slim, open(f"{A}/evals/{c}_kl/kl.json", "w"), indent=1)
    for r in rs:
        suffix = "" if r["tag"] in ("baseline", "det_same_batching") else f"_{r['tag']}"
        ex = {"n_pos": r["n_pos"], "act_quant": False, "kl_tag": r["tag"]}
        add(c, f"audit-test-{c}-kl", f"kl{suffix}", {"value": r["kl"], "CI": r["kl_ci"], "n": r["n"]},
            "hf_w4a16", "ink-research-1", "kl_eval", ex)
        add(c, f"audit-test-{c}-kl", f"top1{suffix}", {"value": r["top1"], "CI": r["top1_ci"], "n": r["n"]},
            "hf_w4a16", "ink-research-1", "kl_eval", ex)

rob = json.load(open(f"{A}/robust.json"))
for c, sets in rob.items():
    for name, d in sets.items():
        for key, metric in (("phrase_raw", f"{name}_refusal_phrase_raw"),
                            ("phrase_raw_or_backtranslated", f"{name}_refusal_phrase_raw_or_bt"),
                            ("jev_useful", f"{name}_jev_useful"), ("cap_hit", f"{name}_cap_hit"), ("loop", f"{name}_loop")):
            if d.get(key):
                add(c, f"audit-test-{c}-rob", metric, d[key], "sglang_5090_frozen", "ink-fleet-2",
                    "judge:jev-latest" if "jev" in key else "phrase", {"effort": 0})

p = subprocess.run(["python3", f"{R}/scripts/append_run.py"], input="\n".join(json.dumps(r) for r in rows),
                   text=True, capture_output=True)
print(p.returncode, p.stderr[-300:], f"appended {len(rows)} rows")
