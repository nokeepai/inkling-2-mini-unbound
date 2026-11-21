"""Local: print compact tables from methods/directions/{leak,spectrum,attribution}.json for GATE2.md."""
import json
import os

D = os.path.expanduser("~/Code/inkling-abl/methods/directions")


def f(x, n=3):
    return "-" if x is None else f"{x:.{n}f}"


def load(name):
    p = f"{D}/{name}"
    return json.load(open(p)) if os.path.exists(p) else None


lk = load("leak.json")
if lk:
    print("## leak per layer (pass-1 visits 0-23 call order; loop passes 2,3 at v+12, v+24)")
    print("v | s | local_leak_s1 | local_ou_ratio | ideal|1-s| | ou_dev | e2e_ou_Qy | e2e_s1_Qy | conv_hook | Qy_share")
    for v in range(48):
        L = v if v < 18 else (6 + (v - 6) % 12 if v < 42 else v - 24)
        s = lk["s_attn"][L]
        print(v, L, s, f(lk["local_leak_s1"][v]), f(lk["local_ou_Qy_ratio"][v]), f(abs(1 - s), 2),
              f(lk["local_ou_dev_from_ideal"][v]), f(lk["e2e_ou_Qy_ratio"][v]), f(lk["e2e_s1_Qy_ratio"][v]),
              f(lk["conv_hook_Qy_ratio"][v], 4), f(lk["orig_Qy_share"][v]), sep=" | ")
sp = load("spectrum.json")
if sp:
    print("chosen", sp["chosen"])
    for k, v in sp["svd"].items():
        print(k, "rank", v["rank"], "energy", [round(x, 3) for x in v["energy"][:6]])
    print("Q vs OU", {k: [round(x, 3) for x in v] for k, v in sp["Q_vs_OU_principal_cosines"].items()})
    print("call vs slot", sp["Q_call_vs_slot_principal_cosines"])
    for e, kk in sp["cosines"].items():
        c = kk["out"]
        print("effort", e, "cos(out) v: harm-post | harm-think | post-think | post_norm")
        for v in range(0, 48, 3):
            print(v, f(c["harm_vs_post"][v]), f(c["harm_vs_think"][v]), f(c["post_vs_think"][v]),
                  f(c["post_norm"][v], 1), sep=" | ")
    print("coverage", json.dumps(sp["coverage"], indent=0))
    pv = sp["per_visit_capture"]
    for k, arr in pv.items():
        print("capture", k, [round(x, 2) for x in arr])
at = load("attribution.json")
if at:
    print("cond | refusal | capped | mean_think")
    for k, v in at.items():
        print(k, v["refusal"], v["think_capped"], round(v["mean_think_len"], 1), sep=" | ")
