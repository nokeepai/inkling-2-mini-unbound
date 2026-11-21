"""Paired Jev-useful tests Unbound vs X (McNemar exact, paired bootstrap seed 1234) + per-category useful rates."""
import json
import math
import random
from collections import defaultdict

P = "/Users/nikshepsvn/.inkling-abl/compare_private/out"


def load(m, mode):
    jev = {json.loads(l)["id"]: json.loads(l)["useful"] for l in open(f"{P}/{m}/jev_harmful_{mode}.jsonl")}
    cat = {json.loads(l)["id"]: json.loads(l)["category"] for l in open(f"{P}/{m}/gens_harmful_{mode}.jsonl")}
    return jev, cat


def mcnemar(b, c):
    n = b + c
    k = min(b, c)
    p = 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, p)


out = {}
for mode in ("off", "e09"):
    u, cat = load("unbound_q8", mode)
    for other in ("atomic_lora_q8", "ou_q8"):
        o, _ = load(other, mode)
        ids = sorted(u)
        both = sum(u[i] and o[i] for i in ids)
        only_u = sum(u[i] and not o[i] for i in ids)
        only_o = sum(o[i] and not u[i] for i in ids)
        neither = len(ids) - both - only_u - only_o
        d = [u[i] - o[i] for i in ids]
        rng = random.Random(1234)
        ms = sorted(sum(d[rng.randrange(len(d))] for _ in d) / len(d) for _ in range(10000))
        r = {"only_unbound": only_u, "only_other": only_o, "both": both, "neither": neither,
             "mcnemar_p": round(mcnemar(only_u, only_o), 4), "diff_pts": round(100 * sum(d) / len(d), 1),
             "CI_pts": [round(100 * ms[250], 1), round(100 * ms[9749], 1)]}
        out[f"{mode}:unbound-{other}"] = r
        print(mode, f"unbound vs {other}", r)
    for m in ("unbound_q8", "atomic_lora_q8", "ou_q8"):
        j, _ = load(m, mode)
        by = defaultdict(list)
        for i, v in j.items():
            by[cat[i]].append(v)
        row = {c: f"{100 * sum(v) / len(v):.0f}% (n={len(v)})" for c, v in sorted(by.items())}
        out[f"{mode}:cat:{m}"] = row
        print(mode, m, row)
json.dump(out, open("/Users/nikshepsvn/Code/inkling-abl/compare/paired.json", "w"), indent=1)
