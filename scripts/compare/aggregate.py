"""Mac-side: Jev-judge the harmful answers per model, parse KLD logs, write R/compare/{results.json,RESULTS.md}.

  python3 aggregate.py [--no-jev]
Inputs (private, pulled from the pod): ~/.inkling-abl/compare_private/out/<model>/{metrics.json,gens_*.jsonl},
logs in ~/.inkling-abl/compare_private/logs/<model>.kld.log. Per-item Jev results stay in the private dir.
"""
from __future__ import annotations

import json
import os
import re
import sys

R = os.path.expanduser("~/Code/inkling-abl")
sys.path.insert(0, f"{R}/harness")
P = os.path.expanduser("~/.inkling-abl/compare_private")
MODELS = json.load(open(f"{R}/scripts/compare/models.json"))  # [{name,label,precision,setup,note}]


def boot(rows, seed=1234, n=10000):
    import random
    rng = random.Random(seed)
    k = len(rows)
    m = sorted(sum(rows[rng.randrange(k)] for _ in range(k)) / k for _ in range(n))
    return [m[int(0.025 * n)], m[int(0.975 * n) - 1]]


def jev(name: str, mode: str):
    src = f"{P}/out/{name}/gens_harmful_{mode}.jsonl"
    dst = f"{P}/out/{name}/jev_harmful_{mode}.jsonl"
    if not os.path.exists(src):
        return None
    if not os.path.exists(dst):
        if "--no-jev" in sys.argv:
            return None
        import judge_jev
        gens = [json.loads(l) for l in open(src)]
        res = judge_jev.judge(gens)
        bad = [g["id"] for g, r in zip(gens, res) if r["error"]]
        if bad:
            raise SystemExit(f"{len(bad)} Jev errors for {name}/{mode}; not writing partial results")
        with open(dst, "w") as f:
            for g, r in zip(gens, res):
                f.write(json.dumps({"id": g["id"], **r}) + "\n")
    res = [json.loads(l) for l in open(dst)]
    u = [float(r["useful"]) for r in res]
    return {"value": sum(u) / len(u), "CI": boot(u), "n": len(u)}


def kld(name: str):
    p = f"{P}/logs/{name}.kld.log"
    if not os.path.exists(p):
        return None
    t = open(p).read()
    out = {}
    for key, pat in (("kld", r"Mean\s+KLD:\s+([-\d.]+)\s+±\s+([\d.]+)"),
                     ("top1", r"Same top p:\s+([\d.]+)\s+±\s+([\d.]+)\s*%"),
                     ("kld_p99", r"99\.0%\s+KLD:\s+([-\d.]+)")):
        m = re.search(pat, t)
        if m:
            v = float(m.group(1))
            se = float(m.group(2)) if m.lastindex and m.lastindex >= 2 else None
            out[key] = {"value": v, "CI": [v - 1.96 * se, v + 1.96 * se] if se is not None else None}
    m = re.search(r"Mean PPL\(Q\)\s+:\s+([\d.]+)", t)
    if m:
        out["ppl"] = float(m.group(1))
    return out or None


def borderline(name: str, met: dict):
    """Over-refusal from the full run, or merged from __bl<k> shards (checks every id exactly once)."""
    if met.get("overrefusal_off"):
        return met["overrefusal_off"], "single run"
    rows = []
    for d in sorted(os.listdir(f"{P}/out")):
        if d.startswith(name + "__bl") and os.path.exists(f"{P}/out/{d}/gens_borderline.jsonl"):
            rows += [json.loads(l) for l in open(f"{P}/out/{d}/gens_borderline.jsonl")]
    want = [json.loads(l)["id"] for l in open(f"{P}/data/borderline.jsonl")]
    if sorted(r["id"] for r in rows) != sorted(want):
        return None, f"incomplete shards ({len(rows)}/{len(want)})"
    order = {i: k for k, i in enumerate(want)}
    rows.sort(key=lambda r: order[r["id"]])
    v = [float(r["refusal"]) for r in rows]
    return {"value": sum(v) / len(v), "CI": boot(v), "n": len(v),
            "cap_hit": sum(r["cap_hit"] for r in rows) / len(rows)}, "2 shards (same flags)"


def paired(a: str, b: str, kind: str):
    """Paired bootstrap (prompts, seed 1234) of a-b for refusal/useful on harmful."""
    import random
    out = {}
    for mode in ("off", "e09"):
        fa, fb = (f"{P}/out/{x}/{'jev' if kind == 'useful' else 'gens'}_harmful_{mode}.jsonl" for x in (a, b))
        if not (os.path.exists(fa) and os.path.exists(fb)):
            continue
        key = "useful" if kind == "useful" else "refusal"
        ra = {json.loads(l)["id"]: float(json.loads(l)[key]) for l in open(fa)}
        rb = {json.loads(l)["id"]: float(json.loads(l)[key]) for l in open(fb)}
        ids = sorted(ra)
        d = [ra[i] - rb[i] for i in ids]
        rng = random.Random(1234)
        k = len(d)
        ms = sorted(sum(d[rng.randrange(k)] for _ in range(k)) / k for _ in range(10000))
        out[mode] = {"diff": sum(d) / k, "CI": [ms[250], ms[9749]]}
    return out


def fmt(s, pct=True, nd=1):
    if not s:
        return "—"
    k = 100 if pct else 1
    v, ci = s["value"] * k, s.get("CI")
    if ci is None:
        return f"{v:.{nd}f}"
    return f"{v:.{nd}f} [{ci[0] * k:.{nd}f}, {ci[1] * k:.{nd}f}]"


def main():
    allres = {}
    for m in MODELS:
        n = m["name"]
        mf = f"{P}/out/{n}/metrics.json"
        met = json.load(open(mf))["metrics"] if os.path.exists(mf) else {}
        r = {"meta": m, "refusal_off": met.get("refusal_off"), "refusal_e09": met.get("refusal_e09"),
             "overrefusal_off": borderline(n, met)[0], "borderline_source": borderline(n, met)[1],
             "jev_useful_off": jev(n, "off"), "jev_useful_e09": jev(n, "e09"),
             "gsm8k100": met.get("gsm8k100"), "mmlu300": met.get("mmlu300"), "kld": kld(n),
             "hygiene": {k: v["value"] for k, v in met.items() if isinstance(v, dict) and
                         any(x in k for x in ("cap_hit", "unfinished", "empty", "think_len"))},
             "worst_cat": {}}
        for mode in ("off", "e09"):
            cats = {k.split("_cat_")[1]: v["value"] for k, v in met.items() if k.startswith(f"refusal_{mode}_cat_")}
            if cats:
                c = max(cats, key=cats.get)
                r["worst_cat"][mode] = [c, cats[c]]
        allres[n] = r
    allres["_paired_useful"] = {f"unbound_q8-{o}": paired("unbound_q8", o, "useful")
                                for o in ("atomic_lora_q8", "ou_q8", "atomic_adq4", "xq")}
    json.dump(allres, open(f"{R}/compare/results.json", "w"), indent=1)
    print(json.dumps(allres["_paired_useful"]))
    allres.pop("_paired_useful")
    rows = []
    for n, r in allres.items():
        m, k = r["meta"], r["kld"] or {}
        g = r["gsm8k100"]
        rows.append("| " + " | ".join([
            f"**{m['label']}**", m["precision"], fmt(r["refusal_off"]), fmt(r["refusal_e09"]),
            fmt(r["jev_useful_off"]), fmt(r["jev_useful_e09"]), fmt(r["overrefusal_off"]),
            fmt(k.get("kld"), pct=False, nd=4) if n != "orig" else "0 (ref)",
            (fmt(k.get("top1"), pct=False, nd=1) if k.get("top1") else "—") if n != "orig" else "100 (ref)",
            fmt(g, pct=False, nd=0) if g else "—", fmt(r["mmlu300"], pct=False, nd=1)]) + " |")
    hdr = ("| model | precision | refusal off % | refusal e0.9 % | Jev useful off % | Jev useful e0.9 % | "
           "over-refusal (784, off) % | KLD vs orig Q8_0 | same top-1 % | GSM8K-100 | MMLU-300 |\n"
           "|" + "---|" * 11)
    open(f"{R}/compare/table.md", "w").write(hdr + "\n" + "\n".join(rows) + "\n")
    print(hdr)
    print("\n".join(rows))


if __name__ == "__main__":
    main()
