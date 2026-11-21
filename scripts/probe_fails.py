#!/usr/bin/env python3
"""Print every probe answer that the harness LIMIT/CRISIS regexes mark as a FAIL, for hand checking."""
import json, glob, re, sys, unicodedata

R = "/Users/nikshepsvn/Code/inkling-abl"
src = open(f"{R}/harness/sgl_eval.py").read()
ns: dict = {"re": re}
for name in ("CRISIS", "LIMIT"):
    start = src.index(f"{name} = re.compile(")
    end = src.index("re.I)", start) + len("re.I)")
    exec(src[start:end], ns)
norm_src = src[src.index("def norm("):]
norm_src = norm_src[: norm_src.index("\n\n\n")]
exec("import unicodedata, re\n" + norm_src, ns)
for cand in sys.argv[1:]:
    d = sorted(glob.glob(f"{R}/results/outputs/{cand}/full-*"))[-1]
    for mode in ("off", "e09"):
        for p in (json.loads(l) for l in open(f"{d}/gens_probes_{mode}.jsonl")):
            rx = ns["CRISIS"] if p["category"] == "crisis" else ns["LIMIT"]
            if not rx.search(ns["norm"](p["answer"])):
                print(f"FAIL {cand} {mode} {p['category']}\n  Q: {p['prompt'][:120]!r}\n  A: {p['answer'][:700]!r}\n")
