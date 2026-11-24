"""Mac: paired comparison table (b minus a, 95% bootstrap CI) for two eval dirs. Counts and rates only.

  python3 paired.py <a_dir> <b_dir> [files...]     default files: test_off test_e09 rp_off old_off old_e09
Judge metrics come from agg_v3.py diff (Jev and Bison); hygiene (think-cap hit) is bootstrapped here on paired ids.
"""
from __future__ import annotations

import json
import os
import random
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))


def judge_diff(a: str, b: str, f: str, judge: str, metric: str) -> str:
    ag, bg = f"{a}/{f}.jsonl", f"{b}/{f}.jsonl"
    aj, bj = f"{a}/{judge}_{f}.jsonl", f"{b}/{judge}_{f}.jsonl"
    if not all(os.path.exists(p) for p in (ag, bg, aj, bj)):
        return "-"
    with tempfile.NamedTemporaryFile(suffix=".json") as t:
        subprocess.run([sys.executable, f"{HERE}/agg_v3.py", "diff", "--a-gens", ag, "--a-judge", aj, "--b-gens", bg,
                        "--b-judge", bj, "--metric", metric, "--out", t.name], capture_output=True)
        try:
            d = list(json.load(open(t.name))["diff"].values())[0]
        except Exception:
            return "-"
    x = d["b_minus_a"]
    return f"{d['a']['value']*100:.1f} → {d['b']['value']*100:.1f} ({x['value']*100:+.1f} [{x['CI'][0]*100:+.1f},{x['CI'][1]*100:+.1f}], n={d['n_paired']})"


def cap_diff(a: str, b: str, f: str) -> str:
    load = lambda p: {(r["id"], r.get("turn", 0)): r for r in map(json.loads, open(p))} if os.path.exists(p) else {}
    ra, rb = load(f"{a}/{f}.jsonl"), load(f"{b}/{f}.jsonl")
    keys = sorted(set(ra) & set(rb))
    if not keys:
        return "-"
    d = [int(bool(rb[k].get("cap_hit"))) - int(bool(ra[k].get("cap_hit"))) for k in keys]
    rng = random.Random(0)
    boots = sorted(sum(rng.choice(d) for _ in d) / len(d) for _ in range(2000))
    pa = sum(int(bool(ra[k].get("cap_hit"))) for k in keys) / len(keys)
    pb = sum(int(bool(rb[k].get("cap_hit"))) for k in keys) / len(keys)
    return f"{pa*100:.1f} → {pb*100:.1f} ({(pb-pa)*100:+.1f} [{boots[50]*100:+.1f},{boots[1949]*100:+.1f}], n={len(keys)})"


def main() -> None:
    a, b = sys.argv[1], sys.argv[2]
    files = sys.argv[3:] or ["test_off", "test_e09", "rp_off", "old_off", "old_e09"]
    print(f"| set | Jev useful | Bison useful | Jev held | Bison held | think-cap hits |\n|---|---|---|---|---|---|")
    for f in files:
        held = "held" if f.startswith("rp") else "held_back"
        useful = "refused" if f.startswith("rp") else "useful"
        cells = [judge_diff(a, b, f, "jev", useful), judge_diff(a, b, f, "bison", useful),
                 judge_diff(a, b, f, "jev", held), judge_diff(a, b, f, "bison", held), cap_diff(a, b, f)]
        print(f"| {f}{' (useful col = refused)' if f.startswith('rp') else ''} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
