"""Mac: build a v2 private data dir where answer-capped samples are replaced by their 8192-cap continuations.

  python3 merge_cont.py <priv_dir> <cont_dir> <out_dir>

gens_*.jsonl: rows replaced by (id, k) from <cont_dir>/gens_*.cont.jsonl. judge_*.jsonl: old judgments of replaced rows
are dropped (they scored truncated answers); the continued harmful rows must then be judged into <out_dir>/judge_cont_*.jsonl.
Prints counts and answer-length stats.
"""
from __future__ import annotations

import glob
import json
import os
import statistics
import sys


def main() -> None:
    priv, cont, out = sys.argv[1:4]
    os.makedirs(out, exist_ok=True)
    replaced = set()
    for gf in sorted(glob.glob(f"{priv}/gens_*.jsonl")):
        base = os.path.basename(gf)
        cf = f"{cont}/{base[:-6]}.cont.jsonl"
        cmap = {(r["id"], r["k"]): r for r in map(json.loads, open(cf))} if os.path.exists(cf) else {}
        n_rep, n_fin, lens = 0, 0, []
        with open(f"{out}/{base}", "w") as f:
            for r in map(json.loads, open(gf)):
                c = cmap.get((r["id"], r["k"]))
                if c is not None:
                    r = c
                    n_rep += 1
                    n_fin += bool(c["finished"])
                    lens.append(c.get("ans_len", 0))
                    replaced.add((r["id"], r["k"], r["sys"], float(r["effort"])))
                f.write(json.dumps(r) + "\n")
        med = statistics.median(lens) if lens else 0
        print(f"{base}: replaced {n_rep}/{len(cmap)} cont rows, now finished {n_fin}, median cont answer tokens {med}")
    for jf in sorted(glob.glob(f"{priv}/judge_*.jsonl")):
        keep = [r for r in map(json.loads, open(jf)) if (r["id"], r["k"], r["sys"], float(r["effort"])) not in replaced]
        with open(f"{out}/{os.path.basename(jf)}", "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in keep)
    print("replaced keys", len(replaced))


if __name__ == "__main__":
    main()
