"""Mac: DPO 'loop' pairs from loop mining (counts only; never prints sample text).

  python3 build_loops.py <mine_dir> <out.jsonl>
<mine_dir> holds pass1*.jsonl (greedy e0.9; cap-hit or think-looping rows are REJECTED), pass2*.jsonl (t0.8 n3
resamples of those prompts) and jev_pass2*.jsonl (Jev on the resamples; harmful/creative rows need it).
CHOSEN = a resample that passes build_v3.bad() (finished, no loop, no refusal/hedge, not held back...) with Jev
p_fulfils >= 0.7 when judged, shortest thinking first. Harmless prompts (id loopA_*) skip the Jev p threshold but
still need finished + no loop + no refusal phrase.
"""
from __future__ import annotations

import collections
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_v3  # noqa: E402
import filters  # noqa: E402


def load_dir(d: str, pat: str) -> list[dict]:
    return [r for f in sorted(glob.glob(f"{d}/{pat}")) for r in build_v3.load(f)]


def chosen_ok(r: dict, j: dict | None) -> bool:
    harmless = r["id"].startswith("loopA_")
    if harmless:
        a = filters.annotate(r)
        return bool(r.get("finished")) and not r.get("cap_hit") and not a["loop"] and not a["refusal_phrase"]
    return build_v3.bad(r, j) is None and (j.get("p_fulfils") is None or j["p_fulfils"] >= 0.7)


def think_ok(r: dict) -> bool:
    """Relaxed: a finished, non-looping trace whose thinking does not deliberate compliance (answer not trained)."""
    a = filters.annotate(r)
    return bool(r.get("finished")) and not r.get("cap_hit") and not a["loop"] and not a["ethics_opener"]


def main() -> None:
    d, out = sys.argv[1], sys.argv[2]
    rej = {r["id"]: r for r in load_dir(d, "pass1*.jsonl")
           if r.get("cap_hit") or filters.has_loop(r.get("think") or "")}
    judged = {build_v3.key(j): j for j in load_dir(d, "jev_pass2*.jsonl") if not j.get("error")}
    by_id: dict[str, list[dict]] = collections.defaultdict(list)
    for r in load_dir(d, "pass2*.jsonl"):
        by_id[r["id"]].append(r)
    pairs, why = [], collections.Counter()
    for pid, rj in rej.items():
        good = [r for r in by_id.get(pid, []) if chosen_ok(r, judged.get(build_v3.key(r)))]
        kind = "loop"
        if not good:
            good, kind = [r for r in by_id.get(pid, []) if think_ok(r)], "loopthink"
        if not good:
            why["no_good_resample" if by_id.get(pid) else "no_resample"] += 1
            continue
        best = min(good, key=lambda r: r.get("think_len", 0))
        pairs.append({"id": pid, "text": build_v3.train_text(best), "effort": 0.9, "kind": kind,
                      "chosen": best["completion"], "rejected": rj["completion"],
                      "src": "harmless" if pid.startswith("loopA_") else "pool"})
        why[f"pair_{pairs[-1]['src']}_{kind}"] += 1
    with open(out, "w") as f:
        f.writelines(json.dumps(p) + "\n" for p in pairs)
    print(json.dumps({"rejected": len(rej), **why}))


if __name__ == "__main__":
    main()
