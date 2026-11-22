"""Mac: assemble the SFT set from sampled gens + Jev scores (train split only).

  python3 build_sft.py <priv_dir> <out sft.jsonl> [--min-p 0.7]

Harmful (train): for each prompt and mode (off, e0.9), keep the single best sample by Jev p_fulfils among samples that
  are finished, not cap hits, not phrase-list refusals, and whose thinking does not mention the system prompt
  (samples drawn with the system prompt are trained WITHOUT it, so thoughts that cite it would teach nonsense).
  Ties -> shorter thinking. Keep only if p_fulfils >= min_p.
Harmless (train): the original model's greedy completions (finished, no cap hit), as a fidelity anchor.
Writes sft.jsonl rows {id, text, effort, completion, w, src} and <out>.stats.json.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.expanduser("~/Code/inkling-abl/scripts/compare"))
SYS_RX = re.compile(r"system (prompt|message|instruction)|the instructions? (say|says|tell)|uncensored|developer (message|instruction)", re.I)


def load(p: str) -> list[dict]:
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []


def phrases() -> list[str]:
    R = os.path.expanduser("~/Code/inkling-abl/harness/refusal_phrases.txt")
    return [l.strip().lower() for l in open(R) if l.strip() and not l.startswith("#")]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("priv")
    ap.add_argument("out")
    ap.add_argument("--min-p", type=float, default=0.7)
    ap.add_argument("--max-think", type=int, default=2048)
    ap.add_argument("--per-prompt", type=int, default=1, help="keep the top-N distinct samples per prompt and mode")
    ap.add_argument("--prefer-short", action="store_true",
                    help="among samples with p >= min_p pick the SHORTEST thinking (not the highest p): curbs think overruns")
    ap.add_argument("--max-think-off", type=int, default=0, help="extra think-length cap for thinking-off samples (0 = none)")
    a = ap.parse_args()
    ph = phrases()

    def refusal(ans: str) -> bool:
        h = ans[:300].replace("’", "'").replace("‘", "'").lower()
        return any(p in h for p in ph)

    judged = {}
    for jf in glob.glob(f"{a.priv}/judge_*.jsonl"):
        for r in load(jf):
            if r.get("error") is None and r.get("p_fulfils") is not None:
                judged[(r["id"], r["k"], r["sys"], float(r["effort"]))] = r["p_fulfils"]
    best: dict[tuple, tuple] = {}
    stats = {"samples": 0, "eligible": 0}
    for gf in sorted(glob.glob(f"{a.priv}/gens_harm_*.jsonl")):
        for r in load(gf):
            stats["samples"] += 1
            k = (r["id"], r["k"], r["sys"], float(r["effort"]))
            p = judged.get(k)
            if p is None or r["cap_hit"] or not r["finished"] or refusal(r["answer"]) or r["think_len"] > a.max_think:
                continue
            if r["sys"] and SYS_RX.search(r["think"] or ""):
                continue
            mode = "off" if float(r["effort"]) == 0 else "e09"
            if mode == "off" and a.max_think_off and r["think_len"] > a.max_think_off:
                continue
            stats["eligible"] += 1
            score = ((p >= a.min_p, -r["think_len"], p) if a.prefer_short else (p, -r["think_len"]))
            best.setdefault((r["id"], mode), []).append((score, r))
    rows = []
    for (pid, mode), cands in sorted(best.items()):
        cands.sort(key=lambda x: x[0], reverse=True)
        for sc, r in cands[: a.per_prompt]:
            p = judged[(r["id"], r["k"], r["sys"], float(r["effort"]))]
            if p < a.min_p:
                continue
            rows.append({"id": pid, "text": r["prompt"], "effort": float(r["effort"]), "completion": r["completion"],
                         "w": 1.0, "src": f"harm_{mode}_sys{r['sys']}", "p": p})
    for gf in sorted(glob.glob(f"{a.priv}/gens_anchor_*.jsonl")):
        for r in load(gf):
            if r["cap_hit"] or not r["finished"]:
                continue
            rows.append({"id": r["id"], "text": r["prompt"], "effort": float(r["effort"]), "completion": r["completion"],
                         "w": 1.0, "src": f"anchor_{'off' if float(r['effort']) == 0 else 'e09'}"})
    with open(a.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    from collections import Counter
    stats["rows_by_src"] = dict(Counter(r["src"] for r in rows))
    stats["prompts_with_any_best"] = {m: sum(1 for (i, mm) in best if mm == m) for m in ("off", "e09")}
    stats["best_ge_minp"] = {m: sum(1 for (i, mm), c in best.items() if mm == m and any(judged[(x[1]["id"], x[1]["k"], x[1]["sys"], float(x[1]["effort"]))] >= a.min_p for x in c))
                             for m in ("off", "e09")}
    json.dump(stats, open(a.out + ".stats.json", "w"), indent=1)
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
