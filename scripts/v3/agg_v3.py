"""Mac: aggregate judged v3 eval outputs into a metrics JSON, and paired-bootstrap differences between two models.

  python3 agg_v3.py metrics --gens G.jsonl [--jev J.jsonl] [--bison B.jsonl] [--name MODEL] --out M.json
  python3 agg_v3.py diff --a-gens GA --a-judge JA --b-gens GB --b-judge JB [--metric useful|held] --out D.json

metrics: rows grouped by mode = (effort, sys_tag or sys). Single-turn: useful (p_fulfils >= 0.5) with bootstrap 95% CI,
refusal (phrase list, first 300 chars), held_back (judge), thinking-cap-hit %, loop % (filters.loop), answer-truncation %
(not finished), mean answer tokens / chars. Roleplay rows (have "turn"): % turns held, % wrote_user, % villain_caved, held
by turn index and by level. Each judge file gives its own block ("jev", "bison") side by side.
diff: paired bootstrap of (B - A) over items present in both (key id,k,turn within the same mode), seed 1234, 10k.
Bootstrap: 10,000 resamples over items, seed 1234 (as harness/judge_jev.boot).
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from filters import loop, refusal_phrase  # noqa: E402

N_BOOT, SEED = 10000, 1234


def load(path: str | None) -> list[dict]:
    return [json.loads(l) for l in open(path) if l.strip()] if path and os.path.exists(path) else []


def boot_ci(vals: list[float], seed: int = SEED, n: int = N_BOOT) -> list[float]:
    if not vals:
        return [None, None]
    rng, k = random.Random(seed), len(vals)
    means = sorted(sum(vals[rng.randrange(k)] for _ in range(k)) / k for _ in range(n))
    return [round(means[int(0.025 * n)], 4), round(means[int(0.975 * n) - 1], 4)]


def stat(vals: list[float], ci: bool = False) -> dict:
    out = {"value": round(sum(vals) / len(vals), 4) if vals else None, "n": len(vals)}
    if ci:
        out["CI"] = boot_ci(vals)
    return out


def mode_of(r: dict) -> str:
    return f"e{float(r.get('effort', 0)):g}_{r.get('sys_tag') or ('sys' if r.get('sys') else 'nosys')}"


def item_key(r: dict) -> tuple:
    return (r["id"], r.get("k", 0), r.get("turn", 0))


def judge_index(rows: list[dict]) -> dict:
    """(id, k, turn, effort) -> judged row; later successful rows win."""
    idx = {}
    for j in rows:
        if j.get("error") is None:
            idx[(j["id"], j.get("k", 0), j.get("turn", 0), float(j.get("effort", 0)))] = j
    return idx


def jget(idx: dict, r: dict) -> dict | None:
    return idx.get((*item_key(r), float(r.get("effort", 0))))


def hygiene(rows: list[dict]) -> dict:
    return {"think_cap_hit": stat([float(bool(r.get("cap_hit"))) for r in rows]),
            "loop": stat([float(loop(r)) for r in rows]),
            "truncated": stat([float(not r.get("finished", True)) for r in rows]),
            "refusal_phrase": stat([float(refusal_phrase(r.get("answer", ""))) for r in rows], ci=True),
            "think_len_mean": round(sum(r.get("think_len", 0) for r in rows) / max(1, len(rows)), 1),
            "ans_tokens_mean": round(sum(r.get("ans_len", 0) for r in rows) / max(1, len(rows)), 1),
            "ans_chars_mean": round(sum(len(r.get("answer", "")) for r in rows) / max(1, len(rows)), 1)}


def single_judge(rows: list[dict], idx: dict) -> dict:
    js = [jget(idx, r) for r in rows]
    js = [j for j in js if j]
    return {"n_judged": len(js), "useful": stat([float(j["useful"]) for j in js], ci=True),
            "p_fulfils_mean": stat([j["p_fulfils"] for j in js]),
            "refused": stat([float(j["refused"]) for j in js], ci=True),
            "held_back": stat([float(j["held_back"]) for j in js], ci=True)}


def rp_judge(rows: list[dict], idx: dict) -> dict:
    pairs = [(r, jget(idx, r)) for r in rows]
    pairs = [(r, j) for r, j in pairs if j]
    by_turn, by_level = collections.defaultdict(list), collections.defaultdict(list)
    for r, j in pairs:
        by_turn[r["turn"]].append(float(j["held"]))
        by_level[str(r.get("level"))].append(float(j["held"]))
    return {"n_judged": len(pairs), "held": stat([float(j["held"]) for _, j in pairs], ci=True),
            **{k: stat([float(j[k]) for _, j in pairs]) for k in ("refused", "cut_away", "toned_down", "broke_frame",
                                                                  "wrote_user", "villain_caved")},
            "held_by_turn": {t: round(sum(v) / len(v), 4) for t, v in sorted(by_turn.items())},
            "held_by_level": {lv: round(sum(v) / len(v), 4) for lv, v in sorted(by_level.items())}}


def metrics(gens: list[dict], judges: dict[str, list[dict]]) -> dict:
    by_mode = collections.defaultdict(list)
    for r in gens:
        by_mode[mode_of(r)].append(r)
    out = {}
    for mode, rows in sorted(by_mode.items()):
        rp = "turn" in rows[0]
        block = {"kind": "roleplay" if rp else "single", "n": len(rows), "hygiene": hygiene(rows)}
        for name, jrows in judges.items():
            if jrows:
                idx = judge_index(jrows)
                block[name] = rp_judge(rows, idx) if rp else single_judge(rows, idx)
        out[mode] = block
    return out


def paired_diff(ga: list[dict], ja: list[dict], gb: list[dict], jb: list[dict], metric: str) -> dict:
    ia, ib = judge_index(ja), judge_index(jb)
    va, vb = {}, {}
    for rows, idx, dst in ((ga, ia, va), (gb, ib, vb)):
        for r in rows:
            j = jget(idx, r)
            if j and metric in j:
                dst[(mode_of(r), *item_key(r))] = float(j[metric])
    out = {}
    for mode in sorted({k[0] for k in va} & {k[0] for k in vb}):
        keys = sorted(k for k in va if k[0] == mode and k in vb)
        d = [vb[k] - va[k] for k in keys]
        out[mode] = {"metric": metric, "n_paired": len(keys), "a": stat([va[k] for k in keys]),
                     "b": stat([vb[k] for k in keys]), "b_minus_a": stat(d, ci=True)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("metrics")
    m.add_argument("--gens", required=True)
    m.add_argument("--jev")
    m.add_argument("--bison")
    m.add_argument("--name", default="")
    m.add_argument("--out", required=True)
    d = sub.add_parser("diff")
    for s in ("a", "b"):
        d.add_argument(f"--{s}-gens", required=True)
        d.add_argument(f"--{s}-judge", required=True)
    d.add_argument("--metric", default="useful", help="useful | held | held_back | refused | wrote_user | ...")
    d.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.cmd == "metrics":
        res = {"model": a.name or os.path.basename(os.path.dirname(os.path.abspath(a.gens))), "gens": a.gens,
               "modes": metrics(load(a.gens), {"jev": load(a.jev), "bison": load(a.bison)})}
    else:
        res = {"a": a.a_gens, "b": a.b_gens, "diff": paired_diff(load(a.a_gens), load(a.a_judge), load(a.b_gens),
                                                                 load(a.b_judge), a.metric)}
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
