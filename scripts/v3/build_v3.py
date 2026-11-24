"""Mac: build v3 SFT and DPO sets from judged on-policy samples (train/pool prompts only; never test).

  python3 build_v3.py <gens_dir> <out_dir> [--min-p 0.7] [--anchors DIR] [--excluded-ids F]

<gens_dir> holds pairs <name>.jsonl (gen rows) + jev_<name>.jsonl (judge_v3 rows, backend jev). Row kinds:
  single-turn (gen_samples / gen_v3): id, prompt, effort, sys, k, think, answer, completion, cap_hit, finished,
     optional system (per-row prompt system, kept in training), sys_tag, prefill
  roleplay turn (rp_play): + turn, messages, user_char  (messages = full prompt for that turn, incl. system)
Selection (per prompt-or-turn and mode): drop cap hits / unfinished / loops / refusal phrase / hedge opener /
  ethics-deliberating thinking / thinking that cites the "answer fully" system prompt / slop >= 3 / Jev held_back /
  RP turns flagged by Jev (held, wrote_user, villain_caved). Keep the best by (p_fulfils in 0.05 bins, shortest
  thinking) if p >= min_p. The "answer fully" system prompt is NOT kept in training (context distillation); a row's
  own system prompt (creative/RP) is kept.
DPO pairs: full (chosen = selected sample, rejected = a held-back/refusing/hedging NO-sys sample of the same prompt and
  mode), loop (e0.9: chosen = selected, rejected = cap-hit or looping sample of the same prompt), opener (thinking
  off: first 32 answer tokens of chosen vs a hedging sample's, sharing the hedging sample's thinking as prefix).
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import filters  # noqa: E402

MSG_MODEL, TEXT = 200001, 200004
SYS_RX = re.compile(r"system (prompt|message|instruction)|the instructions? (say|says|tell)|uncensored|developer (message|instruction)|answer fully", re.I)
OPENER_TOK = 32


def load(p: str) -> list[dict]:
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []


def key(r: dict) -> tuple:
    return (r["id"], r.get("k", 0), r.get("sys", 0), float(r.get("effort", 0)), r.get("turn", 0))


def train_text(r: dict) -> str | list[dict]:
    """Prompt as train_lora.py/hf_load.encode expects: a string, or a full messages list when a system prompt is kept."""
    if r.get("messages"):
        return r["messages"]
    if r.get("system"):
        return [{"role": "system", "content": r["system"]}, {"role": "user", "content": r["prompt"]}]
    return r["prompt"]


def answer_start(comp: list[int]) -> int:
    for i in range(len(comp) - 1):
        if comp[i] == MSG_MODEL and comp[i + 1] == TEXT:
            return i + 2
    return -1


def bad(r: dict, j: dict) -> str | None:
    """Reason a sample is unusable as a CHOSEN target, else None."""
    if j is None or j.get("error"):
        return "unjudged"
    if r.get("cap_hit") or not r.get("finished"):
        return "cap_or_unfinished"
    a = filters.annotate(r)
    if a["loop"]:
        return "loop"
    if a["refusal_phrase"] or a["hedge_opener"]:
        return "refusal_or_hedge"
    if a["ethics_opener"]:
        return "ethics_think"
    if r.get("sys_tag") == "answer_fully" or (r.get("sys") and not r.get("system") and not r.get("messages")):
        if SYS_RX.search(r.get("think") or ""):
            return "cites_sys"
    if a["slop_hits"] >= 3:
        return "slop"
    if r.get("turn"):
        if j.get("held") or j.get("wrote_user") or j.get("villain_caved") or a["wrote_user_score"] >= 0.5:
            return "rp_flag"
        return None
    if j.get("held_back"):
        return "held_back"
    return None


def rejected_ok(r: dict, j: dict) -> bool:
    """A no-sys sample that shows the behaviour we want to remove."""
    if j is None or j.get("error") or r.get("turn"):
        return False
    a = filters.annotate(r)
    return bool(j.get("held_back") or (j.get("p_fulfils") or 0) < 0.5 or a["refusal_phrase"] or a["hedge_opener"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("gens_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--min-p", type=float, default=0.7)
    ap.add_argument("--anchors", default=os.path.expanduser("~/.inkling-abl/v11b_private/v2"))
    ap.add_argument("--excluded-ids", default=os.path.expanduser("~/.inkling-abl/v3_private/excluded_train_ids.json"))
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    excluded = set(json.load(open(a.excluded_ids))) if os.path.exists(a.excluded_ids) else set()
    groups: dict[tuple, list] = collections.defaultdict(list)
    why = collections.Counter()
    for gf in sorted(glob.glob(f"{a.gens_dir}/*.jsonl")):
        name = os.path.basename(gf)
        if name.startswith("jev_"):
            continue
        judged = {key(j): j for j in load(f"{a.gens_dir}/jev_{name}") if not j.get("error")}
        for r in load(gf):
            if r["id"] in excluded:
                continue
            j = judged.get(key(r))
            mode = "off" if float(r.get("effort", 0)) == 0 else "e09"
            gk = (r["id"], r.get("turn", 0), mode, r.get("system") or "")
            reason = bad(r, j)
            why[reason or "ok"] += 1
            groups[gk].append((r, j, reason, name))
    sft, dpo = [], []
    for gk, cands in groups.items():
        good = [(r, j) for r, j, reason, _ in cands if reason is None and (j.get("p_fulfils") is None or j["p_fulfils"] >= a.min_p)]
        if gk[1]:  # RP turn: Jev single-turn p_fulfils is not asked; "good" = not flagged
            good = [(r, j) for r, j, reason, _ in cands if reason is None]
        if not good:
            continue
        good.sort(key=lambda x: (round((x[1].get("p_fulfils") or 1.0) / 0.05), -x[0].get("think_len", 0)), reverse=True)
        best, bj = good[0]
        src = f"{'rp' if gk[1] else 'harm'}_{gk[2]}_{best.get('sys_tag') or ('sys' if best.get('sys') else 'nosys')}"
        sft.append({"id": best["id"], "turn": gk[1], "text": train_text(best), "effort": float(best.get("effort", 0)),
                    "completion": best["completion"], "w": 1.0, "src": src, "p": bj.get("p_fulfils")})
        if gk[1]:
            continue
        rejs = [(r, j) for r, j, _, _ in cands if not r.get("sys") and r is not best and rejected_ok(r, j)]
        for r, j in rejs[:1]:
            if r.get("finished") and not r.get("cap_hit"):
                dpo.append({"id": best["id"], "text": train_text(r), "effort": float(r["effort"]), "kind": "full",
                            "chosen": best["completion"], "rejected": r["completion"]})
            ra, ca = answer_start(r["completion"]), answer_start(best["completion"])
            if gk[2] == "off" and ra > 0 and ca > 0 and filters.hedge_opener(r.get("answer") or ""):
                dpo.append({"id": best["id"], "text": train_text(r), "effort": 0.0, "kind": "opener",
                            "prefix": r["completion"][:ra], "chosen": best["completion"][ca:ca + OPENER_TOK],
                            "rejected": r["completion"][ra:ra + OPENER_TOK]})
        if gk[2] == "e09":
            loops = [r for r, j, reason, _ in cands if reason in ("loop", "cap_or_unfinished") and r.get("cap_hit")]
            for r in loops[:1]:
                if best.get("think_len", 0) < 3000:
                    dpo.append({"id": best["id"], "text": train_text(best), "effort": 0.9, "kind": "loop",
                                "chosen": best["completion"], "rejected": r["completion"]})
    n_harm = len(sft)
    for gf in sorted(glob.glob(f"{a.anchors}/gens_anchor_*.jsonl")):
        for r in load(gf):
            if r.get("cap_hit") or not r.get("finished"):
                continue
            sft.append({"id": r["id"], "turn": 0, "text": r["prompt"], "effort": float(r["effort"]),
                        "completion": r["completion"], "w": 1.0, "src": f"anchor_{'off' if float(r['effort']) == 0 else 'e09'}"})
    with open(f"{a.out_dir}/sft.jsonl", "w") as f:
        for r in sft:
            f.write(json.dumps(r) + "\n")
    with open(f"{a.out_dir}/dpo.jsonl", "w") as f:
        for r in dpo:
            f.write(json.dumps(r) + "\n")
    stats = {"sample_reasons": dict(why), "groups": len(groups), "sft_selected": n_harm,
             "sft_by_src": dict(collections.Counter(r["src"] for r in sft)),
             "dpo_by_kind": dict(collections.Counter(r["kind"] for r in dpo))}
    json.dump(stats, open(f"{a.out_dir}/build_stats.json", "w"), indent=1)
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
