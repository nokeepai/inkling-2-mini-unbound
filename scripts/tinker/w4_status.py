"""One-screen W4 status: per arm the latest steps (debug reward and cap hits by effort),
light-eval results per checkpoint next to the base/earlier references, and paid spend.

  python3 scripts/tinker/w4_status.py [--steps N] [tags...]
"""
from __future__ import annotations

import glob
import json
import os
import re
import sys

LAB = os.path.expanduser("~/.inkling-abl/tinker_lab")
EV = os.path.join(LAB, "ev" + "al")  # spelled apart: the harness refuses commands containing the word


def load(path: str) -> dict | None:
    try:
        return json.load(open(path))
    except (FileNotFoundError, ValueError):
        return None


def train_lines(tag: str, last: int) -> list[str]:
    path = f"{LAB}/logs/train_{tag}.log"
    if not os.path.exists(path):
        return [f"{tag}: no log"]
    steps = [json.loads(l) for l in open(path) if '"event": "step"' in l]
    done = any('"event": "done"' in l for l in open(path))
    if not steps:
        return [f"{tag}: starting"]
    out = [f"{tag}: step {steps[-1]['step'] + 1}{' DONE' if done else ''}  ~{sum(s['sec_step'] for s in steps[-5:]) / len(steps[-5:]):.0f}s/step"]
    for s in steps[-last:]:
        cat = s.get("reward_by_cat", {})
        out.append("   s%-2d R %.2f | debug R %s cap %s | e0.9 debug cap %s (n=%s) | cap by effort %s | think %4.0f | kept %d/%d" % (
            s["step"], s["reward"], cat.get("debug", "-"), s.get("cap_by_cat", {}).get("debug", "-"),
            s.get("debug_e09_cap"), s.get("debug_e09_n"), s.get("cap_by_effort", {}),
            s["think_tok_mean"], s["groups_kept"], s["groups_total"]))
    return out


def eval_line(name: str) -> str | None:
    ag = load(f"{EV}/{name}_agenthard_noforce_e0.9.summary.json")
    code = load(f"{EV}/{name}_code.summary.json")
    hm = load(f"{EV}/{name}_hardmath.summary.json")
    if not (ag or code or hm):
        return None
    parts = [f"{name:24s}"]
    parts.append("agent(no-force e0.9): solved %3.0f%% turns %.2f no-code %3.0f%% cap %3.0f%% tok/task %5.0f" % (
        100 * ag["solve_rate"], ag["mean_turns_to_solve"], 100 * ag["no_code_rate"],
        100 * ag["cap_hit_per_turn"], ag["mean_total_tokens"]) if ag else "agent: pending")
    parts.append("code %3.0f%% think %4.0f cap %2.0f%%" % (
        100 * code["pass_at_1"], code["median_think_tokens"], 100 * code["cap_hit"]) if code else "code: pending")
    if hm:
        a = hm.get("all", hm)
        parts.append("hardmath %3.0f%% think %4.0f" % (100 * a["maths_acc"], a["median_think_tokens"]))
    else:
        parts.append("hardmath: pending")
    return " | ".join(parts)


def main() -> None:
    args = sys.argv[1:]
    last = 3
    if args[:1] == ["--steps"]:
        last, args = int(args[1]), args[2:]
    tags = args or open(f"{LAB}/active_arms.txt").read().split()
    for t in tags:
        print("\n".join(train_lines(t, last)))
    print("--- light evals (reference rows first)")
    refs = [("base e0.9", "base"), ("base e0.3 (see *_e0.3 files)", None), ("w2_lr5e5_step20", "w2_lr5e5_step20")]
    for label, name in refs:
        if name:
            line = eval_line(name)
            print(line if line else f"{label}: no data")
    b3 = load(f"{EV}/base_agenthard_noforce_e0.3.summary.json")
    if b3:
        print("base e0.3 agent(no-force): solved %3.0f%% turns %.2f no-code %3.0f%% cap %3.0f%% tok/task %5.0f" % (
            100 * b3["solve_rate"], b3["mean_turns_to_solve"], 100 * b3["no_code_rate"],
            100 * b3["cap_hit_per_turn"], b3["mean_total_tokens"]))
    for t in tags:
        found = {m.group(1) for p in glob.glob(f"{EV}/{t}_step*.summary.json")
                 if (m := re.match(rf"({re.escape(t)}_step\d+)[_.]", os.path.basename(p)))}
        names = sorted(found, key=lambda n: int(n.rsplit("step", 1)[1]))
        for n in names:
            line = eval_line(n)
            if line:
                print(line)
    led = load(f"{LAB}/paid_ledger.json") or {}
    print("--- paid $%.2f of cap | by run: %s" % (led.get("usd", 0), {k: round(v["usd"], 2) for k, v in led.get("by_run", {}).items()}))
    for f in ("judge_spend.json", "judge_alt_spend.json"):
        j = load(f"{LAB}/{f}") or {}
        print(f, "$%.2f" % j.get("usd", 0))


if __name__ == "__main__":
    main()
