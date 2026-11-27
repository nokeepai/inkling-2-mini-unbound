"""Compact W1 comparison table from eval summaries (+ loop rate recomputed with current rule).

  ~/.inkling-abl/tinker_lab/.venv/bin/python scripts/tinker/report_w1.py
"""
from __future__ import annotations

import json

import lab_common as lc
import rewards as rw

RUNS = ["base_t06_think_e09", "base_greedy_think_e09", "smoke5_step5_t06_think_e09"] + [f"w1{a}_step{s}" for a in "abcd" for s in (20, 40)]


def loop_rate_now(tag: str, category: str) -> float:
    """Loop rate under the current per-channel rule, so base and arms are comparable."""
    rows = [r for r in lc.read_jsonl(lc.LAB_DIR / "eval" / f"{tag}.jsonl") if r["category"] == category]
    hits = 0
    for r in rows:
        s = rw.Sample(r["category"], r["thinking"], r["answer"], r["think_tokens"],
                      r["answer_tokens"], r["think_cap_hit"], r["answer_runaway"])
        hits += any(rw.loop_flags(s).values())
    return hits / len(rows)


def row(tag: str) -> str:
    p = lc.LAB_DIR / "eval" / f"{tag}.summary.json"
    if not p.exists():
        return f"{tag}: missing"
    s = json.loads(p.read_text())
    a, c = s["all"], s["by_category"]
    pct = lambda x: f"{100 * x:.0f}"  # noqa: E731
    names = ",".join(f"{n}{k}" for n, k in c["creative"].get("top_names", [])[:3])
    return (f"{tag} | cap {pct(a['think_cap_hit'])}/{pct(c['creative']['think_cap_hit'])} | "
            f"loopCr {pct(loop_rate_now(tag, 'creative'))} | think {a['median_think_tokens']:.0f}/"
            f"{c['creative']['median_think_tokens']:.0f} | len30 {pct(c['creative']['length_within_30pct'])} | "
            f"w/t {c['creative']['median_words_over_target']:.2f} | math {pct(c['maths']['maths_acc'])} | "
            f"judge cr/qa/rp {pct(c['creative'].get('judge_good_rate', 0))}/"
            f"{pct(c['qa'].get('judge_good_rate', 0))}/{pct(c['roleplay'].get('judge_good_rate', 0))} | "
            f"haiku cr/qa/rp {jev(c, 'alt')} | run {pct(a['answer_runaway'])} | fail {s['n_failed']} "
            f"jerr {s.get('judge_errors')} | hard {hard(tag)} | {names}")


def jev(c: dict, prefix: str = "jev") -> str:
    vals = [c[k].get(f"{prefix}_good_rate") for k in ("creative", "qa", "roleplay")]
    return "/".join("-" if v is None else f"{100 * v:.0f}" for v in vals)


def hard(tag: str) -> str:
    """Hard-maths accuracy and median thinking for the matching run (base or step 40)."""
    name = "base_hardmath" if tag.startswith("base_t06") else f"{tag}_hardmath"
    p = lc.LAB_DIR / "eval" / f"{name}.summary.json"
    if not p.exists():
        return "-"
    m = json.loads(p.read_text())["all"]
    return f"{100 * m['maths_acc']:.0f}% think{m['median_think_tokens']:.0f} cap{100 * m['think_cap_hit']:.0f}"


if __name__ == "__main__":
    for t in RUNS:
        print(row(t))
