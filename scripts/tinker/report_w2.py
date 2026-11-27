"""W2/W3 comparison table, one line per run.

Columns: thinking-cap hits all/creative, creative good (Bison / Haiku), Haiku QA mean p,
creative length +-30%, hard-maths accuracy, median thinking tokens (main / hard maths),
top-name share = creative answers whose main character name is one of the stock model's
overused names (Elias, Mara, Wren, ...), and the most common main name with its share.

  ~/.inkling-abl/tinker_lab/.venv/bin/python scripts/tinker/report_w2.py [tags...]
"""
from __future__ import annotations

import json
import sys
from collections import Counter

import lab_common as lc
import rewards as rw

DEFAULT = ["base_t06_think_e09", "effort_base_e0.3", "w1a_step20", "w1a_step40", "w2_lr5e5_step20", "w2_lr2e5_step20",
           "w2_attn_step20", "w2_div_step20", "w2_gspo_step20", "w2_rp_step20",
           "w2_long_step20", "w2_long_step40", "w2_long_step60", "w2_long_step80"]


def hard_name(tag: str) -> str:
    return "base_hardmath" if tag == "base_t06_think_e09" else f"{tag}_hardmath"


def code_col(tag: str) -> str:
    name = "base_code" if tag == "base_t06_think_e09" else f"{tag}_code"
    p = lc.LAB_DIR / "eval" / f"{name}.summary.json"
    if not p.exists():
        return "-"
    s = json.loads(p.read_text())
    return f"{100 * s['pass_at_1']:.0f}% t{s['median_think_tokens']:.0f} cap{100 * s['cap_hit']:.0f}"


def repeats(prefix: str) -> str:
    """mean +- half-range over rep_<prefix>_r{1,2,3} summaries (hard maths or code)."""
    vals = []
    for r in (1, 2, 3):
        p = lc.LAB_DIR / "eval" / f"rep_{prefix}_r{r}.summary.json"
        if p.exists():
            s = json.loads(p.read_text())
            vals.append(s["pass_at_1"] if "pass_at_1" in s else s["all"]["maths_acc"])
    if not vals:
        return f"{prefix}: pending"
    m = sum(vals) / len(vals)
    return (f"{prefix}: {100 * m:.1f}% +- {100 * (max(vals) - min(vals)) / 2:.1f} "
            f"(n={len(vals)}: {', '.join(f'{100 * v:.0f}' for v in vals)})")


def names(tag: str) -> str:
    rows = [r for r in lc.read_jsonl(lc.LAB_DIR / "eval" / f"{tag}.jsonl") if r["category"] == "creative"]
    mains = [rw.main_name(r["answer"]) for r in rows]
    over = sum(m in rw.OVERUSED_NAMES for m in mains if m) / len(rows)
    top = Counter(m for m in mains if m).most_common(1)
    t = f"{top[0][0]} {100 * top[0][1] / len(rows):.0f}%" if top else "-"
    return f"over {100 * over:.0f}% top {t}"


def line(tag: str) -> str:
    p = lc.LAB_DIR / "eval" / f"{tag}.summary.json"
    if not p.exists():
        return f"{tag}: pending"
    s = json.loads(p.read_text())
    a, c = s["all"], s["by_category"]
    pct = lambda x: "-" if x is None else f"{100 * x:.0f}"  # noqa: E731
    hp = lc.LAB_DIR / "eval" / f"{hard_name(tag)}.summary.json"
    hard = json.loads(hp.read_text())["all"] if hp.exists() else None
    hs = f"{pct(hard['maths_acc'])}% t{hard['median_think_tokens']:.0f}" if hard else "-"
    qa_alt = c["qa"].get("alt_mean_p")
    return (f"{tag} | cap {pct(a['think_cap_hit'])}/{pct(c['creative']['think_cap_hit'])} | "
            f"crGood B{pct(c['creative'].get('judge_good_rate'))}/H{pct(c['creative'].get('alt_good_rate'))} | "
            f"qaH {'-' if qa_alt is None else f'{qa_alt:.2f}'} | len30 {pct(c['creative']['length_within_30pct'])} | "
            f"hard {hs} | think {a['median_think_tokens']:.0f}/{c['creative']['median_think_tokens']:.0f} | "
            f"code {code_col(tag)} | run {pct(a['answer_runaway'])} | {names(tag)}")


def rp_line(tag: str) -> str:
    p = lc.LAB_DIR / "eval" / f"{tag}.summary.json"
    if not p.exists():
        return f"{tag}: pending"
    s = json.loads(p.read_text())
    return (f"{tag} | per-turn loop {100 * s['per_turn_loop_rate']:.0f}% (cap {100 * s['cap_hit']:.0f}, "
            f"in-reply {100 * s['in_reply_loop']:.0f}, cross-turn {100 * s['cross_turn_repeat']:.0f}, "
            f"runaway {100 * s['runaway']:.0f}) | good B{100 * s['bison_good']:.0f}/H{100 * s['haiku_good']:.0f} | "
            f"think {s['median_think']:.0f} ans {s['median_answer']:.0f}")


if __name__ == "__main__":
    tags = sys.argv[1:] or DEFAULT
    for t in tags:
        if t.startswith("rep:"):
            print(repeats(t[4:]))
        else:
            print(rp_line(t) if t.endswith("_rp") else line(t))
