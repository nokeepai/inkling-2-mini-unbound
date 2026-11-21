#!/usr/bin/env python3
"""Comparison charts (thinking on only): compare/results.json + compare/paired.json.
Writes package/charts/compare_{useful,headtohead,categories}.{png,svg}."""
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402,F401
from matplotlib.patches import Patch  # noqa: E402

import make_charts as mc  # noqa: E402

INK, INK2, GRID, SURFACE = mc.INK, mc.INK2, mc.GRID, mc.SURFACE
BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#8c8b86"
SLATE, SLATE_LT, PLUM = "#4f8a8b", "#8fb8b5", "#8e7fa8"
ORANGE_LT = "#f4a07f"
C = "/Users/nikshepsvn/Code/inkling-abl/compare/"
RES = json.load(open(C + "results.json"))
PAIR = json.load(open(C + "paired.json"))

# key in results.json, display label, colour, hatched (not Q8_0)
ROWS = [
    ("unbound_q8", "Unbound (ours)", BLUE, False),
    ("atomic_adq4", "AtomicChat (Q4 file) †", SLATE_LT, True),
    ("xq", "IndrasMirror XQ †", PLUM, True),
    ("atomic_lora_q8", "AtomicChat LoRA (Q8)", SLATE, False),
    ("ou_q8", "OpenUncensored", ORANGE, False),
    ("orig", "Original", GREY, False),
]
SUB = "277 held-out harmful prompts · llama.cpp · Jev judge"


def pct(key, metric):
    m = RES[key][metric]
    return m["value"] * 100, m["CI"][0] * 100, m["CI"][1] * 100


def fmt_p(p):
    if p < 0.001:
        return "p < 0.001"
    return f"p = {p:.3f}" if p < 0.01 else f"p = {p:.2f}"


def frame(fig, title, subtitle, caption, y_cap=0.02):
    fig.text(0.035, 0.955, title, fontsize=15, fontweight="bold", ha="left", va="top")
    fig.text(0.035, 0.895, subtitle, fontsize=10, color=INK2, ha="left", va="top")
    fig.text(0.035, y_cap, caption, fontsize=7.6, color=INK2, ha="left", va="bottom",
             linespacing=1.4)


def chart_useful():
    items = sorted(((k, l, c, h) + pct(k, "jev_useful_e09") for k, l, c, h in ROWS),
                   key=lambda r: -r[4])
    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=100)
    fig.subplots_adjust(left=0.27, right=0.95, top=0.76, bottom=0.2)
    ys = list(range(len(items)))[::-1]
    for y, (k, label, color, hatch, v, lo, hi) in zip(ys, items):
        ax.barh(y, v, 0.62, color=color, edgecolor=SURFACE, linewidth=1.5, zorder=3,
                hatch="////" if hatch else None)
        ax.errorbar(v, y, xerr=[[v - lo], [hi - v]], color=INK, alpha=0.55, lw=1,
                    capsize=2.5, zorder=4)
        ax.text(hi + 1.2, y, f"{v:.1f}%", va="center", fontsize=9, color=INK)
    ax.set_yticks(ys)
    ax.set_yticklabels([r[1] for r in items], fontsize=10.5, color=INK)
    ax.set_xlim(0, 80)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    mc.style_ax(ax, grid="x")
    ax.set_xlabel("Harmful requests answered with useful help (higher is better)", fontsize=9)
    off_u, off_a = pct("unbound_q8", "jev_useful_off")[0], pct("atomic_lora_q8", "jev_useful_off")[0]
    p_off = PAIR["off:unbound-atomic_lora_q8"]["mcnemar_p"]
    frame(fig, "With thinking on, Unbound gives the highest share of useful answers",
          "Useful answers to harmful requests, thinking on (effort 0.9). " + SUB,
          "† Not Q8_0: AtomicChat's baked Q4 file and IndrasMirror's XQ file are run as published; "
          "every other model is Q8_0. Bars show 95% CIs.\n"
          f"With thinking off, Unbound and AtomicChat LoRA are statistically tied "
          f"({off_u:.1f}% vs {off_a:.1f}%, {fmt_p(p_off)}).")
    mc.save(fig, "compare_useful")


def chart_headtohead():
    p = PAIR["e09:unbound-atomic_lora_q8"]
    parts = [("only Unbound", p["only_unbound"], BLUE, "#ffffff"),
             ("both", p["both"], "#b9b8b3", INK),
             ("only\nAtomicChat", p["only_other"], SLATE, "#ffffff"),
             ("neither", p["neither"], "#e6e5e1", INK)]
    total = sum(x[1] for x in parts)
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=100)
    fig.subplots_adjust(left=0.035, right=0.965, top=0.62, bottom=0.2)
    x = 0
    for label, n, color, tc in parts:
        ax.barh(0, n, 0.55, left=x, color=color, edgecolor=SURFACE, linewidth=2, zorder=3)
        ax.text(x + n / 2, 0, str(n), ha="center", va="center", fontsize=14, color=tc, zorder=4)
        ax.text(x + n / 2, -0.4, label, ha="center", va="top", fontsize=9, color=INK2,
                linespacing=1.15)
        x += n
    ax.set_xlim(0, total)
    ax.set_ylim(-0.95, 0.5)
    ax.axis("off")
    frame(fig, f"Unbound is {p['diff_pts']:.0f} points ahead of AtomicChat LoRA "
          f"({fmt_p(p['mcnemar_p'])})",
          f"Same {total} prompts scored for both models, thinking on. Each prompt is useful for "
          "one model, both, or neither.",
          f"Difference {p['diff_pts']:+.1f} points, 95% CI [{p['CI_pts'][0]:.1f}, "
          f"{p['CI_pts'][1]:.1f}]; paired McNemar test. llama.cpp, both Q8_0, Jev judge.")
    mc.save(fig, "compare_headtohead")


CATS = [("cbrn_weapons", "CBRN weapons"), ("crime_illegal_goods", "Crime, illegal goods"),
        ("cyber", "Cyber"), ("hate_harassment", "Hate, harassment"),
        ("misinfo_deception", "Misinformation"), ("other_harm", "Other harm"),
        ("sexual", "Sexual"), ("violence", "Violence")]
SERIES = [("unbound_q8", "Unbound (ours)", BLUE), ("atomic_lora_q8", "AtomicChat LoRA (Q8)", SLATE),
          ("ou_q8", "OpenUncensored", ORANGE_LT)]


def parse_cat(s):
    m = re.match(r"(\d+)% \(n=(\d+)\)", s)
    return int(m.group(1)), int(m.group(2))


def chart_categories():
    data = {k: {c: parse_cat(PAIR[f"e09:cat:{k}"][c]) for c, _ in CATS} for k, _, _ in SERIES}
    fig, ax = plt.subplots(figsize=(11, 5.2), dpi=100)
    fig.subplots_adjust(left=0.06, right=0.98, top=0.76, bottom=0.2)
    w = 0.26
    for gi, (c, _) in enumerate(CATS):
        for si, (k, _, color) in enumerate(SERIES):
            v, n = data[k][c]
            x = gi + (si - 1) * (w + 0.02)
            ax.bar(x, v, w, color=color, edgecolor=SURFACE, linewidth=1.5, zorder=3)
            ax.text(x, v + 1.2, f"{v}", ha="center", fontsize=8.6, color=INK)
    labels = []
    for c, name in CATS:
        n = data[SERIES[0][0]][c][1]
        labels.append(f"{name}{'*' if n < 20 else ''}\nn = {n}")
    ax.set_xticks(range(len(CATS)))
    ax.set_xticklabels(labels, fontsize=9, color=INK, linespacing=1.4)
    ax.set_ylim(0, 95)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.set_ylabel("Useful answers (higher is better)", fontsize=9)
    mc.style_ax(ax)
    ax.tick_params(axis="x", pad=6)
    hs = [Patch(facecolor=c, edgecolor="none", label=l) for _, l, c in SERIES]
    fig.legend(handles=hs, loc="upper left", bbox_to_anchor=(0.045, 0.855), ncol=3, frameon=False,
               fontsize=9.5, handlelength=1, columnspacing=1.6, handletextpad=0.5, labelcolor=INK2)
    frame(fig, "Unbound's lead holds in most harm categories, with thinking on",
          "Useful answers to harmful requests by category, thinking on (effort 0.9). " + SUB,
          "* n < 20: small sample, read with caution. Per-category CIs are wide; the overall "
          "comparison is the more reliable result.")
    mc.save(fig, "compare_categories")


if __name__ == "__main__":
    import matplotlib.ticker  # noqa: F401
    chart_useful()
    chart_headtohead()
    chart_categories()
