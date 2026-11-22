#!/usr/bin/env python3
"""One reply-sized chart: Unbound v2 vs JoaoZaokk Heretic on the sealed test (package_v2/data_test8k.json).
Left: useful answers, thinking off / on, with 95% CIs. Right: head to head on the same 277 prompts.
Writes package_v2/charts/heretic_vs_v2.{png,svg}."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib.pyplot as plt  # noqa: E402

import make_v2 as v  # noqa: E402

MODES = [("off", "Thinking off"), ("e09", "Thinking on")]


def bars(ax):
    keys = ["heretic", "v2"]
    w = 0.36
    for gi, (mode, _) in enumerate(MODES):
        for ki, k in enumerate(keys):
            val, ci = v.m(k, f"useful_{mode}"), v.m(k, f"useful_{mode}_ci")
            x = gi + (ki - 0.5) * (w + 0.03)
            ax.bar(x, val, w, color=v.style(k)[1], hatch="////" if v.not_q8(k) else None,
                   edgecolor=v.SURFACE, linewidth=1.2, zorder=3)
            top = v.whisker(ax, x, val, ci)
            ax.text(x, top + 1.5, f"{val:.0f}%", ha="center", va="bottom", fontsize=10, color=v.INK, zorder=5)
    ax.set_xticks(range(len(MODES)))
    ax.set_xticklabels([t for _, t in MODES], fontsize=10, color=v.INK)
    ax.set_ylim(0, 105)
    ax.yaxis.set_major_formatter(v.PCT)
    ax.set_title("Useful answers to harmful requests", fontsize=10.5, color=v.INK, loc="left", pad=10)
    v.mc.style_ax(ax)


def h2h(ax):
    rows = [(v.DATA["head_to_head"][f"v2_vs_heretic_{m}"], t) for m, t in MODES]
    for i, (d, t) in enumerate(rows):
        y = len(rows) - 1 - i
        v.draw_h2h_row(ax, y, d, v.style("heretic")[1])
        ax.text(-3, y, t, ha="right", va="center", fontsize=10, color=v.INK)
    ax.set_xlim(0, 277)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.set_yticks([])
    ax.set_title("Same 277 prompts: who gave the useful answer", fontsize=10.5, color=v.INK, loc="left", pad=10)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=v.BLUE, label="only v2 useful"), Patch(color="#b9b8b3", label="both"),
                       Patch(color="#e6e5e1", label="neither"), Patch(color=v.style("heretic")[1], label="only Heretic useful")],
              loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=4, frameon=False, fontsize=9, handlelength=1,
              columnspacing=1.2, labelcolor=v.INK2)
    v.mc.style_ax(ax)


def main():
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 4.6), dpi=100, gridspec_kw={"width_ratios": [1, 1.25]})
    fig.subplots_adjust(left=0.07, right=0.97, top=0.72, bottom=0.2, wspace=0.42)
    bars(a)
    h2h(b)
    v.legend(fig, [v.patch("heretic"), v.patch("v2")], y=0.845)
    off2, offh = v.m("v2", "useful_off"), v.m("heretic", "useful_off")
    on2, onh = v.m("v2", "useful_e09"), v.m("heretic", "useful_e09")
    v.frame(fig, f"Unbound v2 vs Heretic: {off2:.0f}% vs {offh:.0f}% useful (thinking off), {on2:.0f}% vs {onh:.0f}% (on)",
            "277 held-out harmful prompts · same prompts, settings, answer cap (8,192) and judge (Jev)",
            "† Heretic ships only 4-bit files that stock llama.cpp can't load; it was run in a compatible llama.cpp build "
            "(0xE2E Inkling engine, Unsloth layout). Unbound v2: Q8_0, stock llama.cpp.")
    v.save(fig, "heretic_vs_v2")


if __name__ == "__main__":
    main()
