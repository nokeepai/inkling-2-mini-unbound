#!/usr/bin/env python3
"""Banner for the HF model card: package/charts/banner.{png,svg}, 1600x800 at 2x.
Numbers are read live (same sources as make_charts.py). The banner is about our
model; the original is the reference, OpenUncensored appears only in the footer."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Ellipse, FancyBboxPatch  # noqa: E402

import make_charts as mc  # noqa: E402

BLUE, GREY = "#2a78d6", "#8c8b86"
INK, INK2, GRID, SURFACE = mc.INK, mc.INK2, mc.GRID, mc.SURFACE
RUN = {k: r for k, _, r, _ in mc.MODELS}


def tm(key, tag, metric, fname="metrics.json"):
    return mc.test_metric(RUN[key], tag, metric, fname)


def live_numbers():
    n = {
        "ref": tm("U", "th", "refusal_off"), "ref9": tm("U", "th", "refusal_e09"),
        "ref_orig": tm("orig", "th", "refusal_off"),
        "use": tm("U", "th", "jev_useful_harmful_off", "jev_metrics.json"),
        "use9": tm("U", "th", "jev_useful_harmful_e09", "jev_metrics.json"),
        "use_orig": tm("orig", "th", "jev_useful_harmful_off", "jev_metrics.json"),
        "use_ou": tm("ou", "th", "jev_useful_harmful_off", "jev_metrics.json"),
    }
    cap = mc.capability()
    inside = 0
    for bkey, _ in mc.BENCHES:
        d = cap.get(bkey, {})
        if "orig" in d and "U" in d and d["orig"][1] <= d["U"][0] <= d["orig"][2]:
            inside += 1
    kl, split = mc.kl_data()
    n.update(inside=inside, total=len(mc.BENCHES), kl_u=kl["U"][0], kl_ou=kl["ou"][0], split=split)
    return n


def mini_bars(fig, x, y, w, vals, vmax):
    """Two thin bars on a shared track: ours (blue) over the original (grey)."""
    h = 0.018
    for i, (v, c) in enumerate(zip(vals, (BLUE, GREY))):
        yy = y - i * 0.034
        fig.patches.append(FancyBboxPatch((x, yy), w, h, boxstyle="round,pad=0,rounding_size=0.008",
                                          fc="#efeeea", ec="none", transform=fig.transFigure))
        if v > 0:
            fig.patches.append(FancyBboxPatch((x, yy), max(w * v / vmax, 0.006), h,
                                              boxstyle="round,pad=0,rounding_size=0.008",
                                              fc=c, ec="none", transform=fig.transFigure))


def callout(fig, x, label, big, sub, ref_text, ref_size=10.5):
    fig.text(x, 0.625, label, fontsize=10, color=INK2, va="bottom")
    fig.text(x, 0.43, big, fontsize=50, color=BLUE, va="bottom")
    fig.text(x, 0.425, sub, fontsize=9.5, color=INK2, va="top", linespacing=1.15)
    fig.text(x, 0.295, "●", fontsize=9, color=GREY, va="center")
    fig.text(x + 0.02, 0.295, ref_text, fontsize=ref_size, color=INK, va="center", linespacing=1.15)


def main():
    n = live_numbers()
    ref, use, inside, total = n["ref"][0], n["use"][0], n["inside"], n["total"]
    n_harm = n["ref"][3]
    both_zero = n["ref"][0] == 0 and n["ref9"][0] == 0
    fig = plt.figure(figsize=(8, 4), dpi=400, facecolor=SURFACE)
    fig.text(0.04, 0.93, "Inkling-2-Mini-Unbound", fontsize=27, va="top")
    fig.text(0.04, 0.80, "Refusal removed. Capability kept.", fontsize=14.5, color=INK2, va="top")
    fig.add_artist(plt.Line2D([0.04, 0.96], [0.70, 0.70], color=GRID, lw=1, transform=fig.transFigure))

    xs = (0.04, 0.37, 0.70)
    setting = "thinking on or off" if both_zero else "thinking off"
    callout(fig, xs[0], "HARMFUL REFUSAL", f"{ref:.0f}%",
            f"of {n_harm} harmful test requests\nrefused, {setting}",
            f"original model: {n['ref_orig'][0]:.1f}%")
    callout(fig, xs[1], "USEFUL ANSWERS", f"{use:.0f}%",
            f"of harmful requests answered usefully\n({n['use9'][0]:.0f}% with thinking)",
            f"original model: {n['use_orig'][0]:.1f}%")
    inside_txt = f"{inside} / {total}"
    callout(fig, xs[2], "CAPABILITY KEPT", inside_txt,
            "benchmarks within the original's\n95% CI",
            "MMLU-Pro, GSM8K, MATH,\nGPQA, IFEval, HumanEval+", ref_size=8.8)
    mini_bars(fig, xs[0], 0.19, 0.25, (ref, n["ref_orig"][0]), 100)
    mini_bars(fig, xs[1], 0.19, 0.25, (use, n["use_orig"][0]), 100)
    for i in range(total):  # one dot per benchmark
        # 2:1 figure, so height is twice the width in figure units to draw a true circle
        fig.patches.append(Ellipse((xs[2] + 0.012 + i * 0.038, 0.165), 0.016, 0.032,
                                   fc=BLUE if i < inside else "#d9d8d3", ec="none",
                                   transform=fig.transFigure))
    for x in xs[1:]:
        fig.add_artist(plt.Line2D([x - 0.025] * 2, [0.14, 0.68], color=GRID, lw=1,
                                  transform=fig.transFigure))

    tag = "" if n["split"] == "test" else " (val KL)"
    fig.add_artist(plt.Line2D([0.04, 0.96], [0.115, 0.115], color=GRID, lw=1, transform=fig.transFigure))
    fig.text(0.04, 0.072, "Drop-in checkpoint  ·  only 20 BF16 attention-output tensors edited  "
             "·  bit-exact reproducible", fontsize=10.5, color=INK2, va="center")
    fig.text(0.04, 0.028, f"vs OpenUncensored: more useful answers ({use:.0f}% vs "
             f"{n['use_ou'][0]:.0f}%), less drift (KL {n['kl_u']:.3f} vs {n['kl_ou']:.3f}){tag}",
             fontsize=8.5, color=INK2, va="center")
    for ext in ("png", "svg"):
        fig.savefig(f"{mc.OUT}banner.{ext}", dpi=400 if ext == "png" else 100)
    print("banner", {k: (v[0] if isinstance(v, tuple) else v) for k, v in n.items()})


if __name__ == "__main__":
    main()
