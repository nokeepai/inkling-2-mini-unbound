#!/usr/bin/env python3
"""Build the model-card charts. Re-reads every source at run time.

Usage: python3 -I scripts/charts/make_charts.py
Writes PNG (2x DPI) + SVG to package/charts/ and package/charts/README_SNIPPET.md.

Sources (aggregate tables only; sealed plaintext is never read):
  audit/test/evals/<cand>_{tb,th}/metrics.json, jev_metrics.json   (sealed TEST)
  results/runs.jsonl                                               (KL, top-1)
  suite/RESULTS.md                                                 (capability)
"""
import json
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

R = "./"
OUT = R + "package/charts/"
os.makedirs(OUT, exist_ok=True)

# ---- models: fixed order, fixed colours -----------------------------------
# Blue = ours (accent). Orange = the public comparison. Neutral grey = baseline.
# Runner-up is a pale tint of the accent so it reads as "ours, secondary".
MODELS = [
    # key, label, run-name, colour
    ("orig", "Original", "orig", "#8c8b86"),
    ("ou", "OpenUncensored", "ou", "#eb6834"),
    ("U", "Unbound (ours)", "c1h_r1_svd1_s3_L4", "#2a78d6"),
]
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
NOISE_FLOOR = 0.0025  # KL noise floor measured by the fidelity agent
PENDING = []  # placeholders still waiting on final numbers

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "text.color": INK,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "svg.fonttype": "none",
})


# ---- loaders ---------------------------------------------------------------
def load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def test_metric(run, tag, key, fname="metrics.json"):
    """(value, lo, hi, n) from the sealed test split, or None."""
    d = load_json(f"{R}audit/test/evals/{run}_{tag}/{fname}")
    m = (d or {}).get("metrics", {}).get(key)
    if not m:
        return None
    lo, hi = m.get("CI", [m["value"], m["value"]])
    return m["value"] * 100, lo * 100, hi * 100, m.get("n")


def kl_rows():
    """latest kl/top1 row per (run, split, metric) on engine hf_w4a16."""
    best = {}
    with open(R + "results/runs.jsonl") as f:
        for line in f:
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get("engine") != "hf_w4a16" or r.get("metric") not in ("kl", "top1"):
                continue
            k = (r["candidate"], r["split"], r["metric"])
            if k not in best or r["time"] > best[k]["time"]:
                best[k] = r
    return best


def parse_cell(cell):
    m = re.match(r"\s*([\d.]+)\s*\[\s*([\d.]+)\s*,\s*([\d.]+)\s*\]", cell)
    return tuple(float(x) for x in m.groups()) if m else None


def capability():
    """{bench: {model_key: (v, lo, hi)}} from the 'Main metrics' table."""
    out = {}
    try:
        text = open(R + "suite/RESULTS.md").read()
    except OSError:
        return out
    sec = re.search(r"## Main metrics\n(.*?)(?:\n## |\Z)", text, re.S)
    if not sec:
        return out
    rows = [ln for ln in sec.group(1).splitlines() if ln.startswith("|")]
    if len(rows) < 3:
        return out
    head = [c.strip() for c in rows[0].strip("|").split("|")]
    colmap = {}
    for i, h in enumerate(head[1:], 1):
        for key, _, run, _ in MODELS:
            if h == run or h == key:
                colmap[i] = key
    for ln in rows[2:]:
        cells = [c.strip() for c in ln.strip("|").split("|")]
        bench = re.sub(r"[^a-z0-9]", "", cells[0].lower())
        for i, key in colmap.items():
            if i < len(cells):
                v = parse_cell(cells[i])
                if v:
                    out.setdefault(bench, {})[key] = v
    return out


# ---- drawing helpers -------------------------------------------------------
def fmt(v):
    return f"{v:.1f}%"


def frame(fig, title, subtitle, caption):
    fig.text(0.035, 0.955, title, fontsize=15, fontweight="bold", ha="left", va="top")
    fig.text(0.035, 0.90, subtitle, fontsize=10, color=INK2, ha="left", va="top")
    fig.text(0.035, 0.018, caption, fontsize=7.2, color=INK2, ha="left", va="bottom")


def legend(fig, y=0.855, models=MODELS):
    hs = [Patch(facecolor=c, edgecolor="none", label=l) for _, l, _, c in models]
    fig.legend(handles=hs, loc="upper left", bbox_to_anchor=(0.03, y), ncol=len(hs),
               frameon=False, fontsize=9, handlelength=1, columnspacing=1.6,
               handletextpad=0.5, labelcolor=INK2)


def style_ax(ax, grid="y"):
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.grid(axis=grid, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0, labelsize=9)


def grouped_bars(ax, groups, data, ymax, ylabel, err=True):
    """data[group][model_key] = (v, lo, hi) or None."""
    n = len(MODELS)
    w = 0.19
    for gi, g in enumerate(groups):
        for mi, (key, _, _, color) in enumerate(MODELS):
            d = data[g].get(key)
            if d is None:
                continue
            v, lo, hi = d[:3]
            x = gi + (mi - (n - 1) / 2) * (w + 0.02)
            ax.bar(x, v, w, color=color, edgecolor=SURFACE, linewidth=1.5, zorder=3)
            top = v
            if err and hi - lo > 0.05:
                ax.errorbar(x, v, yerr=[[v - lo], [hi - v]], color=INK, alpha=0.55,
                            lw=1, capsize=2.5, capthick=1, zorder=4)
                top = hi
            ax.text(x, top + ymax * 0.015, fmt(v), ha="center", va="bottom",
                    fontsize=8.2, color=INK, zorder=5)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups, fontsize=10.5, color=INK)
    ax.set_ylim(0, ymax)
    ax.set_xlim(-0.6, len(groups) - 0.4)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    style_ax(ax)


def save(fig, name):
    for ext in ("png", "svg"):
        fig.savefig(f"{OUT}{name}.{ext}", dpi=200, bbox_inches=None)
    plt.close(fig)
    print("wrote", name)


def collect(tag, key, fname="metrics.json"):
    return {k: test_metric(run, tag, key, fname) for k, _, run, _ in MODELS}


# ---- charts ----------------------------------------------------------------
CAP_TEST = "Source: sealed TEST split, 277 harmful prompts; 95% CIs."


def chart_refusal():
    groups = ["Thinking off", "Thinking effort 0.9"]
    data = {groups[0]: collect("th", "refusal_off"), groups[1]: collect("th", "refusal_e09")}
    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=100)
    fig.subplots_adjust(left=0.09, right=0.97, top=0.78, bottom=0.14)
    grouped_bars(ax, groups, data, 115, "Refused (lower is better)")
    ax.set_ylim(0, 112)
    ax.set_yticks([0, 25, 50, 75, 100])
    frame(fig, "The original refuses 97 in 100 harmful requests; Unbound refuses none",
          "Share of harmful requests refused, by thinking setting.", CAP_TEST)
    legend(fig)
    save(fig, "refusal")


def chart_useful():
    groups = ["Thinking off", "Thinking effort 0.9"]
    data = {groups[0]: collect("th", "jev_useful_harmful_off", "jev_metrics.json"),
            groups[1]: collect("th", "jev_useful_harmful_e09", "jev_metrics.json")}
    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=100)
    fig.subplots_adjust(left=0.09, right=0.97, top=0.78, bottom=0.14)
    grouped_bars(ax, groups, data, 80, "Useful answers (higher is better)")
    frame(fig, "Not refusing is not enough: Unbound also gives the most useful answers",
          "Share of harmful requests answered with substantive, on-task help (judge: Jev, "
          "86% agreement with hand labels).",
          CAP_TEST + " Error bars: 95% CI.")
    legend(fig)
    save(fig, "useful")


def chart_overrefusal():
    groups = ["Benign but borderline prompts"]
    data = {groups[0]: collect("tb", "overrefusal_off")}
    fig, ax = plt.subplots(figsize=(6.4, 4.8), dpi=100)
    fig.subplots_adjust(left=0.11, right=0.97, top=0.78, bottom=0.14)
    grouped_bars(ax, groups, data, 50, "Refused (lower is better)")
    frame(fig, "The original turns away 38% of harmless borderline requests",
          "Over-refusal on 784 benign prompts that merely sound risky; thinking off.",
          "Source: sealed TEST split, 784 prompts; 95% CIs.")
    legend(fig)
    ax.set_xlim(-0.6, 0.6)
    save(fig, "overrefusal")


def auditor_kl():
    """Sealed-test KL from the auditor's file: ({key: (kl, lo, hi, top1%)}, noise) or None."""
    path = R + "audit/test/kl_results.jsonl"
    try:
        rows = [json.loads(ln) for ln in open(path) if ln.strip()]
    except (OSError, ValueError):
        return None
    d, noise = {}, None
    for r in rows:
        if r.get("cand") == "orig" and r.get("tag") == "noise_bs1":
            noise = r["kl"]
        for key, _, run, _ in MODELS[1:]:
            if r.get("cand") == run and r.get("tag") == "baseline":
                d[key] = (r["kl"], r["kl_ci"][0], r["kl_ci"][1], r["top1"] * 100)
    return (d, noise) if len(d) == len(MODELS) - 1 else None


def kl_data():
    global NOISE_FLOOR
    aud = auditor_kl()
    if aud:
        NOISE_FLOOR = aud[1] or NOISE_FLOOR
        return aud[0], "test"
    rows = kl_rows()
    use_test = all((r, "test", "kl") in rows for _, _, r, _ in MODELS[1:])
    split = "test" if use_test else "val"
    if not use_test:
        PENDING.append("drift.png uses VAL-split KL; swap to test automatically when rows land")
    d = {}
    for key, _, run, _ in MODELS[1:]:
        r = rows.get((run, split, "kl"))
        t = rows.get((run, split, "top1"))
        if r:
            d[key] = (r["value"], r["CI"][0], r["CI"][1], t["value"] * 100 if t else None)
    return d, split


def chart_drift():
    d, split = kl_data()
    tag = "" if split == "test" else " (val)"
    fig, ax = plt.subplots(figsize=(8, 4.4), dpi=100)
    fig.subplots_adjust(left=0.22, right=0.95, top=0.72, bottom=0.17)
    items = [(k, l, c) for k, l, _, c in MODELS[1:] if k in d]
    ys = list(range(len(items)))[::-1]
    xmax = max(v[2] for v in d.values()) * 1.18
    for y, (k, label, color) in zip(ys, items):
        v, lo, hi, t1 = d[k]
        ax.barh(y, v, 0.5, color=color, edgecolor=SURFACE, linewidth=1.5, zorder=3)
        ax.errorbar(v, y, xerr=[[v - lo], [hi - v]], color=INK, alpha=0.55, lw=1,
                    capsize=2.5, zorder=4)
        t = f"{v:.3f}" + (f"   top-1 agreement {t1:.1f}%" if t1 else "")
        ax.text(hi + xmax * 0.015, y, t, va="center", fontsize=8.6, color=INK)
    ax.axvline(NOISE_FLOOR, color=INK2, lw=1, ls=(0, (3, 3)), zorder=2)
    ax.text(NOISE_FLOOR + xmax * 0.008, ys[0] + 0.62, "measurement noise floor",
            fontsize=7.8, color=INK2, va="bottom")
    ax.set_yticks(ys)
    ax.set_yticklabels([l for _, l, _ in items], fontsize=10.5, color=INK)
    ax.set_xlim(0, xmax * 1.55)
    ax.set_xlabel("KL divergence from the original model (lower is closer)", fontsize=9)
    style_ax(ax, grid="x")
    frame(fig, "Unbound's outputs stay close to the original's",
          "How far each model's next-token distribution moved from the original's." + tag,
          ("Source: auditor, sealed TEST split, 1,000 prompts; " if split == "test"
           else "Source: results/runs.jsonl, val split, 200 passages; ")
          + f"95% CIs. The original is 0 by definition. Noise floor {NOISE_FLOOR:.4f}.")
    save(fig, "drift")


BENCHES = [("mmlupro", "MMLU-Pro"), ("gsm8k", "GSM8K"), ("math500", "MATH-500"),
           ("gpqadiamond", "GPQA-Diamond"), ("ifeval", "IFEval"), ("humanevalplus", "HumanEval+")]


def chart_capability():
    cap = capability()
    fig, axes = plt.subplots(2, 3, figsize=(10, 6), dpi=100)
    fig.subplots_adjust(left=0.13, right=0.97, top=0.76, bottom=0.12, hspace=0.55, wspace=0.12)
    for ax, (bkey, bname) in zip(axes.flat, BENCHES):
        d = cap.get(bkey, {})
        ax.set_title(bname, fontsize=10.5, fontweight="bold", loc="left", color=INK)
        ys = list(range(len(MODELS)))[::-1]
        vals = [x for v in d.values() for x in v]
        if not vals:
            PENDING.append(f"capability: {bname} has no numbers yet")
            ax.text(0.5, 0.5, "pending", transform=ax.transAxes, ha="center",
                    va="center", color=INK2, fontsize=10)
            style_ax(ax, "x")
            ax.set_yticks([])
            ax.set_xticks([])
            continue
        lo_all, hi_all = min(vals), max(vals)
        pad = max((hi_all - lo_all) * 0.2, 2)
        ax.set_xlim(max(0, lo_all - pad), min(100, hi_all + pad))
        if "orig" in d:  # original's CI as a reference band
            ax.axvspan(d["orig"][1], d["orig"][2], color=MODELS[0][3], alpha=0.16, lw=0, zorder=1)
        for y, (key, label, _, color) in zip(ys, MODELS):
            if key not in d:
                if key in ("U", "F"):
                    PENDING.append(f"capability: {bname} / {label} pending")
                continue
            v, lo, hi = d[key]
            ax.plot([lo, hi], [y, y], color=color, lw=2, solid_capstyle="round", zorder=3)
            ax.plot(v, y, "o", ms=8, color=color, mec=SURFACE, mew=2, zorder=4)
            ax.text(v, y + 0.34, f"{v:.1f}", ha="center", fontsize=7.8, color=INK)
        ax.set_ylim(-0.6, len(MODELS) - 0.3)
        ax.set_yticks(ys)
        ax.set_yticklabels([l.replace(" (ours, runner-up)", " (ours)") for _, l, _, _ in MODELS]
                           if ax in axes[:, 0] else [], fontsize=8.8)
        style_ax(ax, "x")
        ax.tick_params(axis="x", labelsize=8)
    for ax in axes[:, 0]:
        ax.set_yticks(list(range(len(MODELS)))[::-1])
        ax.set_yticklabels([l.replace(" (ours, runner-up)", " (ours)") for _, l, _, _ in MODELS],
                           fontsize=8.8)
    frame(fig, "Uncensoring costs little capability: scores stay in the original's range",
          "Accuracy (%) per benchmark. Dots are scores, lines are 95% CIs, shaded band is the "
          "original's CI.",
          "Source: suite/RESULTS.md, one GPU type for every model, subsets of each benchmark. "
          "GPQA-Diamond at a 4,096-token thinking cap; many items hit the cap, so it partly "
          "measures solving within the cap.")
    save(fig, "capability")


def chart_hero():
    specs = [
        ("Refuses harmful requests", "lower is better", "refusal_off", "th", "metrics.json", 105),
        ("Gives a useful answer", "higher is better", "jev_useful_harmful_off", "th",
         "jev_metrics.json", 75),
        ("Refuses harmless borderline", "lower is better", "overrefusal_off", "tb",
         "metrics.json", 45),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(11, 4.2), dpi=100)
    fig.subplots_adjust(left=0.04, right=0.98, top=0.60, bottom=0.14, wspace=0.12)
    for ax, (title, sub, key, tag, fname, ymax) in zip(axes, specs):
        d = collect(tag, key, fname)
        for i, (k, _, _, color) in enumerate(MODELS):
            if d.get(k) is None:
                continue
            v = d[k][0]
            ax.bar(i, v, 0.72, color=color, edgecolor=SURFACE, linewidth=1.5, zorder=3)
            ax.text(i, v + ymax * 0.02, f"{v:.0f}%", ha="center", fontsize=9.5, color=INK)
        ax.set_ylim(0, ymax)
        ax.set_xlim(-0.6, len(MODELS) - 0.4)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, fontsize=11.5, fontweight="bold", loc="left", color=INK, pad=18)
        ax.text(0, 1.02, sub, transform=ax.transAxes, fontsize=8.5, color=INK2, va="bottom")
        style_ax(ax, "y")
        ax.spines["left"].set_visible(False)
        ax.grid(False)
    frame(fig, "Unbound answers what the original refuses, without breaking the benign cases",
          "Thinking off. Sealed TEST split: 277 harmful prompts, 784 benign borderline prompts. "
          "Useful answers judged by Jev.",
          "Sources: sealed audit, test split. Per-chart detail with 95% CIs in the charts below.")
    legend(fig, y=0.855)
    save(fig, "hero")


SNIPPET = """## Results

![Summary](charts/hero.png)

![Refusal rate on harmful requests](charts/refusal.png)
*Harmful requests refused, thinking off and effort 0.9. Unbound refuses none.*

![Useful answers](charts/useful.png)
*Share of harmful requests that get a genuinely useful answer (Jev judge, 95% CIs).*

![Over-refusal](charts/overrefusal.png)
*Harmless but borderline prompts refused. The original refuses 38%; the uncensored models refuse almost none.*

![KL drift](charts/drift.png)
*How far each model's output distribution moved from the original (lower is closer).*

![Capability](charts/capability.png)
*Capability benchmarks with 95% CIs against the original's range.*
"""


def main():
    chart_refusal()
    chart_useful()
    chart_overrefusal()
    chart_drift()
    chart_capability()
    chart_hero()
    with open(OUT + "README_SNIPPET.md", "w") as f:
        f.write(SNIPPET)
    print("PENDING:")
    for p in sorted(set(PENDING)):
        print(" -", p)


if __name__ == "__main__":
    sys.exit(main())
