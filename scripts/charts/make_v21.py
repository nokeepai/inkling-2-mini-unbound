#!/usr/bin/env python3
"""Model-card charts for Inkling-2-Mini-Unbound (current version v2.1).

Usage: python3 scripts/charts/make_v21.py [--rebuild]
Reads package_v21/data_v21.json (built by data_v21.py; --rebuild runs it first) and writes PNG + SVG to
package_v21/charts/. Same palette and layout conventions as the v2 charts (make_v2.py).
Every comparison is on the 263-prompt basis: v2's sealed harmful test without its CBRN category.
"""
from __future__ import annotations

import json
import os
import sys
import textwrap

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import matplotlib  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
from matplotlib.colors import to_rgb  # noqa: E402
from matplotlib.font_manager import FontProperties  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Patch  # noqa: E402

import make_charts as mc  # noqa: E402  (palette, rcParams, style_ax)

R = os.path.abspath(os.path.join(HERE, "..", "..")) + "/"
if "--rebuild" in sys.argv:
    import data_v21
    data_v21.main()
DATA = json.load(open(R + "package_v21/data_v21.json"))
OUT = R + "package_v21/charts/"
os.makedirs(OUT, exist_ok=True)

INK, INK2, GRID, SURFACE = mc.INK, mc.INK2, mc.GRID, mc.SURFACE
BLUE, BLUE_MID, BLUE_LT, GREY = "#2a78d6", "#6fa3e6", "#b3d0f2", "#8c8b86"
SLATE, SLATE_LT, PLUM, ORANGE, OCHRE = "#4f8a8b", "#8fb8b5", "#8e7fa8", "#eb6834", "#b5893a"
GREEN, RED, TILE = "#3f8f5a", "#b4472f", "#f1f0ec"
BOLD = FontProperties(fname="/System/Library/Fonts/Supplemental/Arial Bold.ttf") \
    if os.path.exists("/System/Library/Fonts/Supplemental/Arial Bold.ttf") else FontProperties(weight="bold")

# key -> (label, colour, ours). Order = legend order.
MODEL_STYLE = {
    "v21": ("Unbound v2.1 (ours)", BLUE, True),
    "v2": ("Unbound v2 (ours)", BLUE_MID, True),
    "v1": ("Unbound v1.0 (ours)", BLUE_LT, True),
    "xq": ("IndrasMirror XQ", PLUM, False),
    "atomic_lora": ("AtomicChat LoRA", SLATE, False),
    "atomic_adq4": ("AtomicChat Q4 file", SLATE_LT, False),
    "ou": ("OpenUncensored", ORANGE, False),
    "heretic": ("JoaoZaokk Heretic", OCHRE, False),
    "orig": ("Original", GREY, False),
}
CATS = {"crime_illegal_goods": "Crime &\nillegal goods", "cyber": "Cyber", "hate_harassment": "Hate &\nharassment",
        "misinfo_deception": "Misinfo &\ndeception", "other_harm": "Other harm", "sexual": "Sexual",
        "violence": "Violence"}
N = DATA["_meta"]["n"]
SUB = f"{N} held-out harmful prompts · llama.cpp Q8_0 · answer cap 8,192 · Jev judge"
BASIS = f"Basis: v2's sealed test without its CBRN category ({N} of 277 prompts), for every model."
PCT = matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0f}%")
MODES = [("off", "Thinking off"), ("e09", "Thinking on (effort 0.9)")]


# ---- data helpers -----------------------------------------------------------
def m(key: str, metric: str) -> float:
    v = DATA["models"][key][metric]
    return v["value"] if isinstance(v, dict) and "value" in v else v


def ci(key: str, metric: str) -> list[float]:
    return DATA["models"][key][metric]["ci"]


def st(key: str, field: str):
    return DATA["static"].get(key, {}).get(field)


def not_q8(key: str) -> bool:
    return bool(st(key, "not_q8"))


def label(key: str) -> str:
    return MODEL_STYLE[key][0] + (" †" if not_q8(key) else "")


def color(key: str) -> str:
    return MODEL_STYLE[key][1]


def others(exclude=("v21",)) -> list[str]:
    return [k for k in MODEL_STYLE if k not in exclude]


def best_other(mode: str, exclude=("v21", "orig")) -> str:
    return max(others(exclude), key=lambda k: m(k, f"useful_{mode}"))


def v21_best() -> bool:
    return all(m("v21", f"useful_{md}") > m(k, f"useful_{md}") for k in others() for md, _ in MODES)


def h2h(other: str, mode: str) -> dict:
    return DATA["head_to_head"][f"v21_vs_{other}_{mode}"]


def fmt_pct(v: float) -> str:
    return f"{v:.1f}%" if v < 10 else f"{v:.0f}%"


def not_q8_note(keys) -> str:
    names = [MODEL_STYLE[k][0] for k in keys if not_q8(k)]
    if not names:
        return ""
    return ("† Not Q8_0 (about 4-bit, run as published): " + ", ".join(names) + ".\nEvery other model is Q8_0 "
            "(AtomicChat LoRA: Q8_0 + LoRA).")


# ---- drawing helpers --------------------------------------------------------
def frame(fig, title, subtitle, caption=""):
    fig.text(0.035, 0.955, title, fontsize=15, ha="left", va="top", fontproperties=BOLD)
    fig.text(0.035, 0.895, subtitle, fontsize=10, color=INK2, ha="left", va="top", linespacing=1.4)
    if caption:
        fig.text(0.035, 0.02, caption, fontsize=7.6, color=INK2, ha="left", va="bottom", linespacing=1.4)


def legend(fig, handles, y=0.855, x=0.03, ncol=None):
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(x, y), ncol=ncol or len(handles), frameon=False,
               fontsize=9.5, handlelength=1, columnspacing=1.6, handletextpad=0.5, labelcolor=INK2)


def patch(key: str) -> Patch:
    return Patch(facecolor=color(key), edgecolor="none", hatch="////" if not_q8(key) else None, label=label(key))


def save(fig, name: str, dpi: int = 200):
    for ext in ("png", "svg"):
        fig.savefig(f"{OUT}{name}.{ext}", dpi=dpi if ext == "png" else 100, bbox_inches=None)
    plt.close(fig)
    print("wrote", name)


def whisker(ax, x, v, c, horizontal=False):
    lo, hi = c
    kw = dict(color=INK, alpha=0.55, lw=1, capsize=2.5, capthick=1, zorder=4)
    if horizontal:
        ax.errorbar(v, x, xerr=[[v - lo], [hi - v]], **kw)
    else:
        ax.errorbar(x, v, yerr=[[v - lo], [hi - v]], **kw)
    return hi


def tint(c, amount=0.8):
    rgb, bg = to_rgb(c), to_rgb(SURFACE)
    return tuple(a * (1 - amount) + b * amount for a, b in zip(rgb, bg))


# ---- 1. banner --------------------------------------------------------------
def chart_banner():
    fig = plt.figure(figsize=(8, 4), dpi=400, facecolor=SURFACE)
    fig.text(0.04, 0.93, "Inkling-2-Mini-Unbound", fontsize=27, va="top")
    fig.text(0.96, 0.905, "v2.1", fontsize=20, va="top", ha="right", color=BLUE)
    off, on = m("v21", "useful_off"), m("v21", "useful_e09")
    lead = "The most useful uncensored Inkling 2 Mini" if v21_best() else "Uncensored Inkling 2 Mini that answers"
    fig.text(0.04, 0.80, f"{lead}: {off:.0f}% useful thinking off, {on:.0f}% thinking on", fontsize=12.5,
             color=INK2, va="top")
    fig.add_artist(Line2D([0.04, 0.96], [0.70, 0.70], color=GRID, lw=1, transform=fig.transFigure))
    for x, title, mode in ((0.04, "USEFUL, THINKING OFF", "off"), (0.52, "USEFUL, THINKING ON", "e09")):
        bo = best_other(mode, ("v21", "v2", "v1", "orig"))
        v, vb, vo = m("v21", f"useful_{mode}"), m(bo, f"useful_{mode}"), m("orig", f"useful_{mode}")
        fig.text(x, 0.625, title, fontsize=10, color=INK2, va="bottom")
        fig.text(x, 0.40, f"{v:.0f}%", fontsize=50, color=BLUE, va="bottom")
        fig.text(x + 0.21, 0.50, f"best other release:\n{MODEL_STYLE[bo][0]} {vb:.0f}%\noriginal: {vo:.1f}%",
                 fontsize=10, color=INK2, va="center", linespacing=1.45)
        for i, (val, c) in enumerate(zip((v, vb, vo), (BLUE, color(bo), GREY))):
            yy = 0.30 - i * 0.045
            fig.add_artist(plt.Rectangle((x, yy), 0.40, 0.022, fc="#efeeea", ec="none", transform=fig.transFigure))
            fig.add_artist(plt.Rectangle((x, yy), max(0.40 * val / 100, 0.003), 0.022, fc=c, ec="none",
                                         transform=fig.transFigure))
    fig.add_artist(Line2D([0.50] * 2, [0.18, 0.68], color=GRID, lw=1, transform=fig.transFigure))
    fig.add_artist(Line2D([0.04, 0.96], [0.12, 0.12], color=GRID, lw=1, transform=fig.transFigure))
    fig.text(0.04, 0.06, SUB, fontsize=9, color=INK2, va="center")
    save(fig, "banner_v21", dpi=400)


# ---- 2. hero: grouped bars, the releases people choose between ---------------
def chart_hero():
    keys = ["orig", "heretic", "ou", "atomic_lora", "xq", "v1", "v2", "v21"]
    fig, ax = plt.subplots(figsize=(10, 5.2), dpi=100)
    fig.subplots_adjust(left=0.075, right=0.98, top=0.72, bottom=0.2)
    w = 0.86 / len(keys)
    for gi, (mode, _) in enumerate(MODES):
        for ki, k in enumerate(keys):
            v = m(k, f"useful_{mode}")
            x = gi + (ki - (len(keys) - 1) / 2) * w
            ax.bar(x, v, w * 0.92, color=color(k), edgecolor=SURFACE, linewidth=1.0, zorder=3,
                   hatch="////" if not_q8(k) else None)
            top = whisker(ax, x, v, ci(k, f"useful_{mode}"))
            ax.text(x, top + 1.5, fmt_pct(v), ha="center", va="bottom", fontsize=8.6, color=INK,
                    fontweight="bold" if k == "v21" else "normal", zorder=5)
        bo = best_other(mode, ("v21", "v2", "v1", "orig"))
        d = m("v21", f"useful_{mode}") - m(bo, f"useful_{mode}")
        ax.text(gi, 101, f"v2.1 {d:+.0f} pts over the best non-Unbound release", ha="center", va="bottom", fontsize=10,
                color=BLUE)
    ax.set_xticks(range(len(MODES)))
    ax.set_xticklabels([t for _, t in MODES], fontsize=10.5, color=INK)
    ax.set_ylim(0, 110)
    ax.set_xlim(-0.52, len(MODES) - 0.48)
    ax.set_yticks(range(0, 101, 20))
    ax.yaxis.set_major_formatter(PCT)
    ax.set_ylabel("Useful answers (higher is better)", fontsize=9)
    mc.style_ax(ax)
    legend(fig, [patch(k) for k in keys], ncol=4, y=0.87)
    title = ("Unbound v2.1 gives a useful answer more often than any other release" if v21_best()
             else "Useful answers: Unbound v2.1 against every other release")
    frame(fig, title, "Useful answers to harmful requests. " + SUB,
          "Same prompts, engine, settings and judge for every model. Whiskers: 95% bootstrap CIs.\n"
          + not_q8_note(keys) + "\n" + BASIS)
    save(fig, "hero_v21")


# ---- 3. all models, horizontal ----------------------------------------------
def chart_compare_useful():
    keys = sorted(MODEL_STYLE, key=lambda k: m(k, "useful_e09"))
    fig, ax = plt.subplots(figsize=(8, 1.9 + 0.55 * len(keys)), dpi=100)
    h = fig.get_figheight()
    fig.subplots_adjust(left=0.27, right=0.86, top=1 - 1.3 / h, bottom=1.3 / h)
    for y, k in enumerate(keys):
        v, off = m(k, "useful_e09"), m(k, "useful_off")
        ax.barh(y, v, 0.62, color=color(k), edgecolor=SURFACE, linewidth=1.5, zorder=3,
                hatch="////" if not_q8(k) else None)
        hi = whisker(ax, y, v, ci(k, "useful_e09"), horizontal=True)
        ax.plot(off, y, "D", ms=6.5, mfc=SURFACE, mec=INK, mew=1.3, zorder=5)
        ax.text(max(hi, off) + 1.5, y, f"{v:.1f}% · off {off:.1f}%", va="center", fontsize=9, color=INK,
                fontweight="bold" if k == "v21" else "normal")
    ax.set_yticks(range(len(keys)))
    ax.set_yticklabels([label(k) for k in keys], fontsize=10.5, color=INK)
    for t, k in zip(ax.get_yticklabels(), keys):
        t.set_fontweight("bold" if k == "v21" else "normal")
    ax.set_xlim(0, 100)
    ax.xaxis.set_major_formatter(PCT)
    mc.style_ax(ax, grid="x")
    ax.set_xlabel("Harmful requests answered with useful help (higher is better)", fontsize=9)
    legend(fig, [Patch(facecolor=BLUE, label="Thinking on (effort 0.9), 95% CI"),
                 Line2D([], [], marker="D", ls="", mfc=SURFACE, mec=INK, mew=1.3, label="Thinking off")],
           y=1 - 0.98 / h)
    title = ("Unbound v2.1 gives the most useful answers of any model tested" if v21_best()
             else "Useful answers, every model tested")
    frame(fig, title, "Sorted by thinking on. " + SUB.replace(" Q8_0", ""),
          not_q8_note(keys) + "\nBars: thinking on with 95% CIs; diamonds: thinking off.\n" + BASIS)
    save(fig, "compare_useful_v21")


# ---- 4. head to head --------------------------------------------------------
def draw_h2h_row(ax, y, d, other_color, total):
    parts = [(d["a_only"], BLUE, "#ffffff"), (d["both"], "#b9b8b3", INK), (d["neither"], "#e6e5e1", INK),
             (d["b_only"], other_color, INK)]
    x, thin = 0, total * 0.03
    for n, c, tc in parts:
        ax.barh(y, n, 0.62, left=x, color=c, edgecolor=SURFACE, linewidth=2, zorder=3)
        if n >= thin:
            ax.text(x + n / 2, y, str(n), ha="center", va="center", fontsize=10.5, color=tc)
        x += n
    if d["b_only"] < thin:
        ax.text(x + total * 0.005, y, str(d["b_only"]), ha="left", va="center", fontsize=10.5, color=INK)


def chart_headtohead():
    rows = [(o, md) for o in ("v2", "xq", "atomic_lora") for md, _ in MODES]
    fig, ax = plt.subplots(figsize=(8.6, 1.9 + 0.72 * len(rows)), dpi=100)
    h = fig.get_figheight()
    fig.subplots_adjust(left=0.25, right=0.85, top=1 - 1.65 / h, bottom=0.55 / h)
    for y, (o, md) in enumerate(reversed(rows)):
        draw_h2h_row(ax, y, h2h(o, md), color(o), N)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"vs {label(o)}\n{'thinking off' if md == 'off' else 'thinking on'}" for o, md in reversed(rows)],
                       fontsize=9.5, color=INK, linespacing=1.3)
    for y, (o, md) in enumerate(reversed(rows)):
        d = h2h(o, md)
        ax.text(N * 1.035, y, f"{d['diff']:+.1f} pts\n[{d['ci'][0]:+.1f}, {d['ci'][1]:+.1f}]", va="center",
                ha="left", fontsize=8.5, color=GREEN if d["ci"][0] > 0 else INK2, linespacing=1.25)
    ax.set_xlim(0, N)
    ax.set_xticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    legend(fig, [Patch(facecolor=BLUE, label="only v2.1 useful"), Patch(facecolor="#b9b8b3", label="both useful"),
                 Patch(facecolor="#e6e5e1", label="neither"),
                 Patch(facecolor=BLUE_MID, label="only the other useful (its colour)")], y=1 - 1.08 / h)
    frame(fig, "Where v2.1 and another model disagree, v2.1 is usually the useful one",
          "Paired on the same prompts: useful for v2.1 only, both, neither, or the other only.\n" + SUB,
          "Right: gain in useful share (v2.1 minus the other), paired bootstrap 95% CI. " + BASIS)
    save(fig, "headtohead_v21")


# ---- 5. categories ----------------------------------------------------------
def chart_categories():
    keys = [k for k in MODEL_STYLE if k != "orig"]
    nbc = DATA["models"]["v21"]["n_by_cat"]
    cats = list(CATS)
    fig, ax = plt.subplots(figsize=(11.5, 5.6), dpi=100)
    fig.subplots_adjust(left=0.06, right=0.985, top=0.72, bottom=0.24)
    w = 0.84 / len(keys)
    for gi, c in enumerate(cats):
        top_k = max(keys, key=lambda k: m(k, "useful_e09_by_cat")[c])
        for si, k in enumerate(keys):
            v = m(k, "useful_e09_by_cat")[c]
            x = gi + (si - (len(keys) - 1) / 2) * w
            ax.bar(x, v, w * 0.9, color=color(k), edgecolor=SURFACE, linewidth=0.8, zorder=3,
                   hatch="////" if not_q8(k) else None)
            ax.text(x, v + 1.2, f"{v:.0f}", ha="center", fontsize=6.4, color=INK,
                    fontweight="bold" if k == top_k else "normal")
    ax.set_xticks(range(len(cats)))
    ax.set_xticklabels([f"{CATS[c]}{'*' if nbc[c] < 20 else ''}\nn = {nbc[c]}" for c in cats], fontsize=9,
                       color=INK, linespacing=1.3)
    ax.set_ylim(0, 108)
    ax.set_xlim(-0.5, len(cats) - 0.5)
    ax.yaxis.set_major_formatter(PCT)
    ax.set_ylabel("Useful answers (higher is better)", fontsize=9)
    mc.style_ax(ax)
    legend(fig, [patch(k) for k in keys], x=0.045, ncol=4, y=0.865)
    leads = sum(1 for c in cats if all(m("v21", "useful_e09_by_cat")[c] >= m(k, "useful_e09_by_cat")[c] for k in keys))
    title = (f"v2.1 leads or ties in every harm category, with thinking on" if leads == len(cats)
             else f"v2.1 leads or ties in {leads} of {len(cats)} harm categories, with thinking on")
    frame(fig, title, "Useful answers to harmful requests by category, thinking on (effort 0.9). " + SUB,
          "* n < 20: small sample, read with caution. Per-category CIs are wide; the overall comparison is the more "
          "reliable result.\n" + not_q8_note(keys) + "\n" + BASIS)
    save(fig, "categories_v21")


# ---- 6. trade-off -----------------------------------------------------------
LABEL_OFFSET = {"atomic_lora": (0, -18, "center"), "atomic_adq4": (9, -3, "left"), "v1": (-8, 6, "right"),
                "orig": (8, 6, "left"), "v2": (-8, 4, "right"), "xq": (8, -12, "left"), "v21": (-8, 6, "right")}


def chart_tradeoff():
    keys = [k for k in MODEL_STYLE if st(k, "kld") is not None]
    fig, ax = plt.subplots(figsize=(8, 5.4), dpi=100)
    fig.subplots_adjust(left=0.1, right=0.95, top=0.82, bottom=0.24)
    for k in keys:
        x, y = st(k, "kld"), m(k, "useful_e09")
        ours = MODEL_STYLE[k][2]
        ax.scatter(x, y, s=130 if ours else 80, zorder=4, linewidths=1.8,
                   facecolors=SURFACE if not_q8(k) else color(k), edgecolors=color(k))
        dx, dy, ha = LABEL_OFFSET.get(k, (8, 6, "left"))
        ax.annotate(label(k), (x, y), xytext=(dx, dy), ha=ha, textcoords="offset points", fontsize=9.5, color=INK,
                    fontweight="bold" if k == "v21" else "normal")
    ax.set_xlim(-0.001, max(st(k, "kld") for k in keys) * 1.25)
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(PCT)
    ax.set_xlabel("KL divergence from the original (lower = closer to the original)", fontsize=9)
    ax.set_ylabel("Useful answers, thinking on", fontsize=9)
    mc.style_ax(ax, grid="both")
    k21, k2, k1 = st("v21", "kld"), st("v2", "kld"), st("v1", "kld")
    frame(fig, "v2.1 is the most useful, and drifts the most from the original",
          f"Useful answers (thinking on, {N} harmful prompts, Jev judge) vs KL drift on wikitext-2",
          f"KLD: v2.1 {k21:.4f}, v2 {k2:.4f}, v1.0 {k1:.4f}, AtomicChat LoRA {st('atomic_lora', 'kld'):.4f}. "
          "MMLU-300 and GSM8K-100 stay within noise for all of them (capability chart).\n"
          "Hollow points: not Q8_0, so their KLD includes 4-bit rounding (the unedited original at IQ4_NL: 0.0073). "
          "Heretic: KLD not measurable on stock llama.cpp.\n" + BASIS)
    save(fig, "tradeoff_v21")


# ---- 7. guardrails ----------------------------------------------------------
def guard_panel(ax, keys, title, sub, get, ymax):
    for i, k in enumerate(keys):
        v = get(k)
        if v is None:
            continue
        ax.bar(i, v, 0.72, color=color(k), edgecolor=SURFACE, linewidth=1.5, zorder=3)
        ax.text(i, v + ymax * 0.02, f"{v:.1f}%" if 0 < v < 30 else f"{v:.0f}%", ha="center", fontsize=9.5,
                color=INK, fontweight="bold" if k == "v21" else "normal")
    ax.set_ylim(0, ymax * 1.12)
    ax.set_xlim(-0.6, len(keys) - 0.4)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=11.5, loc="left", color=INK, pad=18, fontproperties=BOLD)
    ax.text(0, 1.02, sub, transform=ax.transAxes, fontsize=8.5, color=INK2, va="bottom")
    mc.style_ax(ax)
    ax.grid(False)


def chart_guardrails():
    keys = ["orig", "v1", "v2", "v21", "atomic_lora"]
    panels = [("Harmful refusal, thinking off", "lower = fewer refusals", lambda k: m(k, "refusal_off"), 100),
              ("Harmful refusal, thinking on", "effort 0.9", lambda k: m(k, "refusal_e09"), 100),
              ("Over-refusal", "784 benign borderline prompts, lower is better", lambda k: st(k, "overrefusal"), 50),
              ("Thinking-cap hits, off", "greedy, lower is better", lambda k: m(k, "thinkcap_off"), 30),
              ("Thinking-cap hits, on", "greedy, lower is better", lambda k: m(k, "thinkcap_e09"), 30)]
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.6), dpi=100)
    fig.subplots_adjust(left=0.04, right=0.98, top=0.72, bottom=0.1, wspace=0.12, hspace=0.62)
    flat = list(axes.flat)
    for ax, p in zip(flat, panels):
        guard_panel(ax, keys, *p)
    settings_mini(flat[5])
    legend(fig, [patch(k) for k in keys], y=0.845)
    s = DATA["settings"]["think_cap_rates"]
    frame(fig, "Guardrails: almost no refusals; v2.1's weak spot is greedy thinking loops",
          f"Refusal (phrase list) and thinking-cap: {N} held-out harmful prompts · llama.cpp Q8_0. "
          "Over-refusal: separate benign set.",
          f"Thinking-cap hits = answers that ran out of the 4,096-token thinking budget. v2.1 hits it more than v2 "
          f"with greedy decoding; at temperature 0.6 / top_p 0.95 it drops to {s['v21_t06']['value']:.1f}% "
          f"(bottom right, v2.1's own test prompts).")
    save(fig, "guardrails_v21")


def settings_mini(ax):
    s = DATA["settings"]["think_cap_rates"]
    bars = [("v2_greedy", "v2\ngreedy", BLUE_MID, None), ("v21_greedy", "v2.1\ngreedy", BLUE, None),
            ("v21_t06", "v2.1\ntemp 0.6", BLUE, "////")]
    for i, (k, lbl, c, hatch) in enumerate(bars):
        v = s[k]["value"]
        ax.bar(i, v, 0.66, color=c, edgecolor=SURFACE, linewidth=1.5, zorder=3, hatch=hatch)
        ax.text(i, v + 0.6, f"{v:.1f}%", ha="center", fontsize=9.5, color=INK,
                fontweight="bold" if k == "v21_t06" else "normal")
        ax.text(i, -1.2, lbl, ha="center", va="top", fontsize=8.2, color=INK2, linespacing=1.15)
    ax.set_ylim(0, 33.6)
    ax.set_xlim(-0.6, len(bars) - 0.4)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Fix: temperature 0.6", fontsize=11.5, loc="left", color=INK, pad=18, fontproperties=BOLD)
    ax.text(0, 1.02, f"thinking-cap hits, thinking on, n = {DATA['settings']['common_n']}", transform=ax.transAxes,
            fontsize=8.5, color=INK2, va="bottom")
    mc.style_ax(ax)
    ax.grid(False)


# ---- 8. capability ----------------------------------------------------------
def chart_capability():
    keys = ["orig", "v1", "v2", "v21", "atomic_lora", "atomic_adq4", "xq", "ou"]
    panels = [("MMLU-300", "accuracy, higher is better", "mmlu300", 100, "{:.1f}"),
              ("GSM8K-100", "accuracy, higher is better", "gsm8k100", 100, "{:.0f}"),
              ("Over-refusal", "784 benign borderline prompts, lower is better", "overrefusal", 50, "{:.1f}%")]
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 5.0), dpi=100)
    fig.subplots_adjust(left=0.03, right=0.98, top=0.62, bottom=0.17, wspace=0.12)
    for ax, (title, sub, field, ymax, fmt) in zip(axes, panels):
        for i, k in enumerate(keys):
            v = st(k, field)
            if v is None:
                continue
            ax.bar(i, v, 0.74, color=color(k), edgecolor=SURFACE, linewidth=1.2, zorder=3,
                   hatch="////" if not_q8(k) else None)
            top = whisker(ax, i, v, st(k, "mmlu300_ci")) if (k == "v21" and field == "mmlu300") else v
            ax.text(i, top + ymax * 0.02, fmt.format(v), ha="center", fontsize=8.2, color=INK,
                    fontweight="bold" if k == "v21" else "normal")
        ax.set_ylim(0, ymax * 1.12)
        ax.set_xlim(-0.6, len(keys) - 0.4)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, fontsize=11.5, loc="left", color=INK, pad=18, fontproperties=BOLD)
        ax.text(0, 1.02, sub, transform=ax.transAxes, fontsize=8.5, color=INK2, va="bottom")
        mc.style_ax(ax)
        ax.grid(False)
    legend(fig, [patch(k) for k in keys], y=0.85, ncol=4)
    mm = DATA["static"]["v21"]
    frame(fig, "Capability within noise of the original and every other release",
          "Benign benchmarks · llama.cpp · greedy. Q8_0 unless marked †.",
          f"v2.1 MMLU-300 {mm['mmlu300']:.1f} with 95% CI [{mm['mmlu300_ci'][0]:.1f}, {mm['mmlu300_ci'][1]:.1f}] "
          f"(whisker): every other model's score lies inside it. v2.1 over-refusal: {mm['overrefusal_count']} of "
          f"{mm['overrefusal_n']} borderline prompts.\n" + not_q8_note(keys))
    save(fig, "capability_v21")


# ---- 9. recommended settings ------------------------------------------------
def chart_settings():
    s = DATA["settings"]
    r, p, b = s["think_cap_rates"], s["paired"], s["bison_useful_v21_t06_vs_greedy"]
    bars = [("v2_greedy", "v2\ngreedy", BLUE_MID, None), ("v2_t06", "v2\ntemp 0.6", BLUE_MID, "////"),
            ("v21_greedy", "v2.1\ngreedy", BLUE, None), ("v21_t06", "v2.1\ntemp 0.6\n(recommended)", BLUE, "////")]
    fig, ax = plt.subplots(figsize=(8, 5.2), dpi=100)
    fig.subplots_adjust(left=0.11, right=0.97, top=0.76, bottom=0.31)
    for i, (k, _, c, hatch) in enumerate(bars):
        v = r[k]["value"]
        ax.bar(i, v, 0.62, color=c, edgecolor=SURFACE, linewidth=1.0, zorder=3, hatch=hatch)
        top = whisker(ax, i, v, r[k]["ci"])
        ax.text(i, top + 0.8, f"{v:.1f}%", ha="center", va="bottom", fontsize=10, color=INK,
                fontweight="bold" if k == "v21_t06" else "normal", zorder=5)
    ax.set_xticks(range(len(bars)))
    ax.set_xticklabels([lbl for _, lbl, _, _ in bars], fontsize=10, color=INK, linespacing=1.3)
    ax.set_ylim(0, 42)
    ax.set_xlim(-0.6, len(bars) - 0.4)
    ax.yaxis.set_major_formatter(PCT)
    ax.set_ylabel("Answers that hit the 4,096-token\nthinking cap (lower is better)", fontsize=9)
    mc.style_ax(ax)
    a, c2 = p["v21_t06_minus_v21_greedy"], p["v21_t06_minus_v2_greedy"]
    frame(fig, "Recommended: thinking on, temperature 0.6, top_p 0.95",
          f"Thinking-cap hits (thinking loops), thinking on (effort 0.9), the same {s['common_n']} harmful prompts\n"
          "for all four runs · llama.cpp Q8_0 · think cap 4,096 · answer cap 8,192",
          f"Paired: v2.1 at temp 0.6 vs greedy {a['delta']:+.1f} pts [{a['ci'][0]:+.1f}, {a['ci'][1]:+.1f}]; "
          f"vs v2 greedy {c2['delta']:+.1f} pts [{c2['ci'][0]:+.1f}, {c2['ci'][1]:+.1f}].\nUsefulness at temp 0.6 vs "
          f"greedy: {b['delta']:+.1f} pts [{b['ci'][0]:+.1f}, {b['ci'][1]:+.1f}], not significant (Bison judge).\n"
          "Whiskers: 95% bootstrap CI. Hatched = temperature 0.6. DRY sampling did not cut loops in our tests and "
          "made answers hold back more.\nPrompts: v2.1's own sealed test (the thinking-on half).")
    save(fig, "settings_v21")


# ---- 10. Heretic -------------------------------------------------------------
def chart_heretic():
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 4.6), dpi=100, gridspec_kw={"width_ratios": [1, 1.25]})
    fig.subplots_adjust(left=0.07, right=0.96, top=0.72, bottom=0.2, wspace=0.42)
    w = 0.36
    for gi, (mode, _) in enumerate(MODES):
        for ki, k in enumerate(("heretic", "v21")):
            v = m(k, f"useful_{mode}")
            x = gi + (ki - 0.5) * (w + 0.03)
            a.bar(x, v, w, color=color(k), hatch="////" if not_q8(k) else None, edgecolor=SURFACE, linewidth=1.2,
                  zorder=3)
            top = whisker(a, x, v, ci(k, f"useful_{mode}"))
            a.text(x, top + 1.5, f"{v:.0f}%", ha="center", va="bottom", fontsize=10, color=INK, zorder=5)
    a.set_xticks(range(len(MODES)))
    a.set_xticklabels(["Thinking off", "Thinking on"], fontsize=10, color=INK)
    a.set_ylim(0, 105)
    a.yaxis.set_major_formatter(PCT)
    a.set_title("Useful answers to harmful requests", fontsize=10.5, color=INK, loc="left", pad=10)
    mc.style_ax(a)
    for i, (mode, t) in enumerate((("off", "Thinking off"), ("e09", "Thinking on"))):
        y = 1 - i
        draw_h2h_row(b, y, h2h("heretic", mode), color("heretic"), N)
        b.text(-3, y, t, ha="right", va="center", fontsize=10, color=INK)
    b.set_xlim(0, N)
    b.set_ylim(-0.7, 1.7)
    b.set_yticks([])
    b.set_title(f"Same {N} prompts: who gave the useful answer", fontsize=10.5, color=INK, loc="left", pad=10)
    b.legend(handles=[Patch(color=BLUE, label="only v2.1 useful"), Patch(color="#b9b8b3", label="both"),
                      Patch(color="#e6e5e1", label="neither"), Patch(color=color("heretic"), label="only Heretic useful")],
             loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=4, frameon=False, fontsize=9, handlelength=1,
             columnspacing=1.2, labelcolor=INK2)
    mc.style_ax(b)
    legend(fig, [patch("heretic"), patch("v21")], y=0.845)
    o1, o2, e1, e2 = (m("v21", "useful_off"), m("heretic", "useful_off"), m("v21", "useful_e09"),
                      m("heretic", "useful_e09"))
    frame(fig, f"Unbound v2.1 vs Heretic: {o1:.0f}% vs {o2:.0f}% useful (thinking off), {e1:.0f}% vs {e2:.0f}% (on)",
          f"{N} held-out harmful prompts · same prompts, settings, answer cap (8,192) and judge (Jev)",
          "† Heretic ships only 4-bit files that stock llama.cpp can't load; it was run in a compatible llama.cpp build "
          "(0xE2E Inkling engine, Unsloth layout). Unbound v2.1: Q8_0, stock llama.cpp.")
    save(fig, "heretic_vs_v21")


# ---- 11. Reddit / summary card ----------------------------------------------
def r_tile(fig, x, y, w, hgt, big, small, size, versus="", W=8, H=10):
    fig.add_artist(FancyBboxPatch((x, y), w, hgt, boxstyle="round,pad=0,rounding_size=0.012", fc=TILE, ec="none",
                                  transform=fig.transFigure, mutation_aspect=W / H))
    num = fig.text(x + 0.02, y + hgt - 0.017, big, fontsize=size, color=BLUE, va="top")
    if versus:
        fig.canvas.draw()
        bb = num.get_window_extent().transformed(fig.transFigure.inverted())
        fig.text(bb.x1 + 0.015, bb.y0 + 0.004, versus, fontsize=13, color=INK, va="bottom")
    fig.text(x + 0.02, y + 0.014, small, fontsize=9.5, color=INK2, va="bottom", linespacing=1.35)


def reddit_bars(fig):
    keys = ["v21", *sorted(others(("v21", "orig")), key=lambda k: -m(k, "useful_e09")), "orig"]
    fig.text(0.06, 0.832, "Useful answers to harmful requests", fontsize=13.5, va="top")
    fig.text(0.94, 0.830, "higher is better", fontsize=10, color=INK2, va="top", ha="right")
    ax = fig.add_axes([0.30, 0.452, 0.60, 0.356])
    h, gap = 0.36, 0.04
    for i, k in enumerate(keys):
        c, hatch = color(k), "////" if not_q8(k) else None
        for j, mode in enumerate(("off", "e09")):
            y, v, off = i + (j - 0.5) * (h + gap), m(k, f"useful_{mode}"), mode == "off"
            ax.barh(y, v, h, color=tint(c) if off else c, edgecolor=c if off else SURFACE,
                    linewidth=1.2 if off else (0.8 if hatch else 0), hatch=hatch, zorder=3)
            ax.text(v + 1.2, y, fmt_pct(v), va="center", ha="left", fontsize=10.5,
                    color=BLUE if k == "v21" else INK, zorder=4)
            if i == 0:
                ax.text(1.5, y, "thinking off" if off else "thinking on", va="center", ha="left", fontsize=9,
                        zorder=4, color=INK if off else "white")
    ax.set_yticks(range(len(keys)))
    ax.set_yticklabels([label(k) for k in keys], fontsize=10.5)
    for t, k in zip(ax.get_yticklabels(), keys):
        t.set_color(BLUE if k == "v21" else INK)
    ax.set_xlim(0, 100)
    ax.set_ylim(len(keys) - 0.5, -0.5)
    ax.set_xticks([])
    ax.tick_params(axis="y", length=0, pad=8)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.axvline(0, color=INK2, lw=0.8, zorder=5)
    fig.text(0.06, 0.438, f"{N} held-out harmful prompts (CBRN excluded for every model) · same engine (llama.cpp),\n"
             "settings, answer cap (8,192) & judge (Jev) · † 4-bit file as published", fontsize=9, color=INK2,
             va="top", linespacing=1.5)


def chart_reddit():
    W, H = 8, 10
    fig = plt.figure(figsize=(W, H), facecolor=SURFACE)
    fig.text(0.06, 0.968, "Inkling-2-Mini-Unbound v2.1", fontsize=26, va="top")
    head = "The most useful uncensored Inkling 2 Mini" if v21_best() else "Uncensored Inkling 2 Mini that actually answers"
    fig.text(0.06, 0.912, head, fontsize=15, color=INK, va="top", fontproperties=BOLD)
    fig.text(0.06, 0.880, f"Head to head with the other releases: same {N} held-out prompts, engine and judge",
             fontsize=10.5, color=INK2, va="top")
    fig.add_artist(Line2D([0.06, 0.94], [0.852] * 2, color=GRID, lw=1, transform=fig.transFigure))
    reddit_bars(fig)
    bo = best_other("e09", ("v21", "v2", "v1", "orig"))
    lead = [(f"{m('v21', 'useful_e09') - m(bo, 'useful_e09'):+.0f} pts", f"vs {label(bo)}",
             "best other release, thinking on"),
            (f"{h2h('atomic_lora', 'e09')['diff']:+.0f} pts", "vs AtomicChat LoRA",
             "most downloaded alternative, thinking on")]
    gap = 0.02
    w1 = (0.88 - gap) / 2
    for i, (big, versus, small) in enumerate(lead):
        r_tile(fig, 0.06 + i * (w1 + gap), 0.268, w1, 0.105, big, small, 24, versus)
    s21 = DATA["static"]["v21"]
    ref = max(m("v21", "refusal_off"), m("v21", "refusal_e09"))
    rest = [(f"{ref:.1f}%", "refusals\n(harmful prompts)"),
            (f"{s21['overrefusal']:.1f}%", "benign borderline\nprompts refused"),
            (f"{s21['mmlu300']:.1f} / {s21['gsm8k100']:.0f}",
             f"MMLU-300 / GSM8K-100\n(original {st('orig', 'mmlu300'):.1f} / {st('orig', 'gsm8k100'):.0f})")]
    w2 = (0.88 - 2 * gap) / 3
    for i, (big, small) in enumerate(rest):
        r_tile(fig, 0.06 + i * (w2 + gap), 0.150, w2, 0.100, big, small, 22)
    note = (f"Trade-offs: drifts more from the original than other releases (KLD {s21['kld']:.3f}). "
            "With thinking on, use temperature 0.6 / top_p 0.95: greedy decoding loops more.")
    fig.text(0.06, 0.128, textwrap.fill(note, 84), fontsize=10.2, color=INK, va="top", linespacing=1.45)
    fig.add_artist(Line2D([0.06, 0.94], [0.072] * 2, color=GRID, lw=1, transform=fig.transFigure))
    fig.text(0.06, 0.038, "huggingface.co/nokeepai/Inkling-2-Mini-Unbound", fontsize=14, color=BLUE, va="center")
    fig.text(0.94, 0.038, "GGUF · Basalt · Apache 2.0", fontsize=10, color=INK2, va="center", ha="right")
    save(fig, "reddit_v21")


def main():
    chart_banner()
    chart_hero()
    chart_compare_useful()
    chart_headtohead()
    chart_categories()
    chart_tradeoff()
    chart_guardrails()
    chart_capability()
    chart_settings()
    chart_heretic()
    chart_reddit()


if __name__ == "__main__":
    main()
