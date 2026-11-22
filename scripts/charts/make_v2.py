#!/usr/bin/env python3
"""Model-card charts for Inkling-2-Mini-Unbound v2.

Usage: python3 scripts/charts/make_v2.py
Reads package_v2/data_test8k.json; writes PNG + SVG to package_v2/charts/.
Models listed in MODEL_STYLE are drawn automatically when present in the JSON,
so the script can be re-run as more comparison models are added.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

import make_charts as mc  # noqa: E402  (palette, rcParams, style_ax)

R = "/Users/nikshepsvn/Code/inkling-abl/"
DATA = json.load(open(R + "package_v2/data_test8k.json"))
OUT = R + "package_v2/charts/"
os.makedirs(OUT, exist_ok=True)

INK, INK2, GRID, SURFACE = mc.INK, mc.INK2, mc.GRID, mc.SURFACE
BLUE, BLUE_LT, GREY = "#2a78d6", "#94bdec", "#8c8b86"
SLATE, SLATE_LT, PLUM, ORANGE = "#4f8a8b", "#8fb8b5", "#8e7fa8", "#eb6834"

# key -> (display label, colour, ours). Order = legend / series order.
MODEL_STYLE = {
    "v2": ("Unbound v2 (ours)", BLUE, True),
    "v1": ("Unbound v1.0 (ours)", BLUE_LT, True),
    "atomic_lora": ("AtomicChat LoRA", SLATE, False),
    "atomic_adq4": ("AtomicChat (Q4 file)", SLATE_LT, False),
    "xq": ("IndrasMirror XQ", PLUM, False),
    "ou": ("OpenUncensored", ORANGE, False),
    "orig": ("Original", GREY, False),
}
CATS = {"cbrn_weapons": "CBRN & weapons", "crime_illegal_goods": "Crime & illegal goods",
        "cyber": "Cyber", "hate_harassment": "Hate & harassment",
        "misinfo_deception": "Misinfo & deception", "other_harm": "Other harm",
        "sexual": "Sexual", "violence": "Violence"}
N_HARM = 277
SUB = f"{N_HARM} held-out harmful prompts · llama.cpp Q8_0 · answer cap 8,192 · Jev judge"
PCT = matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0f}%")


# ---- data helpers ----------------------------------------------------------
def present(keys=None):
    """Model keys present in `models`, in MODEL_STYLE order (unknown keys last)."""
    models = DATA["models"]
    order = list(MODEL_STYLE) + [k for k in models if k not in MODEL_STYLE]
    return [k for k in order if k in models and (keys is None or k in keys)]


def style(key):
    return MODEL_STYLE.get(key, (DATA["static"].get(key, {}).get("label", key), GREY, False))


def not_q8(key):
    return bool(DATA["static"].get(key, {}).get("not_q8"))


def label(key):
    return style(key)[0] + (" †" if not_q8(key) else "")


def m(key, metric):
    return DATA["models"][key][metric]


# ---- drawing helpers -------------------------------------------------------
def frame(fig, title, subtitle, caption=""):
    fig.text(0.035, 0.955, title, fontsize=15, fontweight="bold", ha="left", va="top")
    fig.text(0.035, 0.895, subtitle, fontsize=10, color=INK2, ha="left", va="top")
    if caption:
        fig.text(0.035, 0.02, caption, fontsize=7.6, color=INK2, ha="left", va="bottom",
                 linespacing=1.4)


def legend(fig, handles, y=0.855, x=0.03, ncol=None):
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(x, y), ncol=ncol or len(handles),
               frameon=False, fontsize=9.5, handlelength=1, columnspacing=1.6,
               handletextpad=0.5, labelcolor=INK2)


def patch(key):
    lbl, color, _ = style(key)
    return Patch(facecolor=color, edgecolor="none", hatch="////" if not_q8(key) else None,
                 label=label(key))


def save(fig, name):
    for ext in ("png", "svg"):
        fig.savefig(f"{OUT}{name}.{ext}", dpi=200, bbox_inches=None)
    plt.close(fig)
    print("wrote", name)


def whisker(ax, x, v, ci, horizontal=False):
    lo, hi = ci
    kw = dict(color=INK, alpha=0.55, lw=1, capsize=2.5, capthick=1, zorder=4)
    if horizontal:
        ax.errorbar(v, x, xerr=[[v - lo], [hi - v]], **kw)
    else:
        ax.errorbar(x, v, yerr=[[v - lo], [hi - v]], **kw)
    return hi


def not_q8_note(keys):
    names = [style(k)[0] for k in keys if not_q8(k)]
    if not names:
        return ""
    return ("† Not Q8_0: " + ", ".join(names) + " run as published; every other model is "
            "Q8_0 (AtomicChat LoRA is Q8_0 + LoRA).\n")


# ---- 1. banner -------------------------------------------------------------
def chart_banner():
    off, e09 = round(m("v2", "useful_off")), round(m("v2", "useful_e09"))
    v1_off, v1_e09 = round(m("v1", "useful_off")), round(m("v1", "useful_e09"))
    fig = plt.figure(figsize=(8, 4), dpi=400, facecolor=SURFACE)
    fig.text(0.04, 0.93, "Inkling-2-Mini-Unbound v2", fontsize=27, va="top")
    fig.text(0.04, 0.80, f"Most useful uncensored Inkling 2 Mini: {off}% useful thinking off, "
             f"{e09}% thinking on", fontsize=12.5, color=INK2, va="top")
    fig.add_artist(Line2D([0.04, 0.96], [0.70, 0.70], color=GRID, lw=1, transform=fig.transFigure))
    callouts = [(0.04, "USEFUL, THINKING OFF", off, v1_off, m("orig", "useful_off")),
                (0.52, "USEFUL, THINKING ON", e09, v1_e09, m("orig", "useful_e09"))]
    for x, title, v, v1, orig in callouts:
        fig.text(x, 0.625, title, fontsize=10, color=INK2, va="bottom")
        fig.text(x, 0.40, f"{v}%", fontsize=50, color=BLUE, va="bottom")
        fig.text(x + 0.21, 0.50, f"v1.0: {v1}%\noriginal: {orig:.1f}%", fontsize=11,
                 color=INK2, va="center", linespacing=1.5)
        banner_bars(fig, x, 0.30, 0.40, (v, v1, orig))
    fig.add_artist(Line2D([0.50] * 2, [0.18, 0.68], color=GRID, lw=1, transform=fig.transFigure))
    fig.add_artist(Line2D([0.04, 0.96], [0.12, 0.12], color=GRID, lw=1, transform=fig.transFigure))
    fig.text(0.04, 0.06, SUB, fontsize=9, color=INK2, va="center")
    for ext in ("png", "svg"):
        fig.savefig(f"{OUT}banner_v2.{ext}", dpi=400 if ext == "png" else 100)
    plt.close(fig)
    print("wrote banner_v2")


def banner_bars(fig, x, y, w, vals):
    """Thin bars on a shared 0-100 track: v2, v1, original."""
    for i, (v, c) in enumerate(zip(vals, (BLUE, BLUE_LT, GREY))):
        yy = y - i * 0.045
        fig.add_artist(plt.Rectangle((x, yy), w, 0.022, fc="#efeeea", ec="none",
                                     transform=fig.transFigure))
        fig.add_artist(plt.Rectangle((x, yy), max(w * v / 100, 0.003), 0.022, fc=c, ec="none",
                                     transform=fig.transFigure))


# ---- 2. v1 vs v2 hero ------------------------------------------------------
def chart_v1_vs_v2():
    # hero: the releases people actually choose between, ours last and highlighted
    keys = [k for k in ("orig", "ou", "atomic_lora", "xq", "v1", "v2") if k in DATA["models"]]
    modes = [("off", "Thinking off"), ("e09", "Thinking on (effort 0.9)")]
    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=100)
    fig.subplots_adjust(left=0.09, right=0.97, top=0.70, bottom=0.17)
    w = 0.84 / len(keys)
    for gi, (mode, _) in enumerate(modes):
        for ki, k in enumerate(keys):
            v, ci = m(k, f"useful_{mode}"), m(k, f"useful_{mode}_ci")
            x = gi + (ki - (len(keys) - 1) / 2) * w
            ax.bar(x, v, w * 0.92, color=style(k)[1], edgecolor=SURFACE, linewidth=1.0, zorder=3)
            top = whisker(ax, x, v, ci)
            bold = "bold" if k == "v2" else "normal"
            ax.text(x, top + 1.5, f"{v:.1f}%" if v < 10 else f"{v:.0f}%", ha="center", va="bottom", fontsize=9 if len(keys) > 3 else 9.5,
                    color=INK, fontweight=bold, zorder=5)
        best_other = max(m(k, f"useful_{mode}") for k in keys if k != "v2")
        d = m("v2", f"useful_{mode}") - best_other
        ax.text(gi, 99, f"v2 +{d:.0f} pts over the next best", ha="center", va="bottom",
                fontsize=10, color=BLUE, fontweight="bold")
    ax.set_xticks(range(len(modes)))
    ax.set_xticklabels([t for _, t in modes], fontsize=10.5, color=INK)
    ax.set_ylim(0, 108)
    ax.set_xlim(-0.55, len(modes) - 0.45)
    ax.yaxis.set_major_formatter(PCT)
    ax.set_ylabel("Useful answers (higher is better)", fontsize=9)
    mc.style_ax(ax)
    legend(fig, [patch(k) for k in keys], ncol=3 if len(keys) > 4 else None, y=0.865)
    frame(fig, "Unbound v2 gives a useful answer far more often than any other release",
          "Useful answers to harmful requests. " + SUB,
          ("Same sealed test prompts, engine, settings and judge for every model. Whiskers show 95% CIs.\n"
           + not_q8_note(keys)).rstrip())
    save(fig, "v1_vs_v2")


# ---- 3. all models ---------------------------------------------------------
def chart_compare_useful():
    keys = sorted(present(), key=lambda k: m(k, "useful_e09"))
    fig, ax = plt.subplots(figsize=(8, 1.9 + 0.55 * len(keys)), dpi=100)
    h = fig.get_figheight()
    fig.subplots_adjust(left=0.27, right=0.86, top=1 - 1.25 / h, bottom=0.95 / h)
    for y, k in enumerate(keys):
        color, hatch = style(k)[1], "////" if not_q8(k) else None
        v, ci = m(k, "useful_e09"), m(k, "useful_e09_ci")
        ax.barh(y, v, 0.62, color=color, edgecolor=SURFACE, linewidth=1.5, zorder=3, hatch=hatch)
        hi = whisker(ax, y, v, ci, horizontal=True)
        off = m(k, "useful_off")
        ax.plot(off, y, "D", ms=6.5, mfc=SURFACE, mec=INK, mew=1.3, zorder=5)
        weight = "bold" if style(k)[2] else "normal"
        ax.text(max(hi, off) + 1.5, y, f"{v:.1f}% · off {off:.1f}%", va="center", fontsize=9,
                color=INK, fontweight=weight)
    ax.set_yticks(range(len(keys)))
    ax.set_yticklabels([label(k) for k in keys], fontsize=10.5, color=INK)
    for t, k in zip(ax.get_yticklabels(), keys):
        t.set_fontweight("bold" if k == "v2" else "normal")
    ax.set_xlim(0, 100)
    ax.xaxis.set_major_formatter(PCT)
    mc.style_ax(ax, grid="x")
    ax.set_xlabel("Harmful requests answered with useful help (higher is better)", fontsize=9)
    legend(fig, [Patch(facecolor=BLUE, label="Thinking on (effort 0.9), 95% CI"),
                 Line2D([], [], marker="D", ls="", mfc=SURFACE, mec=INK, mew=1.3,
                        label="Thinking off")], y=1 - 0.95 / h)
    frame(fig, "Unbound v2 gives the most useful answers of any model tested",
          "Sorted by thinking on. " + SUB.replace(" Q8_0", ""),
          not_q8_note(keys) + "Bars: thinking on with 95% CIs; diamonds: thinking off.")
    save(fig, "compare_useful_v2")


# ---- 4. head to head -------------------------------------------------------
def chart_headtohead():
    h2h = DATA["head_to_head"]
    rows = [(o, mode) for o in ("v1", "atomic_lora") for mode in ("off", "e09")
            if f"v2_vs_{o}_{mode}" in h2h]
    fig, ax = plt.subplots(figsize=(8, 1.9 + 0.75 * len(rows)), dpi=100)
    h = fig.get_figheight()
    fig.subplots_adjust(left=0.25, right=0.97, top=1 - 1.6 / h, bottom=0.35 / h)
    for y, (o, mode) in enumerate(reversed(rows)):
        draw_h2h_row(ax, y, h2h[f"v2_vs_{o}_{mode}"], style(o)[1])
    ax.set_yticks(range(len(rows)))
    names = [f"vs {style(o)[0]}\n{'thinking off' if mode == 'off' else 'thinking on'}"
             for o, mode in reversed(rows)]
    ax.set_yticklabels(names, fontsize=9.5, color=INK, linespacing=1.3)
    ax.set_xlim(0, N_HARM)
    ax.set_xticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    legend(fig, [Patch(facecolor=BLUE, label="only v2 useful"),
                 Patch(facecolor="#b9b8b3", label="both useful"),
                 Patch(facecolor="#e6e5e1", label="neither"),
                 Patch(facecolor=BLUE_LT, label="only other useful (colour of that model)")],
          y=1 - 1.05 / h)
    frame(fig, "Where v2 and the others disagree, v2 is almost always the useful one",
          "Paired on the same prompts: useful for v2 only, both, neither, or the other only.\n"
          + SUB, "")
    save(fig, "headtohead_v2")


def draw_h2h_row(ax, y, d, other_color):
    parts = [(d["v2_only"], BLUE, "#ffffff"), (d["both"], "#b9b8b3", INK),
             (d["neither"], "#e6e5e1", INK), (d["other_only"], other_color, INK)]
    x = 0
    for n, color, tc in parts:
        ax.barh(y, n, 0.62, left=x, color=color, edgecolor=SURFACE, linewidth=2, zorder=3)
        if n >= 8:
            ax.text(x + n / 2, y, str(n), ha="center", va="center", fontsize=10.5, color=tc)
        x += n
    if d["other_only"] < 8:  # too thin for an inside label
        ax.text(x + 1, y, str(d["other_only"]), ha="left", va="center", fontsize=10.5, color=INK)


# ---- 5. categories ---------------------------------------------------------
def chart_categories():
    keys = present([k for k in MODEL_STYLE if k != "orig"])
    nbc = m("v2", "n_by_cat")
    cats = list(CATS)
    fig, ax = plt.subplots(figsize=(11, 5.2), dpi=100)
    fig.subplots_adjust(left=0.06, right=0.98, top=0.76, bottom=0.2)
    w = min(0.8 / len(keys), 0.26)
    for gi, c in enumerate(cats):
        for si, k in enumerate(keys):
            v = m(k, "useful_e09_by_cat").get(c)
            if v is None:
                continue
            x = gi + (si - (len(keys) - 1) / 2) * (w + 0.01)
            ax.bar(x, v, w, color=style(k)[1], edgecolor=SURFACE, linewidth=1.2, zorder=3,
                   hatch="////" if not_q8(k) else None)
            ax.text(x, v + 1.2, f"{v:.0f}", ha="center", fontsize=8 if len(keys) <= 3 else 6.5,
                    color=INK)
    ax.set_xticks(range(len(cats)))
    ax.set_xticklabels([f"{CATS[c]}{'*' if nbc[c] < 20 else ''}\nn = {nbc[c]}" for c in cats],
                       fontsize=9, color=INK, linespacing=1.4)
    ax.set_ylim(0, 108)
    ax.yaxis.set_major_formatter(PCT)
    ax.set_ylabel("Useful answers (higher is better)", fontsize=9)
    mc.style_ax(ax)
    ax.tick_params(axis="x", pad=6)
    legend(fig, [patch(k) for k in keys], x=0.045)
    frame(fig, "v2 leads or ties in every harm category, with thinking on",
          "Useful answers to harmful requests by category, thinking on (effort 0.9). " + SUB,
          ("* n < 20: small sample, read with caution. Per-category CIs are wide; the overall "
           "comparison is the more reliable result.\n" + not_q8_note(keys)).rstrip())
    save(fig, "categories_v2")


# ---- 6. trade-off ----------------------------------------------------------
# label nudges (points) where two models sit close together
LABEL_OFFSET = {"atomic_lora": (0, -20, "center"), "atomic_adq4": (10, -4, "left")}

def chart_tradeoff():
    keys = [k for k in present() if k in DATA["static"]]
    fig, ax = plt.subplots(figsize=(8, 5.2), dpi=100)
    fig.subplots_adjust(left=0.1, right=0.95, top=0.80, bottom=0.2)
    xmax = max(DATA["static"][k]["kld"] for k in keys) * 1.3
    for k in keys:
        x, y = DATA["static"][k]["kld"], m(k, "useful_e09")
        _, color, ours = style(k)
        hollow = not_q8(k)
        ax.scatter(x, y, s=130 if ours else 80, zorder=4, linewidths=1.8,
                   facecolors=SURFACE if hollow else color, edgecolors=color)
        ax.annotate(label(k), (x, y), xytext=LABEL_OFFSET.get(k, (8, 6, "left"))[:2],
                    ha=LABEL_OFFSET.get(k, (8, 6, "left"))[2], textcoords="offset points", fontsize=9.5,
                    color=INK, fontweight="bold" if k == "v2" else "normal")
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(PCT)
    ax.set_xlabel("KL divergence from the original (lower = closer to the original)", fontsize=9)
    ax.set_ylabel("Useful answers, thinking on", fontsize=9)
    mc.style_ax(ax, grid="both")
    v2k, v1k = DATA["static"]["v2"]["kld"], DATA["static"]["v1"]["kld"]
    ak = DATA["static"].get("atomic_lora", {}).get("kld")
    atomic = f", AtomicChat LoRA {ak:.4f}" if ak is not None else ""
    frame(fig, "v2 buys its usefulness with more drift than v1.0",
          f"Useful answers (thinking on) vs KL drift. {SUB}",
          f"KL: v2 {v2k:.4f}, v1.0 {v1k:.4f}{atomic}. v2 drifts {v2k / v1k:.1f}x more than v1.0. "
          "Both axes start at 0. "
          + not_q8_note(keys).replace("† Not Q8_0", "\nHollow † points: not Q8_0").rstrip())
    save(fig, "tradeoff_v2")


# ---- 7. guardrails ---------------------------------------------------------
def chart_guardrails():
    keys = [k for k in ("orig", "v1", "v2", "atomic_lora") if k in DATA["models"]]
    st = DATA["static"]
    panels = [
        ("Harmful refusal, off", "lower = fewer refusals", lambda k: m(k, "refusal_off"), 100),
        ("Harmful refusal, on", "thinking effort 0.9", lambda k: m(k, "refusal_e09"), 100),
        ("Over-refusal", "benign borderline, lower is better",
         lambda k: st.get(k, {}).get("overrefusal"), 50),
        ("MMLU-300", "higher is better", lambda k: st.get(k, {}).get("mmlu300"), 100),
        ("GSM8K-100", "higher is better", lambda k: st.get(k, {}).get("gsm8k100"), 100),
        ("Thinking-cap hits", "thinking on, lower is better",
         lambda k: m(k, "thinkcap_e09"), 25),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.6), dpi=100)
    fig.subplots_adjust(left=0.04, right=0.98, top=0.72, bottom=0.1, wspace=0.12, hspace=0.62)
    for ax, (title, sub, get, ymax) in zip(axes.flat, panels):
        guard_panel(ax, keys, title, sub, get, ymax)
    legend(fig, [patch(k) for k in keys], y=0.845)
    tc2, tc1 = m("v2", "thinkcap_e09"), m("v1", "thinkcap_e09")
    frame(fig, "Guardrails: capability within noise; v2 hits the thinking cap more often",
          "Refusal (phrase list) and thinking-cap: 277 held-out harmful prompts · llama.cpp Q8_0. "
          "Over-refusal, MMLU, GSM8K: separate benign sets.",
          f"Thinking-cap hits = answers that ran out of the 4,096-token thinking budget: "
          f"v2 {tc2:.1f}% vs v1.0 {tc1:.1f}%. This is v2's weak spot.")
    save(fig, "guardrails_v2")


def guard_panel(ax, keys, title, sub, get, ymax):
    for i, k in enumerate(keys):
        v = get(k)
        if v is None:
            continue
        ax.bar(i, v, 0.72, color=style(k)[1], edgecolor=SURFACE, linewidth=1.5, zorder=3)
        txt = f"{v:.1f}%" if 0 < v < 20 else f"{v:.0f}%"
        ax.text(i, v + ymax * 0.02, txt, ha="center", fontsize=9.5, color=INK,
                fontweight="bold" if k == "v2" else "normal")
    ax.set_ylim(0, ymax * 1.12)
    ax.set_xlim(-0.6, len(keys) - 0.4)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=11.5, fontweight="bold", loc="left", color=INK, pad=18)
    ax.text(0, 1.02, sub, transform=ax.transAxes, fontsize=8.5, color=INK2, va="bottom")
    mc.style_ax(ax)
    ax.grid(False)


def main():
    chart_banner()
    chart_v1_vs_v2()
    chart_compare_useful()
    chart_headtohead()
    chart_categories()
    chart_tradeoff()
    chart_guardrails()


if __name__ == "__main__":
    main()
