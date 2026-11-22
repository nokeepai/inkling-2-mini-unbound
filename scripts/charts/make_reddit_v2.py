#!/usr/bin/env python3
"""One-image summary of Inkling-2-Mini-Unbound v2 for a Reddit post.

Usage: python3 scripts/charts/make_reddit_v2.py [data.json] [out_dir]
Defaults: package_v2/data_test8k.json -> package_v2/charts/reddit_v2.{png,svg}.
Every comparison model in MODEL_STYLE that exists in the JSON is drawn, so the
image can be re-rendered as more models are evaluated.
"""
import json
import os
import sys
import textwrap

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import to_rgb  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.font_manager import FontProperties  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

import make_charts as mc  # noqa: E402  (palette, rcParams)

R = "./"
DATA_PATH = sys.argv[1] if len(sys.argv) > 1 else R + "package_v2/data_test8k.json"
OUT = sys.argv[2] if len(sys.argv) > 2 else R + "package_v2/charts/"
DATA = json.load(open(DATA_PATH))
os.makedirs(OUT, exist_ok=True)

INK, INK2, GRID, SURFACE = mc.INK, mc.INK2, mc.GRID, mc.SURFACE
BLUE, BLUE_LT, GREY = "#2a78d6", "#94bdec", "#8c8b86"
SLATE, SLATE_LT, PLUM, ORANGE = "#4f8a8b", "#8fb8b5", "#8e7fa8", "#eb6834"
TILE = "#f1f0ec"
W, H = 8, 10  # 4:5, Reddit's best mobile ratio; 1600x2000 at dpi 200

# key -> (label, colour); same colours as make_v2.py
MODEL_STYLE = {
    "v2": ("Unbound v2 (ours)", BLUE),
    "v1": ("Unbound v1.0 (ours)", BLUE_LT),
    "atomic_lora": ("AtomicChat LoRA", SLATE),
    "atomic_adq4": ("AtomicChat Q4", SLATE_LT),
    "xq": ("IndrasMirror XQ", PLUM),
    "ou": ("OpenUncensored", ORANGE),
    "heretic": ("JoaoZaokk Heretic", "#b5893a"),
    "orig": ("Original", GREY),
}
CAPTION = ("277 held-out harmful prompts · same engine (llama.cpp Q8_0), settings & judge (Jev)\n"
           "answers up to 8,192 tokens · † 4-bit file (Heretic needs a patched llama.cpp build; run in one)")


def m(key, metric):
    return DATA["models"][key][metric]


def static(key, field):
    return DATA["static"][key][field]


def not_q8(key):
    return bool(DATA["static"].get(key, {}).get("not_q8"))


def label(key):
    return MODEL_STYLE[key][0] + (" †" if not_q8(key) else "")


def tint(color, amount=0.5):
    """Mix a colour toward the background; used for the thinking-off bar."""
    rgb, bg = to_rgb(color), to_rgb(SURFACE)
    return tuple(c * (1 - amount) + b * amount for c, b in zip(rgb, bg))


def model_order():
    """v2 first, original last, the rest by thinking-on usefulness."""
    keys = [k for k in MODEL_STYLE if k in DATA["models"]]
    middle = sorted((k for k in keys if k not in ("v2", "orig")),
                    key=lambda k: -m(k, "useful_e09"))
    return [k for k in ["v2", *middle, "orig"] if k in keys]


def hline(fig, y, x0=0.06, x1=0.94):
    fig.add_artist(Line2D([x0, x1], [y, y], color=GRID, lw=1, transform=fig.transFigure))


# ---- claims ----------------------------------------------------------------
OURS = ("v2", "v1")
RELEASE_OF = {"atomic_lora": ("atomic", "AtomicChat"), "atomic_adq4": ("atomic", "AtomicChat"),
              "xq": ("xq", "IndrasMirror XQ"), "ou": ("ou", "OpenUncensored"),
              "heretic": ("heretic", "Heretic")}
BOLD = FontProperties(fname="/System/Library/Fonts/Supplemental/Arial Bold.ttf") \
    if os.path.exists("/System/Library/Fonts/Supplemental/Arial Bold.ttf") else \
    FontProperties(weight="bold")  # house font has no bold face matplotlib can load


def others(exclude=("v2",)):
    return [k for k in model_order() if k not in exclude]


def v2_is_best():
    """True only if v2 beats every other model present in both thinking modes."""
    return all(m("v2", f"useful_{mode}") > m(k, f"useful_{mode}")
               for k in others() for mode in ("off", "e09"))


def best_competitor():
    """(key, v2 lead in pts) against the best non-ours release, thinking on; None if absent."""
    rivals = [k for k in others(OURS + ("orig",))]
    if not rivals:
        return None
    best = max(rivals, key=lambda k: m(k, "useful_e09"))
    return best, m("v2", "useful_e09") - m(best, "useful_e09")


def release_line():
    """Public releases tested, with the date of the version we pulled."""
    seen = {}
    for k in model_order():
        if k in RELEASE_OF:
            rkey, name = RELEASE_OF[k]
            rel = DATA.get("releases", {}).get(rkey, {})
            date = rel.get("modified") or rel.get("created")
            if date and name not in seen:
                seen[name] = date
    if not seen:
        return ""
    return "Public releases tested (version date): " + " · ".join(
        f"{n} {d}" for n, d in seen.items())


def tile(fig, x, y, w, hgt, big, small, size, versus=""):
    fig.add_artist(FancyBboxPatch((x, y), w, hgt, boxstyle="round,pad=0,rounding_size=0.012",
                                  fc=TILE, ec="none", transform=fig.transFigure,
                                  mutation_aspect=W / H))
    num = fig.text(x + 0.02, y + hgt - 0.017, big, fontsize=size, color=BLUE, va="top")
    if versus:  # "vs <who>" sits beside the number so the comparison reads as one phrase
        fig.canvas.draw()
        bb = num.get_window_extent().transformed(fig.transFigure.inverted())
        fig.text(bb.x1 + 0.015, bb.y0 + 0.004, versus, fontsize=14, color=INK, va="bottom")
    fig.text(x + 0.02, y + 0.014, small, fontsize=9.5, color=INK2, va="bottom",
             linespacing=1.35)


# ---- sections --------------------------------------------------------------
def header(fig):
    fig.text(0.06, 0.968, "Inkling-2-Mini-Unbound v2", fontsize=26, va="top")
    if v2_is_best():
        fig.text(0.06, 0.912, "The most useful uncensored Inkling 2 Mini",
                 fontsize=15, color=INK, va="top", fontproperties=BOLD)
        fig.text(0.06, 0.880, "Head to head with the other releases: same 277 held-out prompts, same engine, "
                 "same judge", fontsize=10.5, color=INK2, va="top")
    else:
        fig.text(0.06, 0.912, "Uncensored Inkling 2 Mini that actually answers", fontsize=15,
                 color=INK, va="top")
        fig.text(0.06, 0.880, "nokeep · Apache 2.0 · safetensors · GGUF · Basalt",
                 fontsize=10.5, color=INK2, va="top")
    hline(fig, 0.852)


def bars(fig):
    keys = model_order()
    fig.text(0.06, 0.832, "Useful answers to harmful requests", fontsize=13.5, va="top")
    fig.text(0.94, 0.830, "higher is better", fontsize=10, color=INK2, va="top", ha="right")
    ax = fig.add_axes([0.30, 0.468, 0.60, 0.338])
    h, gap = 0.36, 0.04
    for i, k in enumerate(keys):
        color, hatch = MODEL_STYLE[k][1], "////" if not_q8(k) else None
        for j, mode in enumerate(("off", "e09")):
            y = i + (j - 0.5) * (h + gap)
            v = m(k, f"useful_{mode}")
            off = mode == "off"
            # thinking off: pale fill outlined in the model colour, so it never
            # reads as another model's solid bar (v2-off vs v1-on look alike otherwise)
            fc = tint(color, 0.8) if off else color
            ax.barh(y, v, h, color=fc, edgecolor=color if off else SURFACE,
                    linewidth=1.2 if off else (0.8 if hatch else 0), hatch=hatch, zorder=3)
            ax.text(v + 1.2, y, f"{v:.1f}%" if v < 10 else f"{v:.0f}%", va="center",
                    ha="left", fontsize=11 if k == "v2" else 10.5,
                    color=BLUE if k == "v2" else INK, zorder=4)
            if i == 0:  # key the two shades once, inside the longest bars
                ax.text(1.5, y, "thinking off" if off else "thinking on",
                        va="center", ha="left", fontsize=9, zorder=4,
                        color=INK if off else "white")
    ax.set_yticks(range(len(keys)))
    ax.set_yticklabels([label(k) for k in keys], fontsize=11)
    for t, k in zip(ax.get_yticklabels(), keys):
        t.set_color(BLUE if k == "v2" else INK)
    ax.set_xlim(0, 100)
    ax.set_ylim(len(keys) - 0.5, -0.5)
    ax.set_xticks([])
    ax.tick_params(axis="y", length=0, pad=8)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.axvline(0, color=INK2, lw=0.8, zorder=5)
    caption = CAPTION + ("\n" + release_line() if release_line() else "")
    fig.text(0.06, 0.452, caption, fontsize=9, color=INK2, va="top", linespacing=1.5)


def tiles(fig):
    gap = 0.02
    # row 1: the two comparisons people ask about first
    lead = [(f"{m('v2', 'useful_off') - m('v1', 'useful_off'):+.0f} pts", "vs v1.0",
             "useful answers, thinking off")]
    comp = best_competitor()
    if comp:
        key, d = comp
        lead.append((f"{d:+.0f} pts", f"vs {label(key)}",
                     "best other release, thinking on"))
    w1 = (0.88 - gap * (len(lead) - 1)) / len(lead)
    for i, (big, versus, small) in enumerate(lead):
        tile(fig, 0.06 + i * (w1 + gap), 0.268, w1, 0.105, big, small, 26, versus)
    # row 2: refusals and capability
    v2_ref = m("v2", "refusal_e09")
    ref = f"~{round(v2_ref)}%" if v2_ref >= 0.5 else f"{v2_ref:.1f}%"
    rest = [
        (ref, "refusals\n(harmful prompts)"),
        (f"{static('v2', 'overrefusal'):.1f}%", "benign prompts\nrefused"),
        (f"{static('v2', 'mmlu300'):.1f} / {static('v2', 'gsm8k100'):.0f}",
         f"MMLU-300 / GSM8K-100\n(original {static('orig', 'mmlu300'):.1f} / {static('orig', 'gsm8k100'):.0f})"),
    ]
    w2 = (0.88 - 2 * gap) / 3
    for i, (big, small) in enumerate(rest):
        tile(fig, 0.06 + i * (w2 + gap), 0.150, w2, 0.100, big, small, 22)


def footer(fig):
    kld2, kld1 = static("v2", "kld"), static("v1", "kld")
    note = (f"Trade-offs: drifts more from the original than v1.0 (KLD {kld2:.3f} vs {kld1:.3f})"
            " and thinks longer at high effort. v1.0 stays available.")
    fig.text(0.06, 0.128, textwrap.fill(note, 82), fontsize=10.5, color=INK, va="top",
             linespacing=1.45)
    hline(fig, 0.072)
    fig.text(0.06, 0.038, "huggingface.co/nokeepai/Inkling-2-Mini-Unbound", fontsize=14,
             color=BLUE, va="center")
    fig.text(0.94, 0.038, "GGUF · Basalt · Apache 2.0", fontsize=10, color=INK2,
             va="center", ha="right")


def main():
    fig = plt.figure(figsize=(W, H), facecolor=SURFACE)
    header(fig)
    bars(fig)
    tiles(fig)
    footer(fig)
    for ext in ("png", "svg"):
        fig.savefig(os.path.join(OUT, f"reddit_v2.{ext}"), dpi=200)
    plt.close(fig)
    print("wrote", os.path.join(OUT, "reddit_v2.png"))


if __name__ == "__main__":
    main()
