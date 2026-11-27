#!/usr/bin/env python3
"""Model-card charts for the v3 quality release (the Prime add-on merged into the original and into Unbound v2.1).

Usage: python3 scripts/charts/make_v3.py [--final] [--values PATH] [--out DIR]
Reads package_v3/values.json (or --values) through data_v3.py and writes PNG + SVG to package_v3/charts/ (or --out).
Draft (default): draws everything; TBD values become hatched placeholders, provisional values get a dagger and
the chart gets a "DRAFT" watermark. --final: refuses (exit 1) while any plotted value is TBD or provisional.
Palette and layout follow make_v21.py; make_charts.py is not imported because it pins another checkout's path.
"""
from __future__ import annotations

import os
import re
import sys
import textwrap
from typing import NamedTuple

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.colors  # noqa: E402
import matplotlib.text  # noqa: E402
import matplotlib.ticker  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.font_manager import FontProperties  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402

import data_v3 as d  # noqa: E402
from data_v3 import Key, Pair, Val  # noqa: E402

OUT = os.path.join(d.ROOT, "package_v3", "charts")

INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
BLUE, BLUE_MID, BLUE_LT, GREY = "#2a78d6", "#6fa3e6", "#b3d0f2", "#8c8b86"
SLATE, PLUM, ORANGE = "#4f8a8b", "#8e7fa8", "#eb6834"
GREEN, RED, TILE = "#3f8f5a", "#b4472f", "#f1f0ec"
_ARIAL_BOLD = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
BOLD = FontProperties(fname=_ARIAL_BOLD) if os.path.exists(_ARIAL_BOLD) else FontProperties(weight="bold")

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "text.color": INK,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "svg.fonttype": "none", "hatch.color": GREY, "hatch.linewidth": 0.6,
})

DAGGER = "†"
DAGGER_NOTE = f"{DAGGER} provisional, measured on a candidate before the final add-on."
SOURCE = "Source: our own test set, one pass per model, same prompts and settings for every model."
PUBLIC_SOURCE = "Source: one pass per model, same items and settings for every model."


CI_LABELS = [False]  # set while a chart draws interval bars (chart_xver)


class Slot(NamedTuple):
    """One bar: a value, its colour and the text shown if the value does not exist yet."""
    label: str
    val: Val
    color: str
    missing: str = "TBD"


# ---- formatting -------------------------------------------------------------
def fmt_num(v: Val) -> str:
    x = v.value if v.value is not None else 0.0
    if v.unit == "%":
        return f"{x:.0f}%" if abs(x - round(x)) < 0.05 else f"{x:.1f}%"
    if v.unit == "tokens":
        return f"{x:,.0f}"
    if v.unit == "s":
        return f"{x:.1f} s"
    if v.unit == "x":
        return f"{x:g}×"
    return f"{x:,.0f}" if abs(x - round(x)) < 1e-9 else f"{x:g}"


def fmt_val(v: Val, missing: str = "TBD") -> str:
    if v.is_tbd:
        return missing
    text = v.display if v.display else fmt_num(v)
    return text + (DAGGER if v.is_provisional else "")


def fmt_n(v: Val) -> str:
    return "n=TBD" if v.is_tbd else f"n={fmt_num(v)}" + (DAGGER if v.is_provisional else "")


def base_fixed_names(pair: Pair) -> tuple[str, str]:
    return ("Original", d.prime_name()) if pair == "prime" else ("Unbound v2.1", "Unbound v3")


def prose(name: str) -> str:
    """Tick labels say "Original"; running text says "the original"."""
    return "the original" if name == "Original" else name


# ---- layout helpers ---------------------------------------------------------
def style_ax(ax: Axes, grid: str = "y") -> None:
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.grid(axis=grid, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0, labelsize=9)


def frame(fig: Figure, title: str, subtitle: str, caption: str) -> None:
    fig.text(0.035, 0.955, title, fontsize=15, ha="left", va="top", fontproperties=BOLD)
    fig.text(0.035, 0.895, subtitle, fontsize=10, color=INK2, ha="left", va="top", linespacing=1.4)
    fig.text(0.035, 0.02, caption, fontsize=7.6, color=INK2, ha="left", va="bottom", linespacing=1.4)


def legend(fig: Figure, handles: list[Patch | Line2D], y: float = 0.82, x: float = 0.03) -> None:
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(x, y), ncol=len(handles), frameon=False,
               fontsize=9.5, handlelength=1, columnspacing=1.6, handletextpad=0.5, labelcolor=INK2)


def watermark(fig: Figure) -> None:
    # Diagonal and faint so the chart stays readable, but no screenshot of a draft can pass as final.
    fig.text(0.5, 0.48, "DRAFT · provisional numbers", fontsize=46, color=RED, alpha=0.10, rotation=22,
             ha="center", va="center", fontproperties=BOLD, zorder=100)
    fig.text(0.965, 0.955, "DRAFT · provisional numbers", fontsize=9, color=RED, ha="right", va="top",
             fontproperties=BOLD)


def caption_for(name: str, *lines: str) -> str:
    keys = d.chart_keys()[name]
    out = [ln for ln in lines if ln]
    if any(d.gk(k).is_provisional for k in keys):
        out.append(DAGGER_NOTE)
    if any(d.gk(k).is_tbd for k in keys):
        out.append("Hatched bars and TBD markers are placeholders for numbers not measured yet, not data.")
    return "\n".join(out)


def finish(fig: Figure, name: str, dpi: int = 200) -> None:
    if d.not_final(d.chart_keys()[name]):
        watermark(fig)
    os.makedirs(OUT, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), dpi=dpi if ext == "png" else 100, bbox_inches=None)
    plt.close(fig)
    print("wrote", name)


# ---- bar panels -------------------------------------------------------------
def panel_scale(slots: list[Slot]) -> tuple[float, float]:
    """(axis top, placeholder height). Placeholders get a fixed share of the axis so they never imply a value."""
    measured = [s.val.value for s in slots if not s.val.is_tbd and s.val.value is not None]
    top = max(measured) * 1.28 if measured and max(measured) > 0 else 1.0
    return top, top * (0.45 if measured else 0.55)


def draw_placeholder(ax: Axes, x: float, h: float, w: float, text: str) -> None:
    ax.bar(x, h, w, color=TILE, edgecolor=GREY, linewidth=0.8, linestyle=(0, (3, 2)), hatch="////", zorder=3)
    ax.text(x, h / 2, text, ha="center", va="center", fontsize=8.5 if len(text) < 6 else 7.6, color=INK2,
            rotation=0 if len(text) < 6 else 90, zorder=5,
            bbox=dict(boxstyle="round,pad=0.25", fc=SURFACE, ec="none"))


def draw_slot(ax: Axes, x: float, slot: Slot, top: float, ph: float, w: float) -> None:
    if slot.val.is_tbd or slot.val.value is None:
        draw_placeholder(ax, x, ph, w, slot.missing)
        return
    v = slot.val.value
    ax.bar(x, v, w, color=slot.color, edgecolor=SURFACE, linewidth=1.0, zorder=3)
    label_y = max(v, slot.val.ci[1]) if slot.val.ci and CI_LABELS[0] else v   # above a drawn interval bar
    ax.text(x, label_y + top * 0.02, fmt_val(slot.val), ha="center", va="bottom", fontsize=8.8, color=INK, zorder=5,
            fontproperties=BOLD if slot.color == BLUE else None)


def bar_panel(ax: Axes, title: str, hint: str, slots: list[Slot], xs: list[float], w: float = 0.36) -> None:
    top, ph = panel_scale(slots)
    for x, s in zip(xs, slots):
        draw_slot(ax, x, s, top, ph, w)
    ax.set_ylim(0, top)
    ax.set_xlim(min(xs) - 0.45, max(xs) + 0.45)
    ax.set_xticks(xs)
    ax.set_xticklabels([s.label for s in slots], fontsize=8.8, color=INK)
    ax.set_yticks([])
    ax.set_title(f"{title}\n", fontsize=10.5, loc="left", color=INK, fontproperties=BOLD, pad=2)
    ax.text(0, 1.0, hint, transform=ax.transAxes, fontsize=8.5, color=INK2, va="bottom")
    style_ax(ax)
    ax.grid(False)


def pair_xs(n_pairs: int) -> list[float]:
    return [p + off for p in range(n_pairs) for off in (-0.2, 0.2)]


# ---- (a) hero -----------------------------------------------------------------
def hero_slots(stem: str) -> list[Slot]:
    out: list[Slot] = []
    for pair in d.PAIRS:
        base, fixed = base_fixed_names(pair)
        out.append(Slot(base.replace("Unbound ", "Unbound\n"), d.gk(d.pair_key(pair, stem, False)), GREY))
        out.append(Slot(fixed.replace("Unbound ", "Unbound\n"), d.gk(d.pair_key(pair, stem, True)), BLUE))
    return out


def token_note(ax: Axes, slots: list[Slot]) -> None:
    ratio = d.gk(d.TOKEN_RATIO)
    if not any(s.val.is_tbd for s in slots) or ratio.is_tbd:
        return
    text = f"{ratio.display or fmt_num(ratio)} fewer"
    text += " (provisional)" if ratio.is_provisional else ""
    ax.text(0.5, 0.80, text, transform=ax.transAxes, ha="center", va="center", fontsize=9, color=BLUE,
            fontproperties=BOLD)


def chart_hero() -> None:
    hints = {"debug_solved": "higher is better", "debug_nocode": "lower is better", "debug_tokens": "lower is better"}
    fig, axes = plt.subplots(1, 3, figsize=(12, 5.6), dpi=100)
    fig.subplots_adjust(left=0.035, right=0.975, top=0.66, bottom=0.25, wspace=0.12)
    for ax, (stem, title) in zip(axes, d.HERO_STEMS):
        slots = hero_slots(stem)
        bar_panel(ax, title, hints[stem], slots, pair_xs(2))
        if stem == "debug_tokens":
            token_note(ax, slots)
    n = fmt_n(d.get(d.PAIR_GROUP["prime"], "debug_n_tasks")).replace("n=", "")
    frame(fig, "The Prime add-on makes the model finish the job",
          f"The original vs {d.prime_name()}, and Unbound v2.1 vs Unbound v3, on {n} hard debugging tasks each. "
          "The same add-on in both.", caption_for("hero_v3", SOURCE,
                                              "Stalled turn: a turn that ends without producing any code."))
    legend(fig, [Patch(facecolor=GREY, label="Before the Prime add-on"), Patch(facecolor=BLUE, label="With the Prime add-on")])
    finish(fig, "hero_v3")


# ---- (b) speed ------------------------------------------------------------------
def speed_slots(stem: str) -> list[Slot]:
    names = {"stock": "Original", "prime": d.prime_name(), "v21": "Unbound\nv2.1", "v3": "Unbound\nv3"}
    out: list[Slot] = []
    for pair in d.PAIRS:
        for fixed in (False, True):
            g, k = d.flat_key("speed", pair, stem, fixed)
            out.append(Slot(names[k.rsplit("_", 1)[1]], d.get(g, k), BLUE if fixed else GREY))
    return out


def chart_speed() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 5.6), dpi=100)
    fig.subplots_adjust(left=0.035, right=0.975, top=0.66, bottom=0.25, wspace=0.12)
    for ax, (stem, title, _) in zip(axes, d.SPEED_STEMS):
        bar_panel(ax, title, "lower is better", speed_slots(stem), pair_xs(2))
    main_n = fmt_n(d.get("speed", "main_n"))
    ub_n = fmt_n(d.get("ubench", "n"))
    frame(fig, "Less thinking, faster answers",
          f"The original vs {d.prime_name()}, and Unbound v2.1 vs Unbound v3. Main prompt set ({main_n}); "
          f"uncensored benchmark ({ub_n}).",
          caption_for("speed_v3", SOURCE, "Thinking-cap hit: the model used its whole thinking budget "
                      "without finishing. Seconds per answer on the same hardware for every model."))
    legend(fig, [Patch(facecolor=GREY, label="Before the Prime add-on"), Patch(facecolor=BLUE, label="With the Prime add-on")])
    finish(fig, "speed_v3")


# ---- (c) nothing lost -----------------------------------------------------------
def tbd_band(ax: Axes, y: float) -> None:
    ax.add_patch(Rectangle((2, y - 0.28), 96, 0.56, fc=TILE, ec=GREY, lw=0.6, ls=(0, (3, 2)), hatch="////",
                           zorder=2))
    ax.text(50, y, "TBD", ha="center", va="center", fontsize=8.5, color=INK2, zorder=4,
            bbox=dict(boxstyle="round,pad=0.25", fc=SURFACE, ec="none"))


def dumbbell(ax: Axes, y: float, base: Val, fixed: Val) -> str:
    """Draw one row; returns the colour used for the fixed marker so the value column can match it."""
    if base.is_tbd or fixed.is_tbd or base.value is None or fixed.value is None:
        tbd_band(ax, y)
        for v, c in ((base, GREY), (fixed, BLUE)):
            if v.value is not None:
                ax.plot(v.value, y, "o", ms=8, color=c, zorder=5)
        return INK2
    worse = fixed.value < base.value
    c = RED if worse else BLUE
    ax.plot([base.value, fixed.value], [y, y], color=RED if worse else BLUE_LT, lw=2.4, zorder=3)
    ax.plot(base.value, y, "o", ms=8, color=GREY, zorder=4)
    ax.plot(fixed.value, y, "o", ms=8, color=c, zorder=5)
    return c


def value_text(base: Val, fixed: Val) -> str:
    if base.is_tbd and fixed.is_tbd:
        return "TBD"
    arrow = f"{fmt_val(base)}  to  {fmt_val(fixed)}"
    lower = not (base.is_tbd or fixed.is_tbd) and (fixed.value or 0.0) < (base.value or 0.0)
    return arrow + ("  (lower)" if lower else "")


def level_axis(ax: Axes, rows: list[d.Row]) -> None:
    ax.set_xlim(0, 100)
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{r.label} ({fmt_n(d.gk(r.n))})" for r in rows], fontsize=9.2, color=INK)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    style_ax(ax, grid="x")


def chart_nothing_lost(pair: Pair) -> None:
    name = f"nothing_lost_{pair}_v3"
    base_n, fixed_n = base_fixed_names(pair)
    rows = d.level_rows(pair)
    fig, ax = plt.subplots(figsize=(11, 8.2), dpi=100)
    fig.subplots_adjust(left=0.36, right=0.75, top=0.74, bottom=0.21)
    for y, r in enumerate(rows):
        base, fixed = d.gk(r.base), d.gk(r.fixed)
        colour = dumbbell(ax, y, base, fixed)
        ax.text(1.03, y, value_text(base, fixed), transform=ax.get_yaxis_transform(), va="center", fontsize=8.8,
                color=RED if colour == RED else INK)
    ax.text(1.03, -0.85, f"{base_n} to {fixed_n}", transform=ax.get_yaxis_transform(), fontsize=8.6, color=INK2)
    level_axis(ax, rows)
    frame(fig, f"Nothing lost: {fixed_n} vs {base_n}",
          f"Every level check, {prose(base_n)} (grey) vs {fixed_n} (blue). Sample size per check in brackets.\n"
          f"A check where {fixed_n} scores lower is drawn in red, never hidden.",
          caption_for(name, SOURCE + " Public benchmarks: HumanEval, QuixBugs, GSM8K, MMLU subsets.",
                      "Strict: exact final-answer format. Number correct: the right number in any format."))
    legend(fig, [Line2D([], [], marker="o", ls="", color=GREY, label=base_n),
                 Line2D([], [], marker="o", ls="", color=BLUE, label=fixed_n),
                 Line2D([], [], marker="o", ls="", color=RED, label=f"{fixed_n} below {base_n}")], y=0.83)
    finish(fig, name)


# ---- (d) history ----------------------------------------------------------------
HISTORY_COLORS: dict[str, str] = {"v2": BLUE_LT, "v21": BLUE_MID, "v3": BLUE}
MISSING_V2 = "not measured"


def history_val(group: str, key: str) -> Val:
    # v2 never had these runs; a missing key and a TBD leaf both mean "not measured", not zero.
    return d.get(group, key) if d.has(group, key) else d.parse("TBD")


def history_slots(group: str, stem: str) -> list[Slot]:
    out: list[Slot] = []
    for v, label in d.HISTORY_VERSIONS:
        missing = MISSING_V2 if v == "v2" else "TBD"
        out.append(Slot(label.replace("Unbound ", "Unbound\n"), history_val(group, f"{stem}_{v}"),
                        HISTORY_COLORS[v], missing))
    return out


def chart_history() -> None:
    panels: list[tuple[str, str, str, str]] = [
        ("ubench", "willing", "Willing to answer", "higher is better"),
        ("ubench", "delivery", "Delivered a usable answer", "higher is better"),
        ("speed", "cap_ubench", "Thinking-cap hits", "lower is better"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12, 5.6), dpi=100)
    fig.subplots_adjust(left=0.035, right=0.975, top=0.70, bottom=0.25, wspace=0.12)
    for ax, (group, stem, title, hint) in zip(axes, panels):
        bar_panel(ax, title, hint, history_slots(group, stem), [0.0, 1.0, 2.0], w=0.6)
    frame(fig, "Unbound, version by version",
          f"Unbound v2, v2.1 and v3 on the uncensored benchmark, the same questions for v2.1 and v3 "
          f"({fmt_n(d.get('ubench', 'n_v3'))} graded).",
          caption_for("history_v3", SOURCE,
                      "Unbound v2 was never run on this benchmark, so its bars say 'not measured' (not zero)."))
    finish(fig, "history_v3")


# ---- (e) independent benchmarks: paired-difference forest plot -------------------
def text_or(k: Key, fallback: str) -> str:
    v = d.gk(k)
    return v.text if v.text else fallback


def forest_limits(diffs: list[Val], margin: Val) -> tuple[float, float]:
    """Symmetric around zero so the layout itself never suggests a direction."""
    edges = [abs(x) for v in diffs if v.value is not None for x in (v.ci or (v.value, v.value))]
    if margin.value is not None:
        edges.append(margin.value)
    span = max(edges + [4.0]) * 1.25
    return -span, span


def verdict_colour(verdict: str | None) -> str:
    """Blue = non-inferior, grey = inconclusive (not shown either way), red only for a verdict of worse."""
    if verdict == "WORSE":
        return RED
    if verdict == "inconclusive" or verdict is None:
        return GREY
    return BLUE


def forest_row(ax: Axes, y: float, v: Val, margin: Val, verdict: str | None = None) -> None:
    if v.is_tbd or v.value is None:
        ax.text(0, y, "TBD", ha="center", va="center", fontsize=8.5, color=INK2, zorder=5,
                bbox=dict(boxstyle="round,pad=0.3", fc=TILE, ec=GREY, lw=0.6, ls=(0, (3, 2))))
        return
    lo, hi = v.ci or (v.value, v.value)
    c = verdict_colour(verdict)
    ax.plot([lo, hi], [y, y], color=c, lw=2.2, solid_capstyle="butt", zorder=3)
    ax.plot(v.value, y, "o", ms=7.5, color=c, zorder=4)


def forest_text(v: Val) -> str:
    if v.is_tbd or v.value is None:
        return "TBD"
    ci = f"  [{v.ci[0]:+.1f}, {v.ci[1]:+.1f}]" if v.ci else "  (no interval)"
    return f"{v.value:+.1f} pts{ci}" + (DAGGER if v.is_provisional else "")


def forest_guides(ax: Axes, margin: Val) -> None:
    ax.axvline(0, color=INK, lw=1, zorder=2)
    if margin.value is None:
        ax.text(0.0, 1.01, "non-inferiority margin TBD", transform=ax.transAxes, fontsize=8.5, color=INK2,
                va="bottom")
        return
    ax.axvline(-margin.value, color=INK2, lw=1, ls=(0, (4, 3)), zorder=2)
    ax.text(-margin.value, 1.01, f"non-inferiority margin −{fmt_num(margin)} pts" +
            (DAGGER if margin.is_provisional else ""), transform=ax.get_xaxis_transform(), fontsize=8.2,
            color=INK2, ha="center", va="bottom")


def indep_caption(name: str, pair: Pair) -> str:
    sources = "; ".join(text_or(("indep", f"{b}_source"), f"{label}: source TBD") for b, label in d.INDEP_BENCH
                        if d.has("indep", f"{b}_source"))
    wrapped = textwrap.fill("Benchmarks: " + sources + ".", 150)
    decoding = text_or(d.INDEP_DECODING, "TBD")
    base_n, fixed_n = base_fixed_names(pair)
    return caption_for(name, textwrap.fill(f"{PUBLIC_SOURCE} Decoding: {decoding}.", 150), wrapped,
                       f"Difference = {fixed_n} minus {prose(base_n)} on the same items; bars are 95% intervals. "
                       "Blue: non-inferior; "
                       "grey: inconclusive (a loss beyond the margin is not ruled out); red: worse.")


def chart_indep(pair: Pair) -> None:
    name = f"indep_{pair}_v3"
    base_n, fixed_n = base_fixed_names(pair)
    margin = d.gk(d.INDEP_MARGIN)
    rows = [(label, d.get("indep", f"{b}_{d.INDEP_PAIR[pair]}_n"), d.gk(d.indep_diff_key(pair, b)),
             d.get("indep", f"{b}_{d.INDEP_PAIR[pair]}_verdict").text) for b, label in d.INDEP_BENCH]
    rows += d.pooled_rows(pair)
    fig, ax = plt.subplots(figsize=(11, 7.0 + 0.35 * max(0, len(rows) - 9)), dpi=100)
    fig.subplots_adjust(left=0.27, right=0.74, top=0.74, bottom=0.25)
    for y, (_, _, v, verdict) in enumerate(rows):
        forest_row(ax, y, v, margin, verdict)
        ax.text(1.03, y, forest_text(v), transform=ax.get_yaxis_transform(), va="center", fontsize=8.8, color=INK)
    forest_guides(ax, margin)
    ax.set_xlim(*forest_limits([v for _, _, v, _ in rows], margin))
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{label} ({fmt_n(n)})" for label, n, _, _ in rows], fontsize=9.4, color=INK)
    ax.set_xlabel(f"{fixed_n} minus {base_n}, points", fontsize=9)
    style_ax(ax, grid="x")
    frame(fig, f"Independent benchmarks: {fixed_n} vs {base_n}",
          f"Paired difference, {fixed_n} minus {prose(base_n)}, in points, with 95% intervals. Right of zero = {fixed_n} "
          "higher.\nNo accuracy gain is claimed; this checks nothing was lost.", indep_caption(name, pair))
    finish(fig, name)


# ---- (f) thinking-token distribution --------------------------------------------
def thinkdist_slots(stat: str) -> list[Slot]:
    # Five narrow panels: "Unbound" is dropped from tick labels (the subtitle names it) so labels do not collide.
    names = {"stock": "Original", "prime": d.prime_name(), "v21": "v2.1", "v3": "v3"}
    return [Slot(names[m], d.get("think_dist", f"{stat}_{m}"), BLUE if m in ("prime", "v3") else GREY)
            for m in d.THINK_MODELS]


def chart_thinkdist() -> None:
    fig, axes = plt.subplots(1, len(d.THINK_STATS), figsize=(15, 5.6), dpi=100)
    fig.subplots_adjust(left=0.03, right=0.98, top=0.66, bottom=0.25, wspace=0.14)
    for ax, (stat, title) in zip(axes, d.THINK_STATS):
        hint = "% of prompts, lower is better" if stat == "nonterm" else "tokens, lower is better"
        bar_panel(ax, title, hint, thinkdist_slots(stat), pair_xs(2))
    settings = text_or(d.THINK_SETTINGS, "TBD")
    frame(fig, "How long each model thinks",
          f"Thinking tokens per prompt: the original vs {d.prime_name()}, and Unbound v2.1 vs Unbound v3 "
          f"({fmt_n(d.get('think_dist', 'n'))} prompts per model).",
          caption_for("thinkdist_v3", f"{SOURCE} Settings: {settings}.",
                      "Never finished thinking: the thinking block hit its cap without closing."))
    legend(fig, [Patch(facecolor=GREY, label="Before the Prime add-on"), Patch(facecolor=BLUE, label="With the Prime add-on")])
    finish(fig, "thinkdist_v3")


# ---- (g) across versions and other releases ------------------------------------------
def v21_useful(key: str, mode: str) -> Val:
    cell = d.v21_data()["models"].get(key, {}).get(f"useful_{mode}")
    if not cell:
        return Val(None, "tbd", None, "%", None)
    return Val(float(cell["value"]), "measured", None, "%", None, tuple(cell["ci"]) if cell.get("ci") else None)


def ci_bars(ax: Axes, xs: list[float], vals: list[Val]) -> None:
    for x, v in zip(xs, vals):
        if v.value is not None and v.ci:
            ax.plot([x, x], list(v.ci), color=INK2, lw=1.0, zorder=4)


def text_val(raw: str) -> Val:
    """'87.8% [83.7, 91.6]' -> a % Val with its 95% interval, for hand-set text leaves."""
    ci = re.search(r"\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]", raw)
    return Val(text_pct(raw), "measured", None, "%", None, (float(ci[1]), float(ci[2])) if ci else None)


def useful263_date() -> str:
    """Scoring date of the 263-request re-run, read from the leaf's source note so no date is typed by hand."""
    leaf = d.load()["useful263"]["v3_on"]
    m = re.search(r"\d{4}-\d{2}-\d{2}", str(leaf.get("source", "")) if isinstance(leaf, dict) else "")
    return m.group() if m else "the same day"


def xver_left_slots() -> list[Slot]:
    ours = {"v1", "v2", "v21", "orig"}
    slots: list[Slot] = []
    for k, label in d.XVER_V21_MODELS:
        colour = GREY if k == "orig" else (BLUE_MID if k in ours else SLATE)
        slots.append(Slot(label, v21_useful(k, "e09"), colour, "not\nmeasured"))
        if k == "v21" and d.flag(d.USEFUL263_FLAG[1]):
            slots.append(Slot("Unbound\nv3", text_val(d.get("useful263", "v3_on").text or ""), BLUE))
    return slots


def xver_subtitle() -> str:
    if d.flag(d.USEFUL263_FLAG[1]):
        return (f"Left: useful answers on the v2.1 release's test. Unbound v3 was scored on {useful263_date()} with "
                "the same engine, settings and judge;\nthe other bars are from the v2.1 release run. "
                "Right: this release's debugging test.")
    return (f"Left: useful answers on the v2.1 release's test, from the v2.1 release run. {d.prime_name()} and "
            "Unbound v3 have not been run on it yet,\nso they are not shown there. Right: this release's debugging test.")


def chart_xver() -> None:
    CI_LABELS[0] = True
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 6.0), dpi=100, gridspec_kw={"width_ratios": [2.0, 1]})
    fig.subplots_adjust(left=0.03, right=0.98, top=0.66, bottom=0.27, wspace=0.10)
    n21 = d.v21_data()["_meta"]["n"]
    slots = xver_left_slots()
    xs = [float(i) for i in range(len(slots))]
    bar_panel(a1, f"Useful answers to harmful requests, thinking on (v2.1 release's test, n={n21})",
              "higher = more useful; bars are 95% intervals", slots, xs, w=0.62)
    ci_bars(a1, xs, [s.val for s in slots])
    a1.set_ylim(0, 100)
    names = {"stock": "Original", "prime": d.prime_name(), "v21": "Unbound\nv2.1", "v3": "Unbound\nv3"}
    solved = {"stock": d.gk(d.pair_key("prime", "debug_solved", False)),
              "prime": d.gk(d.pair_key("prime", "debug_solved", True)),
              "v21": d.gk(d.pair_key("unbound", "debug_solved", False)),
              "v3": d.gk(d.pair_key("unbound", "debug_solved", True))}
    s2 = [Slot(names[m], solved[m], BLUE if m in ("prime", "v3") else GREY) for m in d.XVER_V3_MODELS]
    xs2 = [float(i) for i in range(len(s2))]
    n_dbg = fmt_n(d.get(d.PAIR_GROUP["prime"], "debug_n_tasks"))
    bar_panel(a2, f"Hard debugging solved (v3 release, {n_dbg} tasks)", "higher is better; bars are 95% intervals",
              s2, xs2, w=0.6)
    ci_bars(a2, xs2, [s.val for s in s2])
    a2.set_ylim(0, 115)
    frame(fig, "Across versions and other releases", xver_subtitle(),
          caption_for("xver_v3", f"Sources: the v2.1 release run ({n21} requests) and this release's runs, "
                      "one pass per model.",
                      "Not every competitor file is Q8_0 (XQ, AtomicChat AD-Q4_K, Heretic UD-Q4_K_XL); see the v2.1 card."))
    legend(fig, [Patch(facecolor=BLUE_MID, label="Unbound, earlier releases"), Patch(facecolor=SLATE, label="Other releases"),
                 Patch(facecolor=GREY, label="Original"), Patch(facecolor=BLUE, label="With the Prime add-on")])
    finish(fig, "xver_v3")
    CI_LABELS[0] = False


# ---- Reddit chart: "is unbound still uncensored?" (useful answers, thinking on and off) ----------------
class UsefulRow(NamedTuple):
    label: str
    on: float
    off: float
    colour: str
    bold: bool = False


def tint(colour: str, share: float = 0.42) -> str:
    """Lighter tint of a hex colour, blended toward the page background (thinking-off bars)."""
    c = matplotlib.colors.to_rgb(colour)
    bg = matplotlib.colors.to_rgb(SURFACE)
    return matplotlib.colors.to_hex(tuple(share * a + (1 - share) * b for a, b in zip(c, bg)))


def useful_rows() -> list[UsefulRow]:
    card = d.v21_card()

    def u(key: str) -> float:
        return text_pct(d.get("useful263", key).text or "")

    def c(stem: str) -> tuple[float, float]:
        return text_pct(card[f"{stem}_E09"]), text_pct(card[f"{stem}_OFF"])

    rows = [UsefulRow("Unbound v3", u("v3_on"), u("v3_off"), BLUE, True),
            UsefulRow("Unbound v2.1", u("v21_on"), u("v21_off"), BLUE_MID)]
    for label, stem, colour in (("Unbound v2", "V2", BLUE_MID), ("Unbound v1", "V1", BLUE_MID),
                                ("IndrasMirror XQ", "XQ", SLATE), ("AtomicChat LoRA", "ATOM", SLATE),
                                ("OpenUncensored", "OU", SLATE), ("Heretic", "HER", SLATE), ("Original", "ORIG", GREY)):
        rows.append(UsefulRow(label, *c(stem), colour))
    return sorted(rows, key=lambda r: -r.on)


def short_pct(x: float) -> str:
    text = f"{x:.1f}"
    return (text[:-2] if text.endswith(".0") else text) + "%"


def useful_legend(fig: Figure, y: float) -> None:
    handles = [Patch(facecolor=INK, label="thinking on"), Patch(facecolor=tint(INK, 0.3), label="thinking off"),
               Patch(facecolor=BLUE_MID, label="Unbound"), Patch(facecolor=SLATE, label="other releases"),
               Patch(facecolor=GREY, label="the original")]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.035, y), ncol=len(handles), frameon=False,
               fontsize=10.5, handlelength=1.1, columnspacing=1.8, handletextpad=0.5, labelcolor=INK2)


def chart_reddit_useful() -> None:
    fig = plt.figure(figsize=(8, 4.5), dpi=400, facecolor=SURFACE)
    n = d.v21_data()["_meta"]["n"]
    guard_width(fig, fig.text(0.04, 0.955, "Useful answers to harmful requests", fontsize=22, va="top"), 0.92)
    sub = (f"Share of {n} held-out requests where the answer actually does what was asked.\n"
           "Same engine, settings and judge for every model.")
    guard_width(fig, fig.text(0.04, 0.868, sub, fontsize=11.5, color=INK2, va="top", linespacing=1.35), 0.92)
    useful_legend(fig, 0.79)
    rows = useful_rows()
    ax = fig.add_axes((0.215, 0.105, 0.70, 0.61))
    h = 0.44
    for i, r in enumerate(rows):
        for off, val, colour in ((-h / 2, r.on, r.colour), (h / 2, r.off, tint(r.colour))):
            ax.barh(i + off, val, h, color=colour, edgecolor="none", zorder=3)
            ax.text(val + 0.8, i + off, short_pct(val), va="center", ha="left", fontsize=9.5,
                    color=INK if colour == r.colour else INK2, fontproperties=BOLD if r.bold and colour == r.colour else None)
    ax.set_xlim(0, 100)
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r.label for r in rows], fontsize=11.5, color=INK)
    for tick, r in zip(ax.get_yticklabels(), rows):
        if r.bold:
            tick.set_fontproperties(BOLD)
            tick.set_fontsize(11.5)
            tick.set_color(BLUE)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([])
    ax.tick_params(length=0, pad=6)
    fig.add_artist(Line2D([0.04, 0.96], [0.085, 0.085], color=GRID, lw=1, transform=fig.transFigure))
    foot = (f"Unbound v3 and v2.1 scored on {useful263_date()} in one pass; other rows from the v2.1 release run")
    guard_width(fig, fig.text(0.04, 0.042, foot, fontsize=9.5, color=INK2, va="baseline"), 0.66)
    fig.text(0.96, 0.042, "huggingface.co/nokeepai", fontsize=10, color=BLUE, va="baseline", ha="right")
    finish(fig, "reddit_useful_v3", dpi=400)


# ---- model-card banners (same layout as the v2.1 banner, three panels) ---------------
class BannerBar(NamedTuple):
    value: float
    color: str


class BannerPanel(NamedTuple):
    label: str
    big: str
    suffix: str  # small unit after the big number ("tokens"), or ""
    lines: list[str]
    bars: list[BannerBar]
    scale: float  # value that fills a whole bar track


BANNER_X = (0.04, 0.355, 0.67)
BANNER_W = 0.29


def text_pct(raw: str) -> float:
    """First number of a hand-set string like '84.4% [79.8, 88.6]'."""
    m = re.search(r"-?\d+(?:\.\d+)?", raw)
    if not m:
        raise ValueError(f"no number in {raw!r}")
    return float(m.group())


def pct_str(x: float) -> str:
    return fmt_num(Val(x, "measured", None, "%", None))


def val_num(v: Val) -> float:
    return v.value if v.value is not None else 0.0


def fig_width(fig: Figure, t: matplotlib.text.Text) -> float:
    bb = t.get_window_extent(fig.canvas.get_renderer())
    return bb.transformed(fig.transFigure.inverted()).width


def fit_text(fig: Figure, t: matplotlib.text.Text, max_w: float, min_size: float = 6.0) -> None:
    # Shrink rather than clip: labels and comparison lines must stay inside their panel.
    while fig_width(fig, t) > max_w and t.get_fontsize() > min_size:
        t.set_fontsize(t.get_fontsize() - 0.25)


def fig_bottom(fig: Figure, t: matplotlib.text.Text) -> float:
    return t.get_window_extent(fig.canvas.get_renderer()).transformed(fig.transFigure.inverted()).y0


def banner_panel(fig: Figure, x: float, p: BannerPanel) -> None:
    # Three panels are too narrow for v2.1's side-by-side layout, so comparisons sit under the number.
    fit_text(fig, fig.text(x, 0.625, p.label, fontsize=9.5, color=INK2, va="bottom"), BANNER_W)
    big = fig.text(x, 0.455, p.big, fontsize=44, color=BLUE, va="baseline")
    fit_text(fig, big, BANNER_W * 0.8)
    if p.suffix:
        fig.text(x + fig_width(fig, big) + 0.008, 0.455, p.suffix, fontsize=14, color=BLUE, va="baseline")
    side = fig.text(x, 0.405, "\n".join(p.lines), fontsize=10.5, color=INK2, va="top", linespacing=1.4)
    fit_text(fig, side, BANNER_W)
    # Bars start at one height in every panel (room for two comparison lines), lower only if the text runs longer.
    top, floor = min(fig_bottom(fig, side) - 0.03, 0.29), 0.175
    step = min(0.045, (top - floor) / len(p.bars))
    h = step * 0.5
    for i, bar in enumerate(p.bars):
        yy = top - h - i * step
        fig.add_artist(Rectangle((x, yy), BANNER_W, h, fc="#efeeea", ec="none", transform=fig.transFigure))
        w = BANNER_W * min(bar.value / p.scale, 1.0) if p.scale else 0.0
        fig.add_artist(Rectangle((x, yy), max(w, 0.003), h, fc=bar.color, ec="none", transform=fig.transFigure))


def banner(name: str, title: str, version: str, tagline: str, panels: list[BannerPanel], foot: str) -> None:
    fig = plt.figure(figsize=(8, 4), dpi=400, facecolor=SURFACE)
    fit_text(fig, fig.text(0.04, 0.93, title, fontsize=27, va="top"), 0.84 if version else 0.92)
    if version:
        fig.text(0.96, 0.905, version, fontsize=20, va="top", ha="right", color=BLUE)
    fit_text(fig, fig.text(0.04, 0.80, tagline, fontsize=12.5, color=INK2, va="top"), 0.92)
    for y in (0.70, 0.12):
        fig.add_artist(Line2D([0.04, 0.96], [y, y], color=GRID, lw=1, transform=fig.transFigure))
    for x, p in zip(BANNER_X, panels):
        banner_panel(fig, x, p)
    for x in BANNER_X[1:]:
        sep = x - (BANNER_X[1] - BANNER_X[0] - BANNER_W) / 2
        fig.add_artist(Line2D([sep] * 2, [0.18, 0.68], color=GRID, lw=1, transform=fig.transFigure))
    fit_text(fig, fig.text(0.04, 0.06, foot, fontsize=9, color=INK2, va="center"), 0.92)
    finish(fig, name, dpi=400)


def pct_panel(label: str, ours: Val, others: list[tuple[str, Val, str]]) -> BannerPanel:
    lines = [f"{who}: {fmt_val(v)}" for who, v, _ in others]
    bars = [BannerBar(val_num(ours), BLUE)] + [BannerBar(val_num(v), c) for _, v, c in others]
    return BannerPanel(label, fmt_val(ours), "", lines, bars, 100.0)


def think_panel(ours: Val, base: Val, who: str) -> BannerPanel:
    return BannerPanel("THINKING PER ANSWER", fmt_num(ours), "tokens", [f"{who}: {fmt_num(base)} tokens (median)"],
                       [BannerBar(val_num(ours), BLUE), BannerBar(val_num(base), BLUE_LT if who == "v2.1" else GREY)],
                       max(val_num(ours), val_num(base)))


def unbound_answer_panel() -> BannerPanel:
    if d.flag(d.USEFUL263_FLAG[1]):
        v3, v21 = (text_pct(d.get("useful263", k).text or "") for k in ("v3_off", "v21_off"))
        card = d.v21_card()
        xq, orig = text_pct(card["XQ_OFF"]), text_pct(card["ORIG_OFF"])
        return BannerPanel("USEFUL ANSWERS TO HARMFUL REQUESTS", f"{v3:.0f}%", "",
                           [f"v2.1: {v21:.0f}% · original: {card['ORIG_OFF']}",
                            f"best other release: {card['XQ_OFF']}"],
                           [BannerBar(v3, BLUE), BannerBar(v21, BLUE_LT), BannerBar(xq, PLUM), BannerBar(orig, GREY)],
                           100.0)
    return pct_panel("ANSWERS SENSITIVE REQUESTS", d.get("ubench", "willing_v3"),
                     [("v2.1", d.get("ubench", "willing_v21"), BLUE_LT),
                      ("original", d.get("ubench", "willing_stock"), GREY)])


def chart_banner() -> None:
    solved = pct_panel("HARD DEBUGGING TASKS SOLVED", d.gk(d.pair_key("unbound", "debug_solved", True)),
                       [("v2.1", d.gk(d.pair_key("unbound", "debug_solved", False)), BLUE_LT)])
    think = think_panel(d.get("speed", "think_main_v3"), d.get("speed", "think_main_v21"), "v2.1")
    banner("banner_v3", "Inkling-2-Mini-Unbound", "v3",
           "An uncensored Inkling 2 Mini that answers, and now finishes the job",
           [unbound_answer_panel(), solved, think],
           "v3 against v2.1, same questions and settings · 15 debugging tasks · 100 everyday prompts · "
           "details on the card")


def chart_banner_prime() -> None:
    def stock(stem: str) -> tuple[str, Val, str]:
        return "original", d.gk(d.pair_key("prime", stem, False)), GREY

    solved = pct_panel("HARD DEBUGGING TASKS SOLVED", d.gk(d.pair_key("prime", "debug_solved", True)),
                       [stock("debug_solved")])
    stalled = pct_panel("TURNS STALLED WITH NO CODE", d.gk(d.pair_key("prime", "debug_nocode", True)),
                        [stock("debug_nocode")])
    think = think_panel(d.get("speed", "think_main_prime"), d.get("speed", "think_main_stock"), "original")
    banner("banner_prime_v3", d.prime_title(), "",
           "Inkling 2 Mini that stops over-thinking and finishes multi-step work", [solved, stalled, think],
           f"{d.prime_name()} against the original Inkling 2 Mini, same questions and settings · "
           "15 debugging tasks · 100 everyday prompts")


# ---- Reddit image: Prime and Unbound v3 in one picture (16:9, two rows of four cells) ----------------
class RedditCell(NamedTuple):
    label: str
    big: str
    suffix: str
    lines: list[str]
    bars: list[BannerBar]  # after, before; empty for the "kind of model" cell
    scale: float


REDDIT_X = (0.215, 0.392, 0.569, 0.746)
REDDIT_W = (0.165, 0.165, 0.165, 0.21)
REDDIT_ROWS = (0.775, 0.435)
REDDIT_MIN_PT = 8.5  # smallest text that still reads when Reddit shows the image 1200 px wide


def strip_ci(raw: str) -> str:
    """'87.8% [83.7, 91.6]' -> '87.8%': the image shows point values; intervals are on the cards."""
    return raw.split("[")[0].strip()


def guard_width(fig: Figure, t: matplotlib.text.Text, max_w: float) -> None:
    fit_text(fig, t, max_w, REDDIT_MIN_PT)
    if fig_width(fig, t) > max_w:
        print(f"warning: reddit_v3 text too wide: {t.get_text()!r}", file=sys.stderr)


def big_and_unit(v: Val) -> tuple[str, str]:
    if v.unit == "tokens":
        return fmt_num(v), "tokens"
    if v.unit == "s":
        return f"{val_num(v):.1f}", "s"
    return fmt_val(v), ""


def before_text(who: str, v: Val) -> str:
    return f"{who}: {fmt_val(v)}"


def reddit_cell(fig: Figure, x: float, w: float, top: float, c: RedditCell) -> None:
    guard_width(fig, fig.text(x, top, c.label, fontsize=9, color=INK2, va="top", linespacing=1.3), w)
    base = top - 0.165
    big = fig.text(x, base, c.big, fontsize=30, color=BLUE, va="baseline")
    guard_width(fig, big, w)
    if c.suffix:
        fig.text(x + fig_width(fig, big) + 0.006, base, c.suffix, fontsize=12, color=BLUE, va="baseline")
    lines = fig.text(x, top - 0.19, "\n".join(c.lines), fontsize=9.5, color=INK2, va="top", linespacing=1.3)
    guard_width(fig, lines, w)
    for i, bar in enumerate(c.bars):
        yy = top - 0.262 - i * 0.03
        fig.add_artist(Rectangle((x, yy), w, 0.017, fc="#efeeea", ec="none", transform=fig.transFigure))
        bw = w * min(bar.value / c.scale, 1.0) if c.scale else 0.0
        fig.add_artist(Rectangle((x, yy), max(bw, 0.003), 0.017, fc=bar.color, ec="none", transform=fig.transFigure))


def reddit_row(fig: Figure, top: float, name: str, caption: str, cells: list[RedditCell]) -> None:
    fig.text(0.04, top + 0.004, name, fontsize=16, va="top")
    fig.text(0.04, top - 0.075, caption, fontsize=10, color=INK2, va="top", linespacing=1.35)
    for x, w, c in zip(REDDIT_X, REDDIT_W, cells):
        reddit_cell(fig, x, w, top, c)
    for x, prev_x, prev_w in zip(REDDIT_X, (0.04,) + REDDIT_X[:-1], (0.16,) + REDDIT_W[:-1]):
        sep = (prev_x + prev_w + x) / 2
        fig.add_artist(Line2D([sep] * 2, [top - 0.29, top + 0.005], color=GRID, lw=1, transform=fig.transFigure))


def compare_cell(label: str, after: Val, before: Val, who: str, colour: str, scale: float) -> RedditCell:
    big, unit = big_and_unit(after)
    return RedditCell(label, big, unit, [before_text(who, before)],
                      [BannerBar(val_num(after), BLUE), BannerBar(val_num(before), colour)], scale)


def shared_cells(pair: Pair, who: str, colour: str) -> list[RedditCell]:
    """The three before/after cells; token and second bars share one scale across both rows."""
    def sp(stem: str, fixed: bool) -> Val:
        return d.gk(d.flat_key("speed", pair, stem, fixed))

    def col_max(stem: str) -> float:
        return max(val_num(d.gk(d.flat_key("speed", p, stem, f))) for p in d.PAIRS for f in (False, True))

    return [
        compare_cell("HARD DEBUGGING\nTASKS SOLVED", d.gk(d.pair_key(pair, "debug_solved", True)),
                     d.gk(d.pair_key(pair, "debug_solved", False)), who, colour, 100.0),
        compare_cell("THINKING PER\nANSWER (MEDIAN)", sp("think_main", True), sp("think_main", False), who, colour,
                     col_max("think_main")),
        compare_cell("SECONDS PER\nANSWER", sp("sec_main", True), sp("sec_main", False), who, colour,
                     col_max("sec_main")),
    ]


def prime_kind_cell() -> RedditCell:
    return RedditCell("REFUSES LIKE\nTHE ORIGINAL", fmt_val(d.get("ubench", "willing_prime")), "",
                      ["sensitive requests answered",
                       before_text("original", d.get("ubench", "willing_stock"))], [], 100.0)


def unbound_kind_cell() -> RedditCell:
    if not d.flag(d.USEFUL263_FLAG[1]):
        return RedditCell("USEFUL ANSWERS TO\nHARMFUL REQUESTS", "pending", "", [], [], 100.0)
    card = d.v21_card()
    orig = card.get("ORIG_E09", card["ORIG_OFF"])
    return RedditCell("USEFUL ANSWERS TO\nHARMFUL REQUESTS", strip_ci(d.get("useful263", "v3_on").text or ""), "",
                      [f"v2.1: {strip_ci(d.get('useful263', 'v21_on').text or '')}",
                       f"best other release: {card['XQ_E09']}", f"original: {orig}"], [], 100.0)


def reddit_foot() -> str:
    n_dbg = fmt_num(d.gk((d.PAIR_GROUP["prime"], "debug_n_tasks")))
    n_main = fmt_num(d.get("speed", "main_n"))
    n_harm = (d.get("useful263", "bison_n_on").text or "") if d.flag(d.USEFUL263_FLAG[1]) else ""
    harm = f"{n_harm} harmful requests (thinking on) · " if n_harm else ""
    return (f"Same questions and settings for before and after · {n_dbg} debugging tasks · "
            f"{n_main} everyday prompts\n{harm}details on the model cards")


def chart_reddit() -> None:
    fig = plt.figure(figsize=(8, 4.5), dpi=400, facecolor=SURFACE)
    fit_text(fig, fig.text(0.04, 0.955, "Inkling 2 Mini: Prime and Unbound v3", fontsize=24, va="top"), 0.92)
    fit_text(fig, fig.text(0.04, 0.862, "One small add-on that stops the over-thinking. Two models: one censored, "
                           "one not.", fontsize=12.5, color=INK2, va="top"), 0.92)
    for y in (0.805, 0.47, 0.11):
        fig.add_artist(Line2D([0.04, 0.96], [y, y], color=GRID, lw=1, transform=fig.transFigure))
    reddit_row(fig, REDDIT_ROWS[0], d.prime_name(), "original\nInkling 2 Mini\n+ the add-on",
               shared_cells("prime", "original", GREY) + [prime_kind_cell()])
    reddit_row(fig, REDDIT_ROWS[1], "Unbound v3", "uncensored\n+ the same\nadd-on",
               shared_cells("unbound", "v2.1", BLUE_LT) + [unbound_kind_cell()])
    fig.text(0.04, 0.09, reddit_foot(), fontsize=REDDIT_MIN_PT, color=INK2, va="top", linespacing=1.35)
    fig.text(0.96, 0.041, "huggingface.co/nokeepai", fontsize=10, color=BLUE, va="baseline", ha="right")
    finish(fig, "reddit_v3", dpi=400)


# ---- main -------------------------------------------------------------------------
def refuse_if_not_final() -> None:
    bad: list[Key] = d.not_final(d.all_chart_keys())
    if not bad:
        return
    print(f"--final refused: {len(bad)} plotted values are TBD or provisional:", file=sys.stderr)
    for g, k in bad:
        print(f"  {g}.{k}  ({d.get(g, k).status})", file=sys.stderr)
    sys.exit(1)


def main(argv: list[str]) -> int:
    global OUT
    if "--values" in argv:
        d.set_values_path(argv[argv.index("--values") + 1])
    if "--out" in argv:
        OUT = os.path.abspath(os.path.expanduser(argv[argv.index("--out") + 1]))
    if "--final" in argv:
        refuse_if_not_final()
    chart_banner()
    chart_banner_prime()
    chart_reddit()
    if d.flag(d.USEFUL263_FLAG[1]):
        chart_reddit_useful()
    chart_hero()
    chart_speed()
    for pair in d.PAIRS:
        chart_nothing_lost(pair)
    chart_history()
    for pair in d.PAIRS:
        chart_indep(pair)
    chart_thinkdist()
    chart_xver()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
