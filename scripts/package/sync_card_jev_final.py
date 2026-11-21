"""Revert the card's Jev useful numbers to the auditor's FINAL second pass (runs.jsonl / jev_metrics.json,
AUDIT.md corrected 2026-11-21) and add the run-to-run variance note. Fails loudly if a target string is missing."""
import os

P = os.path.expanduser("~/Code/inkling-abl/package/MODEL_CARD_DRAFT.md")
s = open(P).read()
REPL = [
    ("- **Answers usefully.** 53% (thinking off)", "- **Answers usefully.** 54% (thinking off)"),
    ("more useful answers (53% / 63% against\n35% / 43%)", "more useful answers (54% / 63% against\n34% / 43%)"),
    ("| 0.4% [0.0, 1.1] | 35.0% [29.6, 40.8] | **53.1%** [47.3, 58.8] |", "| 0.4% [0.0, 1.1] | 34.3% [28.9, 39.7] | **54.2%** [48.4, 59.9] |"),
    ("| 1.1% [0.0, 2.5] | 43.3% [37.6, 49.1] | **63.2%** [57.4, 69.0] |", "| 1.1% [0.0, 2.5] | 42.6% [36.8, 48.4] | **63.2%** [57.4, 69.0] |"),
    ("About 53–63% of harmful requests", "About 54–63% of harmful requests"),
    ("against 53.1% for the plain\n  request.", "against 54.2% for the plain\n  request."),
    ("""  p ≥ 0.5, a threshold fixed before scoring. On 50 hand-labelled responses the judge agreed 86% of the time
  (Cohen's κ 0.71).""",
     """  p ≥ 0.5, a threshold fixed before scoring. On 50 hand-labelled responses the judge agreed 86% of the time
  (Cohen's κ 0.71). Jev scores vary by up to about 1.5 points between runs on the same answers."""),
]
for old, new in REPL:
    if old not in s:
        raise SystemExit(f"missing in card: {old[:70]!r}")
    s = s.replace(old, new)
open(P, "w").write(s)
print("ok", len(REPL))
