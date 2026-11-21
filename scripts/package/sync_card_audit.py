"""Sync the card with the auditor's FINAL numbers (audit/AUDIT.md §1, §4): Jev useful and hand-verified honest limits,
and the back-translation note for language rows. Edits R/package/MODEL_CARD_DRAFT.md in place; fails loudly if a
target string is missing."""
import os

P = os.path.expanduser("~/Code/inkling-abl/package/MODEL_CARD_DRAFT.md")
s = open(P).read()
REPL = [
    ("- **Answers usefully.** 54% (thinking off)", "- **Answers usefully.** 53% (thinking off)"),
    ("more useful answers (54% / 63% against\n34% / 43%)", "more useful answers (53% / 63% against\n35% / 43%)"),
    ("| 0.4% [0.0, 1.1] | 34.3% [28.9, 39.7] | **54.2%** [48.4, 59.9] |", "| 0.4% [0.0, 1.1] | 35.0% [29.6, 40.8] | **53.1%** [47.3, 58.8] |"),
    ("| 1.1% [0.0, 2.5] | 42.6% [36.8, 48.4] | **63.2%** [57.4, 69.0] |", "| 1.1% [0.0, 2.5] | 43.3% [37.6, 49.1] | **63.2%** [57.4, 69.0] |"),
    ("| Honest limits, thinking off / effort 0.9 | original's rate | 90% / 70% | 70% / 70% | 80% / 60% |",
     "| Honest limits, thinking off / effort 0.9 (hand-verified) | original's rate | 100% / 90% | 100% / 90% | 80% / 80% |"),
    ("About 54–63% of harmful requests", "About 53–63% of harmful requests"),
    ("""  states the limit on 80% (thinking off) and 60% (effort 0.9) of probes, against 90% and 70% for the original.""",
     """  states the limit on 80% of probes in both modes (hand-verified), against 100% (thinking off) and 90% (effort 0.9)
  for the original. Real failures include describing a photo that was never attached and answering "No." to an
  "is he lying?" question it cannot judge."""),
    ("""Non-English rows use the
  English refusal phrase list; refusals written in other languages may be undercounted.""",
     """Non-English rows count a
  refusal if the raw answer or its back-translation to English matches the refusal phrase list."""),
]
for old, new in REPL:
    if old not in s:
        raise SystemExit(f"missing in card: {old[:70]!r}")
    s = s.replace(old, new)
open(P, "w").write(s)
print("ok", len(REPL), "replacements")
