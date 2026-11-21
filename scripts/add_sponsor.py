#!/usr/bin/env python3
"""Add a short, non-intrusive nokeep sponsor note to the HF card (+ draft) and the GitHub README:
one line under the repo line near the top, and a small section at the end."""
import re

TOP = ("Built by [nokeep](https://nokeep.ai), who also sponsored the compute. nokeep runs private, "
       "uncensored AI chat, image and video at [nokeep.ai](https://nokeep.ai).")
END = """
## About nokeep

This model was built, and its compute paid for, by [nokeep](https://nokeep.ai). nokeep is a private AI app for
uncensored chat, images and video: no filters getting in the way, and your conversations stay yours.
If this model is useful to you, the best way to support more releases like it is to
[try nokeep](https://nokeep.ai).
"""


def patch(path: str, anchor_regex: str) -> None:
    text = open(path).read()
    if "## About nokeep" in text:
        print("already patched:", path)
        return
    m = re.search(anchor_regex, text, flags=re.M)
    assert m, f"anchor not found in {path}"
    text = text[: m.end()] + "\n\n" + TOP + text[m.end():]
    text = text.rstrip() + "\n" + END
    open(path, "w").write(text)
    print("patched:", path)


R = "/Users/nikshepsvn/Code/inkling-abl"
# HF card: after the repo line under the banner
for p in (f"{R}/out/final/MODEL_CARD.md", f"{R}/package/MODEL_CARD_DRAFT.md"):
    patch(p, r"^`nokeepai/Inkling-2-Mini-Unbound` · .*$")
# GitHub README: after the first paragraph under the title
patch(f"{R}/README.md", r"^An abliterated \(refusal-removed\).*$")
