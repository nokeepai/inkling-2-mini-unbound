#!/usr/bin/env python3
"""Insert the 'How it compares' section into the main card, the GGUF card and the GitHub README."""
R = "/Users/nikshepsvn/Code/inkling-abl"
SECTION = open(f"{R}/package/compare_section.md").read().strip() + "\n\n"


def insert(path: str, before: str, section: str) -> None:
    t = open(path).read()
    if "## How it compares" in t:
        start = t.index("## How it compares")
        end = t.index("\n## ", start + 5) + 1
        t = t[:start] + t[end:]
    i = t.index(before)
    t = t[:i] + section + t[i:]
    open(path, "w").write(t)
    print("inserted:", path)


insert(f"{R}/out/final/MODEL_CARD.md", "## Evaluation", SECTION)
insert(f"{R}/package/MODEL_CARD_DRAFT.md", "## Evaluation", SECTION)
insert(f"{R}/package/gguf/README.md", "## Files", SECTION)
gh = SECTION.replace("](charts/", "](package/charts/").replace(
    "[GitHub repo](https://github.com/nokeepai/inkling-2-mini-unbound/tree/main/compare)", "[`compare/`](compare/)")
insert(f"{R}/README.md", "## What was edited", gh)
