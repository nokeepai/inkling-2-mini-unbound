#!/usr/bin/env python3
"""After go-live: refresh a PUBLIC repo's card (README.md) and, optionally, its charts as results land.

    golive_card_update.py --repo nokeepai/NAME --card FILE [--charts DIR --chart NAME ...] [--dry-run] --yes-owner-approved

Touches README.md and charts/<NAME>.{png,svg} only. Never deletes, never touches weights. Skips files already
identical on the hub. Same approval gate and card lint as the other go-live steps; refuses a private repo (private
repos are updated with stage_v3.py).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v3rel_common as C  # noqa: E402

BAD_MARKERS = ("{{", "TBD", "DRAFT RENDER", "[MISSING")
ALLOWED = ("nokeepai/Inkling-2-Mini-Prime", "nokeepai/Inkling-2-Mini-Prime-LoRA", "nokeepai/Inkling-2-Mini-Prime-GGUF",
           "nokeepai/Inkling-2-Mini-Unbound", "nokeepai/Inkling-2-Mini-Unbound-GGUF",
           "nokeepai/Inkling-2-Mini-Unbound-Basalt")


def collect(card: Path, charts: Path | None, names: list[str], prefix: str) -> dict[str, Path]:
    text = card.read_text()
    for marker in BAD_MARKERS:
        if marker in text:
            raise SystemExit(f"STOP: card still contains {marker!r}")
    files = {f"{prefix}README.md": card}
    for name in names:
        for ext in ("png", "svg"):
            path = (charts or Path(".")) / f"{name}.{ext}"
            if not path.is_file():
                raise SystemExit(f"STOP: missing chart file {path}")
            files[f"charts/{name}.{ext}"] = path
    return files


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, choices=ALLOWED)
    ap.add_argument("--card", required=True, type=Path)
    ap.add_argument("--path-prefix", default="", help="e.g. lora-v3-fix/ to update a folder README")
    ap.add_argument("--charts", type=Path)
    ap.add_argument("--chart", action="append", default=[])
    ap.add_argument("--message", default="docs(v3): refresh card with results that landed after release")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--yes-owner-approved", action="store_true")
    a = ap.parse_args()
    files = collect(a.card, a.charts, a.chart, a.path_prefix)
    metas = {name: C.local_meta(path) for name, path in files.items()}
    print(f"PLAN {a.repo}: {len(files)} candidate files (README + {len(a.chart)} charts), adds/replacements only")
    if a.dry_run:
        print("DRY RUN: no network calls made.")
        return 0
    C.approval_gate(a.yes_owner_approved)
    C.require_token()
    from huggingface_hub import CommitOperationAdd

    api = C.real_api()
    if C.is_private(api, a.repo) is not False:
        raise SystemExit(f"STOP: {a.repo} is not public; use stage_v3.py for private repos")
    parent, remote = C.remote_files(api, a.repo)
    todo = {name: path for name, path in files.items() if not C.same_content(metas[name], remote.get(name))}
    if not todo:
        print(f"UP TO DATE {a.repo} @ {parent[:12]}")
        return 0
    ops = [CommitOperationAdd(path_in_repo=name, path_or_fileobj=str(path)) for name, path in todo.items()]
    info = api.create_commit(repo_id=a.repo, repo_type="model", operations=ops, parent_commit=parent,
                             commit_message=a.message)
    _, after = C.remote_files(api, a.repo)
    bad = [name for name in todo if not C.same_content(metas[name], after.get(name))]
    lost = sorted(set(remote) - set(after))
    if bad or lost:
        print(f"VERIFY FAIL: differs {bad}, missing {lost[:5]}")
        return 1
    print(f"UPDATED {a.repo} {info.oid[:12]}: {sorted(todo)}; {len(after)} files on main (was {len(remote)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
