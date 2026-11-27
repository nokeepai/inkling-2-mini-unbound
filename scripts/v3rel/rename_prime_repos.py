#!/usr/bin/env python3
"""Owner rule (Prime -> Honed): rename the three staged repos while they are still PRIVATE, so the 20 GB of staged
weights need no re-upload.

    rename_prime_repos.py --to Honed [--from Prime] [--dry-run] --yes-owner-approved

Refuses unless every source repo exists and is private and no destination repo exists. Renames only; no file is
changed. Re-stage afterwards with `stage_v3.py ... --prime-name <to>` so receipts exist under the new names
(identical files are skipped, not re-uploaded).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v3rel_common as C  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="src", default="Prime")
    ap.add_argument("--to", required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--allow-public", action="store_true",
                    help="owner decision after go-live: rename repos that are already public (the old ids redirect)")
    ap.add_argument("--yes-owner-approved", action="store_true")
    a = ap.parse_args()
    old, new = C.names(a.src), C.names(a.to)
    pairs = [(old.prime, new.prime), (old.prime_lora, new.prime_lora), (old.prime_gguf, new.prime_gguf)]
    for s, d in pairs:
        print(f"PLAN rename {s} -> {d}")
    if a.dry_run:
        print("DRY RUN: no network calls made.")
        return 0
    C.approval_gate(a.yes_owner_approved)
    C.require_token()
    api = C.real_api()
    for s, d in pairs:
        private = C.is_private(api, s)
        if private is None or (private is not True and not a.allow_public):
            raise SystemExit(f"STOP: {s} is missing or not private")
        try:
            info = api.model_info(d)
        except Exception:  # noqa: BLE001 - absent is what we want
            continue
        if info.id != s:  # an id that merely redirects to the source repo is free to take back
            raise SystemExit(f"STOP: destination {d} already exists")
    for s, d in pairs:
        was_private = C.is_private(api, s)
        api.move_repo(from_id=s, to_id=d, repo_type="model")
        if C.is_private(api, d) is not was_private:
            raise SystemExit(f"STOP: {d} changed visibility during the rename")
        print(f"RENAMED {s} -> {d} ({'private' if was_private else 'public'})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
