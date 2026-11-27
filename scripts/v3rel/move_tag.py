#!/usr/bin/env python3
"""Show, and with --move re-point, a tag on the public model repos so it names the current head of main.

    move_tag.py [--tag v3.0] [--move --yes-owner-approved] [--repo ID ...]

A card refresh adds a commit to main, and a tag made at go-live then lags behind it. Weights are untouched by card
commits, so moving the tag changes which README and charts a tagged download gets, nothing else. The script refuses
to move a tag when any weight file (LFS) differs between the tagged revision and main.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v3rel_common as C  # noqa: E402

DEFAULT = ("nokeepai/Inkling-2-Mini-Unbound", "nokeepai/Inkling-2-Mini-Unbound-GGUF", "nokeepai/Inkling-2-Mini-Unbound-Basalt")


def lfs_files(api: object, repo: str, rev: str) -> dict[str, str]:
    info = api.model_info(repo, revision=rev, files_metadata=True)  # type: ignore[attr-defined]
    # chart images are LFS files too and change with every card refresh; only the weights must not move
    return {s.rfilename: s.lfs.sha256 for s in info.siblings
            if s.lfs is not None and s.rfilename.endswith((".safetensors", ".gguf", ".bin", ".pt"))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v3.0")
    ap.add_argument("--repo", action="append")
    ap.add_argument("--move", action="store_true")
    ap.add_argument("--yes-owner-approved", action="store_true")
    a = ap.parse_args()
    C.require_token()
    api = C.real_api()
    if a.move:
        C.approval_gate(a.yes_owner_approved)
    for repo in a.repo or DEFAULT:
        refs = api.list_repo_refs(repo)
        head = next(b.target_commit for b in refs.branches if b.name == "main")
        if a.tag not in [t.name for t in refs.tags]:
            print(f"{repo}: no tag {a.tag}; tags: {[t.name for t in refs.tags]}")
            continue
        # a tag made with a message is its own object, so ask the hub which commit the tag resolves to
        at = api.model_info(repo, revision=a.tag).sha
        if at == head:
            print(f"{repo}: {a.tag} already at main {head[:12]}")
            continue
        same = lfs_files(api, repo, at) == lfs_files(api, repo, head)
        print(f"{repo}: {a.tag} at {at[:12]}, main at {head[:12]}, weight files identical: {same}")
        if not a.move:
            continue
        if not same:
            raise SystemExit(f"STOP: weight files differ between {a.tag} and main in {repo}; tag left alone")
        api.delete_tag(repo, tag=a.tag)
        api.create_tag(repo, tag=a.tag, revision=head, tag_message=f"{a.tag}: same weights, current card")
        print(f"  MOVED {a.tag} -> {head[:12]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
