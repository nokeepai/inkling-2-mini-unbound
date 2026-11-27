#!/usr/bin/env python3
"""Rewrite SHA256SUMS and MANIFEST.tsv in the public Prime GGUF repo from the files actually on the hub.

    prime_gguf_manifest.py [--dry-run] --yes-owner-approved

Used once after the repo's GGUF files got their Prime names back: both small text files still listed the old file
names. Sizes and sha256 values are read from the hub's own LFS metadata, so the lists cannot drift from the files.
One commit, two small text files, nothing deleted.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v3rel_common as C  # noqa: E402

REPO = "nokeepai/Inkling-2-Mini-Prime-GGUF"


def build(api: object) -> tuple[str, str, int]:
    info = api.model_info(REPO, files_metadata=True)  # type: ignore[attr-defined]
    if info.private:
        raise SystemExit(f"STOP: {REPO} is private")
    ggufs = sorted((s for s in info.siblings if s.rfilename.endswith(".gguf")), key=lambda s: s.rfilename)
    if any(s.lfs is None for s in ggufs):
        raise SystemExit("STOP: a GGUF file has no LFS metadata")
    if any("Prime" not in s.rfilename and not s.rfilename.startswith("mmproj-") for s in ggufs):
        raise SystemExit("STOP: a GGUF file without the Prime name is still in the repo")
    sums = "".join(f"{s.lfs.sha256}  {s.rfilename}\n" for s in ggufs)
    manifest = "".join(  # same three columns as before: file, bytes, sha256 (no header line)
        f"{s.rfilename}\t{s.size}\t{s.lfs.sha256}\n" for s in ggufs)
    return sums, manifest, len(ggufs)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--yes-owner-approved", action="store_true")
    a = ap.parse_args()
    C.require_token()
    api = C.real_api()
    sums, manifest, n = build(api)
    print(f"PLAN {REPO}: SHA256SUMS and MANIFEST.tsv for {n} GGUF files")
    print(manifest, end="")
    if a.dry_run:
        print("DRY RUN: nothing sent.")
        return 0
    C.approval_gate(a.yes_owner_approved)
    from huggingface_hub import CommitOperationAdd
    commit = api.create_commit(REPO, operations=[
        CommitOperationAdd("SHA256SUMS", sums.encode()), CommitOperationAdd("MANIFEST.tsv", manifest.encode())],
        commit_message="SHA256SUMS and MANIFEST.tsv: list the files under their Prime names")
    print("UPDATED", REPO, commit.oid[:12])
    return 0


if __name__ == "__main__":
    sys.exit(main())
