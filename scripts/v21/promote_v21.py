"""Promote the v2.1 branch to main on a HF repo with SERVER-SIDE copies (no re-upload). HF_TOKEN from env.

  python3 promote_v2.py <repo_id> [--apply] [--tag v2.0]

Every file on `v2` that is new or differs from `main`: LFS files -> CommitOperationCopy(src_revision="v2.1"),
small (non-LFS) files -> re-added from the v2 bytes. Files only on main are kept (v1.0 files stay available).
Dry run by default: prints the plan. With --apply: one commit on main, then (optionally) a tag.
"""
from __future__ import annotations

import sys

from huggingface_hub import CommitOperationAdd, CommitOperationCopy, HfApi, hf_hub_download


def files(api: HfApi, repo: str, rev: str) -> dict:
    out = {}
    for f in api.list_repo_tree(repo, revision=rev, recursive=True, expand=True):
        if getattr(f, "size", None) is None and not hasattr(f, "lfs"):
            continue  # folder
        lfs = getattr(f, "lfs", None)
        out[f.path] = {"oid": (lfs.sha256 if lfs else getattr(f, "blob_id", None)), "lfs": bool(lfs), "size": f.size}
    return out


def main() -> None:
    repo = sys.argv[1]
    apply = "--apply" in sys.argv
    tag = sys.argv[sys.argv.index("--tag") + 1] if "--tag" in sys.argv else None
    api = HfApi()
    v2, main_ = files(api, repo, "v2.1"), files(api, repo, "main")
    ops = []
    for path, meta in sorted(v2.items()):
        if path in main_ and main_[path]["oid"] == meta["oid"]:
            continue
        if meta["lfs"]:
            ops.append(CommitOperationCopy(src_path_in_repo=path, path_in_repo=path, src_revision="v2.1"))
            kind = "copy"
        else:
            local = hf_hub_download(repo, path, revision="v2.1")
            ops.append(CommitOperationAdd(path_in_repo=path, path_or_fileobj=local))
            kind = "add"
        print(f"{kind:4s} {path}  ({(meta['size'] or 0) / 1e9:.2f} GB)")
    print(f"{repo}: {len(ops)} operations; files only on main are kept: {len(set(main_) - set(v2))}")
    if apply and ops:
        c = api.create_commit(repo, operations=ops, revision="main", commit_message="Release v2.1 (promote v2.1 branch)")
        print("COMMITTED", c.commit_url)
    if apply and tag:
        api.create_tag(repo, tag=tag, revision="main", tag_message="Inkling-2-Mini-Unbound v2.1", exist_ok=True)
        print("TAGGED", tag)


if __name__ == "__main__":
    main()
