#!/usr/bin/env python3
"""Go-live step the main script does not cover: put the v3 fix adapter inside the public Unbound repo as
`lora-v3-fix/`, next to the v2.1 LoRA folders. One commit, adds only, never deletes.

    golive_lora_folder.py --adapter DIR --merge-script FILE --readme FILE [--dry-run] --yes-owner-approved

Files written under lora-v3-fix/: adapter_model.safetensors, adapter_config.json, merge_adapter.py, README.md.
Same approval gate as golive_v3.py (flag + ~/.inkling-abl/v3_golive_approved). Run it AFTER unbound-promote so the
folder lands on top of the v3 commit. Verifies the uploaded sha256 by reading the repo back.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v3rel_common as C  # noqa: E402

REPO = "nokeepai/Inkling-2-Mini-Unbound"
FOLDER = "lora-v3-fix"
BAD_MARKERS = ("{{", "TBD", "DRAFT RENDER", "[MISSING")


def plan_files(adapter: Path, merge_script: Path, readme: Path) -> dict[str, Path]:
    files = {
        f"{FOLDER}/adapter_model.safetensors": adapter / "adapter_model.safetensors",
        f"{FOLDER}/adapter_config.json": adapter / "adapter_config.json",
        f"{FOLDER}/merge_adapter.py": merge_script,
        f"{FOLDER}/README.md": readme,
    }
    for name, path in files.items():
        if not path.is_file():
            raise SystemExit(f"STOP: missing {name} source: {path}")
    text = readme.read_text()
    for marker in BAD_MARKERS:
        if marker in text:
            raise SystemExit(f"STOP: README still contains {marker!r}")
    return files


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True, type=Path)
    ap.add_argument("--merge-script", required=True, type=Path)
    ap.add_argument("--readme", required=True, type=Path)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--yes-owner-approved", action="store_true")
    a = ap.parse_args()
    files = plan_files(a.adapter, a.merge_script, a.readme)
    metas = {name: C.local_meta(path) for name, path in files.items()}
    total_gb = sum(m["size"] for m in metas.values()) / 1e9
    print(f"PLAN {REPO}: add {len(files)} files under {FOLDER}/, {total_gb:.3f} GB, one commit, no deletions")
    for name, meta in metas.items():
        print(f"  add {name}  {meta['size']} bytes  sha256 {str(meta.get('sha256', ''))[:16]}")
    if a.dry_run:
        print("DRY RUN: no network calls made.")
        return 0
    C.approval_gate(a.yes_owner_approved)
    C.require_token()
    from huggingface_hub import CommitOperationAdd

    api = C.real_api()
    parent, remote = C.remote_files(api, REPO)
    if "README.md" not in remote:
        raise SystemExit(f"STOP: {REPO} main has no README.md; wrong repo?")
    ops = [CommitOperationAdd(path_in_repo=name, path_or_fileobj=str(path)) for name, path in files.items()]
    info = api.create_commit(repo_id=REPO, repo_type="model", operations=ops, parent_commit=parent,
                             commit_message="v3: add the fix adapter as lora-v3-fix/ (apply to v2.1 to get v3)")
    print(f"COMMITTED {REPO} {info.oid}")
    _, after = C.remote_files(api, REPO)
    bad = [name for name, meta in metas.items() if not C.same_content(meta, after.get(name))]
    if bad:
        print(f"VERIFY FAIL: {bad}")
        return 1
    lost = sorted(set(remote) - set(after))
    if lost:
        print(f"VERIFY FAIL: files missing after commit: {lost[:5]}")
        return 1
    print(f"VERIFY PASS {REPO}: {len(files)} files match local sha256; {len(after)} files on main (was {len(remote)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
