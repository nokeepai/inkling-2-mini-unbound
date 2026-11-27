#!/usr/bin/env python3
"""Publish the v3 release files to the public GitHub repo as ONE ordinary commit on main, through the GitHub API
(`gh api`; no local git; history kept unless the owner asks for --replace-onto).

    gh_publish_v3.py --manifest FILE --message MSG [--dry-run] --yes-owner-approved

Manifest: TSV, one line per file: <local path relative to the worktree>\t<path in the GitHub repo>.
Adds and replaces only; nothing is deleted. Refuses files over 5 MB, binary junk (.DS_Store) and any path whose
text matches a secret pattern. Needs `gh` logged in; the token is never read or printed here.
"""
from __future__ import annotations

import argparse
import base64
import json
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = "nokeepai/inkling-2-mini-unbound"
BRANCH = "main"
APPROVAL = Path.home() / ".inkling-abl" / "v3_golive_approved"
SECRET = re.compile(rb"(hf_[A-Za-z0-9]{30,}|sk-[A-Za-z0-9_-]{30,}|tml-[A-Za-z0-9_-]{20,}|ghp_[A-Za-z0-9]{30,}|AKIA[0-9A-Z]{16}|-----BEGIN [A-Z ]*PRIVATE KEY-----)")
MAX_BYTES = 5 << 20


def gh(args: list[str], body: dict | None = None, tries: int = 5) -> dict:
    for attempt in range(tries):
        proc = subprocess.run(["gh", "api", *args, *(["--input", "-"] if body is not None else [])],
                              input=json.dumps(body) if body is not None else None, capture_output=True, text=True)
        if proc.returncode == 0:
            return json.loads(proc.stdout) if proc.stdout.strip() else {}
        if "rate limit" in proc.stderr.lower() or "secondary" in proc.stderr.lower() or "50" in proc.stderr[:40]:
            time.sleep(20 * (attempt + 1))
            continue
        raise SystemExit(f"gh api {' '.join(args[:2])} failed: {proc.stderr.strip()[:300]}")
    raise SystemExit(f"gh api {' '.join(args[:2])} failed after {tries} tries")


def load_manifest(path: Path, root: Path) -> list[tuple[Path, str]]:
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        local, dest = line.split("\t")
        src = root / local
        if not src.is_file():
            raise SystemExit(f"STOP: missing {src}")
        data = src.read_bytes()
        if len(data) > MAX_BYTES:
            raise SystemExit(f"STOP: {local} is {len(data)} bytes (limit {MAX_BYTES})")
        if src.name == ".DS_Store" or dest.startswith("/") or ".." in dest.split("/"):
            raise SystemExit(f"STOP: bad entry {local} -> {dest}")
        if SECRET.search(data):
            raise SystemExit(f"STOP: secret-looking text in {local}")
        rows.append((src, dest))
    dests = [d for _, d in rows]
    if len(dests) != len(set(dests)):
        raise SystemExit("STOP: duplicate destination paths in the manifest")
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--message", required=True)
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--replace-onto", default="",
                    help="owner request: REPLACE earlier commits instead of adding one. Full sha of the commit to "
                         "build on; the new commit gets it as its only parent and the branch is force-moved")
    ap.add_argument("--replace-commits", default="",
                    help="with --replace-onto: comma-separated full shas, oldest first, of the commits being "
                         "replaced. Refuses unless the branch is exactly <onto> followed by these, nothing else")
    ap.add_argument("--move-tag", default="", help="with --replace-onto: also force-move this existing tag to the new commit")
    ap.add_argument("--yes-owner-approved", action="store_true")
    a = ap.parse_args()
    rows = load_manifest(a.manifest, a.root)
    total = sum(src.stat().st_size for src, _ in rows)
    print(f"PLAN {REPO}@{BRANCH}: {len(rows)} files, {total / 1e6:.1f} MB, one commit, adds/replacements only")
    if a.dry_run:
        tops: dict[str, int] = {}
        for _, dest in rows:
            tops[dest.split("/")[0]] = tops.get(dest.split("/")[0], 0) + 1
        print("  by top-level path:", dict(sorted(tops.items())))
        print("DRY RUN: nothing sent.")
        return 0
    if not a.yes_owner_approved or not APPROVAL.is_file():
        raise SystemExit("REFUSED: needs --yes-owner-approved and the approval file")
    head = gh([f"repos/{REPO}/git/ref/heads/{BRANCH}"])["object"]["sha"]
    parent, force = head, False
    if a.replace_onto:
        expected = [s for s in a.replace_commits.split(",") if s]
        if not expected:
            raise SystemExit("REFUSED: --replace-onto needs --replace-commits")
        cur = head
        for want in reversed(expected):  # walk back from the branch head; every step must match exactly
            c = gh([f"repos/{REPO}/git/commits/{cur}"])
            if cur != want or len(c["parents"]) != 1:
                raise SystemExit(f"STOP: branch history is not the expected one at {cur[:8]} (wanted {want[:8]}); "
                                 "something else landed, nothing was changed")
            cur = c["parents"][0]["sha"]
        if cur != a.replace_onto:
            raise SystemExit(f"STOP: the replaced commits sit on {cur[:8]}, not on {a.replace_onto[:8]}; nothing was changed")
        parent, force = a.replace_onto, True
        print(f"REPLACE: {len(expected)} commit(s) after {parent[:8]} will be replaced by one new commit")
    base_tree = gh([f"repos/{REPO}/git/commits/{parent}"])["tree"]["sha"]
    tree = []
    for i, (src, dest) in enumerate(rows, 1):
        blob = gh([f"repos/{REPO}/git/blobs", "-X", "POST"],
                  {"content": base64.b64encode(src.read_bytes()).decode(), "encoding": "base64"})
        mode = "100755" if src.suffix == ".sh" else "100644"
        tree.append({"path": dest, "mode": mode, "type": "blob", "sha": blob["sha"]})
        if i % 50 == 0:
            print(f"  blobs {i}/{len(rows)}", flush=True)
    # One request with several hundred entries times out on GitHub's side, so the tree is built in slices, each
    # on top of the previous one. Nothing is visible on the branch until the final ref update below.
    new_tree = base_tree
    for i in range(0, len(tree), 80):
        new_tree = gh([f"repos/{REPO}/git/trees", "-X", "POST"], {"base_tree": new_tree, "tree": tree[i:i + 80]})["sha"]
    commit = gh([f"repos/{REPO}/git/commits", "-X", "POST"],
                {"message": a.message, "tree": new_tree, "parents": [parent]})["sha"]
    if force and gh([f"repos/{REPO}/git/ref/heads/{BRANCH}"])["object"]["sha"] != head:
        raise SystemExit("STOP: the branch moved while the commit was being built; nothing was changed")
    gh([f"repos/{REPO}/git/refs/heads/{BRANCH}", "-X", "PATCH"], {"sha": commit, "force": force})
    print(f"COMMITTED https://github.com/{REPO}/commit/{commit} (parent {parent[:8]}{', replaced ' + head[:8] if force else ''})")
    if force and a.move_tag:
        gh([f"repos/{REPO}/git/refs/tags/{a.move_tag}", "-X", "PATCH"], {"sha": commit, "force": True})
        print(f"TAG {a.move_tag} -> {commit[:8]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
