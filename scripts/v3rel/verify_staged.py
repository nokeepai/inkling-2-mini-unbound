#!/usr/bin/env python3
"""Read-only check of the PRIVATE v3 staging repos. Anyone with read access to the nokeepai org can run it.

  python3 verify_staged.py [--receipts DIR] [--prime-name Prime]
        [--local nokeepai/Inkling-2-Mini-Prime=/srv/stack_w/prime ...]   # dirs to diff the small files against
        [--card-dir package_v3/dist]                                     # rendered cards to diff README.md against
  python3 verify_staged.py --selftest                                      # offline, fake hub

For each of Inkling-2-Mini-<P>, -<P>-LoRA, -<P>-GGUF and Inkling-2-Mini-Unbound-v3-staging:
  1. repo_info: private is True (anything else is a FAIL);
  2. the file list equals the receipts' (stage_v3.py), sizes equal, every LFS sha256 and small-file git blob id equal;
  3. README.md has no unrendered {{...}}, no "TBD", no draft banner;
  4. downloads ONLY the small files (configs, tokenizer, chat template, README, adapter_config, scripts; nothing over
     50 MB) into a temp dir and byte-compares them with --local dirs / --card-dir when given.
Makes no write call. Exit 0 only if every check passes.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v3rel_common as C  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
SMALL = 50 << 20


def card_folder(repo: str, N: C.Names) -> str:
    return "Inkling-2-Mini-Unbound" if repo == N.staging else repo.split("/", 1)[1]


def check_repo(api: Any, repo: str, N: C.Names, receipts: Path, local: dict[str, Path], card_dir: Path | None,
               tmp: Path, charts_dir: Path | None = None) -> list[str]:
    fails: list[str] = []
    p = C.is_private(api, repo)
    print(f"[{repo}] private={p}")
    if p is not True:
        fails.append(f"{repo}: private={p!r}")
    rec = C.load_receipts(repo, receipts)
    sha, remote = C.remote_files(api, repo)
    if not rec:
        fails.append(f"{repo}: no receipt in {receipts}")
    fails += [f"{repo}: {f}" for f in C.verify_against(rec, remote, exact=True)]
    print(f"  {len(remote)} files on hub @ {sha[:12]}, {len(rec)} in receipts, "
          f"{sum(m['size'] or 0 for m in remote.values()) / C.GB:.2f} GB")
    for path, m in sorted(remote.items()):
        if path == ".gitattributes" or (m["size"] or 0) >= SMALL:
            continue
        got = Path(api.hf_hub_download(repo_id=repo, filename=path, revision=sha, local_dir=str(tmp / repo.replace("/", "__"))))
        if path == "README.md":
            t = got.read_text(errors="replace")
            fails += [f"{repo}: README.md contains {w!r}" for w in ("{{", "}}", "TBD", "DRAFT RENDER", "[MISSING") if w in t]
            ref = card_dir / card_folder(repo, N) / "README.md" if card_dir else None
        elif path.startswith("charts/"):
            ref = charts_dir / path.split("/", 1)[1] if charts_dir else None
        else:
            ref = local[repo] / path if repo in local else None
        if ref is not None:
            if not ref.is_file():
                print(f"  NOTE no local copy to diff: {ref}")
            elif ref.read_bytes() != got.read_bytes():
                fails.append(f"{repo}: {path} differs from {ref}")
            else:
                print(f"  same as local: {path}")
    return fails


def run(a: argparse.Namespace, api: Any | None) -> int:
    N = C.names(a.prime_name)
    if api is None:
        C.require_token()
        api = C.real_api()
    local = {}
    for spec in a.local or []:
        repo, _, d = spec.partition("=")
        local[repo] = Path(d)
    tmp = Path(tempfile.mkdtemp())
    fails: list[str] = []
    for repo in N.private_allow:
        try:
            fails += check_repo(api, repo, N, Path(a.receipts), local, Path(a.card_dir) if a.card_dir else None, tmp,
                                Path(a.charts_dir) if a.charts_dir else None)
        except Exception as e:  # noqa: BLE001 - a missing repo is a failure, keep checking the others
            fails.append(f"{repo}: {type(e).__name__}: {str(e)[:200]}")
    for f in fails:
        print("FAIL", f)
    print("VERIFY_STAGED", "PASS" if not fails else f"FAIL ({len(fails)})")
    return 0 if not fails else 1


def selftest() -> int:
    os.environ["HF_TOKEN"] = "hf_SELFTEST_SENTINEL_never_printed"
    tmp = Path(tempfile.mkdtemp())
    C.SHA_CACHE = tmp / "cache.json"
    N = C.names("Prime")
    d = tmp / "ck"
    d.mkdir()
    (d / "config.json").write_text('{"a": 1}')
    (d / "model-00001-of-00001.safetensors").write_bytes(os.urandom(11 << 20))
    card = tmp / "dist" / "Inkling-2-Mini-Prime"
    card.mkdir(parents=True)
    (card / "README.md").write_text("# Prime card\n")
    files = {"config.json": d / "config.json", "model-00001-of-00001.safetensors": d / "model-00001-of-00001.safetensors",
             "README.md": card / "README.md"}
    metas = {p: {**C.local_meta(lp), "_src": str(lp)} for p, lp in files.items()}
    repos = {r: C.FakeRepo(True, {"README.md": metas["README.md"]}) for r in N.private_allow}
    repos[N.prime] = C.FakeRepo(True, dict(metas))
    rc_dir = tmp / "rc"
    for r, fr in repos.items():
        C.write_receipt(C.receipt_path(r, "t", rc_dir), {"repo": r, "private": True, "timestamp": C.now(),
                                                         "files": {k: {x: y for x, y in v.items() if x != "_src"}
                                                                   for k, v in fr.files.items()}})
    api = C.FakeApi(repos)
    ap = parser()
    base = ["--receipts", str(rc_dir), "--local", f"{N.prime}={d}", "--card-dir", str(tmp / "dist")]
    ok = True

    def go(extra: list[str]) -> tuple[int, str]:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = run(ap.parse_args(base + extra), api)
        return rc, buf.getvalue()

    rc, o = go([])
    # the Prime-LoRA/-GGUF/staging READMEs are the Prime card here, so their card diff fails; Prime itself must pass
    ok &= "same as local: config.json" in o and not any(f.startswith(f"FAIL {N.prime}:") for f in o.splitlines())
    print(("ok   " if ok else "FAIL ") + "clean Prime repo passes (files, sha256, small-file diff, card diff)")
    api.repos[N.prime].private = False
    rc, o = go([])
    c2 = rc == 1 and f"FAIL {N.prime}: private=False" in o
    print(("ok   " if c2 else "FAIL ") + "public repo detected")
    api.repos[N.prime].private = True
    api.repos[N.prime].files["extra.bin"] = {"size": 3, "sha256": None, "blob_id": "x",
                                              "_src": str(d / "config.json")}
    rc, o = go([])
    c3 = rc == 1 and "unexpected file on hub: extra.bin" in o
    print(("ok   " if c3 else "FAIL ") + "unexpected remote file detected")
    C.die_on_token_leak(o)
    ok = ok and c2 and c3
    print("SELFTEST_OK" if ok else "SELFTEST_FAIL")
    return 0 if ok else 1


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--prime-name", default="Prime")
    ap.add_argument("--receipts", default=str(C.RECEIPTS_DIR))
    ap.add_argument("--local", action="append", help="REPO=DIR: diff that repo's small files against DIR")
    ap.add_argument("--card-dir", default=str(REPO_ROOT / "package_v3" / "dist"))
    ap.add_argument("--charts-dir", default=str(REPO_ROOT / "package_v3" / "charts"))
    return ap


def main() -> int:
    a = parser().parse_args()
    return selftest() if a.selftest else run(a, None)


if __name__ == "__main__":
    sys.exit(main())
