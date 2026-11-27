#!/usr/bin/env python3
"""Stage the v3 release PRIVATELY on the Hugging Face hub (huggingface_hub only). Runs on a pod or locally.

  python3 stage_v3.py prime-main      --ckpt /srv/stack_w/prime   [--card F] [--charts DIR] [--only weights|card|all]
  python3 stage_v3.py prime-lora      --adapter DIR --merge-script F [--card F]
  python3 stage_v3.py prime-gguf-card-only --card F
  python3 stage_v3.py unbound-staging --ckpt /srv/stack_w/unbound [--card F] [--charts DIR] [--only ...]
  common: [--prime-name Prime] [--dry-run] [--receipts DIR] [--budget-gb 45] [--no-skip-same-as-public]
  python3 stage_v3.py --selftest       # offline, in-memory fake hub; needs huggingface_hub installed (classes only)

Writes ONLY to the four private names (Inkling-2-Mini-<P>, -<P>-LoRA, -<P>-GGUF, Inkling-2-Mini-Unbound-v3-staging).
For each repo: create_repo(private=True, exist_ok=True); re-read repo_info and ABORT unless private is True; one
commit with explicit CommitOperationAdd paths (no folder scan, so stray files in the checkpoint dir never go up;
files already on the hub with the same sha256 are skipped, so a re-run resumes); re-read privacy again; verify the
remote file list, sizes and every LFS sha256 (model_info(files_metadata=True)) against locally computed sha256
(small files: git blob id); write a JSON receipt (repo, commit sha, files, sha256s, private flag, timestamp).

unbound-staging uploads only what differs from the PUBLIC Unbound main (normally the 19.5 GB shard, README and the v3
charts; mtp.safetensors and the tokenizer are identical to v2.1) to keep private storage under budget. golive_v3.py
then copies exactly the staging repo's files onto Unbound main.

--dry-run is fully offline: no network call of any kind; it prints the plan (files, sizes, sha256 unless --no-hash).
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v3rel_common as C  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
DIST = REPO_ROOT / "package_v3" / "dist"
CHARTS = REPO_ROOT / "package_v3" / "charts"


# ---------------------------------------------------------------- plans: {path_in_repo: local Path}
def ckpt_files(ckpt: Path) -> dict[str, Path]:
    missing = [f for f in C.STOCK_LAYOUT if not (ckpt / f).is_file()]
    if missing:
        raise SystemExit(f"{ckpt}: missing checkpoint files {missing}")
    return {f: ckpt / f for f in C.STOCK_LAYOUT}


def card_files(card: Path, charts: Path | None, chart_stems: tuple[str, ...] | None) -> dict[str, Path]:
    if not card.is_file():
        raise SystemExit(f"card not found: {card} (render with package_v3/render.py --final first)")
    out = {"README.md": card}
    if charts is not None:
        pics = sorted(p for p in charts.iterdir() if p.suffix in (".png", ".svg") and not p.name.startswith("._"))
        if chart_stems is not None:
            pics = [p for p in pics if p.stem in chart_stems]
            want = {f"{s}{e}" for s in chart_stems for e in (".png", ".svg")}
            lack = sorted(want - {p.name for p in pics})
            if lack:
                raise SystemExit(f"charts missing in {charts}: {lack} (run scripts/charts/make_v3.py --final)")
        out.update({f"charts/{p.name}": p for p in pics})
    return out


def plan_for(a: argparse.Namespace, N: C.Names) -> tuple[str, str, dict[str, Path]]:
    """-> (repo, receipt part name, files)."""
    only = getattr(a, "only", "all")
    if a.cmd in ("prime-main", "unbound-staging"):
        repo = N.prime if a.cmd == "prime-main" else N.staging
        stems = C.PRIME_CHARTS if a.cmd == "prime-main" else None  # Unbound: all of charts/ (package_v3/README.md)
        default_card = DIST / ("Inkling-2-Mini-" + a.prime_name if a.cmd == "prime-main" else "Inkling-2-Mini-Unbound") / "README.md"
        files: dict[str, Path] = {}
        if only in ("weights", "all"):
            if not a.ckpt:
                raise SystemExit("--ckpt is required unless --only card")
            files.update(ckpt_files(Path(a.ckpt)))
        if only in ("card", "all"):
            files.update(card_files(Path(a.card or default_card), Path(a.charts or CHARTS), stems))
        return repo, f"{a.cmd}-{only}", files
    if a.cmd == "prime-lora":
        ad = Path(a.adapter)
        files = {f: ad / f for f in ("adapter_model.safetensors", "adapter_config.json")}
        for f, p in files.items():
            if not p.is_file():
                raise SystemExit(f"adapter file missing: {p}")
        ms = Path(a.merge_script)
        if not ms.is_file():
            raise SystemExit(f"merge script missing: {ms}")
        files[a.merge_script_name or ms.name] = ms
        files.update(card_files(Path(a.card or DIST / f"Inkling-2-Mini-{a.prime_name}-LoRA" / "README.md"), None, None))
        return N.prime_lora, a.cmd, files
    if a.cmd == "prime-gguf-card-only":
        card = Path(a.card or DIST / f"Inkling-2-Mini-{a.prime_name}-GGUF" / "README.md")
        return N.prime_gguf, a.cmd, card_files(card, None, None)
    raise SystemExit(f"unknown command {a.cmd}")


def card_lint(files: dict[str, Path]) -> list[str]:
    """A card must be final: no unrendered {{...}}, no TBD, no draft banner."""
    p = files.get("README.md")
    if p is None:
        return []
    t = p.read_text(encoding="utf-8", errors="replace")
    bad = [w for w in ("{{", "}}", "TBD", "DRAFT RENDER", "[MISSING") if w in t]
    return [f"README.md contains {w!r}" for w in bad]


# ---------------------------------------------------------------- run
def describe(repo: str, metas: dict[str, dict], skip: set[str]) -> None:
    tot = sum(m["size"] for p, m in metas.items() if p not in skip)
    print(f"PLAN {repo}: {len(metas) - len(skip)} files to upload, {tot / C.GB:.2f} GB "
          f"({len(skip)} skipped as identical)", flush=True)
    for p, m in sorted(metas.items()):
        tag = "skip" if p in skip else "add "
        print(f"  {tag} {p:52s} {m['size'] / C.GB:9.3f} GB  {m.get('sha256', '(not hashed)')}", flush=True)


def stage(a: argparse.Namespace, api: Any | None) -> int:
    N = C.names(a.prime_name)
    repo, part, files = plan_for(a, N)
    if repo not in N.private_allow:  # cannot happen through plan_for; kept as the hard rule
        raise SystemExit(f"ABORT: {repo} not in allow-list")
    lint = card_lint(files)
    if lint:
        raise SystemExit("ABORT: card not final: " + "; ".join(lint))
    metas = {p: C.local_meta(lp, hash_it=not a.no_hash) for p, lp in files.items()}
    receipts = Path(a.receipts)
    others = C.staged_bytes(receipts, exclude=repo)
    if a.dry_run:
        describe(repo, metas, set())
        if a.cmd == "unbound-staging" and not a.no_skip_same_as_public:
            print(f"  (live run also skips files identical to public {N.unbound} main: normally mtp/tokenizer/configs)")
        print(f"DRY RUN: no network calls made. Private allow-list: {list(N.private_allow)}. "
              f"Other private repos per receipts: {others / C.GB:.2f} GB of a {a.budget_gb} GB budget.", flush=True)
        return 0
    if a.no_hash:
        raise SystemExit("--no-hash is only for --dry-run (verification needs the sha256)")
    C.require_token()
    api = api or C.real_api()
    print(f"huggingface_hub {C.hub_version()} on {C.host()}", flush=True)
    C.ensure_private_repo(api, repo, N.private_allow)
    skip: set[str] = set()
    _, remote = C.remote_files(api, repo)
    skip |= {p for p, m in metas.items() if C.same_content(m, remote.get(p))}
    if a.cmd == "unbound-staging" and not a.no_skip_same_as_public:
        _, pub = C.remote_files(api, N.unbound)  # read only
        same_pub = {p for p, m in metas.items() if C.same_content(m, pub.get(p))}
        print(f"identical to public {N.unbound} main, not staged: {sorted(same_pub)}", flush=True)
        for p in same_pub:
            metas.pop(p)
            skip.discard(p)
    describe(repo, metas, skip)
    new_bytes = sum(m["size"] for p, m in metas.items() if p not in skip and p not in remote)
    have = sum(m["size"] for m in remote.values())
    if (others + have + new_bytes) / C.GB > a.budget_gb:
        raise SystemExit(f"ABORT: private storage would reach {(others + have + new_bytes) / C.GB:.1f} GB "
                         f"> --budget-gb {a.budget_gb} (receipts: {others / C.GB:.1f} GB elsewhere)")
    todo = [p for p in sorted(metas) if p not in skip]
    if todo:
        from huggingface_hub import CommitOperationAdd
        ops = [CommitOperationAdd(path_in_repo=p, path_or_fileobj=str(files[p])) for p in todo]
        C.assert_private(api, repo, N.private_allow)  # immediately before the bytes leave
        c = api.create_commit(repo, operations=ops, commit_message=f"v3 staging ({part}): {len(ops)} files")
        print(f"COMMITTED {repo} {getattr(c, 'oid', c)}", flush=True)
    C.assert_private(api, repo, N.private_allow)
    sha, remote = C.remote_files(api, repo)
    fails = C.verify_against(metas, remote, exact=False)
    status = "PASS" if not fails else "FAIL"
    for f in fails:
        print("FAIL", f, flush=True)
    C.write_receipt(C.receipt_path(repo, part, receipts), {
        "repo": repo, "part": part, "commit": sha, "private": C.is_private(api, repo), "timestamp": C.now(),
        "host": C.host(), "huggingface_hub": C.hub_version(), "verify": status, "failures": fails,
        "files": metas, "sources": {p: str(lp) for p, lp in files.items() if p in metas}})
    print(f"STAGE {status} {repo} @ {sha}", flush=True)
    return 0 if not fails else 1


# ---------------------------------------------------------------- selftest
def selftest() -> int:
    import contextlib
    import io
    os.environ["HF_TOKEN"] = "hf_SELFTEST_SENTINEL_never_printed"
    C.SHA_CACHE = Path(tempfile.mkdtemp()) / "cache.json"
    tmp = Path(tempfile.mkdtemp())
    ck = tmp / "ck"
    for f in C.STOCK_LAYOUT:
        p = ck / f
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes((f * 50).encode() if f not in C.LARGE_FILES else os.urandom(11 << 20))
    (ck / "README.md").write_text("stock card, must never be uploaded")
    (ck / "merge_report.json").write_text("{}")
    card = tmp / "README.md"
    card.write_text("---\nlicense: apache-2.0\n---\n# Prime\n")
    charts = tmp / "charts"
    charts.mkdir()
    for s in C.PRIME_CHARTS + ("history_v3",):
        for e in (".png", ".svg"):
            (charts / f"{s}{e}").write_bytes(s.encode())
    N = C.names("Prime")
    base = ["--receipts", str(tmp / "rc"), "--budget-gb", "45"]
    ap = parser()
    out = io.StringIO()
    ok = True

    def run(argv: list[str], api: Any) -> tuple[int, str]:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            try:
                rc = stage(ap.parse_args(argv), api)
            except SystemExit as e:
                print("EXIT", e)
                rc = 2
        out.write(buf.getvalue())
        return rc, buf.getvalue()

    def expect(cond: bool, msg: str) -> None:
        nonlocal ok
        print(("ok   " if cond else "FAIL ") + msg)
        ok &= cond

    # 1. normal: new repo created private, uploaded, verified, receipt written; stray files not uploaded
    api = C.FakeApi()
    rc, o = run(["prime-main", "--ckpt", str(ck), "--card", str(card), "--charts", str(charts)] + base, api)
    r = api.repos.get(N.prime)
    expect(rc == 0 and r is not None and r.private, "prime-main: created private, exit 0")
    expect(r is not None and "README.md" in r.files and "merge_report.json" not in r.files
           and "charts/history_v3.png" not in r.files and "charts/hero_v3.svg" in r.files,
           "only listed files uploaded (card, prime charts, stock layout)")
    rec = C.receipt_path(N.prime, "prime-main-all", tmp / "rc")
    expect(rec.is_file() and '"verify": "PASS"' in rec.read_text(), "receipt written with verify PASS")
    # 2. re-run: everything identical -> no new commit
    n = r.commits if r else -1
    rc, _ = run(["prime-main", "--ckpt", str(ck), "--card", str(card), "--charts", str(charts)] + base, api)
    expect(rc == 0 and api.repos[N.prime].commits == n, "re-run skips identical files (no commit)")
    # 3. existing PUBLIC repo -> abort before any upload
    api = C.FakeApi({N.prime_lora: C.FakeRepo(private=False)})
    ad = tmp / "ad"
    ad.mkdir()
    (ad / "adapter_model.safetensors").write_bytes(os.urandom(1 << 20))
    (ad / "adapter_config.json").write_text("{}")
    ms = tmp / "merge_adapter.py"
    ms.write_text("print(1)\n")
    rc, o = run(["prime-lora", "--adapter", str(ad), "--merge-script", str(ms), "--card", str(card)] + base, api)
    expect(rc != 0 and not any(c[0] == "create_commit" for c in api.calls), "public repo: aborted, no commit")
    # 4. hub returns a wrong sha256 -> verify FAIL, exit 1
    api = C.FakeApi(corrupt=lambda p: p == "model-00001-of-00001.safetensors")
    rc, o = run(["unbound-staging", "--ckpt", str(ck), "--card", str(card), "--charts", str(charts),
                 "--no-skip-same-as-public"] + base, api)
    expect(rc == 1 and "sha256 differs: model-00001" in o, "corrupted upload detected (sha256 mismatch)")
    # 5. unfinished card -> abort
    bad = tmp / "bad.md"
    bad.write_text("x {{files.lora_sha}} TBD")
    rc, o = run(["prime-gguf-card-only", "--card", str(bad)] + base, C.FakeApi())
    expect(rc != 0 and "card not final" in o, "card with placeholders/TBD refused")
    # 6. budget
    api = C.FakeApi()
    rc, o = run(["prime-main", "--ckpt", str(ck), "--only", "weights", "--receipts", str(tmp / "rc"),
                 "--budget-gb", "0.001"], api)
    expect(rc != 0 and "budget" in o and not any(c[0] == "create_commit" for c in api.calls), "budget guard")
    # 7. allow-list: a repo outside it cannot be made private-asserted
    try:
        C.ensure_private_repo(C.FakeApi(), N.unbound, N.private_allow)
        expect(False, "allow-list")
    except SystemExit:
        expect(True, "allow-list refuses the public Unbound repo")
    C.die_on_token_leak(out.getvalue() + "".join(p.read_text() for p in (tmp / "rc").glob("*.json")))
    expect(True, "token sentinel absent from all output and receipts")
    print("SELFTEST_OK" if ok else "SELFTEST_FAIL")
    return 0 if ok else 1


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    sub = ap.add_subparsers(dest="cmd")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--prime-name", default="Prime")
    common.add_argument("--dry-run", action="store_true")
    common.add_argument("--no-hash", action="store_true", help="dry run only: skip sha256 (fast)")
    common.add_argument("--receipts", default=str(C.RECEIPTS_DIR))
    common.add_argument("--budget-gb", type=float, default=45.0)
    common.add_argument("--card")
    for name in ("prime-main", "unbound-staging"):
        s = sub.add_parser(name, parents=[common])
        s.add_argument("--ckpt")
        s.add_argument("--charts")
        s.add_argument("--only", choices=("weights", "card", "all"), default="all")
        s.add_argument("--no-skip-same-as-public", action="store_true")
    s = sub.add_parser("prime-lora", parents=[common])
    s.add_argument("--adapter", required=True)
    s.add_argument("--merge-script", required=True)
    s.add_argument("--merge-script-name", help="name in the repo (default: the file's own name)")
    sub.add_parser("prime-gguf-card-only", parents=[common])
    return ap


def main() -> int:
    ap = parser()
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not a.cmd:
        ap.print_help()
        return 2
    for k in ("no_skip_same_as_public", "only"):
        if not hasattr(a, k):
            setattr(a, k, False if k != "only" else "all")
    return stage(a, None)


if __name__ == "__main__":
    sys.exit(main())
