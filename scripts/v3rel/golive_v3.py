#!/usr/bin/env python3
"""v3 GO-LIVE on the Hugging Face hub. Written ahead; run only on the owner's word.

  python3 golive_v3.py check [--gguf-dir D --prime-gguf-dir D --basalt-pack F ...]   # read-only; writes the plan file
  python3 golive_v3.py prime-public     --yes-owner-approved
  python3 golive_v3.py unbound-promote  --yes-owner-approved [--with-deletions] [--from-local CKPT --card F --charts D]
  python3 golive_v3.py gguf-upload      --yes-owner-approved --which unbound|prime --gguf-dir D --card F [--manifest-rows F]
  python3 golive_v3.py basalt-upload    --yes-owner-approved --basalt-pack F --card F --verify-log F
  python3 golive_v3.py tag              --yes-owner-approved [--tag v3.0]
  python3 golive_v3.py cleanup          [--yes-owner-approved --delete-staging]
  python3 golive_v3.py rollback         --yes-owner-approved --step prime-public|unbound-promote|gguf-unbound|gguf-prime|basalt|tag
  python3 golive_v3.py --selftest       # offline, in-memory fake hub

Every mutating step needs --yes-owner-approved AND the file ~/.inkling-abl/v3_golive_approved on the machine it runs
on (the owner creates it). Every mutating step recomputes its part of the plan and REFUSES unless it is identical to
what `check` saved in ~/.inkling-abl/v3_golive_plan.json (same parent commit, same file operations), so nothing
happens that `check` did not print. Each commit to an existing repo pins parent_commit to the commit `check` saw.

Deletions on public repos: none by default. `check --propose-delete PATTERN` (fnmatch, repeatable; e.g.
'charts/._*' for the macOS AppleDouble junk on Unbound main) lists candidate deletions in the plan; unbound-promote
applies them only with --with-deletions, and never deletes a path the staging repo carries.

Order: check -> prime-public -> unbound-promote -> tag (Unbound) -> gguf-upload x2 -> basalt-upload -> tag (again
for GGUF/Basalt, idempotent) -> cleanup --delete-staging. GGUF uploads go only to PUBLIC repos (private storage cap),
so Prime-GGUF must be flipped by prime-public first.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v3rel_common as C  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
SMALL_REPLACEABLE = ("README.md", "MANIFEST.tsv", "SHA256SUMS")
MODEL_SHARD = "model-00001-of-00001.safetensors"


# ================================================================ plan building (read-only)
def plan_prime_public(api: Any, N: C.Names, receipts: Path) -> list[dict]:
    out = []
    for repo in (N.prime, N.prime_lora, N.prime_gguf):
        rec = C.load_receipts(repo, receipts)
        sha, remote = C.remote_files(api, repo)
        fails = C.verify_against(rec, remote, exact=True) if rec else [f"no receipt for {repo}"]
        out.append({"repo": repo, "commit": sha, "private_now": C.is_private(api, repo),
                    "files": len(remote), "gb": round(sum(m["size"] or 0 for m in remote.values()) / C.GB, 3),
                    "verify_fail": fails, "readme_lint": readme_lint(api, repo, sha, remote)})
    return out


def readme_lint(api: Any, repo: str, rev: str, remote: dict) -> list[str]:
    if "README.md" not in remote:
        return ["no README.md"]
    t = Path(api.hf_hub_download(repo_id=repo, filename="README.md", revision=rev)).read_text(errors="replace")
    return [f"README contains {w!r}" for w in ("{{", "TBD", "DRAFT RENDER") if w in t]


def plan_unbound(api: Any, N: C.Names, receipts: Path, propose_delete: list[str]) -> dict:
    src_sha, src = C.remote_files(api, N.staging)
    dst_sha, dst = C.remote_files(api, N.unbound)
    rec = C.load_receipts(N.staging, receipts)
    stage_fail = C.verify_against(rec, src, exact=True) if rec else ["no staging receipt"]
    ops = []
    for p, m in sorted(src.items()):
        if p == ".gitattributes" or C.same_content({**m, "size": m["size"]}, dst.get(p)):
            continue
        ops.append({"op": "copy" if m["sha256"] else "add", "path": p, "size": m["size"],
                    "sha256": m["sha256"], "blob_id": m["blob_id"], "replaces": p in dst})
    dels = sorted(p for p in dst if C.match_any(p, propose_delete) and p not in src and p != ".gitattributes")
    kept = sorted(p for p in dst if p not in src and p not in dels)
    return {"src": N.staging, "src_commit": src_sha, "dst": N.unbound, "parent": dst_sha, "ops": ops,
            "deletions": dels, "kept": kept, "staging_verify_fail": stage_fail,
            "staging_private": C.is_private(api, N.staging),
            "warn": [] if any(o["path"] == MODEL_SHARD for o in ops) else [f"{MODEL_SHARD} not among the changes"]}


def gguf_local(which: str, gguf_dir: Path, prefix: str, N: C.Names) -> dict[str, Path]:
    types = C.GGUF_TYPES_UNBOUND if which == "unbound" else C.GGUF_TYPES_PRIME
    files = {f"{prefix}-{t}.gguf": gguf_dir / f"{prefix}-{t}.gguf" for t in types}
    if which == "prime":
        for extra in ("mmproj-Inkling-2-Mini-BF16.gguf", "mmproj-Inkling-2-Mini-Q8_0.gguf", "SHA256SUMS", "MANIFEST.tsv"):
            if (gguf_dir / extra).is_file():
                files[extra] = gguf_dir / extra
    lack = [str(p) for p in files.values() if not p.is_file()]
    if lack:
        raise SystemExit(f"GGUF files missing: {lack}")
    return files


def check_sums(files: dict[str, Path], metas: dict[str, dict], sums: Path) -> list[str]:
    if not sums.is_file():
        return [f"{sums} not found: GGUF sha256 not cross-checked against the build's SHA256SUMS"]
    want = dict(reversed(line.split()) for line in sums.read_text().splitlines() if line.strip())
    want = {k.lstrip("*"): v for k, v in want.items()}
    return [f"{p}: sha256 differs from SHA256SUMS" for p in files if p.endswith(".gguf") and p in want
            and metas[p]["sha256"] != want[p]]


def plan_upload(api: Any, repo: str, files: dict[str, Path], replaceable: tuple[str, ...]) -> dict:
    sha, remote = C.remote_files(api, repo)
    metas = {p: C.local_meta(lp) for p, lp in files.items()}
    adds, repl, same = [], [], []
    for p, m in sorted(metas.items()):
        if C.same_content(m, remote.get(p)):
            same.append(p)
        elif p in remote:
            repl.append(p)
        else:
            adds.append(p)
    bad = [p for p in repl if p not in replaceable]
    return {"repo": repo, "parent": sha, "private_now": C.is_private(api, repo), "adds": adds, "replaces": repl,
            "identical": same, "refused_replacements": bad,
            "files": {p: {k: v for k, v in metas[p].items()} for p in metas}, "sources": {p: str(v) for p, v in files.items()}}


def manifest_file(api: Any, repo: str, rows: Path, tmpdir: Path) -> Path:
    """Unbound-GGUF: remote MANIFEST.tsv + the v3 rows (idempotent: rows already present are not appended twice)."""
    cur = Path(api.hf_hub_download(repo_id=repo, filename="MANIFEST.tsv", revision="main")).read_text().rstrip("\n")
    have = set(cur.splitlines())
    new = [r for r in rows.read_text().splitlines() if r.strip() and r not in have]
    out = tmpdir / "MANIFEST.tsv"
    out.write_text(cur + "\n" + "".join(r + "\n" for r in new))
    return out


def plan_gguf(api: Any, N: C.Names, which: str, a: argparse.Namespace, tmpdir: Path) -> dict:
    gdir = Path(a.gguf_dir if which == "unbound" else a.prime_gguf_dir)
    repo = N.unbound_gguf if which == "unbound" else N.prime_gguf
    prefix = a.unbound_gguf_prefix if which == "unbound" else (a.prime_gguf_prefix or f"Inkling-2-Mini-{a.prime_name}")
    files = gguf_local(which, gdir, prefix, N)
    card = a.card_unbound_gguf if which == "unbound" else a.card_prime_gguf
    if card:
        files["README.md"] = Path(card)
    if which == "unbound":
        if not a.manifest_rows:
            raise SystemExit("--manifest-rows is required for the Unbound GGUF repo (v3 rows appended to MANIFEST.tsv)")
        files["MANIFEST.tsv"] = manifest_file(api, repo, Path(a.manifest_rows), tmpdir)
    sec = plan_upload(api, repo, files, SMALL_REPLACEABLE)
    sec["sum_check"] = check_sums(files, sec["files"], gdir / "SHA256SUMS")
    return sec


def plan_basalt(api: Any, N: C.Names, a: argparse.Namespace) -> dict:
    pack = Path(a.basalt_pack)
    files = {C.UNBOUND_BASALT_FILE: pack}
    if a.card_basalt:
        files["README.md"] = Path(a.card_basalt)
    sec = plan_upload(api, N.unbound_basalt, files, ("README.md",))
    sec["verify_log_ok"] = verify_log_ok(Path(a.verify_log), sec["files"][C.UNBOUND_BASALT_FILE]["sha256"]) \
        if a.verify_log else "no --verify-log given"
    return sec


def verify_log_ok(log: Path, sha: str) -> str:
    if not log.is_file():
        return f"{log} not found"
    t = log.read_text(errors="replace")
    if "RESULT: VERIFY PASS" not in t:
        return "verify log has no 'RESULT: VERIFY PASS'"
    if f"PACK_SHA256 {sha}" not in t:
        return "verify log is for a different pack (PACK_SHA256 line does not match)"
    return "ok"


def plan_tag(api: Any, N: C.Names, tag: str) -> list[dict]:
    out = []
    for repo in (N.unbound, N.unbound_gguf, N.unbound_basalt):
        sha, remote = C.remote_files(api, repo)
        tags = [t.name for t in api.list_repo_refs(repo).tags]
        out.append({"repo": repo, "tag": tag, "at": sha, "exists": tag in tags})
    return out


def plan_cleanup(api: Any, N: C.Names) -> dict:
    try:
        _, src = C.remote_files(api, N.staging)
    except Exception as e:  # noqa: BLE001 - absent repo
        return {"staging": N.staging, "exists": False, "note": type(e).__name__}
    _, dst = C.remote_files(api, N.unbound)
    missing = [p for p, m in src.items() if p != ".gitattributes" and not C.same_content(m, dst.get(p))]
    return {"staging": N.staging, "exists": True, "not_yet_on_unbound_main": missing}


# ================================================================ printing and plan file
def print_plan(plan: dict) -> None:
    print(f"=== GO-LIVE PLAN (read-only check, {plan['created']}, huggingface_hub {plan['hub']}, "
          f"cross-repo LFS copy supported: {plan['cross_repo_copy']})")
    for s in plan.get("prime_public", []):
        print(f"[prime-public] {s['repo']}: private={s['private_now']} -> PUBLIC; {s['files']} files {s['gb']} GB; "
              f"verify vs receipt: {'OK' if not s['verify_fail'] else s['verify_fail']}; card: {s['readme_lint'] or 'OK'}")
    u = plan.get("unbound_promote")
    if u:
        print(f"[unbound-promote] {u['src']}@{u['src_commit'][:12]} -> {u['dst']} main (parent {u['parent'][:12]}), ONE commit:")
        for o in u["ops"]:
            print(f"    {o['op']:4s} {'REPLACE' if o['replaces'] else 'NEW    '} {o['path']}  {o['size'] / C.GB:.3f} GB")
        print(f"    delete (only with --with-deletions): {u['deletions'] or 'none'}")
        print(f"    kept unchanged on main: {len(u['kept'])} files (e.g. {u['kept'][:6]})")
        print(f"    staging private={u['staging_private']}, verify vs receipt: {u['staging_verify_fail'] or 'OK'}; "
              f"warnings: {u['warn'] or 'none'}")
    for k in ("gguf_unbound", "gguf_prime", "basalt"):
        s = plan.get(k)
        if s:
            print(f"[{k}] {s['repo']} (parent {s['parent'][:12]}, private={s['private_now']}): add {s['adds']}; "
                  f"replace {s['replaces']}; identical {len(s['identical'])}; deletes: none")
            for p in s["adds"] + s["replaces"]:
                print(f"    {p}  {s['files'][p]['size'] / C.GB:.3f} GB  {s['files'][p]['sha256']}")
            if s["refused_replacements"]:
                print(f"    REFUSED (would overwrite existing public files): {s['refused_replacements']}")
            for w in s.get("sum_check", []):
                print("    WARN", w)
            if "verify_log_ok" in s:
                print(f"    basalt verify log: {s['verify_log_ok']}")
    for t in plan.get("tag", []):
        print(f"[tag] {t['repo']}: {t['tag']} at {t['at'][:12]} ({'EXISTS, skipped' if t['exists'] else 'new'})")
    c = plan.get("cleanup")
    if c:
        print(f"[cleanup] {c['staging']}: exists={c['exists']}; deleted only with --delete-staging; "
              f"blocked while not on Unbound main: {c.get('not_yet_on_unbound_main', [])}")


STABLE = {"prime_public": ("repo", "commit"), "unbound_promote": ("src_commit", "parent", "ops", "deletions"),
          "gguf_unbound": ("repo", "parent", "adds", "replaces", "files"), "gguf_prime": ("repo", "parent", "adds", "replaces", "files"),
          "basalt": ("repo", "parent", "adds", "replaces", "files"), "tag": ("repo", "tag", "at")}


def stable(key: str, sec: Any) -> str:
    keep = STABLE[key]
    pick = (lambda d: {k: d[k] for k in keep}) if keep else (lambda d: d)
    val = [pick(x) for x in sec] if isinstance(sec, list) else pick(sec)
    return json.dumps(val, sort_keys=True)


def require_same(key: str, fresh: Any, plan_path: Path) -> None:
    try:
        saved = json.loads(plan_path.read_text())[key]
    except (OSError, KeyError, ValueError):
        raise SystemExit(f"REFUSED: no '{key}' section in {plan_path}; run `check` (with the same inputs) first")
    if stable(key, saved) != stable(key, fresh):
        raise SystemExit(f"REFUSED: the hub or the local files changed since `check` for '{key}'; run `check` again "
                         "and read what it prints")
    print(f"PLAN MATCH {key}", flush=True)


# ================================================================ mutating steps
def applied_path(plan_path: Path) -> Path:
    return plan_path.with_name(plan_path.stem + ".applied.json")


def record_applied(plan_path: Path, key: str, sec: Any) -> None:
    """What a step actually changed, kept apart from the plan (a later `check` rewrites the plan); rollback reads it."""
    p = applied_path(plan_path)
    d = json.loads(p.read_text()) if p.is_file() else {}
    d[key] = {"at": C.now(), "section": sec}
    p.write_text(json.dumps(d, indent=1, sort_keys=True) + "\n")
    print(f"APPLIED LOG {p} [{key}]", flush=True)


def do_prime_public(api: Any, N: C.Names, a: argparse.Namespace) -> int:
    fresh = plan_prime_public(api, N, Path(a.receipts))
    require_same("prime_public", fresh, Path(a.plan))
    bad = [s for s in fresh if s["verify_fail"] or s["readme_lint"]]
    if bad:
        raise SystemExit(f"REFUSED: {[(s['repo'], s['verify_fail'], s['readme_lint']) for s in bad]}")
    for s in fresh:
        api.update_repo_settings(s["repo"], private=False)
        p = C.is_private(api, s["repo"])
        print(f"{'PUBLIC' if p is False else 'STILL PRIVATE?'} {s['repo']} (private={p})", flush=True)
    return 0


def local_source(path: str, a: argparse.Namespace) -> Path:
    if path == "README.md":
        return Path(a.card)
    if path.startswith("charts/"):
        return Path(a.charts) / path.split("/", 1)[1]
    return Path(a.from_local) / path


def unbound_ops(api: Any, u: dict, a: argparse.Namespace) -> list:
    from huggingface_hub import CommitOperationAdd, CommitOperationCopy, CommitOperationDelete
    cross = C.supports_cross_repo_copy() and not a.from_local
    if not cross and not a.from_local:
        raise SystemExit(f"huggingface_hub {C.hub_version()} has no cross-repo CommitOperationCopy; "
                         "pass --from-local CKPT --card F --charts D (upload from the pod)")
    ops = []
    for o in u["ops"]:
        if cross and o["op"] == "copy":
            ops.append(CommitOperationCopy(src_path_in_repo=o["path"], path_in_repo=o["path"], src_revision=u["src_commit"],
                                           src_repo_id=u["src"], src_repo_type="model"))
            continue
        lp = Path(api.hf_hub_download(repo_id=u["src"], filename=o["path"], revision=u["src_commit"])) if cross \
            else local_source(o["path"], a)
        m = C.local_meta(lp)
        if not C.same_content(m, {"size": o["size"], "sha256": o["sha256"], "blob_id": o["blob_id"]}):
            raise SystemExit(f"REFUSED: {lp} is not the staged {o['path']} (sha/blob id differs)")
        ops.append(CommitOperationAdd(path_in_repo=o["path"], path_or_fileobj=str(lp)))
    if a.with_deletions:
        ops += [CommitOperationDelete(path_in_repo=p) for p in u["deletions"]]
    elif u["deletions"]:
        print(f"NOT deleting (no --with-deletions): {u['deletions']}", flush=True)
    return ops


def do_unbound_promote(api: Any, N: C.Names, a: argparse.Namespace) -> int:
    if not Path(a.plan).is_file():
        raise SystemExit(f"REFUSED: {a.plan} missing; run `check` first")
    fresh = plan_unbound(api, N, Path(a.receipts), json.loads(Path(a.plan).read_text()).get("propose_delete", []))
    require_same("unbound_promote", fresh, Path(a.plan))
    if fresh["staging_verify_fail"] or not fresh["ops"]:
        raise SystemExit(f"REFUSED: staging check {fresh['staging_verify_fail']} / ops {len(fresh['ops'])}")
    _, before = C.remote_files(api, N.unbound)
    ops = unbound_ops(api, fresh, a)
    c = api.create_commit(N.unbound, operations=ops, parent_commit=fresh["parent"],
                          commit_message="Release v3: Unbound v2.1 + the Prime fix")
    print(f"COMMITTED {N.unbound} {getattr(c, 'commit_url', c)}", flush=True)
    record_applied(Path(a.plan), "unbound_promote", {**fresh, "with_deletions": bool(a.with_deletions)})
    sha, after = C.remote_files(api, N.unbound)
    want = {o["path"]: {"size": o["size"], "sha256": o["sha256"], "blob_id": o["blob_id"]} for o in fresh["ops"]}
    fails = C.verify_against(want, after, exact=False)
    fails += [f"kept file changed: {p}" for p in fresh["kept"] if after.get(p) != before.get(p)]
    gone = fresh["deletions"] if a.with_deletions else []
    fails += [f"deletion not applied: {p}" for p in gone if p in after]
    fails += [f"file vanished: {p}" for p in before if p not in after and p not in gone]
    print("PROMOTE", "PASS" if not fails else f"FAIL {fails}", sha, flush=True)
    return 1 if fails else 0


def do_upload(api: Any, sec: dict, a: argparse.Namespace, label: str) -> int:
    from huggingface_hub import CommitOperationAdd
    if sec["refused_replacements"]:
        raise SystemExit(f"REFUSED: would overwrite {sec['refused_replacements']}")
    if C.is_private(api, sec["repo"]) is not False:
        raise SystemExit(f"REFUSED: {sec['repo']} is not public; large uploads never go to private repos (storage cap). "
                         "Run prime-public first.")
    todo = sec["adds"] + sec["replaces"]
    record_applied(Path(a.plan), label, sec)  # before the first byte: a partial upload can be rolled back too
    big = [p for p in todo if sec["files"][p]["size"] >= 50 << 20]
    small = [p for p in todo if p not in big]
    for p in big:  # one commit per large file: a failure leaves a clean, resumable state
        c = api.create_commit(sec["repo"], operations=[CommitOperationAdd(path_in_repo=p, path_or_fileobj=sec["sources"][p])],
                              commit_message=f"v3: {p}")
        _, r = C.remote_files(api, sec["repo"])
        f = C.verify_against({p: sec["files"][p]}, r, exact=False)
        print(f"{'UPLOADED' if not f else 'FAIL'} {p} {getattr(c, 'oid', '')} {f}", flush=True)
        if f:
            return 1
    if small:
        api.create_commit(sec["repo"], operations=[CommitOperationAdd(path_in_repo=p, path_or_fileobj=sec["sources"][p])
                                                   for p in small], commit_message=f"v3: {label} card and manifest")
    _, r = C.remote_files(api, sec["repo"])
    fails = C.verify_against({p: sec["files"][p] for p in todo}, r, exact=False)
    print(f"{label.upper()} {'PASS' if not fails else 'FAIL ' + str(fails)}", flush=True)
    return 1 if fails else 0


def do_gguf(api: Any, N: C.Names, a: argparse.Namespace, tmpdir: Path) -> int:
    key = f"gguf_{a.which}"
    fresh = plan_gguf(api, N, a.which, a, tmpdir)
    require_same(key, fresh, Path(a.plan))
    if fresh["sum_check"] and not a.no_sums_ok:
        raise SystemExit(f"REFUSED: {fresh['sum_check']} (pass --no-sums-ok only if no SHA256SUMS exists by design)")
    return do_upload(api, fresh, a, key)


def do_basalt(api: Any, N: C.Names, a: argparse.Namespace) -> int:
    fresh = plan_basalt(api, N, a)
    require_same("basalt", fresh, Path(a.plan))
    if fresh["verify_log_ok"] != "ok":
        raise SystemExit(f"REFUSED: basalt verify log: {fresh['verify_log_ok']}")
    return do_upload(api, fresh, a, "basalt")


def do_tag(api: Any, N: C.Names, a: argparse.Namespace) -> int:
    fresh = plan_tag(api, N, a.tag)
    require_same("tag", fresh, Path(a.plan))
    staged = C.load_receipts(N.staging, Path(a.receipts)).get(MODEL_SHARD)
    for t in fresh:
        if t["exists"]:
            print(f"TAG EXISTS {t['repo']} {t['tag']} (left alone)")
            continue
        if t["repo"] == N.unbound:
            _, r = C.remote_files(api, N.unbound, t["at"])
            if not staged or not C.same_content(staged, r.get(MODEL_SHARD)):
                raise SystemExit("REFUSED: Unbound main does not carry the staged v3 shard; promote first")
        api.create_tag(t["repo"], tag=t["tag"], revision=t["at"], tag_message=f"Inkling-2-Mini-Unbound {t['tag']}")
        record_applied(Path(a.plan), f"tag:{t['repo']}", t)
        print(f"TAGGED {t['repo']} {t['tag']} @ {t['at'][:12]}", flush=True)
    return 0


def do_cleanup(api: Any, N: C.Names, a: argparse.Namespace) -> int:
    c = plan_cleanup(api, N)
    print(json.dumps(c, indent=1))
    if not c["exists"]:
        return 0
    if c["not_yet_on_unbound_main"]:
        raise SystemExit(f"REFUSED: staging files not on Unbound main yet: {c['not_yet_on_unbound_main']}")
    if not a.delete_staging:
        print(f"would delete {N.staging}; pass --yes-owner-approved --delete-staging to do it")
        return 0
    C.approval_gate(a.yes_owner_approved, Path(a.approval_file))
    if C.is_private(api, N.staging) is not True:
        raise SystemExit(f"REFUSED: {N.staging} is not private; it is not the repo this script made")
    api.delete_repo(N.staging)
    print(f"DELETED {N.staging}", flush=True)
    return 0


# ================================================================ rollback
def revert_paths(api: Any, repo: str, parent: str, paths: list[str], msg: str) -> int:
    """Restore each path to its state at `parent` (intra-repo LFS copy, small files re-added) or delete it if new."""
    from huggingface_hub import CommitOperationAdd, CommitOperationCopy, CommitOperationDelete
    _, old = C.remote_files(api, repo, parent)
    _, cur = C.remote_files(api, repo)
    ops = []
    for p in paths:
        if p in old and C.same_content(old[p], cur.get(p)):
            continue
        if p not in old:
            if p in cur:
                ops.append(CommitOperationDelete(path_in_repo=p))
        elif old[p]["sha256"]:
            ops.append(CommitOperationCopy(src_path_in_repo=p, path_in_repo=p, src_revision=parent))
        else:
            ops.append(CommitOperationAdd(path_in_repo=p, path_or_fileobj=api.hf_hub_download(repo_id=repo, filename=p,
                                                                                             revision=parent)))
    for o in ops:
        print(f"  rollback {type(o).__name__.replace('CommitOperation', '').lower():6s} {o.path_in_repo}")
    if ops:
        api.create_commit(repo, operations=ops, commit_message=msg)
    _, now_ = C.remote_files(api, repo)
    fails = [p for p in paths if (p in old) != (p in now_) or (p in old and not C.same_content(old[p], now_.get(p)))]
    print(f"ROLLBACK {repo} {'PASS' if not fails else 'FAIL ' + str(fails)}", flush=True)
    return 1 if fails else 0


def do_rollback(api: Any, N: C.Names, a: argparse.Namespace) -> int:
    ap_ = applied_path(Path(a.plan))
    applied = json.loads(ap_.read_text()) if ap_.is_file() else {}
    if a.step == "prime-public":
        for repo in (N.prime, N.prime_lora, N.prime_gguf):
            api.update_repo_settings(repo, private=True)
            print(f"private={C.is_private(api, repo)} {repo}")
        return 0
    if a.step == "tag":  # only tags this script created (recorded in the applied log)
        for k, v in applied.items():
            t = v["section"]
            if k.startswith("tag:") and t["tag"] == a.tag and a.tag in [x.name for x in api.list_repo_refs(t["repo"]).tags]:
                api.delete_tag(t["repo"], tag=a.tag)
                print(f"TAG DELETED {t['repo']} {a.tag}")
        return 0
    key = {"unbound-promote": "unbound_promote", "gguf-unbound": "gguf_unbound", "gguf-prime": "gguf_prime",
           "basalt": "basalt"}[a.step]
    if key not in applied:
        raise SystemExit(f"nothing to roll back: no '{key}' entry in {ap_}")
    s = applied[key]["section"]
    if key == "unbound_promote":
        paths = [o["path"] for o in s["ops"]] + (s["deletions"] if s.get("with_deletions") else [])
        return revert_paths(api, s["dst"], s["parent"], paths, "Roll back v3 promotion (restore v2.1 files)")
    return revert_paths(api, s["repo"], s["parent"], s["adds"] + s["replaces"], f"Roll back v3 {key}")


# ================================================================ main
def build_check(api: Any, N: C.Names, a: argparse.Namespace, tmpdir: Path) -> dict:
    plan: dict[str, Any] = {"created": C.now(), "hub": C.hub_version(), "cross_repo_copy": C.supports_cross_repo_copy(),
                            "host": C.host(), "propose_delete": a.propose_delete or []}
    plan["prime_public"] = plan_prime_public(api, N, Path(a.receipts))
    plan["unbound_promote"] = plan_unbound(api, N, Path(a.receipts), a.propose_delete or [])
    if a.gguf_dir:
        plan["gguf_unbound"] = plan_gguf(api, N, "unbound", a, tmpdir)
    if a.prime_gguf_dir:
        plan["gguf_prime"] = plan_gguf(api, N, "prime", a, tmpdir)
    if a.basalt_pack:
        plan["basalt"] = plan_basalt(api, N, a)
    plan["tag"] = plan_tag(api, N, a.tag)
    plan["cleanup"] = plan_cleanup(api, N)
    return plan


def run(a: argparse.Namespace, api: Any | None) -> int:
    N = C.names(a.prime_name)
    if a.cmd != "check" and not (a.cmd == "cleanup" and not a.delete_staging):
        C.approval_gate(a.yes_owner_approved, Path(a.approval_file))
    if api is None:
        C.require_token()
        api = C.real_api()
    tmpdir = Path(tempfile.mkdtemp())
    if a.cmd == "check":
        plan = build_check(api, N, a, tmpdir)
        print_plan(plan)
        Path(a.plan).parent.mkdir(parents=True, exist_ok=True)
        Path(a.plan).write_text(json.dumps(plan, indent=1, sort_keys=True) + "\n")
        print(f"PLAN SAVED {a.plan}")
        return 0
    # keep the plan's own inputs for the gguf/basalt re-plans
    steps = {"prime-public": lambda: do_prime_public(api, N, a), "unbound-promote": lambda: do_unbound_promote(api, N, a),
             "gguf-upload": lambda: do_gguf(api, N, a, tmpdir), "basalt-upload": lambda: do_basalt(api, N, a),
             "tag": lambda: do_tag(api, N, a), "cleanup": lambda: do_cleanup(api, N, a),
             "rollback": lambda: do_rollback(api, N, a)}
    return steps[a.cmd]()


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", nargs="?", choices=("check", "prime-public", "unbound-promote", "gguf-upload", "basalt-upload",
                                               "tag", "cleanup", "rollback"))
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--prime-name", default="Prime")
    ap.add_argument("--yes-owner-approved", action="store_true")
    ap.add_argument("--approval-file", default=str(C.APPROVAL_FILE))
    ap.add_argument("--plan", default=str(C.PLAN_FILE))
    ap.add_argument("--receipts", default=str(C.RECEIPTS_DIR))
    ap.add_argument("--tag", default="v3.0")
    ap.add_argument("--propose-delete", action="append", help="check: fnmatch pattern of Unbound-main deletions to propose")
    ap.add_argument("--with-deletions", action="store_true")
    ap.add_argument("--from-local", help="unbound-promote without cross-repo copy: the v3 checkpoint dir on the pod")
    ap.add_argument("--card", help="unbound-promote --from-local: the Unbound README.md")
    ap.add_argument("--charts", help="unbound-promote --from-local: package_v3/charts")
    ap.add_argument("--which", choices=("unbound", "prime"))
    ap.add_argument("--gguf-dir", help="dir with Inkling-2-Mini-Unbound-v3-<TYPE>.gguf (+ SHA256SUMS)")
    ap.add_argument("--prime-gguf-dir", help="dir with Inkling-2-Mini-<P>-<TYPE>.gguf, SHA256SUMS, MANIFEST.tsv")
    ap.add_argument("--unbound-gguf-prefix", default="Inkling-2-Mini-Unbound-v3")
    ap.add_argument("--prime-gguf-prefix")
    ap.add_argument("--manifest-rows", help="TSV rows for the v3 files, appended to the Unbound GGUF MANIFEST.tsv")
    ap.add_argument("--card-unbound-gguf")
    ap.add_argument("--card-prime-gguf")
    ap.add_argument("--no-sums-ok", action="store_true")
    ap.add_argument("--basalt-pack")
    ap.add_argument("--card-basalt")
    ap.add_argument("--verify-log", help="output of verify_basalt_v3.py for this pack")
    ap.add_argument("--delete-staging", action="store_true")
    ap.add_argument("--step", choices=("prime-public", "unbound-promote", "gguf-unbound", "gguf-prime", "basalt", "tag"))
    return ap


def normalize(a: argparse.Namespace) -> argparse.Namespace:
    """gguf-upload --which X --gguf-dir D --card F maps onto the check-time option names."""
    if a.cmd == "gguf-upload":
        if not a.which or not a.gguf_dir:
            raise SystemExit("gguf-upload needs --which and --gguf-dir")
        if a.which == "prime":
            a.prime_gguf_dir, a.card_prime_gguf = a.gguf_dir, a.card or a.card_prime_gguf
        else:
            a.card_unbound_gguf = a.card or a.card_unbound_gguf
    if a.cmd == "basalt-upload":
        a.card_basalt = a.card or a.card_basalt
        if not a.basalt_pack:
            raise SystemExit("basalt-upload needs --basalt-pack")
    if a.cmd == "rollback" and not a.step:
        raise SystemExit("rollback needs --step")
    return a


# ================================================================ selftest
def selftest() -> int:
    import contextlib
    import io
    from huggingface_hub import CommitOperationAdd  # noqa: F401  (real classes, fake hub)
    os.environ["HF_TOKEN"] = "hf_SELFTEST_SENTINEL_never_printed"
    tmp = Path(tempfile.mkdtemp())
    C.SHA_CACHE = tmp / "cache.json"
    N = C.names("Prime")
    ok = True
    out = io.StringIO()

    def mk(name: str, data: bytes) -> Path:
        p = tmp / "src" / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
        return p

    def meta(p: Path) -> dict:
        m = C.local_meta(p)
        m["_src"] = str(p)
        return m

    def expect(cond: bool, msg: str) -> None:
        nonlocal ok
        print(("ok   " if cond else "FAIL ") + msg)
        ok &= cond

    # fake hub: public Unbound v2.1 main (+ AppleDouble junk), private staging and Prime repos with receipts
    v21 = {MODEL_SHARD: meta(mk("v21.st", os.urandom(11 << 20))), "mtp.safetensors": meta(mk("mtp", os.urandom(11 << 20))),
           "README.md": meta(mk("v21.md", b"v2.1 card")), "lora_v2.1/peft-r16/adapter_config.json": meta(mk("lc", b"{}")),
           "charts/._banner_v21.png": meta(mk("junk", b"x" * 163)), "charts/hero_v21.png": meta(mk("h21", b"png21"))}
    v3 = {MODEL_SHARD: meta(mk("v3.st", os.urandom(11 << 20))), "README.md": meta(mk("v3.md", b"v3 card")),
          "charts/hero_v3.png": meta(mk("h3", b"png3"))}
    prime = {MODEL_SHARD: meta(mk("p.st", os.urandom(11 << 20))), "README.md": meta(mk("p.md", b"prime card"))}
    repos = {N.unbound: C.FakeRepo(False, dict(v21)), N.staging: C.FakeRepo(True, dict(v3)),
             N.prime: C.FakeRepo(True, dict(prime)), N.prime_lora: C.FakeRepo(True, {"README.md": prime["README.md"]}),
             N.prime_gguf: C.FakeRepo(True, {"README.md": prime["README.md"]}),
             N.unbound_gguf: C.FakeRepo(False, {"MANIFEST.tsv": meta(mk("man", b"file\tsha\n"))}),
             N.unbound_basalt: C.FakeRepo(False, {})}
    api = C.FakeApi(repos)
    for r in repos:
        api.snapshot(r)
    api.repos[N.unbound].tagmap["v2.1"] = api.repo_info(N.unbound).sha
    rc_dir = tmp / "rc"
    for repo, files in ((N.staging, v3), (N.prime, prime), (N.prime_lora, {"README.md": prime["README.md"]}),
                        (N.prime_gguf, {"README.md": prime["README.md"]})):
        C.write_receipt(C.receipt_path(repo, "t", rc_dir), {"repo": repo, "private": True, "timestamp": C.now(),
                                                            "files": {k: {x: y for x, y in v.items() if x != "_src"}
                                                                      for k, v in files.items()}})
    gdir = tmp / "gguf"
    gdir.mkdir()
    for t in C.GGUF_TYPES_UNBOUND:
        (gdir / f"Inkling-2-Mini-Unbound-v3-{t}.gguf").write_bytes(os.urandom(1 << 16))
    rows = mk("rows.tsv", b"Inkling-2-Mini-Unbound-v3-Q8_0.gguf\tabc\n")
    approval = tmp / "approved"
    base = ["--plan", str(tmp / "plan.json"), "--receipts", str(rc_dir), "--approval-file", str(approval)]
    ap = parser()

    def go(argv: list[str]) -> tuple[int, str]:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            try:
                rc = run(normalize(ap.parse_args(argv + base)), api)
            except SystemExit as e:
                print("EXIT", e)
                rc = 2
        out.write(buf.getvalue())
        return rc, buf.getvalue()

    rc, o = go(["check", "--propose-delete", "charts/._*", "--gguf-dir", str(gdir), "--manifest-rows", str(rows)])
    expect(rc == 0 and "copy REPLACE model-00001" in o and "charts/._banner_v21.png" in o, "check prints copy ops and proposed deletion")
    expect(all(c[0] not in ("create_commit", "update_repo_settings", "create_tag", "delete_repo") for c in api.calls),
           "check made no mutating call")
    rc, o = go(["prime-public"])
    expect(rc != 0 and "--yes-owner-approved" in o, "refused without --yes-owner-approved")
    rc, o = go(["prime-public", "--yes-owner-approved"])
    expect(rc != 0 and "does not exist" in o, "refused without the approval file")
    approval.write_text("ok")
    rc, o = go(["gguf-upload", "--yes-owner-approved", "--which", "unbound", "--gguf-dir", str(gdir),
                "--manifest-rows", str(rows), "--no-sums-ok"])
    expect(rc == 0 and "GGUF_UNBOUND PASS" in o, "gguf-upload to public Unbound-GGUF (plan matched)")
    rc, o = go(["prime-public", "--yes-owner-approved"])
    expect(rc == 0 and api.repos[N.prime].private is False, "prime-public flips Prime repos")
    api.repos[N.unbound].files["README.md"] = meta(mk("edit.md", b"someone edited main"))  # main moved after check
    api.repos[N.unbound].commits += 1
    api.snapshot(N.unbound)
    rc, o = go(["unbound-promote", "--yes-owner-approved"])
    expect(rc != 0 and "changed since `check`" in o, "promote refused when main moved after check")
    go(["check", "--propose-delete", "charts/._*"])
    rc, o = go(["unbound-promote", "--yes-owner-approved"])
    f = api.repos[N.unbound].files
    expect(rc == 0 and f[MODEL_SHARD]["sha256"] == v3[MODEL_SHARD]["sha256"] and "lora_v2.1/peft-r16/adapter_config.json" in f
           and "charts/._banner_v21.png" in f and "charts/hero_v3.png" in f, "promote: one commit, v2.1 LoRA kept, no deletion without flag")
    rc, o = go(["check"])
    rc, o = go(["tag", "--yes-owner-approved"])
    expect(rc == 0 and "v3.0" in api.repos[N.unbound].tags, "tag v3.0 created after promote")
    rc, o = go(["cleanup"])
    expect(rc == 0 and N.staging in api.repos, "cleanup without --delete-staging deletes nothing")
    rc, o = go(["rollback", "--yes-owner-approved", "--step", "unbound-promote"])
    expect(rc == 0 and api.repos[N.unbound].files[MODEL_SHARD]["sha256"] == v21[MODEL_SHARD]["sha256"]
           and "charts/hero_v3.png" not in api.repos[N.unbound].files, "rollback restores v2.1 shard and removes v3 additions")
    rc, o = go(["cleanup", "--yes-owner-approved", "--delete-staging"])
    expect(rc != 0 and N.staging in api.repos, "cleanup refused once main no longer carries the staged files")
    C.die_on_token_leak(out.getvalue() + (tmp / "plan.json").read_text())
    expect(True, "token sentinel absent from output and plan")
    print("SELFTEST_OK" if ok else "SELFTEST_FAIL")
    return 0 if ok else 1


def main() -> int:
    ap = parser()
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not a.cmd:
        ap.print_help()
        return 2
    return run(normalize(a), None)


if __name__ == "__main__":
    sys.exit(main())
