"""Hash manifest proving a baked checkpoint differs from the original only in the edited tensors.

  python3 manifest.py <orig_dir> <baked_dir> <out_prefix> [--expect bake_log.json] [--workers N]

One sequential read per file (orig and baked files hashed in parallel processes): file-level sha256 for every
file, and for every safetensors file the sha256 of each tensor's byte range plus the header bytes.
Writes <out_prefix>.json (MANIFEST.json) and <out_prefix>.md (MANIFEST.md).
Exit 0 only if: same file list (README.md may differ), headers byte-identical, and the set of differing tensors
equals the expected edited set (from the bake log when given).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys
import time
from concurrent.futures import ProcessPoolExecutor

CHUNK = 1 << 24
SKIP = {".cache"}
README_OK = {"README.md"}  # the model card replaces the base README
# files the release adds next to the model: the card's charts and the edit/reproduction kit (lead, 2026-11-21)
EXTRA_OK = ("charts/", "abliteration/")


def list_files(d: str) -> list[str]:
    out = []
    for root, dirs, files in os.walk(d):
        dirs[:] = sorted(x for x in dirs if x not in SKIP)
        out += [os.path.normpath(os.path.relpath(os.path.join(root, f), d)) for f in sorted(files)]
    return sorted(out)


def hash_file(path: str) -> dict:
    """File sha256; for .safetensors also header sha256 and per-tensor sha256, in one sequential pass."""
    fh = hashlib.sha256()
    res: dict = {"size": os.path.getsize(path)}
    with open(path, "rb") as f:
        if not path.endswith(".safetensors"):
            for c in iter(lambda: f.read(CHUNK), b""):
                fh.update(c)
            res["sha256"] = fh.hexdigest()
            return res
        raw_n = f.read(8)
        n = struct.unpack("<Q", raw_n)[0]
        raw_h = f.read(n)
        fh.update(raw_n), fh.update(raw_h)
        header = json.loads(raw_h)
        res["header_sha256"] = hashlib.sha256(raw_n + raw_h).hexdigest()
        metas = sorted(((k, v) for k, v in header.items() if k != "__metadata__"), key=lambda kv: kv[1]["data_offsets"][0])
        tensors, pos = {}, 0
        for name, m in metas:
            b, e = m["data_offsets"]
            if b != pos:  # padding / gap between tensors: hash into the file digest only
                gap = f.read(b - pos)
                fh.update(gap)
            th = hashlib.sha256()
            left = e - b
            while left:
                c = f.read(min(CHUNK, left))
                th.update(c), fh.update(c)
                left -= len(c)
            pos = e
            tensors[name] = {"dtype": m["dtype"], "shape": m["shape"], "sha256": th.hexdigest()}
        for c in iter(lambda: f.read(CHUNK), b""):  # trailing bytes, if any
            fh.update(c)
        res["sha256"] = fh.hexdigest()
        res["tensors"] = tensors
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("orig"), ap.add_argument("baked"), ap.add_argument("out_prefix")
    ap.add_argument("--expect", help="bake log json; its patched tensors are the expected edited set")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    t0 = time.time()
    fo, fb_all = list_files(a.orig), list_files(a.baked)
    selfref = {"abliteration/MANIFEST.json", "abliteration/MANIFEST.md"}  # copies of this manifest: cannot hash itself
    added = [f for f in fb_all if f.startswith(EXTRA_OK) and f not in fo and f not in selfref]
    fb = [f for f in fb_all if f not in added and f not in selfref]
    jobs = {("orig", f): os.path.join(a.orig, f) for f in fo} | {("baked", f): os.path.join(a.baked, f) for f in fb}
    jobs |= {("added", f): os.path.join(a.baked, f) for f in added}
    with ProcessPoolExecutor(a.workers) as ex:
        futs = {k: ex.submit(hash_file, p) for k, p in jobs.items()}
        H = {k: v.result() for k, v in futs.items()}
    added_files = [{"file": f, "size": H[("added", f)]["size"], "sha256": H[("added", f)]["sha256"]} for f in added]
    files, tensors, problems = [], [], []
    if fo != fb:
        problems.append(f"file lists differ: only orig {sorted(set(fo) - set(fb))}, only baked {sorted(set(fb) - set(fo))}")
    for f in sorted(set(fo) | set(fb)):
        o, b = H.get(("orig", f)), H.get(("baked", f))
        row = {"file": f, "size": (b or o)["size"], "sha256_orig": o and o["sha256"], "sha256_baked": b and b["sha256"]}
        row["identical"] = bool(o and b and o["sha256"] == b["sha256"])
        files.append(row)
        if o and b and "tensors" in o:
            if o["header_sha256"] != b["header_sha256"]:
                problems.append(f"{f}: safetensors header changed")
            for name in sorted(set(o["tensors"]) | set(b["tensors"])):
                to, tb = o["tensors"].get(name), b["tensors"].get(name)
                tensors.append({"file": f, "tensor": name, "dtype": (tb or to)["dtype"], "shape": (tb or to)["shape"],
                                "sha256_orig": to and to["sha256"], "sha256_baked": tb and tb["sha256"],
                                "differs": not (to and tb and to["sha256"] == tb["sha256"])})
        elif not row["identical"] and f not in README_OK:
            problems.append(f"{f}: non-tensor file differs")
    differ = sorted(t["tensor"] for t in tensors if t["differs"])
    expected = None
    if a.expect:
        log = json.load(open(a.expect))
        expected = sorted(p["tensor"] for p in log["patched"] if p["changed"])
        if differ != expected:
            problems.append(f"differing tensors {differ} != expected {expected}")
    nonbf16 = [t["tensor"] for t in tensors if t["differs"] and t["dtype"] != "BF16"]
    if nonbf16:
        problems.append(f"non-BF16 tensors differ: {nonbf16}")
    man = {"orig_dir": os.path.abspath(a.orig), "baked_dir": os.path.abspath(a.baked),
           "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "seconds": round(time.time() - t0, 1),
           "n_files": len(files), "n_tensors": len(tensors), "n_differing_tensors": len(differ),
           "differing_tensors": differ, "expected_differing": expected, "ok": not problems, "problems": problems,
           "files": files, "added_files": added_files, "tensors": tensors}
    json.dump(man, open(a.out_prefix + ".json", "w"), indent=1)
    write_md(man, a.out_prefix + ".md")
    print(json.dumps({k: man[k] for k in ("ok", "n_files", "n_tensors", "n_differing_tensors", "seconds", "problems")}))
    sys.exit(0 if man["ok"] else 5)


def write_md(m: dict, path: str) -> None:
    L = ["# Hash manifest", "",
         f"Original: `{m['orig_dir']}` · Baked: `{m['baked_dir']}` · {m['time']}", "",
         f"**Verdict: {'PASS' if m['ok'] else 'FAIL'}.** {m['n_tensors']} tensors hashed (sha256 of each tensor's bytes); "
         f"{m['n_differing_tensors']} differ, {m['n_tensors'] - m['n_differing_tensors']} are byte-identical. "
         f"Safetensors headers (names, order, dtypes, shapes, offsets) are byte-identical.", ""]
    if m["problems"]:
        L += ["## Problems", ""] + [f"- {p}" for p in m["problems"]] + [""]
    L += ["## Edited tensors", "", "| tensor | dtype | shape | sha256 original | sha256 baked |", "|---|---|---|---|---|"]
    L += [f"| `{t['tensor']}` | {t['dtype']} | {t['shape']} | `{t['sha256_orig'][:16]}…` | `{t['sha256_baked'][:16]}…` |"
          for t in m["tensors"] if t["differs"]]
    L += ["", "## Files", "", "| file | bytes | sha256 original | sha256 baked | same |", "|---|---|---|---|---|"]
    L += [f"| `{f['file']}` | {f['size']} | `{(f['sha256_orig'] or '-')[:16]}…` | `{(f['sha256_baked'] or '-')[:16]}…` | {'yes' if f['identical'] else 'no'} |"
          for f in m["files"]]
    if m.get("added_files"):
        L += ["", "## Files added by the release (not in the original)", "", "| file | bytes | sha256 |", "|---|---|---|"]
        L += [f"| `{f['file']}` | {f['size']} | `{f['sha256'][:16]}…` |" for f in m["added_files"]]
    L += ["", "Full per-tensor hashes for all tensors (identical and edited) are in MANIFEST.json."]
    open(path, "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
