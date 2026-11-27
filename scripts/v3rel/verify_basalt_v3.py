#!/usr/bin/env python3
"""Verify a v3 Basalt pack: every byte is either the official pack's or what Basalt v0.5.5's converter makes from
the merged v3 checkpoint. Replaces scripts/v11b/verify_basalt_v2.py for v3, whose expert codes changed.

  python3 verify_basalt_v3.py OURS.basalt OFFICIAL.basalt MERGED_CKPT_DIR --basalt-src /root/b/src
        [--stock-checkpoint STOCK_DIR] [--workers 24] [--allow-changed REGEX] [--official-sha SHA | --skip-official-sha]
  python3 verify_basalt_v3.py --selftest     # offline: the classifier and the meta diff on synthetic inputs

OFFICIAL.basalt must be the Basalt 0.5.5 pack the v1/v2/v2.1 packs were checked against: jesdga/Inkling-2-Mini-Basalt
`Inkling-2-Mini-NVFP4-DQ8.basalt` (formerly `-q8`, same bytes) at revision 6c9fbf0730a6d2e8813a1b262c2ab3548cf0239e,
sha256 f7b4e5f8e17681c4735a7dfc51ffd8e805c44184fad4e1778b13187d79dc3787. It is no longer on that repo's main
(replaced 2026-11-23 by the 0.5.6 `-MX8` pack, a different layout), so download it by revision.

Why a new check. v2's verifier required the experts section to be byte-identical to the official pack and only the
5 attention projections to differ. v3 changes the routed experts (NVFP4 codes in layers 0-19, MXFP8 values in 20-23,
re-quantised by merge_adapter.py; scales kept) and attn_rel (wr_du) as well, so that check fails by design. This one
recomputes every weight with the converter's own transform (checkpoint_inkling2.py at the pinned Basalt commit:
NVFP4 codes/scales repacked as stored + FP32 tail amax/denominator; MXFP8 -> ggml reference Q8_0; BF16 projections
-> Q8_0; everything else BF16/F32 as stored) and classifies each tensor / expert:
  identical   pack == official == recomputed-from-merged
  changed     pack == recomputed-from-merged != official   (gguf: only tensors matching --allow-changed)
  FAIL        pack != recomputed-from-merged, or a changed gguf tensor outside --allow-changed
Checks, in order (any FAIL -> exit 1, last line "RESULT: VERIFY FAIL"):
  1. every section SHA-256 of our pack matches its bytes; official file sha256 equals the known one;
  2. same section names; tokenizer/* and experts.layout byte-identical to the official (header digests);
     experts.layout also equals the layout the converter derives from the merged checkpoint;
  3. experts: per layer and expert, the class above (counts per layer); with --stock-checkpoint also that the
     official blob == converter(stock) and changed == (merged expert bytes != stock expert bytes);
  4. gguf: tensor table (type, shape, bytes, offset) identical to the official; every tensor classified as above;
     independent cross-check of the changed Q8_0 tensors with this file's own numpy ggml-reference Q8_0 and the
     RMS rounding error vs the BF16 source;
  5. vision/mmproj: all tensors byte-identical, metadata differs only in general.name;
  6. meta: identical after dropping the declared fields (name, sources, tensors [checked in 4], checkpoint.dir,
     gguf_kv.general.name); hf_config / checkpoint.config_sha256 may differ only if they equal the merged config.json.
Prints "PACK_SHA256 <sha>" (golive_v3.py basalt-upload matches it) and "RESULT: VERIFY PASS".
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

OFFICIAL_SHA = "f7b4e5f8e17681c4735a7dfc51ffd8e805c44184fad4e1778b13187d79dc3787"
ALLOW_DEFAULT = r"^blk\.\d+\.attn_(q|k|v|output|gate|rel)\.weight$"  # the 6 LoRA-targeted attention tensors per layer
DECLARED_META = ("name", "sources", "tensors")
fails: list[str] = []
B: dict[str, Any] = {}  # basalt modules, loaded after --basalt-src is known (inherited by forked workers)


def check(ok: bool, msg: str) -> None:
    print(("PASS " if ok else "FAIL ") + msg, flush=True)
    if not ok:
        fails.append(msg)


def load_basalt(src: Path) -> None:
    conv = src / "tools" / "convert"
    for p in (conv, src / "tools"):
        sys.path.insert(0, str(p))
    import checkpoint as Cb  # noqa: E402
    import checkpoint_inkling2 as K  # noqa: E402
    import model_sections as S  # noqa: E402
    from gguf_reader import GGUFFile  # noqa: E402
    from model_file import ModelFile  # noqa: E402
    B.update(C=Cb, K=K, S=S, GGUFFile=GGUFFile, ModelFile=ModelFile)


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(64 << 20), b""):
            h.update(b)
    return h.hexdigest()


def read_at(path: str, off: int, n: int) -> bytes:
    with open(path, "rb") as f:
        f.seek(off)
        return f.read(n)


# ---------------------------------------------------------------- classification (pure, selftested)
def classify(ours: bytes, official: bytes, recomputed: bytes) -> str:
    if ours != recomputed:
        return "mismatch"
    return "identical" if ours == official else "changed"


def own_q8_0(x: np.ndarray) -> bytes:
    """Independent ggml reference Q8_0 (quantize_row_q8_0_ref): d = amax/127 stored fp16, q = roundf(x / d) with ties
    away from zero, computed as x * (1/d) in float32 as ggml does."""
    b = x.reshape(-1, 32).astype(np.float32)
    d = (np.abs(b).max(axis=1) / np.float32(127.0)).astype(np.float32)
    idd = np.zeros_like(d)
    np.divide(np.float32(1.0), d, out=idd, where=d != 0)
    v = b * idd[:, None]
    q = np.trunc(v + np.copysign(np.float32(0.5), v)).astype(np.int8)
    out = np.empty((b.shape[0], 34), np.uint8)
    out[:, :2] = d.astype(np.float16).view(np.uint8).reshape(-1, 2)
    out[:, 2:] = q.view(np.uint8)
    return out.tobytes()


def q8_rms(got: bytes, x: np.ndarray) -> float:
    q = np.frombuffer(got, np.uint8).reshape(-1, 34)
    d = q[:, :2].copy().view(np.float16).astype(np.float32)
    v = (q[:, 2:].view(np.int8).astype(np.float32) * d).reshape(-1)
    x = x.reshape(-1)
    return float(np.sqrt(np.mean((v - x) ** 2)) / max(np.sqrt(np.mean(x ** 2)), 1e-30))


def meta_diff(a: Any, b: Any, path: str = "") -> list[str]:
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for k in sorted(set(a) | set(b)):
            out += meta_diff(a.get(k), b.get(k), f"{path}.{k}" if path else k)
        return out
    return [] if a == b else [path]


# ---------------------------------------------------------------- experts (one job = one layer chunk)
def _expert_job(j: tuple) -> tuple:
    K = B["K"]
    (l, kind, src, stock_src, h, ff, e0, n, blob, ours_p, ours_base, off_p, off_base) = j
    rec = K._expert_blobs(kind, src, h, ff, e0, n)
    if isinstance(rec, str):
        return l, e0, None, rec
    ours = read_at(ours_p, ours_base + e0 * blob, n * blob)
    off = read_at(off_p, off_base + e0 * blob, n * blob)
    stock = K._expert_blobs(kind, stock_src, h, ff, e0, n) if stock_src else None
    res = []
    for e in range(n):
        sl = slice(e * blob, (e + 1) * blob)
        c = classify(ours[sl], off[sl], rec[sl])
        s_ok = None
        if isinstance(stock, bytes):
            # official == converter(stock), and "changed" exactly where merged differs from stock
            s_ok = stock[sl] == off[sl] and ((rec[sl] != stock[sl]) == (c == "changed"))
        res.append((e0 + e, c, s_ok))
    return l, e0, res, None


def expert_sources(ck: Any, l: int) -> dict:
    K = B["K"]
    p = f"model.llm.layers.{l}.mlp.experts."
    return {k: (str(ck.t[p + k][0]), ck.t[p + k][1]) for k in K.EXPERT_PARTS if p + k in ck.t}


def check_experts(ours: Any, off: Any, ck: Any, stock: Any | None, workers: int) -> None:
    K = B["K"]
    ex = K.Experts(ck)
    check(ours.read("experts.layout") == ex.layout_text(), "experts.layout == the layout the converter derives from the merged checkpoint")
    ob, _ = ours.section("experts")
    fb, _ = off.section("experts")
    jobs = []
    for l, kind, _, offset, blob in ex.layers:
        src, ssrc = expert_sources(ck, l), (expert_sources(stock, l) if stock else None)
        for s in range(0, ex.n_expert, K.EXPERTS_PER_JOB):
            n = min(K.EXPERTS_PER_JOB, ex.n_expert - s)
            jobs.append((l, kind, src, ssrc, ex.h, ex.ff, s, n, blob, ours.path, ob + offset, off.path, fb + offset))
    per: dict[int, dict[str, int]] = {}
    bad, stock_bad, errs = [], [], []
    with multiprocessing.Pool(workers) as pool:
        for l, e0, res, err in pool.imap_unordered(_expert_job, jobs):
            if err:
                errs.append(f"layer {l} experts {e0}..: {err}")
                continue
            d = per.setdefault(l, {"identical": 0, "changed": 0, "mismatch": 0})
            for e, c, s_ok in res:
                d[c] += 1
                if c == "mismatch":
                    bad.append(f"L{l}E{e}")
                if s_ok is False:
                    stock_bad.append(f"L{l}E{e}")
    kinds = {l: kind for l, kind, _, _, _ in ex.layers}
    for l in sorted(per):
        print(f"  experts layer {l:2d} ({kinds[l]}): identical {per[l]['identical']:4d}  changed {per[l]['changed']:4d}  "
              f"mismatch {per[l]['mismatch']}", flush=True)
    tot = {k: sum(d[k] for d in per.values()) for k in ("identical", "changed", "mismatch")}
    check(not errs, f"converter accepted every expert chunk of the merged checkpoint {errs[:3] or ''}")
    check(not bad, f"experts: every blob equals the converter's output from the merged checkpoint "
                   f"(identical {tot['identical']}, changed {tot['changed']}, mismatch {tot['mismatch']}) {bad[:10] or ''}")
    if stock:
        check(not stock_bad, f"experts vs stock: official == converter(stock) and changed exactly where merged != stock {stock_bad[:10] or ''}")


# ---------------------------------------------------------------- gguf
def check_gguf(ours: Any, off: Any, ck: Any, allow: re.Pattern, workers: int) -> None:
    K, S = B["K"], B["S"]
    mo, mf = json.loads(ours.read("meta")), json.loads(off.read("meta"))
    to, tf = mo["tensors"], mf["tensors"]
    check(to == tf, f"gguf tensor table (type, shape, bytes, offset) identical to the official ({len(to)} tensors)")
    plan = K.Plan(ck)
    check(sorted(plan.tensors) == sorted(to), "gguf tensor names == the converter's plan for the merged checkpoint")
    ob, _ = ours.section("gguf")
    fb, _ = off.section("gguf")
    counts = {"identical": 0, "changed": 0, "mismatch": 0, "unexpected": 0}
    changed, bad, unexpected = [], [], []
    indep_ok, worst = 0, 0.0
    with multiprocessing.Pool(workers) as pool:
        for n in sorted(to):
            t = to[n]
            if n not in plan.tensors:
                continue
            ours_b = read_at(ours.path, ob + t["offset"], t["bytes"])
            off_b = read_at(off.path, fb + tf[n]["offset"], tf[n]["bytes"]) if n in tf else b""
            info = plan.tensors[n]
            rec = b"".join(plan.chunks(n, pool)) if info.type_name == "Q8_0" else plan.float_bytes(n)
            c = classify(ours_b, off_b, rec)
            if c == "changed" and not allow.match(n):
                c = "unexpected"
                unexpected.append(n)
            counts[c] += 1
            if c == "mismatch":
                bad.append(n)
            if c == "changed":
                changed.append(n)
                if info.type_name == "Q8_0":
                    x = plan.values(n).reshape(-1)
                    indep_ok += own_q8_0(x) == ours_b
                    worst = max(worst, q8_rms(ours_b, x))
    groups: dict[str, int] = {}
    for n in changed:
        g = n.split(".", 2)[2] if n.startswith("blk.") else n
        groups[g] = groups.get(g, 0) + 1
    print(f"  gguf: identical {counts['identical']}, changed {counts['changed']} {groups}, "
          f"unexpected {counts['unexpected']}, mismatch {counts['mismatch']}", flush=True)
    check(not bad, f"gguf: every tensor equals the converter's output from the merged checkpoint {bad[:10] or ''}")
    check(not unexpected, f"gguf: changed tensors are all allowed ({allow.pattern}) {unexpected[:10] or ''}")
    nq = sum(1 for n in changed if plan.tensors[n].type_name == "Q8_0")
    print(f"  independent numpy Q8_0 reproduces {indep_ok}/{nq} changed Q8_0 tensors byte for byte "
          f"(informational); worst relative RMS rounding error {worst:.4%}", flush=True)


# ---------------------------------------------------------------- vision, meta, sections
def _tbytes(g: Any, i: int) -> bytes:
    t = g.tensors[i]
    starts = sorted(x.offset for x in g.tensors) + [g.path.stat().st_size - g.data_start]
    end = next(s for s in starts if s > t.offset)
    n = t.expected_bytes() or (end - t.offset)
    return read_at(str(g.path), g.data_start + t.offset, n)


def check_vision(ours: Any, off: Any, tmp: Path) -> None:
    GG = B["GGUFFile"]
    vo, vf = tmp / "vo.gguf", tmp / "vf.gguf"
    vo.write_bytes(ours.read("vision/mmproj"))
    vf.write_bytes(off.read("vision/mmproj"))
    go, gf = GG(vo), GG(vf)
    ho = {t.name: hashlib.sha256(_tbytes(go, i)).hexdigest() for i, t in enumerate(go.tensors)}
    hf = {t.name: hashlib.sha256(_tbytes(gf, i)).hexdigest() for i, t in enumerate(gf.tensors)}
    check(ho == hf, f"vision/mmproj: all {len(ho)} tensors byte-identical")
    kd = sorted(k for k in set(go.metadata) | set(gf.metadata) if go.metadata.get(k) != gf.metadata.get(k))
    check(set(kd) <= {"general.name"}, f"vision/mmproj metadata differs only in general.name {kd}")


def check_meta(ours: Any, off: Any, ck_dir: Path) -> None:
    mo, mf = json.loads(ours.read("meta")), json.loads(off.read("meta"))
    cfg_bytes = (ck_dir / "config.json").read_bytes()
    print(f"  pack name: {mo.get('name')!r} (official {mf.get('name')!r})")
    for m in (mo, mf):
        for k in DECLARED_META:
            m.pop(k, None)
        m.get("gguf_kv", {}).pop("general.name", None)
        m.get("checkpoint", {}).pop("dir", None)
    if mo.get("hf_config") != mf.get("hf_config"):
        check(mo.get("hf_config") == json.loads(cfg_bytes), "meta hf_config differs from the official only because "
              f"the merged config.json does (keys: {meta_diff(mo.get('hf_config'), mf.get('hf_config'))[:8]})")
        mo.pop("hf_config"), mf.pop("hf_config")
    cs = mo.get("checkpoint", {}).get("config_sha256")
    if cs != mf.get("checkpoint", {}).get("config_sha256"):
        check(cs == hashlib.sha256(cfg_bytes).hexdigest(), "meta checkpoint.config_sha256 == sha256(merged config.json)")
        mo["checkpoint"].pop("config_sha256"), mf["checkpoint"].pop("config_sha256")
    d = meta_diff(mo, mf)
    check(d == [], f"meta identical apart from the declared fields (name, sources, tensors, checkpoint.dir, "
                   f"gguf_kv.general.name) {d[:12] or ''}")


def check_sections(ours: Any, off: Any, args: argparse.Namespace) -> None:
    bad = [n for n in ours.names() if not ours.verify(n)]
    check(not bad, f"our pack: all {len(ours.names())} section SHA-256s match their bytes {bad or ''}")
    if not args.skip_official_sha:
        s = file_sha256(off.path)
        check(s == args.official_sha, f"official pack sha256 {s[:16]}.. == expected {args.official_sha[:16]}..")
    check(sorted(ours.names()) == sorted(off.names()), f"same section names as the official pack {sorted(ours.names())}")
    weight = {"gguf", "meta", "vision/mmproj", "experts"}
    differ = sorted(n for n in ours.names() if n in off.sections and n not in weight
                    and ours.sections[n][2] != off.sections[n][2])
    check(not differ, f"non-weight sections (tokenizer/*, experts.layout) byte-identical to the official {differ or ''}")


def main_verify(a: argparse.Namespace) -> int:
    import tempfile
    load_basalt(Path(a.basalt_src))
    MF, Cb = B["ModelFile"], B["C"]
    ours, off = MF(a.ours), MF(a.official)
    ck = Cb.Checkpoint(a.checkpoint, side={"mtp.safetensors": "mtp."})
    stock = Cb.Checkpoint(a.stock_checkpoint, side={"mtp.safetensors": "mtp."}) if a.stock_checkpoint else None
    check_sections(ours, off, a)
    check_experts(ours, off, ck, stock, a.workers)
    check_gguf(ours, off, ck, re.compile(a.allow_changed), a.workers)
    check_vision(ours, off, Path(tempfile.mkdtemp()))
    check_meta(ours, off, Path(a.checkpoint))
    print(f"PACK_SHA256 {file_sha256(a.ours)}", flush=True)
    print("RESULT:", "VERIFY PASS" if not fails else f"VERIFY FAIL ({len(fails)}): {fails}")
    return 1 if fails else 0


def selftest() -> int:
    ok = True

    def t(c: bool, m: str) -> None:
        nonlocal ok
        print(("ok   " if c else "FAIL ") + m)
        ok &= c
    t(classify(b"a", b"a", b"a") == "identical", "classify identical")
    t(classify(b"b", b"a", b"b") == "changed", "classify changed")
    t(classify(b"b", b"b", b"a") == "mismatch", "pack == official but merged differs -> mismatch")
    t(classify(b"c", b"a", b"b") == "mismatch", "unexplained difference -> mismatch")
    rng = np.random.default_rng(0)
    x = rng.standard_normal(32 * 64).astype(np.float32)
    x[:32] = 0.0
    x[32:64] = 0.0
    x[32:36] = [127.0, -63.5, -61.5, 2.5]  # d = 1 exactly, so these are exact .5 ties
    q = own_q8_0(x)
    t(len(q) == 64 * 34 and q8_rms(q, x) < 0.01, f"own Q8_0: shape and rounding error {q8_rms(q, x):.4%}")
    v = np.frombuffer(q, np.uint8).reshape(-1, 34)[1, 2:].view(np.int8)
    t([int(i) for i in v[:4]] == [127, -64, -62, 3], "ties round away from zero (ggml roundf)")
    t(meta_diff({"a": 1, "b": {"c": 2}}, {"a": 1, "b": {"c": 3}}) == ["b.c"], "meta diff names the nested key")
    t(re.compile(ALLOW_DEFAULT).match("blk.7.attn_rel.weight") is not None
      and re.compile(ALLOW_DEFAULT).match("blk.7.attn_sink.weight") is None
      and re.compile(ALLOW_DEFAULT).match("dspark.blk.0.attn_q.weight") is None, "allow-list regex")
    print("SELFTEST_OK" if ok else "SELFTEST_FAIL")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ours", nargs="?")
    ap.add_argument("official", nargs="?")
    ap.add_argument("checkpoint", nargs="?")
    ap.add_argument("--basalt-src", default="/root/b/src")
    ap.add_argument("--stock-checkpoint")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--allow-changed", default=ALLOW_DEFAULT)
    ap.add_argument("--official-sha", default=OFFICIAL_SHA)
    ap.add_argument("--skip-official-sha", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not (a.ours and a.official and a.checkpoint):
        ap.error("OURS OFFICIAL CHECKPOINT are required")
    return main_verify(a)


if __name__ == "__main__":
    sys.exit(main())
