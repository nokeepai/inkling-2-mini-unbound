"""v3 GGUF build: patch the official ggml-org BF16 GGUF with a merged checkpoint whose attention (BF16) AND expert
tensors (NVFP4 layers 0-19, MXFP8 layers 20-23) differ from stock. Runs ON the pod (CPU only); --selftest is offline.

  python3 gguf_v3.py map    --ckpt DIR --gguf OFFICIAL.gguf [--out map.json]
  python3 gguf_v3.py prove  --base-ckpt STOCK --gguf OFFICIAL.gguf --out proof.json [--sample N --experts K --rows R | --full]
  python3 gguf_v3.py patch  --base-ckpt STOCK --merged-ckpt DIR --gguf OFFICIAL.gguf --proof proof.json --out NEW.gguf
                            [--name NAME] [--report patch.json] [--force-unproven]
  python3 gguf_v3.py verify --gguf NEW.gguf --official OFFICIAL.gguf --merged-ckpt DIR --proof proof.json --out-dir DIR
  python3 gguf_v3.py --selftest

--base-ckpt must be the checkpoint the official GGUF was converted from: STOCK (thinkingmachines/Inkling-2-Mini
@ed40e7da), also for a merge built on v2.1. Every tensor where merged differs from stock is then replaced, which for a
v2.1-based merge includes v2.1's own attention edits.

Mapping evidence (official header, package/gguf/proof/tensor_manifest.tsv, checkpoint index + hf_quant_config):
  * GGUF expert tensors are BF16 blk.L.ffn_gate_up_exps.weight ne=[2304,2304,128] and blk.L.ffn_down_exps.weight
    ne=[1152,2304,128]. ggml ne is innermost first, so numpy shapes are [128, 2304, 2304] and [128, 2304, 1152]: the
    same [expert, out, in] order as the checkpoint's w13 [E, 2I, D] and w2 [E, D, I] once codes are unpacked.
  * down is not square, so its orientation is fixed by shape. gate_up is square (2I = D = 2304), so orientation and
    row order (checkpoint rows interleave gate/up, merge_adapter.py docstring) are NOT fixed by metadata: prove tests
    every row order x transpose x decode op order x BF16 rounding and keeps only an exact byte match.
  * attention: make_map_v11b.py's five rules (proven byte-exact in v2/v2.1) plus wr_du -> attn_rel (name, and shape
    [d_rel*heads=384, 2304]); prove checks it byte-for-byte like the others.

Patch rule per element: where the merged decoded value equals the base decoded value, the official bytes are kept;
elsewhere the merged value goes through the proven order and rounding. With an exact proof this equals converting
the whole merged tensor, and all other tensors and header bytes stay identical (only general.name may change).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import shutil
import struct
import sys
import time
from dataclasses import asdict, dataclass
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import merge_adapter as ma  # noqa: E402  (codecs, checkpoint index/header helpers; never edited from here)

ATTN_RULE = {"wq_du": "attn_q", "wk_dv": "attn_k", "wv_dv": "attn_v", "wo_ud": "attn_output",
             "gate_attn": "attn_gate", "wr_du": "attn_rel"}
EXPERT_RULE = {"w13": "ffn_gate_up_exps", "w2": "ffn_down_exps"}
GGUF_EXPERT_RE = re.compile(r"^blk\.(\d+)\.(ffn_gate_up_exps|ffn_down_exps)\.weight$")
GGML_TYPES = {0: ("F32", 4), 1: ("F16", 2), 30: ("BF16", 2)}
BF16 = 30
CHUNK = 1 << 26


def say(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def die(msg: str) -> None:
    raise SystemExit(f"FAIL gguf_v3: {msg}")


# ================================================================ GGUF header (own parser: works on a header prefix)

@dataclass(frozen=True)
class GTensor:
    name: str
    ne: tuple[int, ...]
    gtype: int
    rel_offset: int

    @property
    def shape(self) -> tuple[int, ...]:          # numpy order
        return tuple(reversed(self.ne))

    @property
    def esize(self) -> int:
        if self.gtype not in GGML_TYPES:
            die(f"{self.name}: ggml type {self.gtype} not supported (expected an F32/F16/BF16 file)")
        return GGML_TYPES[self.gtype][1]

    @property
    def nbytes(self) -> int:
        n = 1
        for d in self.ne:
            n *= d
        return n * self.esize


@dataclass
class KV:
    key: str
    vtype: int
    value: Any
    start: int          # first byte of the key length
    value_start: int    # first byte of the value (after the type word)
    end: int


@dataclass
class Gguf:
    path: str
    version: int
    kvs: dict[str, KV]
    tensors: dict[str, GTensor]
    ti_end: int
    alignment: int
    data_offset: int
    header_sha256: str

    def abs_offset(self, t: GTensor) -> int:
        return self.data_offset + t.rel_offset


SCALAR_FMT = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}


class Cursor:
    def __init__(self, buf: bytes) -> None:
        self.buf, self.pos = buf, 0

    def take(self, fmt: str) -> Any:
        n = struct.calcsize(fmt)
        if self.pos + n > len(self.buf):
            raise EOFError
        v = struct.unpack_from(fmt, self.buf, self.pos)[0]
        self.pos += n
        return v

    def string(self) -> str:
        n = self.take("<Q")
        if self.pos + n > len(self.buf):
            raise EOFError
        s = self.buf[self.pos:self.pos + n].decode("utf-8", "replace")
        self.pos += n
        return s


def read_value(c: Cursor, vt: int) -> Any:
    """Scalars and strings are returned; long arrays (the tokenizer) are skipped and returned as their length only."""
    if vt == 8:
        return c.string()
    if vt == 9:
        et, n = c.take("<I"), c.take("<Q")
        items = [read_value(c, et) for _ in range(n)] if et in (8, 9) else None
        if items is None:
            size = struct.calcsize(SCALAR_FMT[et])
            if c.pos + n * size > len(c.buf):
                raise EOFError
            items = [struct.unpack_from(SCALAR_FMT[et], c.buf, c.pos + i * size)[0] for i in range(min(n, 64))]
            c.pos += n * size
        return items if n <= 64 else {"array_len": n}
    return c.take(SCALAR_FMT[vt])


def parse_header(buf: bytes, path: str) -> Gguf:
    c = Cursor(buf)
    if buf[:4] != b"GGUF":
        die(f"{path}: not a GGUF file")
    c.pos = 4
    version, n_t, n_kv = c.take("<I"), c.take("<Q"), c.take("<Q")
    kvs: dict[str, KV] = {}
    for _ in range(n_kv):
        start = c.pos
        key, vt = c.string(), c.take("<I")
        vstart = c.pos
        kvs[key] = KV(key, vt, read_value(c, vt), start, vstart, c.pos)
    tensors: dict[str, GTensor] = {}
    for _ in range(n_t):
        name, nd = c.string(), c.take("<I")
        ne = tuple(c.take("<Q") for _ in range(nd))
        tensors[name] = GTensor(name, ne, c.take("<I"), c.take("<Q"))
    align = int(kvs["general.alignment"].value) if "general.alignment" in kvs else 32
    return Gguf(path, version, kvs, tensors, c.pos, align, c.pos + (-c.pos) % align,
                hashlib.sha256(buf[:c.pos]).hexdigest())


def read_gguf(path: str) -> Gguf:
    """Reads only as much of the file as the header needs (doubling), so a header-only prefix file also works."""
    size, n = os.path.getsize(path), 1 << 24
    with open(path, "rb") as f:
        while True:
            f.seek(0)
            buf = f.read(min(n, size))
            try:
                return parse_header(buf, path)
            except EOFError:
                if n >= size:
                    die(f"{path}: header truncated ({size} bytes)")
                n *= 2


# ================================================================ checkpoint side + map

@dataclass(frozen=True)
class Entry:
    ckpt: str            # checkpoint weight key
    gguf: str            # GGUF tensor name
    cls: str             # attn | w13.nvfp4 | w2.nvfp4 | w13.mxfp8 | w2.mxfp8
    layer: int
    which: str           # attention module, or w13 / w2
    fmt: str             # bf16 | nvfp4 | mxfp8
    shape: tuple[int, ...]   # logical numpy shape in the checkpoint (codes unpacked)


def text_config(ckpt: str) -> dict[str, Any]:
    cfg = json.load(open(os.path.join(ckpt, "config.json")))
    return cfg.get("text_config", cfg)


def ckpt_keys(ckpt: str) -> list[str]:
    return list(ma.load_ckpt(ckpt).weight_map)


def stored_meta(ck: ma.Ckpt, key: str) -> dict[str, Any] | None:
    """Safetensors header entry when the shard is on disk (on the pod), else None (map run on the Mac from the index)."""
    fn = ck.weight_map[key]
    if not os.path.exists(os.path.join(ck.base, fn)):
        return None
    return ck.meta(key)[2]


def attn_entries(ck: ma.Ckpt, g: Gguf) -> list[Entry]:
    out = []
    pat = re.compile(r"^model\.llm\.layers\.(\d+)\.attn\.(" + "|".join(ATTN_RULE) + r")\.weight$")
    for k in sorted(ck.weight_map):
        m = pat.match(k)
        if not m:
            continue
        layer, mod = int(m.group(1)), m.group(2)
        gname = f"blk.{layer}.{ATTN_RULE[mod]}.weight"
        if gname not in g.tensors:
            die(f"{k}: GGUF has no {gname}")
        t = g.tensors[gname]
        meta = stored_meta(ck, k)
        if meta is not None and (meta["dtype"] != "BF16" or tuple(meta["shape"]) != t.shape):
            die(f"{k}: checkpoint {meta['dtype']} {meta['shape']} vs GGUF {gname} {t.shape}")
        if t.gtype != BF16:
            die(f"{gname}: GGUF type {t.gtype}, expected BF16")
        out.append(Entry(k, gname, "attn", layer, mod, "bf16", t.shape))
    return out


def expert_entries(ck: ma.Ckpt, g: Gguf, fmts: dict[str, str]) -> list[Entry]:
    tc = text_config(ck.base)
    n_exp, d, inter = int(tc["n_routed_experts"]), int(tc["hidden_size"]), int(tc["intermediate_size"])
    out = []
    for layer in range(int(tc["num_hidden_layers"])):
        for which in ("w13", "w2"):
            key = ma.ckpt_key_expert(layer, which)
            if key not in ck.weight_map:
                continue
            if key not in fmts:
                die(f"{key} is in no NVFP4/MXFP8 group of hf_quant_config.json")
            gname = f"blk.{layer}.{EXPERT_RULE[which]}.weight"
            shape = (n_exp, 2 * inter, d) if which == "w13" else (n_exp, d, inter)
            t = g.tensors.get(gname)
            if t is None or t.shape != shape or t.gtype != BF16:
                die(f"{key}: expected GGUF {gname} BF16 {shape}, header has {t}")
            check_stored_expert(ck, ma.ExpertTensor(key, fmts[key], layer, which), n_exp, d, inter)
            out.append(Entry(key, gname, f"{which}.{fmts[key]}", layer, which, fmts[key], shape))
    return out


def check_stored_expert(ck: ma.Ckpt, t: ma.ExpertTensor, n_exp: int, d: int, inter: int) -> None:
    for key, (dtype, shape) in ma.expected_shapes(t, n_exp, d, inter).items():
        meta = stored_meta(ck, key) if key in ck.weight_map else {"dtype": "missing", "shape": []}
        if meta is not None and (meta["dtype"] != dtype or meta["shape"] != shape):
            die(f"{key}: stored {meta['dtype']} {meta['shape']}, expected {dtype} {shape}")


def build_map(ckpt: str, g: Gguf) -> list[Entry]:
    ck = ma.load_ckpt(ckpt)
    entries = attn_entries(ck, g) + expert_entries(ck, g, ma.quant_formats(ckpt))
    mapped = {e.gguf for e in entries}
    stray = [n for n in g.tensors if GGUF_EXPERT_RE.match(n) and n not in mapped]
    if stray:
        die(f"GGUF expert tensors with no checkpoint source: {stray[:4]}")
    if len(mapped) != len(entries):
        die("two checkpoint tensors map to one GGUF tensor")
    return entries


def map_digest(entries: list[Entry]) -> str:
    return hashlib.sha256(json.dumps([asdict(e) for e in entries], sort_keys=True).encode()).hexdigest()


def cmd_map(a: argparse.Namespace) -> None:
    g = read_gguf(a.gguf)
    entries = build_map(a.ckpt, g)
    counts: dict[str, int] = {}
    for e in entries:
        counts[e.cls] = counts.get(e.cls, 0) + 1
    square = sorted({e.cls for e in entries if e.cls != "attn" and e.shape[1] == e.shape[2]})
    doc = {"gguf_header_sha256": g.header_sha256, "map_sha256": map_digest(entries), "counts": counts,
           "orientation_open": square, "entries": [asdict(e) for e in entries]}
    for e in entries[:2] + [x for x in entries if x.cls != "attn"][:4]:
        say(f"{e.ckpt} -> {e.gguf} {e.cls} {list(e.shape)}")
    say(f"map: {len(entries)} pairs {counts}; square (orientation/row order left to prove): {square}")
    if a.out:
        json.dump(doc, open(a.out, "w"), indent=1)


# ================================================================ hypotheses: checkpoint -> GGUF element order/values

@dataclass(frozen=True)
class Hyp:
    order: str       # as_stored | gate_up_split | up_gate_split
    transpose: bool
    path: str        # vsf = (code*scale)*factor (merge_adapter / transformers) | v_sf = code*(scale*factor) | copy
    rounding: str    # rne | trunc | copy

    @property
    def key(self) -> str:
        return f"{self.order}|T{int(self.transpose)}|{self.path}|{self.rounding}"

    @staticmethod
    def parse(s: str) -> "Hyp":
        o, t, p, r = s.split("|")
        return Hyp(o, t == "T1", p, r)


COPY = Hyp("as_stored", False, "copy", "copy")


def hypotheses(e: Entry) -> list[Hyp]:
    if e.cls == "attn":
        return [COPY]
    orders = ["as_stored", "gate_up_split", "up_gate_split"] if e.which == "w13" else ["as_stored"]
    trans = [False, True] if e.shape[1] == e.shape[2] else [False]
    paths = ["vsf", "v_sf"] if e.fmt == ma.NVFP4 else ["vsf"]
    return [Hyp(o, t, p, r) for o in orders for t in trans for p in paths for r in ("rne", "trunc")]


def reorder(x: Any, h: Hyp) -> Any:
    """[n, R, C] checkpoint order -> GGUF order. Checkpoint w13 row 2j = gate j, 2j+1 = up j."""
    import torch
    if h.order == "gate_up_split":
        x = torch.cat([x[:, 0::2], x[:, 1::2]], 1)
    elif h.order == "up_gate_split":
        x = torch.cat([x[:, 1::2], x[:, 0::2]], 1)
    return x.transpose(1, 2) if h.transpose else x


def to_bf16_bits(x: Any, rounding: str) -> Any:
    """f32 -> bf16 bit patterns as int16. rne matches torch and gguf-py (NaN aside); trunc = round toward zero."""
    import torch
    if rounding == "rne":
        return x.to(torch.bfloat16).view(torch.int16)
    return (x.contiguous().view(torch.int32) >> 16).to(torch.int16)


def bits_to_f32(b: Any) -> Any:
    import torch
    return (b.to(torch.int32) << 16).view(torch.float32)


# ================================================================ reading

class ExpertSource:
    """One checkpoint expert tensor, decoded chunk-wise with merge_adapter's codecs (and one alternative op order)."""

    def __init__(self, ckpt: str, e: Entry) -> None:
        from safetensors import safe_open
        self.e = e
        ck = ma.load_ckpt(ckpt)
        self.f = safe_open(os.path.join(ckpt, ck.weight_map[e.ckpt]), "pt")
        self.t = ma.ExpertTensor(e.ckpt, e.fmt, e.layer, e.which)
        self.factor = ma.nvfp4_factor(self.t, self.f, "cpu")

    def decode(self, e0: int, e1: int, path: str) -> Any:
        q = self.f.get_slice(self.e.ckpt)[e0:e1]
        sc = self.f.get_slice(self.e.ckpt + "_scale")[e0:e1]
        if self.e.fmt == ma.MXFP8:
            return ma.mxfp8_dequant(q, sc)
        if path == "vsf":
            return ma.nvfp4_dequant(q, sc, self.factor[e0:e1])
        return nvfp4_dequant_sf(q, sc, self.factor[e0:e1])

    def stored_bytes(self, e0: int, e1: int) -> list[bytes]:
        keys = [self.e.ckpt + s for s in ("", "_scale")]
        return [ma.tensor_bytes(self.f.get_slice(k)[e0:e1]) for k in keys]


def nvfp4_dequant_sf(codes: Any, scale: Any, factor: Any) -> Any:
    """Alternative op order a converter might use: code * (block scale * per-expert factor)."""
    import torch
    grid = torch.tensor(ma.E2M1_VALUES, device=codes.device)
    b = codes.long()
    vals = torch.stack((grid[b & 0xF], grid[b >> 4]), dim=-1)
    n, rows, nb = scale.shape
    sf = scale.float() * factor.float()[:, None, None]
    return (vals.reshape(n, rows, nb, 16) * sf[..., None]).reshape(n, rows, nb * 16)


def read_gguf_rows(fh: Any, g: Gguf, t: GTensor, e0: int, e1: int, rows: list[int] | None) -> Any:
    """BF16 bits (int16) of experts e0:e1, all rows or the listed GGUF rows, as [n, len(rows), C]."""
    import numpy as np
    import torch
    n_r, n_c = t.shape[-2], t.shape[-1]
    base = g.abs_offset(t)
    if rows is None:
        fh.seek(base + e0 * n_r * n_c * 2)
        arr = np.frombuffer(fh.read((e1 - e0) * n_r * n_c * 2), dtype=np.int16).reshape(e1 - e0, n_r, n_c)
        return torch.from_numpy(arr.copy())
    out = np.empty((e1 - e0, len(rows), n_c), dtype=np.int16)
    for i, e in enumerate(range(e0, e1)):
        for j, r in enumerate(rows):
            fh.seek(base + (e * n_r + r) * n_c * 2)
            out[i, j] = np.frombuffer(fh.read(n_c * 2), dtype=np.int16)
    return torch.from_numpy(out)


def read_range(path: str, start: int, n: int) -> bytes:
    with open(path, "rb") as f:
        f.seek(start)
        return f.read(n)


def ckpt_tensor_range(ckpt: str, key: str) -> tuple[str, int, int]:
    ck = ma.load_ckpt(ckpt)
    fn, start, meta = ck.meta(key)
    s, e = meta["data_offsets"]
    return os.path.join(ckpt, fn), start + s, e - s


# ================================================================ prove

@dataclass
class Score:
    n: int = 0
    byte_eq: int = 0
    value_eq: int = 0
    max_abs: float = 0.0          # |official - expected after rounding|
    max_abs_f32: float = 0.0      # |official - exact f32 decode|

    def add(self, got: Any, want_bits: Any, want_f32: Any) -> None:
        go = bits_to_f32(got)
        self.n += got.numel()
        self.byte_eq += int((got == want_bits).sum())
        self.value_eq += int((go == bits_to_f32(want_bits)).sum())
        self.max_abs = max(self.max_abs, float((go - bits_to_f32(want_bits)).abs().max()) if got.numel() else 0.0)
        if want_f32 is not None:
            self.max_abs_f32 = max(self.max_abs_f32, float((go - want_f32).abs().max()))

    def summary(self) -> dict[str, Any]:
        return {"elements": self.n, "byte_exact": self.byte_eq / max(self.n, 1),
                "value_exact": self.value_eq / max(self.n, 1), "max_abs_diff": self.max_abs,
                "max_abs_diff_vs_f32_decode": self.max_abs_f32}


def expected_chunk(src: ExpertSource, e0: int, e1: int, h: Hyp, rows: list[int] | None) -> tuple[Any, Any]:
    x = reorder(src.decode(e0, e1, h.path), h)
    if rows is not None:
        x = x[:, rows]
    return to_bf16_bits(x, h.rounding), x


def prove_attn(a: argparse.Namespace, g: Gguf, e: Entry) -> dict[str, Any]:
    import numpy as np
    import torch
    path, start, n = ckpt_tensor_range(a.base_ckpt, e.ckpt)
    t = g.tensors[e.gguf]
    if n != t.nbytes:
        return {"pass": False, "error": f"byte sizes differ {n} vs {t.nbytes}"}
    s = Score()
    want = torch.from_numpy(np.frombuffer(read_range(path, start, n), dtype=np.int16).copy())
    got = torch.from_numpy(np.frombuffer(read_range(g.path, g.abs_offset(t), n), dtype=np.int16).copy())
    s.add(got, want, None)
    sm = s.summary()
    return {"scan": {COPY.key: sm["byte_exact"]}, "best": COPY.key, **sm, "pass": sm["byte_exact"] == 1.0}


def sample_rows(n_rows: int, k: int, rnd: random.Random) -> list[int] | None:
    return None if k <= 0 or k >= n_rows else sorted(rnd.sample(range(n_rows), k))


def prove_expert(a: argparse.Namespace, g: Gguf, e: Entry, rnd: random.Random) -> dict[str, Any]:
    src, t = ExpertSource(a.base_ckpt, e), g.tensors[e.gguf]
    n_exp = e.shape[0]
    experts = list(range(n_exp)) if a.experts <= 0 else sorted(rnd.sample(range(n_exp), min(a.experts, n_exp)))
    rows = sample_rows(t.shape[1], a.rows, rnd)
    with open(g.path, "rb") as fh:
        got0 = read_gguf_rows(fh, g, t, experts[0], experts[0] + 1, rows)
        scan = {}
        for h in hypotheses(e):         # scan every hypothesis on the first sampled expert
            bits, _ = expected_chunk(src, experts[0], experts[0] + 1, h, rows)
            scan[h.key] = float((bits == got0).float().mean())
        best = Hyp.parse(max(scan, key=lambda k: (scan[k], -list(scan).index(k))))
        s = Score()
        for e_ in experts:              # then check the winner on every sampled expert
            bits, x = expected_chunk(src, e_, e_ + 1, best, rows)
            s.add(read_gguf_rows(fh, g, t, e_, e_ + 1, rows), bits, x)
    sm = s.summary()
    exact = [k for k, v in scan.items() if v == 1.0]
    return {"scan": scan, "best": best.key, "exact_hypotheses_on_scan": exact, "experts_checked": len(experts),
            "rows_per_expert": len(rows) if rows else t.shape[1], **sm, "pass": sm["byte_exact"] == 1.0}


def pick_entries(entries: list[Entry], n: int, rnd: random.Random) -> list[Entry]:
    """All entries, or a sample of n that holds at least one tensor of every class."""
    if n <= 0 or n >= len(entries):
        return entries
    by_cls: dict[str, list[Entry]] = {}
    for e in entries:
        by_cls.setdefault(e.cls, []).append(e)
    chosen = {rnd.choice(v).gguf for v in by_cls.values()}
    rest = [e for e in entries if e.gguf not in chosen]
    chosen |= {e.gguf for e in rnd.sample(rest, max(0, min(len(rest), n - len(chosen))))}
    return [e for e in entries if e.gguf in chosen]


def decide(entries: list[Entry], results: dict[str, dict[str, Any]]) -> tuple[dict[str, str], list[str]]:
    """One hypothesis per class, required to be exact on every checked tensor of that class."""
    problems, decision = [], {}
    classes = sorted({e.cls for e in entries})
    for cls in classes:
        rs = [(e.gguf, results[e.gguf]) for e in entries if e.cls == cls and e.gguf in results]
        if not rs:
            problems.append(f"{cls}: no tensor checked")
            continue
        bad = [n for n, r in rs if not r["pass"]]
        bests = {r["best"] for _, r in rs}
        if bad:
            problems.append(f"{cls}: {len(bad)} tensor(s) not byte-exact, e.g. {bad[:3]}")
        if len(bests) != 1:
            problems.append(f"{cls}: tensors disagree on the mapping {sorted(bests)}")
        decision[cls] = sorted(bests)[0]
    return decision, problems


def cmd_prove(a: argparse.Namespace) -> None:
    if a.full:
        a.sample, a.experts, a.rows = 0, 0, 0
    g = read_gguf(a.gguf)
    entries = build_map(a.base_ckpt, g)
    rnd = random.Random(a.seed)
    chosen = pick_entries(entries, a.sample, rnd)
    results: dict[str, dict[str, Any]] = {}
    for e in chosen:
        r = prove_attn(a, g, e) if e.cls == "attn" else prove_expert(a, g, e, rnd)
        results[e.gguf] = {"ckpt": e.ckpt, "cls": e.cls, **r}
        say(f"{'OK ' if r['pass'] else 'BAD'} {e.gguf:34s} {e.cls:10s} best={r['best']} "
            f"byte_exact={r['byte_exact']:.6f} value_exact={r['value_exact']:.6f} max_abs={r['max_abs_diff']:.3g}")
    decision, problems = decide(chosen, results)
    full = a.sample == 0 and a.experts == 0 and a.rows == 0
    doc = {"passed": not problems, "mode": "full" if full else "sample", "problems": problems, "decision": decision,
           "gguf_header_sha256": g.header_sha256, "map_sha256": map_digest(entries), "base_ckpt": a.base_ckpt,
           "args": {k: getattr(a, k) for k in ("sample", "experts", "rows", "seed")},
           "tensors_checked": len(chosen), "tensors_mapped": len(entries), "results": results}
    json.dump(doc, open(a.out, "w"), indent=1)
    for p in problems:
        say(f"PROBLEM {p}")
    say(f"decision: {decision}")
    say(("PROVE_PASSED" if not problems else "PROVE_FAILED") + f" ({doc['mode']}, {len(chosen)}/{len(entries)} tensors)")
    if problems:
        sys.exit(1)


# ================================================================ diff base vs merged checkpoint

def range_sha(path: str, start: int, n: int) -> str:
    with open(path, "rb") as f:
        return ma.range_sha(f, start, n)


def tensor_differs(base: str, merged: str, key: str) -> bool:
    pb, sb, nb = ckpt_tensor_range(base, key)
    pm, sm_, nm = ckpt_tensor_range(merged, key)
    if nb != nm:
        die(f"{key}: byte size differs between base and merged")
    with open(pb, "rb") as fa, open(pm, "rb") as fb:
        return not ma.same_range(fa, fb, sb, sm_, nb)


def group_keys(e: Entry, keys: set[str]) -> list[str]:
    if e.cls == "attn":
        return [e.ckpt]
    return [k for k in (e.ckpt + s for s in ("", "_scale", "_amax", "_outer_scale_denominator")) if k in keys]


def changed_entries(base: str, merged: str, entries: list[Entry]) -> list[Entry]:
    """Mapped tensors whose bytes (or scales) differ; any other differing checkpoint tensor aborts the patch."""
    kb, km = set(ckpt_keys(base)), set(ckpt_keys(merged))
    if kb != km:
        die(f"base and merged tensor names differ: {sorted(kb ^ km)[:5]}")
    grouped = {k: e for e in entries for k in group_keys(e, kb)}
    changed, stray = set(), []
    for k in sorted(kb):
        if tensor_differs(base, merged, k):
            if k in grouped:
                changed.add(grouped[k].gguf)
            else:
                stray.append(k)
    if stray:
        die(f"{len(stray)} changed checkpoint tensor(s) have no GGUF mapping: {stray[:5]}")
    return [e for e in entries if e.gguf in changed]


# ================================================================ patch

@dataclass
class PatchStats:
    elements: int = 0
    changed_vs_official: int = 0
    changed_values: int = 0
    base_mismatch: int = 0
    worst_rel_err: float = 0.0

    def summary(self) -> dict[str, Any]:
        return {**asdict(self), "frac_changed_vs_official": self.changed_vs_official / max(self.elements, 1)}


def patch_expert_chunk(bs: ExpertSource, ms: ExpertSource, e0: int, e1: int, h: Hyp, off: Any,
                       st: PatchStats) -> Any:
    """New BF16 bits for experts e0:e1: official where the merged value equals base, else rounded merged."""
    import torch
    xb, xm = reorder(bs.decode(e0, e1, h.path), h), reorder(ms.decode(e0, e1, h.path), h)
    st.base_mismatch += int((to_bf16_bits(xb, h.rounding) != off).sum())
    moved = xb.contiguous().view(torch.int32) != xm.contiguous().view(torch.int32)
    mb = to_bf16_bits(xm, h.rounding)
    new = torch.where(moved, mb, off)
    st.elements += off.numel()
    st.changed_values += int(moved.sum())
    st.changed_vs_official += int((new != off).sum())
    nz = moved & (xm != 0)
    if bool(nz.any()):
        rel = ((bits_to_f32(mb) - xm).abs() / xm.abs())[nz]
        st.worst_rel_err = max(st.worst_rel_err, float(rel.max()))
    return new


def patch_expert(a: argparse.Namespace, g: Gguf, e: Entry, h: Hyp, dst: Any, data_start: int) -> PatchStats:
    bs, ms = ExpertSource(a.base_ckpt, e), ExpertSource(a.merged_ckpt, e)
    t, st = g.tensors[e.gguf], PatchStats()
    per = t.shape[1] * t.shape[2] * 2
    with open(g.path, "rb") as fh:
        for e0 in range(0, e.shape[0], a.expert_chunk):
            e1 = min(e.shape[0], e0 + a.expert_chunk)
            off = read_gguf_rows(fh, g, t, e0, e1, None)
            new = patch_expert_chunk(bs, ms, e0, e1, h, off, st)
            dst.seek(data_start + t.rel_offset + e0 * per)
            dst.write(new.contiguous().numpy().tobytes())
    return st


def patch_attn(a: argparse.Namespace, g: Gguf, e: Entry, dst: Any, data_start: int) -> PatchStats:
    import numpy as np
    t = g.tensors[e.gguf]
    pb, sb, n = ckpt_tensor_range(a.base_ckpt, e.ckpt)
    pm, sm_, _ = ckpt_tensor_range(a.merged_ckpt, e.ckpt)
    off = np.frombuffer(read_range(g.path, g.abs_offset(t), n), dtype=np.int16)
    base = np.frombuffer(read_range(pb, sb, n), dtype=np.int16)
    merged = np.frombuffer(read_range(pm, sm_, n), dtype=np.int16)
    new = np.where(merged != base, merged, off)
    dst.seek(data_start + t.rel_offset)
    dst.write(new.tobytes())
    return PatchStats(int(off.size), int((new != off).sum()), int((merged != base).sum()), int((base != off).sum()))


def new_header(g: Gguf, name: str | None) -> bytes:
    """Official header bytes, with general.name re-spliced when asked, padded to the alignment. Tensor offsets are
    relative to the data section, so they stay valid whatever the header length."""
    raw = read_range(g.path, 0, g.ti_end)
    if name:
        kv = g.kvs.get("general.name")
        if kv is None or kv.vtype != 8:
            die("official GGUF has no string general.name; refusing to guess a header edit")
        enc = name.encode()
        raw = raw[:kv.value_start] + struct.pack("<Q", len(enc)) + enc + raw[kv.end:]
    return raw + b"\x00" * ((-len(raw)) % g.alignment)


def check_proof(a: argparse.Namespace, g: Gguf, entries: list[Entry]) -> dict[str, str]:
    proof = json.load(open(a.proof)) if a.proof and os.path.exists(a.proof) else None
    problems = []
    if proof is None:
        problems.append(f"no proof file ({a.proof})")
    else:
        if not proof.get("passed"):
            problems.append(f"proof did not pass: {proof.get('problems')}")
        if proof.get("gguf_header_sha256") != g.header_sha256:
            problems.append("proof was made against a different GGUF header")
        if proof.get("map_sha256") != map_digest(entries):
            problems.append("proof was made for a different mapping")
        if proof.get("mode") != "full":
            say("NOTE proof is sample-mode; every patched tensor is still re-proven against base below")
    decision = dict(proof.get("decision", {})) if proof else {}
    for cls in {e.cls for e in entries} - set(decision):
        problems.append(f"no proven mapping for class {cls}")
        decision[cls] = COPY.key if cls == "attn" else "as_stored|T0|vsf|rne"
    if problems and not a.force_unproven:
        die("refusing to patch: " + "; ".join(problems) + " (run prove, or pass --force-unproven)")
    for p in problems:
        say(f"WARNING --force-unproven: {p}")
    if problems:
        say("WARNING " + "!" * 60 + "\nWARNING writing a GGUF whose mapping is NOT proven; do not release it\n"
            "WARNING " + "!" * 60)
    return decision


def cmd_patch(a: argparse.Namespace) -> None:
    g = read_gguf(a.gguf)
    entries = build_map(a.base_ckpt, g)
    decision = check_proof(a, g, entries)
    todo = changed_entries(a.base_ckpt, a.merged_ckpt, entries)
    kinds: dict[str, int] = {}
    for e in todo:
        kinds[e.cls] = kinds.get(e.cls, 0) + 1
    say(f"{len(todo)} tensors differ from base: {kinds}")
    hdr = new_header(g, a.name)
    tmp = a.out + ".part"
    with open(g.path, "rb") as src, open(tmp, "wb") as dst:
        dst.write(hdr)
        src.seek(g.data_offset)
        shutil.copyfileobj(src, dst, CHUNK)
    report: dict[str, Any] = {"official": a.gguf, "decision": decision, "name": a.name, "tensors": {}}
    with open(tmp, "r+b") as dst:
        for e in sorted(todo, key=lambda x: g.tensors[x.gguf].rel_offset):
            st = patch_attn(a, g, e, dst, len(hdr)) if e.cls == "attn" else \
                patch_expert(a, g, e, Hyp.parse(decision[e.cls]), dst, len(hdr))
            report["tensors"][e.gguf] = {"ckpt": e.ckpt, "cls": e.cls, **st.summary()}
            say(f"patched {e.gguf:34s} changed_vs_official={st.changed_vs_official}/{st.elements} "
                f"({st.changed_vs_official / max(st.elements, 1):.4f}) worst_rel_err={st.worst_rel_err:.3g} "
                f"base_mismatch={st.base_mismatch}")
            if st.base_mismatch and not a.force_unproven:
                os.remove(tmp)
                die(f"{e.gguf}: base checkpoint does not reproduce the official bytes ({st.base_mismatch} elements)")
    report["counts"] = kinds
    if a.report:
        json.dump(report, open(a.report, "w"), indent=1)
    os.replace(tmp, a.out)
    say(f"PATCH_DONE wrote {a.out}: {len(todo)} tensors replaced {kinds}")


# ================================================================ verify

def gguf_sha(path: str, start: int, n: int) -> str:
    return range_sha(path, start, n)


def kv_raw(g: Gguf) -> dict[str, bytes]:
    with open(g.path, "rb") as f:
        out = {}
        for k, kv in g.kvs.items():
            f.seek(kv.start)
            out[k] = f.read(kv.end - kv.start)
        return out


def header_problems(gn: Gguf, go: Gguf) -> list[str]:
    probs = []
    if [(t.name, t.ne, t.gtype, t.rel_offset) for t in gn.tensors.values()] != \
            [(t.name, t.ne, t.gtype, t.rel_offset) for t in go.tensors.values()]:
        probs.append("tensor infos differ")
    kn, ko = kv_raw(gn), kv_raw(go)
    diff = sorted(k for k in set(kn) | set(ko) if kn.get(k) != ko.get(k))
    if diff not in ([], ["general.name"]):
        probs.append(f"KV keys differ beyond general.name: {diff}")
    return probs


def verify_expert(a: argparse.Namespace, gn: Gguf, e: Entry, h: Hyp) -> int:
    """Elements of the new GGUF tensor that differ from the merged checkpoint pushed through the proven mapping."""
    src, t, bad = ExpertSource(a.merged_ckpt, e), gn.tensors[e.gguf], 0
    with open(gn.path, "rb") as fh:
        for e0 in range(0, e.shape[0], a.expert_chunk):
            e1 = min(e.shape[0], e0 + a.expert_chunk)
            want, _ = expected_chunk(src, e0, e1, h, None)
            bad += int((read_gguf_rows(fh, gn, t, e0, e1, None) != want).sum())
    return bad


def verify_attn(a: argparse.Namespace, gn: Gguf, e: Entry) -> int:
    import numpy as np
    path, start, n = ckpt_tensor_range(a.merged_ckpt, e.ckpt)
    want = np.frombuffer(read_range(path, start, n), dtype=np.int16)
    got = np.frombuffer(read_range(gn.path, gn.abs_offset(gn.tensors[e.gguf]), n), dtype=np.int16)
    return int((want != got).sum())


def cmd_verify(a: argparse.Namespace) -> None:
    gn, go = read_gguf(a.gguf), read_gguf(a.official)
    proof = json.load(open(a.proof))
    entries = {e.gguf: e for e in build_map(a.merged_ckpt, go)}
    problems = header_problems(gn, go)
    os.makedirs(a.out_dir, exist_ok=True)
    rows, n_diff, counts = [], 0, {}
    for t in go.tensors.values():
        so, sn = gguf_sha(go.path, go.abs_offset(t), t.nbytes), gguf_sha(gn.path, gn.abs_offset(t), t.nbytes)
        e = entries.get(t.name)
        bad = 0
        if e is not None:
            bad = verify_attn(a, gn, e) if e.cls == "attn" else verify_expert(a, gn, e, Hyp.parse(proof["decision"][e.cls]))
            if bad:
                problems.append(f"{t.name}: {bad} elements differ from the merged checkpoint")
        elif so != sn:
            problems.append(f"{t.name}: unmapped tensor changed")
        if so != sn:
            n_diff += 1
            counts[e.cls if e else "unmapped"] = counts.get(e.cls if e else "unmapped", 0) + 1
        rows.append(f"{t.name}\t{GGML_TYPES[t.gtype][0]}\t{'x'.join(map(str, t.ne))}\t{so}\t{sn}\t{int(so == sn)}\t"
                    f"{e.cls if e else '-'}\t{bad}")
    open(os.path.join(a.out_dir, "tensor_manifest.tsv"), "w").write(
        "name\ttype\tne\tsha256_official\tsha256_new\tsame\tclass\telements_off_expected\n" + "\n".join(rows) + "\n")
    doc = {"passed": not problems, "problems": problems, "n_tensors": len(rows), "n_differ": n_diff,
           "differ_by_class": counts}
    json.dump(doc, open(os.path.join(a.out_dir, "verify.json"), "w"), indent=1)
    for p in problems[:20]:
        say(f"PROBLEM {p}")
    say(f"{n_diff}/{len(rows)} tensors differ from official {counts}")
    say("VERIFY_PASSED" if not problems else "VERIFY_FAILED")
    if problems:
        sys.exit(1)


# ================================================================ selftest (offline, synthetic)

def st_gguf_write(path: str, kvs: list[tuple[str, int, Any]], tensors: list[tuple[str, Any, int]], align: int = 32) -> None:
    """Minimal GGUF v3 writer: kvs (key, vtype, value) with vtype 4/8/(9, str list); tensors (name, numpy, ggml type)."""
    def s(x: str) -> bytes:
        b = x.encode()
        return struct.pack("<Q", len(b)) + b

    out = [b"GGUF", struct.pack("<IQQ", 3, len(tensors), len(kvs) + 1)]
    out.append(s("general.alignment") + struct.pack("<II", 4, align))
    for k, vt, v in kvs:
        if vt == 8:
            out.append(s(k) + struct.pack("<I", 8) + s(v))
        elif vt == 9:
            out.append(s(k) + struct.pack("<IIQ", 9, 8, len(v)) + b"".join(s(x) for x in v))
        else:
            out.append(s(k) + struct.pack("<I", vt) + struct.pack(SCALAR_FMT[vt], v))
    blobs, off = [], 0
    for name, arr, gt in tensors:
        out.append(s(name) + struct.pack("<I", arr.ndim) + b"".join(struct.pack("<Q", d) for d in reversed(arr.shape))
                   + struct.pack("<IQ", gt, off))
        b = arr.tobytes()
        blobs.append(b + b"\x00" * ((-len(b)) % align))
        off += len(blobs[-1])
    hdr = b"".join(out)
    with open(path, "wb") as f:
        f.write(hdr + b"\x00" * ((-len(hdr)) % align) + b"".join(blobs))


ST_E, ST_D, ST_I, ST_L = 4, 64, 32, 2      # w13 [4, 64, 64] is square like the real one


def st_quant_config() -> dict[str, Any]:
    nv = {"weight_transform": "nvfp4_nearest", "weight_scale_format": "amax_denominator", "weight_block_shape": [1, 16],
          "tensors": [ma.ckpt_key_expert(0, w) for w in ("w13", "w2")]}
    mx = {"weight_transform": "mxfp8_nearest", "weight_block_shape": [32, 32],
          "tensors": [ma.ckpt_key_expert(1, w) for w in ("w13", "w2")]}
    return {"model": {"parameter_groups": [nv, mx]}}


def st_tensors(g: Any) -> dict[str, Any]:
    import torch
    t: dict[str, Any] = {"model.llm.embed.weight": torch.randn(50, ST_D, generator=g).to(torch.bfloat16),
                         "model.llm.layers.0.mlp_norm.weight": torch.randn(ST_D, generator=g).to(torch.bfloat16)}
    shapes = {"wq_du": (96, ST_D), "wk_dv": (32, ST_D), "wv_dv": (32, ST_D), "wo_ud": (ST_D, 96),
              "gate_attn": (96, ST_D), "wr_du": (48, ST_D)}
    for layer in range(ST_L):
        for mod, sh in shapes.items():
            t[f"model.llm.layers.{layer}.attn.{mod}.weight"] = (torch.randn(*sh, generator=g) * 0.05).to(torch.bfloat16)
    for which, (r, c) in (("w13", (2 * ST_I, ST_D)), ("w2", (ST_D, ST_I))):
        k0, k1 = ma.ckpt_key_expert(0, which), ma.ckpt_key_expert(1, which)
        t[k0] = torch.randint(0, 256, (ST_E, r, c // 2), generator=g, dtype=torch.uint8)
        t[k0 + "_scale"] = (torch.rand(ST_E, r, c // 16, generator=g) * 3 + 0.25).to(torch.float8_e4m3fn)
        t[k0 + "_amax"] = torch.rand(ST_E, generator=g) * 0.3 + 0.1
        t[k0 + "_outer_scale_denominator"] = torch.tensor([2688.0])
        t[k1] = (torch.randn(ST_E, r, c, generator=g) * 40).to(torch.float8_e4m3fn)
        t[k1 + "_scale"] = torch.randint(118, 124, (ST_E, r, c // 32), generator=g, dtype=torch.uint8)
    return t


def st_write_ckpt(root: str, t: dict[str, Any]) -> None:
    from safetensors.torch import save_file
    os.makedirs(root, exist_ok=True)
    fn = "model-00001-of-00001.safetensors"
    save_file({k: v.contiguous() for k, v in t.items()}, os.path.join(root, fn))
    json.dump({"weight_map": {k: fn for k in t}}, open(os.path.join(root, "model.safetensors.index.json"), "w"))
    json.dump(st_quant_config(), open(os.path.join(root, "hf_quant_config.json"), "w"))
    json.dump({"text_config": {"hidden_size": ST_D, "intermediate_size": ST_I, "n_routed_experts": ST_E,
                               "num_hidden_layers": ST_L}}, open(os.path.join(root, "config.json"), "w"))


def st_reference_convert(t: dict[str, Any], order: str) -> list[tuple[str, Any, int]]:
    """Stand-in for llama.cpp's converter, written independently of the prove code: dequantise, put w13 into the given
    row order, round to BF16 (RNE)."""
    import torch
    out: list[tuple[str, Any, int]] = [("token_embd.weight", t["model.llm.embed.weight"].view(torch.int16).numpy(), BF16),
                                       ("blk.0.ffn_norm.weight", t["model.llm.layers.0.mlp_norm.weight"].float().numpy(), 0)]
    for layer in range(ST_L):
        for mod, gn in ATTN_RULE.items():
            out.append((f"blk.{layer}.{gn}.weight", t[f"model.llm.layers.{layer}.attn.{mod}.weight"].view(torch.int16).numpy(), BF16))
        for which in ("w13", "w2"):
            k = ma.ckpt_key_expert(layer, which)
            if layer == 0:
                f = t[k + "_amax"] / t[k + "_outer_scale_denominator"]
                x = ma.nvfp4_dequant(t[k], t[k + "_scale"], f)
            else:
                x = ma.mxfp8_dequant(t[k], t[k + "_scale"])
            if which == "w13" and order == "gate_up_split":
                x = torch.cat([x[:, 0::2], x[:, 1::2]], 1)
            out.append((f"blk.{layer}.{EXPERT_RULE[which]}.weight", x.to(torch.bfloat16).view(torch.int16).numpy(), BF16))
    out.append(("output.weight", t["model.llm.embed.weight"].view(torch.int16).numpy(), BF16))
    return out


def st_merged(base: dict[str, Any], g: Any) -> dict[str, Any]:
    """Merged = base with: two attention tensors edited, NVFP4 w13 codes and one block scale changed, MXFP8 w2 values
    nudged by one code. NVFP4 w2 and MXFP8 w13 stay identical (must not be patched)."""
    import torch
    m = {k: v.clone() for k, v in base.items()}
    for k in ("model.llm.layers.0.attn.wq_du.weight", "model.llm.layers.1.attn.wr_du.weight"):
        m[k] = (m[k].float() + torch.randn(m[k].shape, generator=g) * 0.01).to(torch.bfloat16)
    k = ma.ckpt_key_expert(0, "w13")
    flip = torch.rand(m[k].shape, generator=g) < 0.3
    m[k] = torch.where(flip, m[k] ^ 0x01, m[k])
    sc = m[k + "_scale"].view(torch.uint8).clone()
    sc[0, 0, 0] += 1
    m[k + "_scale"] = sc.view(torch.float8_e4m3fn)
    k = ma.ckpt_key_expert(1, "w2")
    u = m[k].view(torch.uint8).clone()
    nudge = (torch.rand(u.shape, generator=g) < 0.2) & ((u & 0x7F) < 0x7D) & ((u & 0x7F) > 0)
    m[k] = torch.where(nudge, u + 1, u).view(torch.float8_e4m3fn)
    return m


def st_run(argv: list[str]) -> int:
    try:
        main(argv)
        return 0
    except SystemExit as ex:
        return 0 if ex.code in (0, None) else 1


def st_expect(cond: bool, msg: str) -> None:
    if not cond:
        raise SystemExit(f"SELFTEST_FAIL {msg}")
    say(f"selftest ok: {msg}")


def st_ground_truth(base: dict[str, Any], merged: dict[str, Any]) -> dict[str, int]:
    """Elements per GGUF tensor that the reference converter puts at different bytes for merged vs base."""
    gb = {n: a for n, a, _ in st_reference_convert(base, "gate_up_split")}
    gm = {n: a for n, a, _ in st_reference_convert(merged, "gate_up_split")}
    return {n: int((gb[n] != gm[n]).sum()) for n in gb if (gb[n] != gm[n]).any()}


def selftest() -> None:
    import tempfile
    import torch
    g = torch.Generator().manual_seed(7)
    base_t = st_tensors(g)
    merged_t = st_merged(base_t, g)
    tmp = tempfile.mkdtemp(prefix="gguf_v3_st_")
    base, merged = os.path.join(tmp, "base"), os.path.join(tmp, "merged")
    st_write_ckpt(base, base_t)
    st_write_ckpt(merged, merged_t)
    kvs = [("general.architecture", 8, "inkling2"), ("general.name", 8, "Inkling-2-Mini"),
           ("tokenizer.ggml.tokens", 9, [f"t{i}" for i in range(100)]), ("x.count", 4, 7)]
    off = os.path.join(tmp, "official.gguf")
    st_gguf_write(off, kvs, st_reference_convert(base_t, "gate_up_split"))
    with open(off, "rb") as f:
        open(os.path.join(tmp, "header_only.gguf"), "wb").write(f.read(read_gguf(off).ti_end))
    p = lambda n: os.path.join(tmp, n)  # noqa: E731

    st_expect(st_run(["map", "--ckpt", base, "--gguf", p("header_only.gguf"), "--out", p("map.json")]) == 0,
              "map from a header-only prefix")
    st_expect(json.load(open(p("map.json")))["counts"] == {"attn": 12, "w13.nvfp4": 1, "w2.nvfp4": 1,
                                                            "w13.mxfp8": 1, "w2.mxfp8": 1}, "map counts")
    rc = st_run(["prove", "--base-ckpt", base, "--gguf", off, "--out", p("proof_s.json"),
                 "--sample", "3", "--experts", "2", "--rows", "8"])
    ps = json.load(open(p("proof_s.json")))
    st_expect(rc == 0 and ps["tensors_checked"] >= 5, "sample prove passes and covers every class")
    st_expect(st_run(["prove", "--base-ckpt", base, "--gguf", off, "--out", p("proof.json"), "--full"]) == 0, "full prove")
    pf = json.load(open(p("proof.json")))
    st_expect(pf["decision"]["w13.nvfp4"].startswith("gate_up_split|T0|") and
              pf["decision"]["w13.mxfp8"].startswith("gate_up_split|T0|"), "prove finds the gate/up row order")
    st_expect(pf["decision"]["w2.nvfp4"] == "as_stored|T0|vsf|rne", "prove finds the decode/rounding")

    st_expect(st_run(["patch", "--base-ckpt", base, "--merged-ckpt", merged, "--gguf", off, "--out", p("new0.gguf")]) != 0
              and not os.path.exists(p("new0.gguf")), "patch refuses without a proof")
    st_expect(st_run(["patch", "--base-ckpt", base, "--merged-ckpt", merged, "--gguf", off, "--proof", p("proof.json"),
                      "--out", p("new.gguf"), "--name", "Inkling-2-Mini-Unbound-v3-selftest",
                      "--report", p("patch.json")]) == 0, "patch with proof")
    rep = json.load(open(p("patch.json")))
    truth = st_ground_truth(base_t, merged_t)
    got = {n: r["changed_vs_official"] for n, r in rep["tensors"].items()}
    st_expect(got == truth, f"changed elements match the reference converter {truth}")
    st_expect(read_gguf(p("new.gguf")).kvs["general.name"].value == "Inkling-2-Mini-Unbound-v3-selftest", "renamed")
    vargs = ["verify", "--gguf", p("new.gguf"), "--official", off, "--merged-ckpt", merged, "--proof", p("proof.json")]
    st_expect(st_run(vargs + ["--out-dir", p("ver")]) == 0, "verify passes")
    st_negative(tmp, base, merged, off, base_t, vargs)
    shutil.rmtree(tmp)
    print("SELFTEST_OK")


def st_corrupt(src: str, dst: str, tensor: str, at: int = 3) -> None:
    shutil.copyfile(src, dst)
    g = read_gguf(dst)
    with open(dst, "r+b") as f:
        f.seek(g.abs_offset(g.tensors[tensor]) + at)
        b = f.read(1)
        f.seek(-1, 1)
        f.write(bytes([b[0] ^ 0x40]))


def st_negative(tmp: str, base: str, merged: str, off: str, base_t: dict[str, Any], vargs: list[str]) -> None:
    p = lambda n: os.path.join(tmp, n)  # noqa: E731
    for name in ("output.weight", "blk.0.ffn_gate_up_exps.weight", "blk.1.ffn_gate_up_exps.weight"):
        st_corrupt(p("new.gguf"), p("bad.gguf"), name)
        st_expect(st_run(["verify", "--gguf", p("bad.gguf")] + vargs[3:] + ["--out-dir", p("verbad")]) != 0,
                  f"verify catches a flipped byte in {name}")
    st_corrupt(off, p("off_bad.gguf"), "blk.0.ffn_down_exps.weight")
    st_expect(st_run(["prove", "--base-ckpt", base, "--gguf", p("off_bad.gguf"), "--out", p("proof_bad.json"),
                      "--full"]) != 0, "prove fails on an official file that is not exactly decode -> BF16")
    st_expect(st_run(["patch", "--base-ckpt", base, "--merged-ckpt", merged, "--gguf", p("off_bad.gguf"),
                      "--proof", p("proof_bad.json"), "--out", p("x.gguf")]) != 0, "patch refuses a failed proof")
    st_expect(st_run(["patch", "--base-ckpt", base, "--merged-ckpt", merged, "--gguf", p("off_bad.gguf"),
                      "--proof", p("proof_bad.json"), "--out", p("x.gguf"), "--force-unproven"]) == 0,
              "--force-unproven writes anyway (with warnings)")
    asst = os.path.join(tmp, "asst.gguf")
    st_gguf_write(asst, [("general.name", 8, "x")], st_reference_convert(base_t, "as_stored"))
    st_expect(st_run(["prove", "--base-ckpt", base, "--gguf", asst, "--out", p("proof_as.json"), "--full"]) == 0 and
              json.load(open(p("proof_as.json")))["decision"]["w13.nvfp4"].startswith("as_stored|T0|"),
              "prove also recognises the interleaved (as stored) order")
    st_expect(st_run(["patch", "--base-ckpt", base, "--merged-ckpt", merged, "--gguf", off, "--proof", p("proof_as.json"),
                      "--out", p("x.gguf")]) != 0, "patch refuses a passed proof made against another GGUF")


# ================================================================ CLI

def parse_args(argv: list[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("map")
    m.add_argument("--ckpt", required=True)
    m.add_argument("--gguf", required=True, help="official GGUF, or a header-only prefix of it")
    m.add_argument("--out")
    pr = sub.add_parser("prove")
    pr.add_argument("--base-ckpt", required=True)
    pr.add_argument("--gguf", required=True)
    pr.add_argument("--out", required=True)
    pr.add_argument("--sample", type=int, default=8, help="tensors to check (0 = all); every class is always included")
    pr.add_argument("--experts", type=int, default=4, help="experts per expert tensor (0 = all)")
    pr.add_argument("--rows", type=int, default=64, help="rows per expert (0 = all)")
    pr.add_argument("--full", action="store_true", help="every tensor, expert and row")
    pr.add_argument("--seed", type=int, default=0)
    pa = sub.add_parser("patch")
    pa.add_argument("--base-ckpt", required=True, help="checkpoint the official GGUF was made from (stock)")
    pa.add_argument("--merged-ckpt", required=True)
    pa.add_argument("--gguf", required=True)
    pa.add_argument("--proof")
    pa.add_argument("--out", required=True)
    pa.add_argument("--name", help="new general.name (as rel21.sh's patcher did)")
    pa.add_argument("--report")
    pa.add_argument("--expert-chunk", type=int, default=8)
    pa.add_argument("--force-unproven", action="store_true")
    ve = sub.add_parser("verify")
    ve.add_argument("--gguf", required=True)
    ve.add_argument("--official", required=True)
    ve.add_argument("--merged-ckpt", required=True)
    ve.add_argument("--proof", required=True)
    ve.add_argument("--out-dir", required=True)
    ve.add_argument("--expert-chunk", type=int, default=8)
    return ap.parse_args(argv)


def main(argv: list[str]) -> None:
    if argv == ["--selftest"]:
        selftest()
        return
    a = parse_args(argv)
    {"map": cmd_map, "prove": cmd_prove, "patch": cmd_patch, "verify": cmd_verify}[a.cmd](a)


if __name__ == "__main__":
    main(sys.argv[1:])
