"""Merge a PEFT LoRA adapter (trained on the stock Inkling-2-Mini) into an Inkling-2-Mini HF checkpoint (e.g. Unbound
v2.1): merged = base + scale * (alpha/r) * B @ A, written as a full checkpoint directory SGLang/vLLM can serve.

  python3 merge_adapter.py --base DIR --adapter DIR --out DIR --parts attn|attn_experts [--scale 1.0]
        [--report FILE] [--device cpu|cuda] [--expert-chunk 8] [--expert-scales keep|bump] [--swap-w1-w3]
  python3 merge_adapter.py --base DIR --adapter A --adapter B [--scales 1.0,0.5] --out DIR ...   # several adapters
  python3 merge_adapter.py --selftest          # offline, synthetic tensors (needs torch + safetensors)

Several adapters: repeat --adapter (one directory each; no comma lists, so paths may contain commas). --scales is a
comma list with one user scale per adapter, in --adapter order; without it every adapter gets --scale. Per tensor the
deltas sum_i (scale_i * alpha_i/r_i) * B_i @ A_i are added in full precision (f64 attention, f32 experts, as for one
adapter) and rounded ONCE, so stacking adapters does not round twice. Adapters may differ in rank and in the modules
they cover; a tensor touched by one adapter gets only that one's delta. One adapter at the default scale takes exactly
the old code path (same op order, same RNG keys), so its output is byte-identical to the single-adapter script.

Layout (evidence, all read from this repo):
  * checkpoint names: vendor/ckpt/orig/model.safetensors.index.json (single shard model-00001-of-00001.safetensors):
    model.llm.layers.N.attn.{wq_du,wk_dv,wv_dv,gate_attn,wo_ud,wr_du}.weight, BF16 [out, in] (shapes in
    out/final/MANIFEST.json); experts model.llm.layers.N.mlp.experts.{w13,w2}_weight (+ _weight_scale, and on NVFP4
    layers _weight_amax and _weight_outer_scale_denominator).
  * adapter names: language_model.layers.N.attn.X.lora_{A,B}.weight -> model.llm.layers.N.attn.X.weight (the
    transformers conversion renames model.llm. -> model.; the adapter uses the checkpoint's own module names).
  * expert formats: vendor/ckpt/orig/hf_quant_config.json: layers 0-19 NVFP4 (weight_transform nvfp4_nearest, block
    [1,16], weight_scale_format amax_denominator), layers 20-23 MXFP8 (mxfp8_nearest). Dequantisation exactly as
    vendor/transformers-pr51050/src/transformers/core_model_loading.py Inkling2LoopDequantizeExperts: NVFP4 = two E2M1
    codes per byte (even element in the low nibble) * float8_e4m3 scale per 16 values * per-expert amax/denominator;
    MXFP8 = float8_e4m3 values * 2**(uint8 scale - 127) per 32 values.
  * w13 row order: conversion_mapping.py applies Interleave(dim=1) to w13 to get gate_up_proj = [gate; up], so stored
    row 2j is gate row j and row 2j+1 is up row j. Adapter w1 -> gate, w3 -> up, w2 -> down: the usual Mixtral/Mistral
    convention (conversion_mapping.py, mistral_large4: "w1 (gate) and w3 (up)"). UNVERIFIED for the training service's
    own export; fallback --swap-w1-w3.

Requantisation (--parts attn_experts): per expert chunk, dequantise, add the delta, re-encode on the ORIGINAL scales
(--expert-scales keep: scale/amax/denominator tensors stay byte-identical, values past a block's range saturate and
are counted) or raise a block's scale just enough to hold the new values (bump). Before merging, every chunk is
re-encoded with a zero delta and must reproduce the stored bytes exactly; a mismatch aborts (format misread).
Delta survival per layer = 1 - |Q(W+d) - (W+d)| / |d| (Frobenius over all experts of w13 and w2).

Rounding (--rounding nearest|stochastic --seed N): nearest is the original behaviour. stochastic rounds W + delta
(f32) to one of its two neighbouring stored values with P proportional to proximity, so E[stored] = W + delta; for
BF16 attention, NVFP4 experts (stored scales kept) and MXFP8 experts. Elements whose delta is exactly zero stay
byte-identical; the random draw is keyed by (seed, tensor name, expert index). Both modes are always measured and
reported (report["fidelity"][mode][group]: change_ratio = |realised|/|intended|, cosine, projected_survival =
scale along the delta) plus a sampled spectral check (report["spectral"]); only --rounding is written.
UNVERIFIED: that SGLang reads amax only through amax/denominator as transformers does. keep mode leaves every scale,
amax and denominator tensor byte-identical, so it does not depend on this; bump mode does (fallback: keep).

Memory: one tensor (attention) or one chunk of experts (experts) at a time; the base shard is copied once to the output
and patched in place (same names, dtypes, shapes, byte ranges), every other file is hard-linked. Output is built in
<out>.tmp and renamed to <out> only after a full byte-level verification, so <out> existing means the merge is good.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import struct
import sys
import time
from dataclasses import dataclass, field
from typing import Any

ATTN_MODULES = ("wq_du", "wk_dv", "wv_dv", "gate_attn", "wo_ud", "wr_du")
EXPERT_MODULES = ("w1", "w2", "w3")
CKPT_PREFIX = "model.llm.layers."
NAME_RE = re.compile(r"^(?:base_model\.model\.)?(?:model\.)?language_model\.layers\.(\d+)\."
                     r"(?:attn\.(" + "|".join(ATTN_MODULES) + r")|mlp\.experts\.(w1|w2|w3))\.lora_(A|B)\.weight$")
E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)
E4M3_MAX_CODE = 0x7E  # 448.0; 0x7F is NaN in float8_e4m3fn
NVFP4 = "nvfp4"
MXFP8 = "mxfp8"


# ---------------------------------------------------------------- small helpers

def say(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def read_header(path: str) -> tuple[int, dict[str, Any]]:
    """(byte offset where tensor data starts, header dict) of a safetensors file."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return 8 + n, json.loads(f.read(n))


def tensor_bytes(t: Any) -> bytes:
    import torch
    return t.contiguous().cpu().view(torch.uint8).numpy().tobytes()


def fail(msg: str) -> None:
    raise SystemExit(f"FAIL merge_adapter: {msg}")


# ---------------------------------------------------------------- adapter -> checkpoint placement

@dataclass(frozen=True)
class Module:
    layer: int
    name: str            # attention module name, or w1 / w2 / w3
    kind: str            # attn | expert
    a_key: str
    b_key: str


def parse_name(name: str) -> tuple[int, str, str, str] | None:
    """(layer, module, kind, A|B) for an adapter tensor name, None if it cannot be placed."""
    m = NAME_RE.match(name)
    if not m:
        return None
    layer, attn, expert, ab = m.groups()
    return int(layer), attn or expert, "attn" if attn else "expert", ab


def group_modules(names: list[str]) -> list[Module]:
    """Pairs lora_A / lora_B per module; any unplaceable or unpaired tensor aborts."""
    parts: dict[tuple[int, str, str], dict[str, str]] = {}
    bad = []
    for n in names:
        p = parse_name(n)
        if p is None:
            bad.append(n)
            continue
        parts.setdefault((p[0], p[1], p[2]), {})[p[3]] = n
    if bad:
        fail(f"{len(bad)} adapter tensor(s) cannot be placed, e.g. {bad[:5]}")
    unpaired = [k for k, v in parts.items() if set(v) != {"A", "B"}]
    if unpaired:
        fail(f"modules without both lora_A and lora_B: {unpaired[:5]}")
    return [Module(l, n, k, v["A"], v["B"]) for (l, n, k), v in sorted(parts.items())]


def ckpt_key_attn(m: Module) -> str:
    return f"{CKPT_PREFIX}{m.layer}.attn.{m.name}.weight"


def ckpt_key_expert(layer: int, which: str) -> str:
    """which = w13 | w2; the stored weight tensor (scales are <key>_scale etc.)."""
    return f"{CKPT_PREFIX}{layer}.mlp.experts.{which}_weight"


def lora_scale(cfg: dict[str, Any], user_scale: float) -> float:
    """alpha/r * user scale. Patterns, rsLoRA and DoRA change the maths, so they abort rather than merge wrongly.
    UNVERIFIED: that the training service applies the same alpha/r to its expert LoRAs as to the attention ones
    (PEFT config is global; the expert layout is the service's own). Fallback: --scale to rescale everything."""
    for k in ("rank_pattern", "alpha_pattern"):
        if cfg.get(k):
            fail(f"adapter_config {k} is set; per-module ranks are not supported")
    if cfg.get("use_rslora") or cfg.get("use_dora"):
        fail("rsLoRA / DoRA adapters are not supported")
    if cfg.get("peft_type", "LORA") != "LORA":
        fail(f"peft_type {cfg.get('peft_type')} is not LORA")
    return float(cfg["lora_alpha"]) / float(cfg["r"]) * user_scale


# ---------------------------------------------------------------- quant formats (from the checkpoint's own config)

def quant_formats(base: str) -> dict[str, str]:
    """weight tensor name -> nvfp4 | mxfp8, from hf_quant_config.json parameter groups."""
    path = os.path.join(base, "hf_quant_config.json")
    if not os.path.exists(path):
        fail(f"{path} missing: cannot establish the expert format")
    out: dict[str, str] = {}
    for g in json.load(open(path))["model"]["parameter_groups"]:
        wt = g.get("weight_transform")
        if wt == "nvfp4_nearest":
            if g.get("weight_scale_format") != "amax_denominator" or g.get("weight_block_shape") != [1, 16]:
                fail(f"unexpected NVFP4 group settings: { {k: v for k, v in g.items() if k != 'tensors'} }")
            out.update({t: NVFP4 for t in g["tensors"]})
        elif wt == "mxfp8_nearest":
            out.update({t: MXFP8 for t in g["tensors"]})
    return out


# ---------------------------------------------------------------- NVFP4 / MXFP8 codecs (torch)

def e2m1_round(x: Any) -> Any:
    """Nearest-even onto {0,.5,1,1.5,2,3,4,6}, saturating at 6 (ported from scripts/parity/hf_load.py _e2m1_round).
    The sign follows signbit(x) so -0.0 stays -0.0 and stored -0 codes survive the zero-delta round trip."""
    import torch
    a = x.abs()
    r = torch.where(a <= 0.25, 0.0, torch.where(a < 0.75, 0.5, torch.where(a <= 1.25, 1.0, torch.where(
        a < 1.75, 1.5, torch.where(a <= 2.5, 2.0, torch.where(a < 3.5, 3.0, torch.where(a <= 5.0, 4.0, 6.0)))))))
    return torch.where(torch.signbit(x), -r, r)


def e2m1_codes(r: Any) -> Any:
    """Grid values (output of e2m1_round) -> 4-bit codes: magnitude index 0..7, bit 3 = sign."""
    import torch
    lut = torch.zeros(13, dtype=torch.uint8, device=r.device)
    for i, v in enumerate(E2M1_VALUES[:8]):
        lut[int(v * 2)] = i
    mag = lut[(r.abs() * 2).long()]
    return mag | (torch.signbit(r).to(torch.uint8) << 3)


def nvfp4_dequant(codes: Any, scale: Any, factor: Any) -> Any:
    """codes u8 [n,R,C/2], scale e4m3 [n,R,C/16], factor f32 [n] -> f32 [n,R,C]; same op order as transformers."""
    import torch
    grid = torch.tensor(E2M1_VALUES, device=codes.device)
    b = codes.long()
    vals = torch.stack((grid[b & 0xF], grid[b >> 4]), dim=-1)          # even element in the low nibble
    n, rows, nb = scale.shape
    vals = vals.reshape(n, rows, nb, 16) * scale.float()[..., None]
    return (vals * factor.float()[:, None, None, None]).reshape(n, rows, nb * 16)


def e4m3_ceil(x: Any) -> Any:
    """Smallest float8_e4m3fn value >= x (x >= 0), capped at 448."""
    import torch
    c = x.clamp(max=448.0).to(torch.float8_e4m3fn)
    low = c.float() < x
    u = c.view(torch.uint8).clone()
    u[low] = torch.clamp(u[low].int() + 1, max=E4M3_MAX_CODE).to(torch.uint8)
    return u.view(torch.float8_e4m3fn)


@dataclass
class QStats:
    saturated: int = 0
    bumped_blocks: int = 0


# ---------------------------------------------------------------- stochastic rounding
# Stored base weights sit exactly on the storage grid, so round-to-nearest drops every change under half a step (the
# pod run kept 0.2% of the expert delta and ~23% of attention elements). Stochastic rounding picks one of the two
# neighbouring representable values with probability proportional to proximity, so E[stored] = W + delta exactly.

def rng(seed: int, name: str, idx: int) -> Any:
    """Generator keyed by (seed, tensor name, expert index): reproducible and independent of chunk size."""
    import torch
    return torch.Generator().manual_seed(int(hashlib.sha256(f"{seed}|{name}|{idx}".encode()).hexdigest()[:15], 16))


def uniform_chunk(seed: int, name: str, e0: int, e1: int, tail: tuple[int, ...], dev: str) -> Any:
    """U[0,1) of shape [e1-e0, *tail], one generator per expert."""
    import torch
    return torch.stack([torch.rand(tail, generator=rng(seed, name, e)) for e in range(e0, e1)]).to(dev)


def bf16_stochastic(x: Any, g: Any) -> Any:
    """f32 -> bf16 by adding uniform 16-bit noise to the dropped mantissa bits and truncating. Values already exact
    in bf16 (low 16 bits zero) cannot move, so zero-delta elements stay byte-identical."""
    import torch
    bits = x.float().contiguous().view(torch.int32)
    r = torch.randint(0, 1 << 16, tuple(bits.shape), generator=g, dtype=torch.int32).to(bits.device)
    return ((bits + r) & -65536).view(torch.float32).to(torch.bfloat16)


def e2m1_stochastic(y: Any, u: Any) -> Any:
    """y in grid units (|y| clamped to 6) -> lower or upper E2M1 neighbour with P(upper) = proximity."""
    import torch
    grid = torch.tensor(E2M1_VALUES[:8], device=y.device)
    a = y.abs().clamp(max=6.0)
    i = (torch.bucketize(a, grid, right=True) - 1).clamp(0, 7)
    lo, hi = grid[i], grid[(i + 1).clamp(max=7)]
    p = torch.where(hi > lo, (a - lo) / torch.where(hi > lo, hi - lo, 1.0), torch.zeros_like(a))
    r = torch.where(u < p, hi, lo)
    return torch.where(torch.signbit(y), -r, r)


def e4m3_stochastic(y: Any, u: Any) -> Any:
    """y (|y| clamped to 448) -> lower or upper float8_e4m3fn neighbour (as f32) with P(upper) = proximity."""
    import torch
    a = y.abs().clamp(max=448.0)
    m = a.to(torch.float8_e4m3fn)
    code = m.view(torch.uint8).int()
    below = m.float() <= a
    lo_c = torch.where(below, code, code - 1).clamp(0, E4M3_MAX_CODE)
    hi_c = torch.where(below, code + 1, code).clamp(0, E4M3_MAX_CODE)
    lo = lo_c.to(torch.uint8).view(torch.float8_e4m3fn).float()
    hi = hi_c.to(torch.uint8).view(torch.float8_e4m3fn).float()
    p = torch.where(hi > lo, (a - lo) / torch.where(hi > lo, hi - lo, 1.0), torch.zeros_like(a))
    r = torch.where(u < p, hi, lo)
    return torch.where(torch.signbit(y), -r, r)


@dataclass(frozen=True)
class Rounding:
    """How to round onto the stored grid. keep marks elements whose delta is exactly zero: they always take the
    nearest value, which the zero-delta check proves is the stored one (so they stay byte-identical)."""
    mode: str = "nearest"            # nearest | stochastic
    u: Any = None                    # U[0,1) shaped like the weight chunk (stochastic only)
    keep: Any = None                 # bool, shaped like the weight chunk


def nvfp4_quant(x: Any, scale: Any, factor: Any, mode: str,
                rnd: Rounding = Rounding()) -> tuple[Any, Any, QStats]:
    """f32 [n,R,C] on the given scales -> (codes u8 [n,R,C/2], scale e4m3, stats)."""
    import torch
    n, rows, cols = x.shape
    xb = x.reshape(n, rows, cols // 16, 16)
    f = factor.float()[:, None, None]
    st = QStats()
    if mode == "bump":
        need = xb.abs().amax(-1) / (6.0 * f)
        grow = need > scale.float()
        st.bumped_blocks = int(grow.sum())
        if st.bumped_blocks:
            scale = torch.where(grow, e4m3_ceil(need).float(), scale.float()).to(torch.float8_e4m3fn)
    denom = (scale.float() * f)[..., None]
    y = torch.where(denom > 0, xb / torch.where(denom > 0, denom, 1.0), torch.zeros_like(xb))
    st.saturated = int((y.abs() > 6.0 * (1 + 1e-5)).sum()) + int(((denom == 0) & (xb != 0)).sum())  # 1e-5: fp32 division noise
    r = e2m1_round(y)
    if rnd.mode == "stochastic":
        sr = e2m1_stochastic(y, rnd.u.reshape(y.shape))
        r = sr if rnd.keep is None else torch.where(rnd.keep.reshape(y.shape), r, sr)
    c = e2m1_codes(r).reshape(n, rows, cols)
    packed = c[..., 0::2] | (c[..., 1::2] << 4)
    return packed.contiguous(), scale, st


def mxfp8_dequant(vals: Any, scale: Any) -> Any:
    """vals e4m3 [n,R,C], scale u8 [n,R,C/32] -> f32 [n,R,C]."""
    import torch
    n, rows, cols = vals.shape
    v = vals.float().reshape(n, rows, cols // 32, 32)
    return (v * torch.exp2(scale.float() - 127)[..., None]).reshape(n, rows, cols)


def uniform_over_32_rows(scale: Any) -> bool:
    """True when every 32-row group shares its scales: the stored per-row scales then encode 32x32 blocks."""
    n, rows, nb = scale.shape
    if rows % 32:
        return False
    g = scale.reshape(n, rows // 32, 32, nb)
    return bool((g == g[:, :, :1]).all())


def mxfp8_quant(x: Any, scale: Any, mode: str, rnd: Rounding = Rounding()) -> tuple[Any, Any, QStats]:
    """f32 [n,R,C] on the given power-of-two scales -> (e4m3 values, u8 scale, stats). In bump mode a block whose
    original scales were uniform over 32 rows (the config's [32,32] block) is bumped as a whole 32x32 block."""
    import torch
    n, rows, cols = x.shape
    xb = x.reshape(n, rows, cols // 32, 32)
    st = QStats()
    if mode == "bump":
        amax = xb.abs().amax(-1)
        need = torch.where(amax > 0, torch.ceil(torch.log2(amax / 448.0)) + 127, torch.zeros_like(amax))
        need = need.clamp(0, 254)
        if uniform_over_32_rows(scale):
            need = need.reshape(n, rows // 32, 32, -1).amax(2, keepdim=True).expand(-1, -1, 32, -1).reshape(n, rows, -1)
        grow = need > scale.float()
        st.bumped_blocks = int(grow.sum())
        scale = torch.where(grow, need, scale.float()).to(torch.uint8)
    y = xb / torch.exp2(scale.float() - 127)[..., None]
    st.saturated = int((y.abs() > 448.0 * (1 + 1e-6)).sum())
    q = y.clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    if rnd.mode == "stochastic":
        sr = e4m3_stochastic(y, rnd.u.reshape(y.shape))
        if rnd.keep is not None:
            sr = torch.where(rnd.keep.reshape(y.shape), q.float(), sr)   # f32 holds every e4m3 value exactly
        q = sr.to(torch.float8_e4m3fn)
    return q.reshape(n, rows, cols), scale, st


# ---------------------------------------------------------------- expert deltas

def expert_delta_w13(ad: dict[str, Any], e0: int, e1: int, s: float, swap: bool, dev: str) -> Any:
    """Per-expert gate/up delta for experts e0:e1, interleaved like the stored w13 rows ([n, 2I, D], f32).
    w1/w3: lora_A [1 or E, r, D] (shared over experts when dim0 = 1), lora_B [E, I, r]."""
    import torch
    g_name, u_name = ("w3", "w1") if swap else ("w1", "w3")

    def one(name: str) -> Any:
        a, b = ad[name]
        a = a if a.shape[0] == 1 else a[e0:e1]
        b = b if b.shape[0] == 1 else b[e0:e1]
        return torch.matmul(b.to(dev, torch.float32), a.to(dev, torch.float32)) * s     # [n, I, D]

    gate, up = one(g_name), one(u_name)
    n, inter, d = gate.shape
    return torch.stack((gate, up), dim=2).reshape(n, 2 * inter, d)


def expert_delta_w2(ad: dict[str, Any], e0: int, e1: int, s: float, dev: str) -> Any:
    """Down-projection delta [n, D, I]: lora_A [E, r, I] per expert, lora_B [1 or E, D, r]."""
    import torch
    a, b = ad["w2"]
    a = a if a.shape[0] == 1 else a[e0:e1]
    b = b if b.shape[0] == 1 else b[e0:e1]
    return torch.matmul(b.to(dev, torch.float32), a.to(dev, torch.float32)) * s


# ---------------------------------------------------------------- survival accounting

SUM_KEYS = ("d2", "err2", "proj", "w2", "r2")


def chunk_sums(w: Any, d: Any, q: Any) -> dict[str, float]:
    """Sums for the fidelity metrics; realised change r = q - w, intended change d (computed once per chunk)."""
    wd, dd = w.double(), d.double()
    r = q.double() - wd
    return {"d2": float((dd * dd).sum()), "err2": float(((r - dd) ** 2).sum()), "proj": float((r * dd).sum()),
            "w2": float((wd * wd).sum()), "r2": float((r * r).sum())}


@dataclass
class Survival:
    d2: float = 0.0          # |delta|^2 (intended)
    err2: float = 0.0        # |Q(W+d) - (W+d)|^2
    proj: float = 0.0        # <Q(W+d) - W, d>
    w2: float = 0.0          # |W|^2
    r2: float = 0.0          # |Q(W+d) - W|^2 (realised change)
    changed: int = 0         # stored weight bytes (or bf16 elements) that changed
    total: int = 0
    saturated: int = 0
    bumped_blocks: int = 0

    def add_sums(self, sums: dict[str, float], changed: int, total: int, st: QStats) -> None:
        for k in SUM_KEYS:
            setattr(self, k, getattr(self, k) + sums[k])
        self.changed, self.total = self.changed + changed, self.total + total
        self.saturated += st.saturated
        self.bumped_blocks += st.bumped_blocks

    def add(self, w: Any, d: Any, q: Any, changed: int, total: int, st: QStats) -> None:
        self.add_sums(chunk_sums(w, d, q), changed, total, st)

    def summary(self) -> dict[str, float]:
        nan = float("nan")
        d = math.sqrt(self.d2) if self.d2 > 0 else nan
        return {"survival": 1.0 - math.sqrt(self.err2) / d,
                # projected_survival = scale of the realised change along the delta, <r,d>/|d|^2
                "projected_survival": self.proj / self.d2 if self.d2 else nan,
                "change_ratio": math.sqrt(self.r2) / d,                       # |realised| / |intended|
                "cosine": self.proj / math.sqrt(self.r2 * self.d2) if self.r2 * self.d2 > 0 else nan,
                "rel_delta": d / math.sqrt(self.w2) if self.w2 else nan,
                "changed_frac": self.changed / max(self.total, 1), "saturated": self.saturated,
                "bumped_blocks": self.bumped_blocks}


def merge_survival(items: list[Survival]) -> Survival:
    out = Survival()
    for s in items:
        out.add_sums({k: getattr(s, k) for k in SUM_KEYS}, s.changed, s.total, QStats(s.saturated, s.bumped_blocks))
    return out


class Fidelity:
    """Survival accumulators per (rounding mode, group); groups such as attn, attn.wq_du, experts.nvfp4, all."""

    def __init__(self) -> None:
        self.acc: dict[str, dict[str, Survival]] = {}

    def add(self, mode: str, groups: list[str], sums: dict[str, float], changed: int, total: int, st: QStats) -> None:
        for g in groups + ["all"]:
            self.acc.setdefault(mode, {}).setdefault(g, Survival()).add_sums(sums, changed, total, st)

    def summary(self) -> dict[str, dict[str, dict[str, float]]]:
        return {m: {g: s.summary() for g, s in sorted(gs.items())} for m, gs in self.acc.items()}


# ---------------------------------------------------------------- spectral check (cheap, sampled)
SPECTRAL_ATTN_LAYERS = (0, 11, 23)
SPECTRAL_EXPERT_LAYERS = (0, 10, 19, 21)       # three NVFP4 layers and one MXFP8 layer
SPECTRAL_EXPERTS = (0, 77)
SPECTRAL_RANK = 32


def spectral_ratios(d: Any, err: Any, g: Any) -> dict[str, float]:
    """Does the low-rank signal stand above the rounding noise? d = intended delta [out,in], err = realised - intended.
    ratio_top = |err V| / |d V| for V = top right-singular vectors of d (rank 32); ratio_rand the same for random
    unit vectors; noise_top_over_rand = |err V| / |err V_rand| (1 = the noise is isotropic)."""
    import torch
    d, err = d.double().cpu(), err.double().cpu()
    k = min(SPECTRAL_RANK, min(d.shape))
    # Randomised SVD (q = k + 16, 2 power iterations): exact for the rank-32 attention/w2 deltas, an approximation of
    # the top 32 for w13, whose interleaved gate+up delta has rank up to 64. Cheap by design; it is a diagnostic.
    torch.manual_seed(0)                                     # svd_lowrank draws from the global generator
    v = torch.svd_lowrank(d, q=min(k + 16, min(d.shape)), niter=2)[2][:, :k]
    vr = torch.randn(d.shape[1], k, generator=g, dtype=torch.float64)
    vr = vr / vr.norm(dim=0, keepdim=True)
    dv, ev, dr, er = (d @ v).norm(), (err @ v).norm(), (d @ vr).norm(), (err @ vr).norm()
    return {"ratio_top": float(ev / dv), "ratio_rand": float(er / dr), "noise_top_over_rand": float(ev / er)}


# ---------------------------------------------------------------- checkpoint I/O

@dataclass
class Ckpt:
    base: str
    weight_map: dict[str, str]
    headers: dict[str, tuple[int, dict[str, Any]]] = field(default_factory=dict)

    def meta(self, key: str) -> tuple[str, int, dict[str, Any]]:
        if key not in self.weight_map:
            fail(f"checkpoint has no tensor {key} (index lists e.g. {sorted(self.weight_map)[:3]})")
        fn = self.weight_map[key]
        if fn not in self.headers:
            self.headers[fn] = read_header(os.path.join(self.base, fn))
        start, hdr = self.headers[fn]
        return fn, start, hdr[key]


def load_ckpt(base: str) -> Ckpt:
    idx = os.path.join(base, "model.safetensors.index.json")
    if not os.path.exists(idx):
        fail(f"{idx} missing")
    return Ckpt(base, json.load(open(idx))["weight_map"])


def link_or_copy_tree(base: str, dst: str, copy_files: set[str]) -> None:
    """Hard-link every file of base into dst (copy where linking fails), real copies for the shards we patch."""
    for root, dirs, files in os.walk(base):
        dirs[:] = [d for d in dirs if d != ".cache"]
        rel = os.path.relpath(root, base)
        os.makedirs(os.path.join(dst, rel), exist_ok=True)
        for f in files:
            s, d = os.path.join(root, f), os.path.join(dst, rel, f)
            relf = os.path.normpath(os.path.join(rel, f))
            if relf in copy_files:
                say(f"copying {relf} ({os.path.getsize(s) / 1e9:.1f} GB)")
                shutil.copyfile(s, d)
                continue
            try:
                os.link(s, d)
            except OSError:
                shutil.copy2(s, d)


def write_range(fh: Any, data_start: int, meta: dict[str, Any], offset: int, raw: bytes) -> None:
    s, e = meta["data_offsets"]
    if offset + len(raw) > e - s:
        fail("write past the end of a tensor's byte range")
    fh.seek(data_start + s + offset)
    fh.write(raw)


def check_dtype_shape(key: str, meta: dict[str, Any], dtype: str, shape: list[int]) -> None:
    if meta["dtype"] != dtype or meta["shape"] != shape:
        fail(f"{key}: checkpoint has {meta['dtype']} {meta['shape']}, expected {dtype} {shape}")


# ---------------------------------------------------------------- attention merge

def check_attn_shapes(ck: Ckpt, mods: list[Module], ad: dict[str, Any]) -> None:
    """Before any copying: every attention target exists, is BF16 and matches [B.out, A.in]."""
    for m in mods:
        key = ckpt_key_attn(m)
        a, b = ad[m.a_key], ad[m.b_key]
        if a.dim() != 2 or b.dim() != 2 or a.shape[0] != b.shape[1]:
            fail(f"{m.a_key}: lora shapes A{list(a.shape)} B{list(b.shape)} do not pair")
        check_dtype_shape(key, ck.meta(key)[2], "BF16", [b.shape[0], a.shape[1]])


def attn_candidates(w: Any, delta: Any, key: str, seed: int) -> dict[str, Any]:
    """Both roundings of W + delta to BF16. nearest: fp64 sum, one rounding (as v3/merge_stack.py; unchanged).
    stochastic: f32 sum, unbiased rounding keyed by (seed, tensor name)."""
    import torch
    x = torch.where(delta == 0, w.float(), w.float() + delta.float())   # keeps -0.0 elements byte-identical
    return {"nearest": (w.double() + delta).to(torch.bfloat16),
            "stochastic": bf16_stochastic(x, rng(seed, key, 0))}


def add_delta(key: str, acc: Any, term: Any) -> Any:
    """acc += term; a shape disagreement would broadcast silently, so it aborts naming the tensor."""
    if tuple(acc.shape) != tuple(term.shape):
        fail(f"{key}: adapter deltas disagree in shape {list(acc.shape)} vs {list(term.shape)}")
    return acc.add_(term)        # acc is always a fresh matmul result, so in place is safe and saves a copy


def attn_groups(ads: list[Adapter]) -> dict[str, list[tuple[int, Module]]]:
    """checkpoint key -> [(adapter index, module)], in (layer, module) order like the single-adapter loop."""
    pairs = sorted(((m.layer, m.name, i), m) for i, ad in enumerate(ads) for m in ad.attn)
    out: dict[str, list[tuple[int, Module]]] = {}
    for (_, _, i), m in pairs:
        out.setdefault(ckpt_key_attn(m), []).append((i, m))
    return out


def attn_delta(key: str, terms: list[tuple[int, Module]], ads: list[Adapter]) -> Any:
    """sum_i s_i * B_i @ A_i in f64. With one term this is exactly the old expression (no extra op)."""
    delta = None
    for i, m in terms:
        la, lb = ads[i].tensors[m.a_key], ads[i].tensors[m.b_key]
        term = ads[i].scale * (lb.double() @ la.double())
        delta = term if delta is None else add_delta(key, delta, term)
    return delta


def merge_attention(ck: Ckpt, ads: list[Adapter], fhs: dict[str, Any], report: dict[str, Any],
                    hashes: dict[str, Any], a: argparse.Namespace, fid: Fidelity) -> None:
    """W' = round_bf16(W + sum_i s_i * B_i @ A_i) per attention tensor; both roundings are measured, --rounding is
    written."""
    from safetensors import safe_open
    for key, terms in attn_groups(ads).items():
        m = terms[0][1]
        fn, start, meta = ck.meta(key)
        with safe_open(os.path.join(ck.base, fn), "pt") as f:
            w = f.get_tensor(key)
        delta = attn_delta(key, terms, ads)
        cands = attn_candidates(w, delta, key, a.seed)
        entry: dict[str, Any] = {"adapter": m.a_key.rsplit(".lora_A", 1)[0],
                                 "rel_delta": float(delta.norm() / w.double().norm())}
        if len(ads) > 1:
            entry["adapters"] = [i for i, _ in terms]
        for mode, wn in cands.items():
            changed = int((wn != w).sum())
            fid.add(mode, ["attn", f"attn.{m.name}"], chunk_sums(w, delta, wn), changed, w.numel(), QStats())
            entry[f"changed_frac_{mode}"] = changed / w.numel()
            if a.spectral and m.layer in SPECTRAL_ATTN_LAYERS:
                err = wn.double() - w.double() - delta
                report["spectral"].append({"tensor": key, "expert": None, "mode": mode,
                                           **spectral_ratios(delta, err, rng(a.seed, key + "#spec", 0))})
        wn = cands[a.rounding]
        raw = tensor_bytes(wn)
        write_range(fhs[fn], start, meta, 0, raw)
        hashes[key] = hashlib.sha256(raw)
        report["attn"][key] = entry


# ---------------------------------------------------------------- expert merge

@dataclass(frozen=True)
class ExpertTensor:
    key: str
    fmt: str
    layer: int
    which: str      # w13 | w2


def expert_targets(ck: Ckpt, fmts: dict[str, str], layers: list[int]) -> list[ExpertTensor]:
    out = []
    for layer in layers:
        for which in ("w13", "w2"):
            key = ckpt_key_expert(layer, which)
            ck.meta(key)
            if key not in fmts:
                fail(f"{key} is not in any NVFP4/MXFP8 group of hf_quant_config.json")
            out.append(ExpertTensor(key, fmts[key], layer, which))
    return out


def expected_shapes(t: ExpertTensor, n_exp: int, d: int, inter: int) -> dict[str, tuple[str, list[int]]]:
    rows, cols = (2 * inter, d) if t.which == "w13" else (d, inter)
    if t.fmt == NVFP4:
        return {t.key: ("U8", [n_exp, rows, cols // 2]), t.key + "_scale": ("F8_E4M3", [n_exp, rows, cols // 16]),
                t.key + "_amax": ("F32", [n_exp])}
    return {t.key: ("F8_E4M3", [n_exp, rows, cols]), t.key + "_scale": ("U8", [n_exp, rows, cols // 32])}


def read_chunk(f: Any, key: str, e0: int, e1: int, dev: str) -> Any:
    return f.get_slice(key)[e0:e1].to(dev)


def dequant_chunk(t: ExpertTensor, f: Any, e0: int, e1: int, factor: Any, dev: str) -> tuple[Any, Any, Any]:
    """(stored weight chunk, stored scale chunk, dequantised f32)."""
    q, sc = read_chunk(f, t.key, e0, e1, dev), read_chunk(f, t.key + "_scale", e0, e1, dev)
    w = nvfp4_dequant(q, sc, factor[e0:e1]) if t.fmt == NVFP4 else mxfp8_dequant(q, sc)
    return q, sc, w


def requant(t: ExpertTensor, x: Any, sc: Any, factor: Any, e0: int, e1: int, mode: str,
            rnd: Rounding = Rounding()) -> tuple[Any, Any, QStats]:
    if t.fmt == NVFP4:
        return nvfp4_quant(x, sc, factor[e0:e1], mode, rnd)
    return mxfp8_quant(x, sc, mode, rnd)


def dequant(t: ExpertTensor, q: Any, sc: Any, factor: Any, e0: int, e1: int) -> Any:
    return nvfp4_dequant(q, sc, factor[e0:e1]) if t.fmt == NVFP4 else mxfp8_dequant(q, sc)


@dataclass
class ChunkCtx:
    t: ExpertTensor
    e0: int
    e1: int
    q: Any               # stored weight chunk
    sc: Any              # stored scale chunk
    w: Any               # dequantised f32
    d: Any               # intended delta f32
    factor: Any


def requant_both(c: ChunkCtx, a: argparse.Namespace) -> dict[str, tuple[Any, Any, QStats]]:
    """W + delta re-encoded with nearest and stochastic rounding (same scale policy); zero-delta elements kept."""
    import torch
    keep = c.d == 0
    rnd = Rounding("stochastic", uniform_chunk(a.seed, c.t.key, c.e0, c.e1, tuple(c.w.shape[1:]), a.device), keep)
    x = c.w + c.d
    x_keep = torch.where(keep, c.w, x)        # -0.0 + 0.0 = +0.0 would flip stored -0 codes; pin them
    return {"nearest": requant(c.t, x, c.sc, c.factor, c.e0, c.e1, a.expert_scales),
            "stochastic": requant(c.t, x_keep, c.sc, c.factor, c.e0, c.e1, a.expert_scales, rnd)}


def measure_chunk(c: ChunkCtx, outs: dict[str, tuple[Any, Any, QStats]], a: argparse.Namespace, fid: Fidelity,
                  spectral: list[dict[str, Any]]) -> None:
    groups = ["experts", f"experts.{c.t.fmt}", f"experts.layer{c.t.layer}"]
    for mode, (qn, scn, st) in outs.items():
        back = dequant(c.t, qn, scn, c.factor, c.e0, c.e1)
        fid.add(mode, groups, chunk_sums(c.w, c.d, back), changed_bytes(qn, c.q), qn.numel(), st)
        if not a.spectral or c.t.layer not in SPECTRAL_EXPERT_LAYERS:
            continue
        for e in SPECTRAL_EXPERTS:
            if c.e0 <= e < c.e1:
                k = e - c.e0
                err = back[k].double() - c.w[k].double() - c.d[k].double()
                spectral.append({"tensor": c.t.key, "expert": e, "mode": mode,
                                 **spectral_ratios(c.d[k], err, rng(a.seed, c.t.key + "#spec", e))})


def nvfp4_factor(t: ExpertTensor, f: Any, dev: str) -> Any:
    """Per-expert amax / outer_scale_denominator (denominator may be a single value)."""
    if t.fmt != NVFP4:
        return None
    return (f.get_tensor(t.key + "_amax").float() / f.get_tensor(t.key + "_outer_scale_denominator").float()).to(dev)


ExpertSrc = tuple[dict[int, dict[str, tuple[Any, Any]]], float]     # (per-layer expert adapter, scale), one per adapter


def expert_delta(t: ExpertTensor, srcs: list[ExpertSrc], e0: int, e1: int, a: argparse.Namespace) -> Any:
    """sum_i of each adapter's chunk delta in f32 (adapters without this layer skip). With one adapter this is exactly
    the old single expression. Extra peak memory: the accumulated delta chunk while the next term is built."""
    d = None
    for ead, s in srcs:
        if t.layer not in ead:
            continue
        term = expert_delta_w13(ead[t.layer], e0, e1, s, a.swap_w1_w3, a.device) if t.which == "w13" \
            else expert_delta_w2(ead[t.layer], e0, e1, s, a.device)
        d = term if d is None else add_delta(t.key, d, term)
    if d is None:
        fail(f"{t.key}: no adapter covers layer {t.layer}")
    return d


def load_chunk(t: ExpertTensor, f: Any, e0: int, e1: int, factor: Any, srcs: list[ExpertSrc],
               a: argparse.Namespace) -> ChunkCtx:
    """Read + dequantise one expert chunk, prove the format by a zero-delta byte-exact re-encode, build the delta."""
    q, sc, w = dequant_chunk(t, f, e0, e1, factor, a.device)
    q0, sc0, _ = requant(t, w, sc, factor, e0, e1, "keep")
    if tensor_bytes(q0) != tensor_bytes(q) or tensor_bytes(sc0) != tensor_bytes(sc):
        fail(f"{t.key} experts {e0}:{e1}: zero-delta re-encode does not reproduce the stored bytes "
             f"({t.fmt} format misread); nothing written for this variant")
    d = expert_delta(t, srcs, e0, e1, a)
    if d.shape != w.shape:
        fail(f"{t.key}: delta shape {tuple(d.shape)} != weight shape {tuple(w.shape)}")
    return ChunkCtx(t, e0, e1, q, sc, w, d, factor)


def merge_expert_tensor(t: ExpertTensor, ck: Ckpt, srcs: list[ExpertSrc], a: argparse.Namespace,
                        fhs: dict[str, Any], hashes: dict[str, Any], fid: Fidelity,
                        spectral: list[dict[str, Any]]) -> tuple[Survival, int]:
    """Dequantise, add the per-expert delta, requantise (both roundings measured, --rounding written), patch the
    copied shard in place, chunk by chunk. hashes[key] accumulates sha256 of the bytes each tensor should hold."""
    from safetensors import safe_open
    hashes[t.key], hashes[t.key + "_scale"] = hashlib.sha256(), hashlib.sha256()
    fn, start, meta = ck.meta(t.key)
    _, s_start, s_meta = ck.meta(t.key + "_scale")
    n_exp = meta["shape"][0]
    surv, rescaled = Survival(), 0
    with safe_open(os.path.join(ck.base, fn), "pt") as f:
        factor = nvfp4_factor(t, f, a.device)
        for e0 in range(0, n_exp, a.expert_chunk):
            c = load_chunk(t, f, e0, min(n_exp, e0 + a.expert_chunk), factor, srcs, a)
            outs = requant_both(c, a)
            measure_chunk(c, outs, a, fid, spectral)
            qn, scn, st = outs[a.rounding]
            raw_q, raw_s = tensor_bytes(qn), tensor_bytes(scn)
            surv.add(c.w, c.d, dequant(t, qn, scn, factor, c.e0, c.e1), changed_bytes(qn, c.q), len(raw_q), st)
            per_q, per_s = len(raw_q) // (c.e1 - e0), len(raw_s) // (c.e1 - e0)
            write_range(fhs[fn], start, meta, e0 * per_q, raw_q)
            hashes[t.key].update(raw_q)
            hashes[t.key + "_scale"].update(raw_s)
            if raw_s != tensor_bytes(c.sc):
                rescaled += 1
                write_range(fhs[fn], s_start, s_meta, e0 * per_s, raw_s)
    return surv, rescaled


def changed_bytes(new: Any, old: Any) -> int:
    import torch
    return int((new.contiguous().view(torch.uint8) != old.contiguous().view(torch.uint8)).sum())


def load_expert_adapter(ad_all: dict[str, Any], mods: list[Module]) -> dict[int, dict[str, tuple[Any, Any]]]:
    out: dict[int, dict[str, tuple[Any, Any]]] = {}
    for m in mods:
        out.setdefault(m.layer, {})[m.name] = (ad_all[m.a_key], ad_all[m.b_key])
    for layer, d in out.items():
        if set(d) != set(EXPERT_MODULES):
            fail(f"layer {layer}: expert adapter has {sorted(d)}, need w1, w2, w3")
    return out


def check_expert_shapes(targets: list[ExpertTensor], ck: Ckpt, ead: dict[int, dict[str, tuple[Any, Any]]]) -> None:
    """Derive (E, I, D) from the adapter and require every stored expert tensor to match."""
    for t in targets:
        a1, b1 = ead[t.layer]["w1"]
        n_exp, inter, d = b1.shape[0], b1.shape[1], a1.shape[2]
        a2, b2 = ead[t.layer]["w2"]
        if tuple(a2.shape[1:]) != (a1.shape[1], inter) or b2.shape[1] != d or a1.shape[0] not in (1, n_exp):
            fail(f"layer {t.layer}: inconsistent expert adapter shapes w1 A{list(a1.shape)} B{list(b1.shape)} "
                 f"w2 A{list(a2.shape)} B{list(b2.shape)}")
        for k, (dt, shape) in expected_shapes(t, n_exp, d, inter).items():
            check_dtype_shape(k, ck.meta(k)[2], dt, shape)
        if t.fmt == NVFP4:
            den = ck.meta(t.key + "_outer_scale_denominator")[2]
            if den["dtype"] != "F32" or den["shape"] not in ([1], [n_exp], []):
                fail(f"{t.key}_outer_scale_denominator: unexpected {den['dtype']} {den['shape']}")


# ---------------------------------------------------------------- verification

def verify_output(base: str, out: str, expected: dict[str, str]) -> dict[str, int]:
    """Headers identical; every rewritten tensor reads back with the sha256 we wrote; every other tensor in the
    shards is byte-identical to the base (so only the intended bytes moved)."""
    ck = load_ckpt(base)
    res = {"rewritten_verified": 0, "untouched_identical": 0}
    for fn in sorted(set(ck.weight_map.values())):
        sa, ha = read_header(os.path.join(base, fn))
        sb, hb = read_header(os.path.join(out, fn))
        if ha != hb or sa != sb:
            fail(f"header of {fn} changed")
        with open(os.path.join(base, fn), "rb") as fa, open(os.path.join(out, fn), "rb") as fb:
            for k, m in ha.items():
                if k == "__metadata__":
                    continue
                s, e = m["data_offsets"]
                if k in expected:
                    if range_sha(fb, sb + s, e - s) != expected[k]:
                        fail(f"read-back mismatch for {k}")
                    res["rewritten_verified"] += 1
                elif not same_range(fa, fb, sa + s, sb + s, e - s):
                    fail(f"untouched tensor differs: {k}")
                else:
                    res["untouched_identical"] += 1
    if res["rewritten_verified"] != len(expected):
        fail(f"verified {res['rewritten_verified']} of {len(expected)} rewritten tensors")
    return res


def range_sha(fh: Any, offset: int, n: int, step: int = 1 << 26) -> str:
    h, done = hashlib.sha256(), 0
    while done < n:
        k = min(step, n - done)
        fh.seek(offset + done)
        h.update(fh.read(k))
        done += k
    return h.hexdigest()


def same_range(fa: Any, fb: Any, oa: int, ob: int, n: int, step: int = 1 << 26) -> bool:
    done = 0
    while done < n:
        k = min(step, n - done)
        fa.seek(oa + done)
        fb.seek(ob + done)
        if fa.read(k) != fb.read(k):
            return False
        done += k
    return True


# ---------------------------------------------------------------- main

def load_adapter(path: str) -> tuple[dict[str, Any], dict[str, Any]]:
    from safetensors.torch import load_file
    cfg_p, st_p = os.path.join(path, "adapter_config.json"), os.path.join(path, "adapter_model.safetensors")
    for p in (cfg_p, st_p):
        if not os.path.exists(p):
            fail(f"{p} missing")
    return json.load(open(cfg_p)), load_file(st_p)


def mapping_line(m: Module, swap: bool) -> str:
    """'adapter pattern -> checkpoint pattern' for one module, layer number replaced by N."""
    if m.kind == "attn":
        src, tgt = f"attn.{m.name}", ckpt_key_attn(m)
    else:
        src, tgt = f"mlp.experts.{m.name}", ckpt_key_expert(m.layer, "w2" if m.name == "w2" else "w13")
        gate = "w3" if swap else "w1"
        tgt += " (down)" if m.name == "w2" else (" rows 0::2 (gate)" if m.name == gate else " rows 1::2 (up)")
    return f"language_model.layers.N.{src} -> " + re.sub(r"layers\.\d+\.", "layers.N.", tgt)


def mapping_report(mods: list[Module], skipped: list[Module], swap: bool) -> list[str]:
    """One line per mapping with how many layers use it, for the log."""
    counts: dict[str, int] = {}
    for m in mods:
        line = mapping_line(m, swap)
        counts[line] = counts.get(line, 0) + 1
    out = [f"  {n:3d} x {line}" for line, n in sorted(counts.items())]
    if skipped:
        out.append(f"  {len(skipped):3d} x expert modules SKIPPED (--parts attn): "
                   + ", ".join(sorted({m.name for m in skipped})))
    return out


def plan(a: argparse.Namespace, ad: dict[str, Any]) -> tuple[list[Module], list[Module], list[Module]]:
    mods = group_modules(sorted(ad))
    attn = [m for m in mods if m.kind == "attn"]
    experts = [m for m in mods if m.kind == "expert"]
    if a.parts == "attn":
        return attn, [], experts
    return attn, experts, []


@dataclass(frozen=True)
class Adapter:
    """One loaded adapter with its effective scale and the modules it covers after --parts."""
    path: str
    cfg: dict[str, Any]
    tensors: dict[str, Any]
    user_scale: float
    scale: float                                     # alpha/r * user_scale
    attn: list[Module]
    experts: list[Module]
    skipped: list[Module]
    ead: dict[int, dict[str, tuple[Any, Any]]]       # layer -> w1/w2/w3 (A, B)


def adapter_paths(a: argparse.Namespace) -> list[str]:
    """--adapter is a list from the CLI; callers building a Namespace by hand (the selftest) may pass one string."""
    paths = [a.adapter] if isinstance(a.adapter, str) else list(a.adapter or [])
    if not paths:
        fail("no --adapter given")
    return paths


def adapter_scales(a: argparse.Namespace, n: int) -> list[float]:
    raw = getattr(a, "scales", "") or ""
    if not raw:
        return [a.scale] * n
    vals = [float(x) for x in raw.split(",")]
    if len(vals) != n:
        fail(f"--scales has {len(vals)} value(s) for {n} adapter(s)")
    return vals


def load_adapters(a: argparse.Namespace) -> list[Adapter]:
    paths = adapter_paths(a)
    out = []
    for path, us in zip(paths, adapter_scales(a, len(paths))):
        cfg, ad = load_adapter(path)
        attn, experts, skipped = plan(a, ad)
        ead = load_expert_adapter(ad, experts) if experts else {}
        out.append(Adapter(path, cfg, ad, us, lora_scale(cfg, us), attn, experts, skipped, ead))
        say(f"adapter {path}: {len(ad)} tensors, r={cfg['r']} alpha={cfg['lora_alpha']} -> scale {out[-1].scale}")
    return out


def check_adapter_shapes(ck: Ckpt, ads: list[Adapter], targets: list[ExpertTensor]) -> None:
    """Each adapter against the checkpoint on its own modules; with several, the failure also names the adapter."""
    for ad in ads:
        try:
            check_attn_shapes(ck, ad.attn, ad.tensors)
            check_expert_shapes([t for t in targets if t.layer in ad.ead], ck, ad.ead)
        except SystemExit as e:
            if len(ads) == 1:
                raise
            raise SystemExit(f"{e} [adapter {ad.path}]") from None


def adapter_report(ads: list[Adapter]) -> list[dict[str, Any]]:
    return [{"path": ad.path, "r": ad.cfg["r"], "lora_alpha": ad.cfg["lora_alpha"], "user_scale": ad.user_scale,
             "scale": ad.scale,
             "modules": sorted({ckpt_key_attn(m) for m in ad.attn}
                               | {f"{CKPT_PREFIX}{m.layer}.mlp.experts.{m.name}" for m in ad.experts}),
             "skipped_modules": [m.a_key for m in ad.skipped]} for ad in ads]


def base_report(a: argparse.Namespace, ads: list[Adapter]) -> dict[str, Any]:
    """One adapter: the old keys and values. Several: adapter is a list and scale/user_scale are null (the
    per-adapter values are under "adapters")."""
    one = len(ads) == 1
    mods = [m for ad in ads for m in ad.attn + ad.experts]
    skipped = [m for ad in ads for m in ad.skipped]
    return {"base": a.base, "adapter": ads[0].path if one else [ad.path for ad in ads], "parts": a.parts,
            "scale": ads[0].scale if one else None, "user_scale": ads[0].user_scale if one else None,
            "adapters": adapter_report(ads), "swap_w1_w3": a.swap_w1_w3, "expert_scales": a.expert_scales,
            "mapping": mapping_report(mods, skipped, a.swap_w1_w3), "rounding": a.rounding, "seed": a.seed,
            "attn": {}, "experts": {}, "skipped_modules": [m.a_key for m in skipped], "spectral": []}


def run_merge(a: argparse.Namespace) -> None:
    import torch
    torch.set_grad_enabled(False)
    t0 = time.time()
    ads = load_adapters(a)
    ck = load_ckpt(a.base)
    say("name mapping (adapter -> checkpoint):\n" + "\n".join(base_report(a, ads)["mapping"]))
    layers = sorted({layer for ad in ads for layer in ad.ead})
    targets = expert_targets(ck, quant_formats(a.base), layers) if layers else []
    check_adapter_shapes(ck, ads, targets)
    shards = {ck.weight_map[k] for k in attn_groups(ads)} | {ck.weight_map[t.key] for t in targets}
    tmp = a.out.rstrip("/") + ".tmp"
    prepare_out(a, tmp, shards)
    report = base_report(a, ads)
    hashes: dict[str, Any] = {}
    fid = Fidelity()
    fhs = {fn: open(os.path.join(tmp, fn), "r+b") for fn in shards}
    try:
        merge_attention(ck, ads, fhs, report, hashes, a, fid)
        say(f"attention: {len(report['attn'])} tensors merged ({a.rounding}); mean rel_delta "
            f"{sum(v['rel_delta'] for v in report['attn'].values()) / max(len(report['attn']), 1):.3e}")
        merge_all_experts(targets, ck, [(ad.ead, ad.scale) for ad in ads], a, fhs, report, hashes, fid)
    finally:
        for fh in fhs.values():
            fh.close()
    report["fidelity"] = fid.summary()
    say_fidelity(report)
    report["verify"] = verify_output(a.base, tmp, {k: h.hexdigest() for k, h in hashes.items()})
    report["seconds"] = round(time.time() - t0, 1)
    finish(a, tmp, report)


def merge_all_experts(targets: list[ExpertTensor], ck: Ckpt, srcs: list[ExpertSrc],
                      a: argparse.Namespace, fhs: dict[str, Any], report: dict[str, Any],
                      hashes: dict[str, Any], fid: Fidelity) -> None:
    per_layer: dict[int, list[Survival]] = {}
    for t in targets:
        sv, rescaled = merge_expert_tensor(t, ck, srcs, a, fhs, hashes, fid, report["spectral"])
        per_layer.setdefault(t.layer, []).append(sv)
        report["experts"][t.key] = {"format": t.fmt, "scale_chunks_rewritten": rescaled, **sv.summary()}
        say(f"{t.key} [{t.fmt}] survival {sv.summary()['survival']:.4f} rel_delta {sv.summary()['rel_delta']:.2e} "
            f"saturated {sv.saturated} bumped {sv.bumped_blocks}")
    if targets:
        report["survival_per_layer"] = {str(l): merge_survival(v).summary() for l, v in sorted(per_layer.items())}
        report["survival_all"] = merge_survival([x for v in per_layer.values() for x in v]).summary()
        say("delta survival per layer: " + " ".join(f"L{l}={v['survival']:.3f}"
                                                    for l, v in report["survival_per_layer"].items()))
        say(f"delta survival overall: {report['survival_all']['survival']:.4f} "
            f"(projected {report['survival_all']['projected_survival']:.4f})")


def say_fidelity(report: dict[str, Any]) -> None:
    """Log: per rounding mode, |realised|/|intended|, cosine and scale along the delta for the main groups,
    then the spectral sample's median ratios (realised - intended along top singular vs random directions)."""
    for mode, groups in report["fidelity"].items():
        for g in ("attn", "experts.nvfp4", "experts.mxfp8", "all"):
            if g in groups:
                x = groups[g]
                say(f"fidelity [{mode:10s}] {g:14s} change_ratio {x['change_ratio']:.3f} cosine {x['cosine']:.3f} "
                    f"scale_along_delta {x['projected_survival']:.3f} changed {x['changed_frac']:.3f}")
    for mode in ("nearest", "stochastic"):
        rows = [r for r in report["spectral"] if r["mode"] == mode]
        if rows:
            med = {k: sorted(r[k] for r in rows)[len(rows) // 2] for k in ("ratio_top", "ratio_rand",
                                                                          "noise_top_over_rand")}
            say(f"spectral [{mode:10s}] n={len(rows)} median " + " ".join(f"{k} {v:.3g}" for k, v in med.items()))


def prepare_out(a: argparse.Namespace, tmp: str, shards: set[str]) -> None:
    if os.path.exists(a.out):
        fail(f"{a.out} exists; delete it to rebuild")
    if os.path.exists(tmp):
        shutil.rmtree(tmp)
    need = sum(os.path.getsize(os.path.join(a.base, fn)) for fn in shards)
    os.makedirs(os.path.dirname(os.path.abspath(tmp)), exist_ok=True)
    free = shutil.disk_usage(os.path.dirname(os.path.abspath(tmp))).free
    if free < need + (2 << 30):
        fail(f"not enough disk: need {need / 1e9:.1f} GB + 2 GB, free {free / 1e9:.1f} GB")
    link_or_copy_tree(a.base, tmp, shards)


def finish(a: argparse.Namespace, tmp: str, report: dict[str, Any]) -> None:
    rp = a.report or a.out.rstrip("/") + ".merge_report.json"
    with open(rp, "w") as f:
        json.dump(report, f, indent=1)
    os.rename(tmp, a.out)
    say(f"MERGE_OK {a.out} parts={a.parts} attn={len(report['attn'])} experts={len(report['experts'])} "
        f"verify={json.dumps(report['verify'])} report={rp}")


def parse_args(argv: list[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base")
    ap.add_argument("--adapter", action="append", help="adapter dir; repeat to sum several adapters (one rounding)")
    ap.add_argument("--scales", default="", help="comma list, one user scale per --adapter (default: --scale for all)")
    ap.add_argument("--out")
    ap.add_argument("--parts", choices=["attn", "attn_experts"], default="attn")
    ap.add_argument("--scale", type=float, default=1.0, help="multiplies alpha/r")
    ap.add_argument("--report", default="", help="default <out>.merge_report.json")
    ap.add_argument("--device", default="cpu", help="torch device for the expert maths (cpu or cuda)")
    ap.add_argument("--expert-chunk", type=int, default=8, help="experts per chunk (memory)")
    ap.add_argument("--expert-scales", choices=["keep", "bump"], default="keep",
                    help="keep: scale tensors untouched, out-of-range values saturate; bump: raise block scales")
    ap.add_argument("--swap-w1-w3", action="store_true", help="adapter w1 = up, w3 = gate (UNVERIFIED convention)")
    ap.add_argument("--rounding", choices=["nearest", "stochastic"], default="nearest",
                    help="how W + delta is rounded onto the stored grid (both are measured; this one is written)")
    ap.add_argument("--seed", type=int, default=0, help="stochastic rounding seed (keyed with each tensor's name)")
    ap.add_argument("--no-spectral", dest="spectral", action="store_false", help="skip the sampled spectral check")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if not a.selftest and not (a.base and a.adapter and a.out):
        ap.error("--base, --adapter and --out are required")
    return a


# ---------------------------------------------------------------- multi-adapter selftest
# Lives here (not in selftest_merge.py) to keep the multi-adapter change in one file; reuses that file's synthetic
# checkpoint and adapter builders. Synthetic tensors only.

def st_ns(**kw: Any) -> argparse.Namespace:
    base = dict(base="", adapter=[], scales="", out="", parts="attn_experts", scale=1.0, report="", device="cpu",
                expert_chunk=3, expert_scales="keep", swap_w1_w3=False, selftest=False, rounding="nearest", seed=0,
                spectral=True)
    return argparse.Namespace(**{**base, **kw})


def st_merge(**kw: Any) -> dict[str, Any]:
    run_merge(st_ns(**kw))
    return json.load(open(kw["out"] + ".merge_report.json"))


def st_adapter(tensors: dict[str, Any], r: int, alpha: int, user_scale: float = 1.0) -> Adapter:
    """In-memory Adapter as load_adapters builds it (no files)."""
    cfg = {"peft_type": "LORA", "r": r, "lora_alpha": alpha}
    mods = group_modules(sorted(tensors))
    attn, experts = [m for m in mods if m.kind == "attn"], [m for m in mods if m.kind == "expert"]
    ead = load_expert_adapter(tensors, experts) if experts else {}
    return Adapter("mem", cfg, tensors, user_scale, lora_scale(cfg, user_scale), attn, experts, [], ead)


def st_rand(g: Any, r: int, attn: list[tuple[int, str]], expert_layers: list[int], scale_b: float) -> dict[str, Any]:
    """Rank-r adapter tensors (selftest_merge shapes) on the given attention modules and expert layers only."""
    import torch
    import selftest_merge as S
    t: dict[str, Any] = {}
    for layer, name in attn:
        o, i = S.ATTN_SHAPES[name]
        p = f"language_model.layers.{layer}.attn.{name}"
        t[p + ".lora_A.weight"] = torch.randn(r, i, generator=g) * 0.1
        t[p + ".lora_B.weight"] = torch.randn(o, r, generator=g) * 0.1 * scale_b
    for layer in expert_layers:
        p = f"language_model.layers.{layer}.mlp.experts."
        for w in ("w1", "w3"):
            t[p + w + ".lora_A.weight"] = torch.randn(1, r, S.D, generator=g) * 0.1
            t[p + w + ".lora_B.weight"] = torch.randn(S.E, S.I, r, generator=g) * 0.1 * scale_b
        t[p + "w2.lora_A.weight"] = torch.randn(S.E, r, S.I, generator=g) * 0.1
        t[p + "w2.lora_B.weight"] = torch.randn(1, S.D, r, generator=g) * 0.1 * scale_b
    return t


def st_write(root: str, t: dict[str, Any], r: int, alpha: int) -> None:
    from safetensors.torch import save_file
    os.makedirs(root, exist_ok=True)
    save_file(t, os.path.join(root, "adapter_model.safetensors"))
    json.dump({"peft_type": "LORA", "r": r, "lora_alpha": alpha, "use_rslora": False, "use_dora": False},
              open(os.path.join(root, "adapter_config.json"), "w"))


def st_delta_identity() -> None:
    """(a, delta level) one adapter through the summing helpers is bit-equal to the old single expressions."""
    import torch
    import selftest_merge as S
    print("multi-adapter (a): one adapter through the new path == old single-adapter maths:")
    ad = st_adapter(S.rand_adapter(torch.Generator().manual_seed(11)), S.R, 8, 0.7)
    groups = attn_groups([ad])
    ok = len(groups) == 12 and all(
        torch.equal(attn_delta(k, ts, [ad]),
                    ad.scale * (ad.tensors[ts[0][1].b_key].double() @ ad.tensors[ts[0][1].a_key].double()))
        for k, ts in groups.items())
    S.check(ok, "attention: summed delta bit-equal to s * (B.double() @ A.double())")
    a = st_ns()
    ok = True
    for layer in (0, 1):
        for which in ("w13", "w2"):
            t = ExpertTensor(ckpt_key_expert(layer, which), NVFP4, layer, which)
            old = expert_delta_w13(ad.ead[layer], 1, 3, ad.scale, False, "cpu") if which == "w13" \
                else expert_delta_w2(ad.ead[layer], 1, 3, ad.scale, "cpu")
            ok &= torch.equal(expert_delta(t, [(ad.ead, ad.scale)], 1, 3, a), old)
    S.check(ok, "experts: summed chunk delta bit-equal to expert_delta_w13 / expert_delta_w2")


def st_single_end_to_end(tmp: str, base: str, base_t: dict[str, Any], g: Any) -> None:
    """(a, end to end) --adapter given as a list (new CLI) == the old string form, byte for byte, both roundings;
    nearest also equals selftest_merge's independent old-maths references."""
    import selftest_merge as S
    adp = os.path.join(tmp, "one")
    ad = S.rand_adapter(g, scale_b=20.0)
    S.write_adapter(adp, ad)
    shard = "model-00001-of-00001.safetensors"
    for rounding in ("nearest", "stochastic"):
        st_merge(base=base, adapter=adp, out=os.path.join(tmp, f"str_{rounding}"), rounding=rounding, seed=7)
        rep = st_merge(base=base, adapter=[adp], scales="1.0", out=os.path.join(tmp, f"lst_{rounding}"),
                       rounding=rounding, seed=7)
        b1, b2 = (open(os.path.join(tmp, f"{x}_{rounding}", shard), "rb").read() for x in ("str", "lst"))
        S.check(b1 == b2 and rep["scale"] == 8 / S.R and rep["adapter"] == adp,
                f"{rounding}: list form + --scales 1.0 byte-identical to the single-adapter call; report keys kept")
    got = S.load(os.path.join(tmp, "lst_nearest"))
    want = S.expected_experts(base_t, ad, 8 / S.R)
    S.check(S.check_attn(base_t, ad, got, 8 / S.R) and all(tensor_bytes(got[k]) == tensor_bytes(v)
                                                          for k, v in want.items()),
            "nearest: equals bf16(W + s B A) and whole-tensor expert requant (old references)")


def st_sum_reference() -> None:
    """(b) two adapters of rank 4 and 8: summed delta == float64 reference before any rounding."""
    import torch
    import selftest_merge as S
    print("multi-adapter (b): two adapters summed == float64 reference (before rounding):")
    g = torch.Generator().manual_seed(12)
    a1 = st_adapter(st_rand(g, 4, [(0, "wo_ud")], [0], 1.0), 4, 8, 1.0)
    a2 = st_adapter(st_rand(g, 8, [(0, "wo_ud")], [0], 1.0), 8, 4, 0.5)
    key = ckpt_key_attn(a1.attn[0])
    got = attn_delta(key, attn_groups([a1, a2])[key], [a1, a2])
    m1, m2 = a1.attn[0], a2.attn[0]
    bcat = torch.cat([a1.scale * a1.tensors[m1.b_key].double(), a2.scale * a2.tensors[m2.b_key].double()], 1)
    acat = torch.cat([a1.tensors[m1.a_key].double(), a2.tensors[m2.a_key].double()], 0)
    S.check(torch.allclose(got, bcat @ acat, rtol=0, atol=1e-12) and got.dtype == torch.float64,
            "attention: s1 B1 A1 + s2 B2 A2 == [s1 B1 | s2 B2] @ [A1; A2] (f64, ranks 4 + 8)")
    a = st_ns()
    srcs = [(a1.ead, a1.scale), (a2.ead, a2.scale)]
    ok = True
    for which in ("w13", "w2"):
        d = expert_delta(ExpertTensor(ckpt_key_expert(0, which), NVFP4, 0, which), srcs, 0, S.E, a)
        ref = sum(st_ref_expert(x.ead[0], which, x.scale) for x in (a1, a2))
        ok &= d.dtype == torch.float32 and torch.allclose(d.double(), ref, rtol=1e-5, atol=1e-7)
    S.check(ok, "experts: w13 (interleaved gate/up) and w2 summed delta == float64 per-expert reference")
    S.expect_fail(lambda: add_delta("t.weight", torch.zeros(2, 3), torch.zeros(3, 2)), "delta shape mismatch")


def st_ref_expert(ead: dict[str, tuple[Any, Any]], which: str, s: float) -> Any:
    """Per-expert float64 delta written out longhand (no broadcasting tricks), stored-row order."""
    import torch
    import selftest_merge as S
    out = []
    for e in range(S.E):
        if which == "w2":
            a, b = ead["w2"]
            out.append(s * b[0].double() @ a[e].double())
            continue
        rows = [s * ead[w][1][e].double() @ ead[w][0][0].double() for w in ("w1", "w3")]
        out.append(torch.stack(rows, 1).reshape(2 * S.I, S.D))      # row 2j gate j, row 2j+1 up j
    return torch.stack(out)


def st_expected_layer(base_t: dict[str, Any], ead: dict[str, tuple[Any, Any]], layer: int, s: float) -> dict[str, Any]:
    """Whole-tensor nearest requant of one expert layer with one adapter's delta (layer 0 NVFP4, 1 MXFP8)."""
    import selftest_merge as S
    out = {}
    for which in ("w13", "w2"):
        k = ckpt_key_expert(layer, which)
        d = expert_delta_w13(ead, 0, S.E, s, False, "cpu") if which == "w13" else expert_delta_w2(ead, 0, S.E, s, "cpu")
        if layer == 0:
            f = base_t[k + "_amax"] / base_t[k + "_outer_scale_denominator"]
            out[k] = nvfp4_quant(nvfp4_dequant(base_t[k], base_t[k + "_scale"], f) + d, base_t[k + "_scale"], f,
                                 "keep")[0]
        else:
            out[k] = mxfp8_quant(mxfp8_dequant(base_t[k], base_t[k + "_scale"]) + d, base_t[k + "_scale"], "keep")[0]
    return out


def st_disjoint_attn_ok(base_t: dict[str, Any], got: dict[str, Any], ads: list[Adapter]) -> bool:
    """Every attention tensor == bf16(W + sum over the adapters that touch it), others untouched."""
    import torch
    import selftest_merge as S
    ok = True
    for layer in (0, 1):
        for name in S.ATTN_SHAPES:
            k = f"{CKPT_PREFIX}{layer}.attn.{name}.weight"
            delta = None
            for ad in ads:
                for m in ad.attn:
                    if ckpt_key_attn(m) == k:
                        term = ad.scale * (ad.tensors[m.b_key].double() @ ad.tensors[m.a_key].double())
                        delta = term if delta is None else delta + term
            want = base_t[k] if delta is None else (base_t[k].double() + delta).to(torch.bfloat16)
            ok &= torch.equal(got[k], want)
    return ok


def st_disjoint(tmp: str, base: str, base_t: dict[str, Any], g: Any) -> None:
    """(c) rank 4 on layer-0 attention + layer-1 wq_du + layer-0 experts; rank 8 on layer-1 attention + layer-1
    experts. Overlap only on layer-1 wq_du."""
    import selftest_merge as S
    print("multi-adapter (c): differing ranks, disjoint module sets, end to end:")
    names = list(S.ATTN_SHAPES)
    ta = st_rand(g, 4, [(0, n) for n in names] + [(1, "wq_du")], [0], 20.0)
    tb = st_rand(g, 8, [(1, n) for n in names], [1], 20.0)
    pa, pb = os.path.join(tmp, "ra4"), os.path.join(tmp, "rb8")
    st_write(pa, ta, 4, 8)
    st_write(pb, tb, 8, 8)
    rep = st_merge(base=base, adapter=[pa, pb], scales="1.0,0.5", out=os.path.join(tmp, "multi"))
    got = S.load(os.path.join(tmp, "multi"))
    ads = [st_adapter(ta, 4, 8, 1.0), st_adapter(tb, 8, 8, 0.5)]
    S.check(st_disjoint_attn_ok(base_t, got, ads), "attention: each tensor = bf16(W + sum of the adapters on it)")
    want = {**st_expected_layer(base_t, ads[0].ead[0], 0, ads[0].scale),
            **st_expected_layer(base_t, ads[1].ead[1], 1, ads[1].scale)}
    S.check(all(tensor_bytes(got[k]) == tensor_bytes(v) for k, v in want.items()),
            "experts: layer 0 (NVFP4) only adapter 1's delta, layer 1 (MXFP8) only adapter 2's")
    st_disjoint_report(rep, pa, pb)
    st_disjoint_failures(tmp, base, pa, tb)


def st_disjoint_report(rep: dict[str, Any], pa: str, pb: str) -> None:
    import selftest_merge as S
    ads = rep["adapters"]
    wq1 = f"{CKPT_PREFIX}1.attn.wq_du.weight"
    S.check([x["path"] for x in ads] == [pa, pb] and [x["r"] for x in ads] == [4, 8]
            and [x["user_scale"] for x in ads] == [1.0, 0.5] and [x["scale"] for x in ads] == [2.0, 0.5]
            and rep["scale"] is None and rep["adapter"] == [pa, pb], "report: per-adapter path, r, alpha, scales")
    wq0 = f"{CKPT_PREFIX}0.attn.wq_du.weight"
    S.check(len(ads[0]["modules"]) == 7 + 3 and len(ads[1]["modules"]) == 6 + 3
            and rep["attn"][wq1]["adapters"] == [0, 1] and rep["attn"][wq0]["adapters"] == [0],
            "report: modules each adapter touched; summed tensor lists both adapters")
    S.check(rep["verify"]["rewritten_verified"] == 12 + 8 and "survival_all" in rep
            and {"attn", "experts.nvfp4", "experts.mxfp8"} <= set(rep["fidelity"]["stochastic"]),
            f"report: verify / fidelity / survival on the summed delta ({rep['verify']})")


def st_disjoint_failures(tmp: str, base: str, pa: str, tb: dict[str, Any]) -> None:
    import torch
    import selftest_merge as S
    bad = dict(tb)
    bad["language_model.layers.1.attn.wk_dv.lora_B.weight"] = torch.zeros(48, 8)
    pbad = os.path.join(tmp, "bad8")
    st_write(pbad, bad, 8, 8)
    try:
        st_merge(base=base, adapter=[pa, pbad], out=os.path.join(tmp, "z"))
        raise AssertionError("shape mismatch in the second adapter did not fail")
    except SystemExit as e:
        S.check("layers.1.attn.wk_dv" in str(e) and pbad in str(e) and not os.path.exists(os.path.join(tmp, "z")),
                f"shape mismatch names tensor and adapter ({str(e)[:100]})")
    S.expect_fail(lambda: st_merge(base=base, adapter=[pa, pbad], scales="1.0", out=os.path.join(tmp, "z2")),
                  "--scales count != adapter count")


def run_multi_selftest() -> None:
    import tempfile
    import torch
    import selftest_merge as S
    torch.set_grad_enabled(False)
    st_delta_identity()
    st_sum_reference()
    g = torch.Generator().manual_seed(13)
    with tempfile.TemporaryDirectory() as tmp:
        base = os.path.join(tmp, "base")
        os.makedirs(base)
        base_t = S.build_base(base, g)
        st_single_end_to_end(tmp, base, base_t, g)
        st_disjoint(tmp, base, base_t, g)
    print("MULTI_SELFTEST_OK", flush=True)


def main(argv: list[str]) -> None:
    a = parse_args(argv)
    if a.selftest:
        from selftest_merge import run_selftest  # kept apart so this file stays readable
        run_selftest()
        run_multi_selftest()
        return
    run_merge(a)


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    main(sys.argv[1:])
