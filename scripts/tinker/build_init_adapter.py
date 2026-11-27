"""Combine PEFT LoRA adapters by rank concatenation into ONE adapter whose delta is exactly their sum, in the Tinker
export layout, so it can be uploaded as a starting checkpoint (external_weights_upload_start/finish) and trained on.

Use: start Tinker RL from "stock + v2.1 edit + P" by uploading cat(v2.1 r64 SVD LoRA, P) as a rank-96 adapter.

  PY=~/.inkling-abl/tinker_lab/.venv/bin/python
  $PY scripts/tinker/build_init_adapter.py --src ~/.inkling-abl/tinker_lab/adapters/v21_r64 --src <P dir> \
      --rank 96 [--alpha 32] [--template <fresh rank-96 export dir>] --out <dir> --verify
  $PY scripts/tinker/build_init_adapter.py --src A --src B --out <dir> --verify-only      # re-check an existing build
  $PY scripts/tinker/build_init_adapter.py --selftest                                     # synthetic, offline

--src DIR[@SCALE]: a PEFT dir (adapter_config.json + adapter_model.safetensors); SCALE is a user multiplier (default 1).
Maths, per module present in k >= 1 sources: delta = sum_i s_i B_i A_i with s_i = SCALE_i * alpha_i / r_i. Output scale
s = alpha / R. Each source's factors are multiplied by sqrt(s_i / s) (balanced, so A and B keep comparable magnitudes
for Adam), A is stacked and B concatenated on the rank axis, then the rank is padded to R. Padding: A gets small random
rows (PEFT's kaiming-uniform bound 1/sqrt(in), or the template's own init rows), B gets zero columns, so the delta is
unchanged and the padded slots still receive gradient (zero A AND zero B would never train).
Expert layout (Tinker): w1/w3 lora_A [1, r, in] shared + lora_B [E, out, r]; w2 lora_A [E, r, in] + lora_B [1, out, r].
The rank axis is -2 for A and -1 for B in every layout, so concatenation is the same code path; a shared (dim0 = 1)
factor is only broadcast to E when a source or the template needs per-expert.
Names: HF names of the v2.1 LoRA (base_model.model.model.language_model.layers.N.self_attn.{q,k,v}_proj.linear,
o_proj.linear, o_proj.gate_proj) are renamed to the Tinker/checkpoint names (attn.wq_du, wk_dv, wv_dv, wo_ud,
gate_attn): the v2.1 SVD was computed on checkpoint tensors and only renamed (scripts/v21/lora_build_v21.py), and
methods/directions/MODEL_NOTES.md section 3 gives the same correspondence, so this is a pure rename.
Single source = rename/rescale only: e.g. the v2.1 r64 LoRA in Tinker names, which merge_adapter.py can parse
(its NAME_RE rejects the HF self_attn.* names) for the final "(trained) - (v2.1 r64)" merge.

--template DIR: a save_weights_external export of a FRESH training client created with the target LoraConfig (rank R,
same train_* flags). Upload accepts only "exactly what save_weights_external exports" (tinker rest_client.py:521-524),
so with a template the output copies its adapter_config.json verbatim, must match its tensor names, shapes and dtypes
exactly, and uses its init A rows for the padded slots. Without a template the names follow the Tinker export seen on
P (language_model.layers.N...lora_{A,B}.weight, F32), and alpha defaults to 32 (P's export: r=32, lora_alpha=32; the
value Tinker writes for r=96 is UNVERIFIED, which is why the template path exists).
Memory: one module at a time; the output safetensors file is streamed (header first, then tensors in order).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import struct
import sys
import tempfile
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray

Arr = NDArray[Any]
ATTN = ("wq_du", "wk_dv", "wv_dv", "gate_attn", "wo_ud", "wr_du")
HF_TO_TINKER = {"self_attn.q_proj.linear": "attn.wq_du", "self_attn.k_proj.linear": "attn.wk_dv",
                "self_attn.v_proj.linear": "attn.wv_dv", "self_attn.o_proj.linear": "attn.wo_ud",
                "self_attn.o_proj.gate_proj": "attn.gate_attn"}
KEY_RE = re.compile(r"^(?:base_model\.model\.)?(?:model\.)?language_model\.layers\.(\d+)\.(.+)\.lora_(A|B)\.weight$")
TINKER_SUB_RE = re.compile(r"^(?:attn\.(?:" + "|".join(ATTN) + r")|mlp\.experts\.w[123])$")
DEFAULT_ALPHA = 32.0
VERIFY_TOL = 1e-5


def fail(msg: str) -> None:
    raise SystemExit(f"build_init_adapter: {msg}")


# ---------------------------------------------------------------- reading sources

def module_of(key: str) -> tuple[str, str] | None:
    """('language_model.layers.N.<tinker sub>', 'A'|'B') for a LoRA tensor name, None if not a LoRA tensor."""
    m = KEY_RE.match(key)
    if not m:
        return None
    layer, sub, ab = m.groups()
    sub = HF_TO_TINKER.get(sub, sub)
    if not TINKER_SUB_RE.match(sub):
        fail(f"cannot map {key!r} to a Tinker module (sub-module {sub!r})")
    return f"language_model.layers.{layer}.{sub}", ab


@dataclass
class Adapter:
    path: Path
    cfg: dict[str, Any]
    scale: float                                   # user scale * alpha / r
    keys: dict[str, dict[str, str]]                # module -> {"A": key, "B": key}
    handle: Any = field(repr=False)

    def shape(self, key: str) -> list[int]:
        return list(self.handle.get_slice(key).get_shape())

    def tensor(self, key: str) -> Arr:
        return self.handle.get_tensor(key)


def pair_keys(names: list[str], where: Path) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for n in names:
        mod = module_of(n)
        if mod is None:
            fail(f"{where}: tensor {n!r} is not a LoRA A/B weight (unembed or other modules are not handled)")
        if mod[1] in out.setdefault(mod[0], {}):
            fail(f"{where}: two tensors map to {mod[0]} lora_{mod[1]}")
        out[mod[0]][mod[1]] = n
    bad = [m for m, v in out.items() if set(v) != {"A", "B"}]
    if bad:
        fail(f"{where}: modules without both lora_A and lora_B: {bad[:4]}")
    return out


def check_cfg(cfg: dict[str, Any], where: Path) -> None:
    """Per-module ranks/alphas, rsLoRA and DoRA change the maths; refuse rather than combine wrongly."""
    for k in ("rank_pattern", "alpha_pattern"):
        if cfg.get(k):
            fail(f"{where}: {k} is set; per-module ranks/alphas are not supported")
    if cfg.get("use_rslora") or cfg.get("use_dora"):
        fail(f"{where}: rsLoRA/DoRA adapters are not supported")
    if cfg.get("peft_type", "LORA") != "LORA":
        fail(f"{where}: peft_type {cfg.get('peft_type')} is not LORA")


def open_adapter(spec: str) -> Adapter:
    """spec = DIR or DIR@SCALE (a path may contain '@' only if a scale is also given)."""
    path_s, _, scale_s = spec.rpartition("@") if re.search(r"@-?[0-9.eE+-]+$", spec) else (spec, "", "1")
    path = Path(path_s).expanduser()
    from safetensors import safe_open
    cfg_p, st_p = path / "adapter_config.json", path / "adapter_model.safetensors"
    for p in (cfg_p, st_p):
        if not p.exists():
            fail(f"{p} missing")
    cfg = json.loads(cfg_p.read_text())
    check_cfg(cfg, path)
    handle = safe_open(str(st_p), framework="np")
    keys = pair_keys(list(handle.keys()), path)
    scale = float(scale_s) * float(cfg["lora_alpha"]) / float(cfg["r"])
    return Adapter(path, cfg, scale, keys, handle)


# ---------------------------------------------------------------- planning the output

@dataclass(frozen=True)
class Plan:
    module: str
    a_name: str
    b_name: str
    a_shape: tuple[int, ...]
    b_shape: tuple[int, ...]
    used: int                                      # sum of source ranks; the rest is padding


def rank_of(a_shape: list[int]) -> int:
    return a_shape[-2]


def lead_dim(shapes: list[list[int]], module: str, what: str) -> int:
    """Leading expert dim for 3-D factors: 1 if every source shares it, else the common E."""
    leads = {s[0] for s in shapes if len(s) == 3}
    if not leads:
        return 0
    big = leads - {1}
    if len(big) > 1:
        fail(f"{module} {what}: expert counts disagree {sorted(leads)}")
    return big.pop() if big else 1


def derived_shapes(srcs: list[Adapter], module: str, rank: int) -> tuple[tuple[int, ...], tuple[int, ...], int]:
    a_shapes = [s.shape(s.keys[module]["A"]) for s in srcs if module in s.keys]
    b_shapes = [s.shape(s.keys[module]["B"]) for s in srcs if module in s.keys]
    if len({len(x) for x in a_shapes + b_shapes}) != 1:
        fail(f"{module}: sources mix 2-D and 3-D factors")
    ins, outs = {x[-1] for x in a_shapes}, {x[-2] for x in b_shapes}
    if len(ins) != 1 or len(outs) != 1:
        fail(f"{module}: in/out dims disagree between sources (in {ins}, out {outs})")
    used = sum(rank_of(x) for x in a_shapes)
    if used > rank:
        fail(f"{module}: sources need rank {used} > output rank {rank}")
    la, lb = lead_dim(a_shapes, module, "A"), lead_dim(b_shapes, module, "B")
    a = ((la,) if la else ()) + (rank, ins.pop())
    b = ((lb,) if lb else ()) + (outs.pop(), rank)
    return a, b, used


def template_pairs(tpl: Adapter) -> dict[str, tuple[str, str, tuple[int, ...], tuple[int, ...]]]:
    return {m: (k["A"], k["B"], tuple(tpl.shape(k["A"])), tuple(tpl.shape(k["B"]))) for m, k in tpl.keys.items()}


def check_against_template(m: str, a: tuple[int, ...], b: tuple[int, ...], t: tuple[Any, ...]) -> None:
    """Sources' shapes must fit the template: same dims, and a shared (1) lead only where the template shares too."""
    for mine, theirs, what in ((a, t[2], "A"), (b, t[3], "B")):
        if len(mine) != len(theirs) or mine[1 if len(mine) == 3 else 0:] != theirs[1 if len(theirs) == 3 else 0:]:
            fail(f"{m} lora_{what}: sources give {mine}, template has {theirs}")
        if len(mine) == 3 and mine[0] not in (1, theirs[0]):
            fail(f"{m} lora_{what}: sources are per-expert ({mine[0]}) where the template shares ({theirs[0]})")


def make_plans(srcs: list[Adapter], rank: int, tpl: Adapter | None) -> list[Plan]:
    mods = sorted({m for s in srcs for m in s.keys}, key=module_sort_key)
    if tpl is None:
        out = []
        for m in mods:
            a, b, used = derived_shapes(srcs, m, rank)
            out.append(Plan(m, f"{m}.lora_A.weight", f"{m}.lora_B.weight", a, b, used))
        return out
    tp = template_pairs(tpl)
    missing = [m for m in mods if m not in tp]
    if missing:
        fail(f"{len(missing)} source module(s) are absent from the template, e.g. {missing[:4]} "
             "(was the template exported with the same train_attn/train_mlp flags?)")
    plans = []
    for m in sorted(tp, key=module_sort_key):
        t = tp[m]
        if rank_of(list(t[2])) != rank:
            fail(f"template {m} has rank {rank_of(list(t[2]))}, expected {rank}")
        used = 0
        if m in mods:
            a, b, used = derived_shapes(srcs, m, rank)
            check_against_template(m, a, b, t)
        plans.append(Plan(m, t[0], t[1], t[2], t[3], used))
    return plans


def module_sort_key(m: str) -> tuple[int, str]:
    mm = re.match(r"language_model\.layers\.(\d+)\.(.+)$", m)
    return (int(mm.group(1)), mm.group(2)) if mm else (-1, m)


# ---------------------------------------------------------------- building one module

def broadcast_lead(x: Arr, shape: tuple[int, ...]) -> Arr:
    """Expand a shared expert factor (dim0 = 1) to per-expert when the output is per-expert."""
    if x.ndim == 3 and x.shape[0] == 1 and shape[0] > 1:
        return np.ascontiguousarray(np.broadcast_to(x, (shape[0],) + x.shape[1:]))
    return x


def pad_a(p: Plan, pad: int, seed: int, mode: str, tpl_a: Arr | None) -> Arr:
    lead, fan_in = p.a_shape[:-2], p.a_shape[-1]
    if tpl_a is not None:            # the service's own init rows for the unused slots
        return np.ascontiguousarray(tpl_a[..., p.used:, :], dtype=np.float32)
    if mode == "zero" or pad == 0:
        return np.zeros(lead + (pad, fan_in), np.float32)
    rng = np.random.default_rng([seed, zlib.crc32(p.module.encode())])
    bound = 1.0 / math.sqrt(fan_in)  # PEFT kaiming_uniform_(a=sqrt(5)) on lora_A
    return rng.uniform(-bound, bound, lead + (pad, fan_in)).astype(np.float32)


def build_module(p: Plan, srcs: list[Adapter], s_out: float, seed: int, pad_mode: str,
                 tpl: Adapter | None) -> tuple[Arr, Arr]:
    a_parts, b_parts = [], []
    for s in srcs:
        if p.module not in s.keys:
            continue
        f = math.sqrt(abs(s.scale) / s_out)
        sign = -1.0 if s.scale < 0 else 1.0                   # negative user scale goes on B only
        a = s.tensor(s.keys[p.module]["A"]).astype(np.float64) * f
        b = s.tensor(s.keys[p.module]["B"]).astype(np.float64) * (f * sign)
        a_parts.append(broadcast_lead(a.astype(np.float32), p.a_shape))
        b_parts.append(broadcast_lead(b.astype(np.float32), p.b_shape))
    pad = p.a_shape[-2] - p.used
    tpl_a = tpl.tensor(p.a_name) if tpl is not None else None
    a_parts.append(pad_a(p, pad, seed, pad_mode, tpl_a))
    b_parts.append(np.zeros(p.b_shape[:-1] + (pad,), np.float32))
    a_out = np.concatenate([broadcast_lead(x, p.a_shape) for x in a_parts], axis=-2)
    b_out = np.concatenate([broadcast_lead(x, p.b_shape) for x in b_parts], axis=-1)
    if a_out.shape != p.a_shape or b_out.shape != p.b_shape:
        fail(f"{p.module}: built {a_out.shape}/{b_out.shape}, planned {p.a_shape}/{p.b_shape}")
    return a_out, b_out


# ---------------------------------------------------------------- writing

@dataclass(frozen=True)
class FileStyle:
    """How the reference export lays out its file; mimicked in case the upload check is stricter than names/shapes."""
    metadata: dict[str, str] | None
    pad_header: bool
    order: dict[str, int]                          # tensor name -> data offset in the reference


def read_style(st_path: Path | None) -> FileStyle:
    if st_path is None:
        return FileStyle({"format": "pt"}, True, {})
    with open(st_path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        h = json.loads(fh.read(n))
    meta = h.pop("__metadata__", None)
    return FileStyle(meta, n % 8 == 0, {k: v["data_offsets"][0] for k, v in h.items()})


def style_source(srcs: list[Adapter], tpl: Adapter | None) -> Path | None:
    """The template, else a Tinker-exported source (P: target_modules all-linear), else none."""
    ref = tpl or next((s for s in srcs if s.cfg.get("target_modules") == "all-linear"), None)
    return ref.path / "adapter_model.safetensors" if ref else None


TensorItem = tuple[str, tuple[int, ...], Plan, int]   # name, shape, module plan, 0 = A / 1 = B


def tensor_order(plans: list[Plan], style: FileStyle) -> list[TensorItem]:
    """Reference export order, per tensor (Tinker writes w1.A, w3.A, w1.B, w3.B); unknown names go last."""
    big = 1 << 62
    items = [it for p in plans for it in ((p.a_name, p.a_shape, p, 0), (p.b_name, p.b_shape, p, 1))]
    return sorted(items, key=lambda it: (style.order.get(it[0], big), module_sort_key(it[2].module), it[3]))


def write_safetensors_streaming(path: Path, plans: list[Plan], make: Callable[[Plan], tuple[Arr, Arr]],
                                style: FileStyle) -> None:
    """Header from planned shapes, then tensors in reference order; a module is built when its first tensor is due
    and its other half kept only until written, so memory stays at a few modules."""
    items = tensor_order(plans, style)
    header: dict[str, Any] = {"__metadata__": style.metadata} if style.metadata else {}
    off = 0
    for name, shape, _, _ in items:
        n = 4 * math.prod(shape)
        header[name] = {"dtype": "F32", "shape": list(shape), "data_offsets": [off, off + n]}
        off += n
    hb = json.dumps(header, separators=(",", ":")).encode()
    hb += b" " * ((-len(hb) % 8) if style.pad_header else 0)
    tmp = path.with_suffix(".safetensors.tmp")
    pending: dict[str, list[Arr | None]] = {}
    with open(tmp, "wb") as fh:
        fh.write(struct.pack("<Q", len(hb)))
        fh.write(hb)
        for i, (_, _, p, ab) in enumerate(items):
            if p.module not in pending:
                pending[p.module] = list(make(p))
            t = pending[p.module][ab]
            fh.write(np.ascontiguousarray(t, dtype="<f4").tobytes())
            pending[p.module][ab] = None
            if all(x is None for x in pending[p.module]):
                del pending[p.module]
            if (i + 1) % 100 == 0:
                print(f"  wrote {i + 1}/{len(items)} tensors", flush=True)
    os.replace(tmp, path)


def out_config(srcs: list[Adapter], tpl: Adapter | None, rank: int, alpha: float) -> dict[str, Any]:
    if tpl is not None:
        return dict(tpl.cfg)
    # Base on a Tinker-exported source (all-linear, no layers_to_transform) so the fields look like an export.
    base = next((s.cfg for s in srcs if s.cfg.get("target_modules") == "all-linear"), None)
    cfg = dict(base) if base else {"peft_type": "LORA", "task_type": "CAUSAL_LM", "target_modules": "all-linear",
                                   "lora_dropout": 0.0, "bias": "none", "use_rslora": False, "use_dora": False}
    cfg.update({"r": rank, "lora_alpha": int(alpha) if float(alpha).is_integer() else alpha})
    return cfg


def resolve_rank_alpha(a: argparse.Namespace, tpl: Adapter | None) -> tuple[int, float]:
    if tpl is not None:
        r, al = int(tpl.cfg["r"]), float(tpl.cfg["lora_alpha"])
        if (a.rank and a.rank != r) or (a.alpha is not None and a.alpha != al):
            fail(f"--rank/--alpha conflict with the template (r={r}, lora_alpha={al})")
        return r, al
    return a.rank, (a.alpha if a.alpha is not None else DEFAULT_ALPHA)


def build(srcs: list[Adapter], out: Path, rank: int, alpha: float, seed: int, pad_mode: str,
          tpl: Adapter | None) -> list[Plan]:
    if not rank:
        rank = max(sum(rank_of(s.shape(s.keys[m]["A"])) for s in srcs if m in s.keys)
                   for m in {m for s in srcs for m in s.keys})
        print(f"output rank (max summed source rank): {rank}")
    style = read_style(style_source(srcs, tpl))
    plans = make_plans(srcs, rank, tpl)
    s_out = alpha / rank
    out.mkdir(parents=True, exist_ok=True)
    write_safetensors_streaming(out / "adapter_model.safetensors", plans,
                                lambda p: build_module(p, srcs, s_out, seed, pad_mode, tpl), style)
    (out / "adapter_config.json").write_text(json.dumps(out_config(srcs, tpl, rank, alpha), indent=2) + "\n")
    write_provenance(out, srcs, plans, rank, alpha, tpl)
    return plans


def write_provenance(out: Path, srcs: list[Adapter], plans: list[Plan], rank: int, alpha: float,
                     tpl: Adapter | None) -> None:
    """Sidecar NEXT TO the output dir: the upload must contain only the exported file set."""
    used = {}
    for p in plans:
        used[p.used] = used.get(p.used, 0) + 1
    info = {"sources": [{"path": str(s.path), "r": s.cfg["r"], "lora_alpha": s.cfg["lora_alpha"], "scale": s.scale,
                         "modules": len(s.keys)} for s in srcs],
            "rank": rank, "alpha": alpha, "template": str(tpl.path) if tpl else None,
            "modules": len(plans), "modules_by_used_rank": used}
    Path(str(out).rstrip("/") + ".provenance.json").write_text(json.dumps(info, indent=1) + "\n")


# ---------------------------------------------------------------- verification

def factor_at(x: Arr, e: int) -> Arr:
    return x if x.ndim == 2 else x[0 if x.shape[0] == 1 else e]


def module_delta(ad: Adapter, a_key: str, b_key: str, scale: float, e: int) -> Arr:
    a = factor_at(ad.tensor(a_key), e).astype(np.float64)
    b = factor_at(ad.tensor(b_key), e).astype(np.float64)
    return scale * (b @ a)


def experts_of(ad: Adapter, keys: dict[str, str]) -> int:
    sh = [ad.shape(keys["A"]), ad.shape(keys["B"])]
    return max((x[0] for x in sh if len(x) == 3), default=0)


def sample_modules(out: Adapter, srcs: list[Adapter], n: int, seed: int) -> list[str]:
    """n random modules that sources touch, plus one of every module kind present (attention + each expert type)."""
    touched = sorted({m for s in srcs for m in s.keys}, key=module_sort_key)
    rng = np.random.default_rng(seed)
    pick = set(rng.choice(touched, size=min(n, len(touched)), replace=False).tolist()) if touched else set()
    for kind in sorted({m.split(".", 3)[3] for m in touched}):
        pick.add(next(m for m in touched if m.endswith(kind)))
    return sorted(pick, key=module_sort_key)


def verify_module(m: str, out: Adapter, srcs: list[Adapter], rng: np.random.Generator) -> dict[str, float]:
    ne = experts_of(out, out.keys[m])
    idx = sorted(set(rng.choice(ne, size=min(2, ne), replace=False).tolist())) if ne else [0]
    worst = {"max_rel": 0.0, "fro_rel": 0.0}
    for e in idx:
        ref = sum(module_delta(s, s.keys[m]["A"], s.keys[m]["B"], s.scale, e) for s in srcs if m in s.keys)
        got = module_delta(out, out.keys[m]["A"], out.keys[m]["B"], out.scale, e)
        den = float(np.abs(ref).max()) or 1.0
        worst["max_rel"] = max(worst["max_rel"], float(np.abs(got - ref).max()) / den)
        worst["fro_rel"] = max(worst["fro_rel"], float(np.linalg.norm(got - ref) / (np.linalg.norm(ref) or 1.0)))
    return worst


def check_untouched_zero(out: Adapter, srcs: list[Adapter]) -> int:
    """Template-only modules (no source) must contribute nothing: their B is all zero."""
    touched = {m for s in srcs for m in s.keys}
    bad = [m for m in out.keys if m not in touched and np.any(out.tensor(out.keys[m]["B"]))]
    if bad:
        fail(f"{len(bad)} module(s) with no source have a nonzero B, e.g. {bad[:3]}")
    return len(out.keys) - len(touched & set(out.keys))


def verify(out_dir: Path, srcs: list[Adapter], n: int, seed: int, tpl: Adapter | None) -> float:
    out = open_adapter(str(out_dir))
    missing = [m for s in srcs for m in s.keys if m not in out.keys]
    if missing:
        fail(f"output lacks {len(missing)} source module(s), e.g. {missing[:3]}")
    if tpl is not None:
        check_template_match(out, tpl)
    rng = np.random.default_rng(seed + 1)
    worst = 0.0
    for m in sample_modules(out, srcs, n, seed):
        r = verify_module(m, out, srcs, rng)
        worst = max(worst, r["max_rel"])
        print(f"  {m:45s} max_rel {r['max_rel']:.2e}  fro_rel {r['fro_rel']:.2e}")
    n_pad_only = check_untouched_zero(out, srcs)
    print(f"verify: {len(out.keys)} modules, r={out.cfg['r']} alpha={out.cfg['lora_alpha']}, "
          f"{n_pad_only} padding-only, worst max_rel {worst:.2e} (tol {VERIFY_TOL:g})")
    if worst > VERIFY_TOL:
        fail(f"delta mismatch {worst:.2e} > {VERIFY_TOL:g}")
    return worst


def check_template_match(out: Adapter, tpl: Adapter) -> None:
    """Upload accepts only the exported file set: same names, shapes, dtypes and config as the template."""
    on, tn = set(out.handle.keys()), set(tpl.handle.keys())
    if on != tn:
        fail(f"tensor names differ from the template: {len(on - tn)} extra, {len(tn - on)} missing")
    for k in sorted(tn):
        so, st = out.handle.get_slice(k), tpl.handle.get_slice(k)
        if so.get_shape() != st.get_shape() or so.get_dtype() != st.get_dtype():
            fail(f"{k}: {so.get_dtype()}{so.get_shape()} vs template {st.get_dtype()}{st.get_shape()}")
    if out.handle.metadata() != tpl.handle.metadata():
        fail(f"safetensors metadata {out.handle.metadata()} differs from the template's {tpl.handle.metadata()}")
    if out.cfg != tpl.cfg:
        fail("adapter_config.json differs from the template")
    print(f"template match: {len(tn)} tensors, names/shapes/dtypes/config identical")


# ---------------------------------------------------------------- selftest

def save_peft(d: Path, tensors: dict[str, Arr], r: int, alpha: float, extra: dict[str, Any] | None = None) -> str:
    from safetensors.numpy import save_file
    d.mkdir(parents=True)
    save_file({k: np.ascontiguousarray(v, dtype=np.float32) for k, v in tensors.items()}, str(d / "adapter_model.safetensors"))
    cfg = {"peft_type": "LORA", "r": r, "lora_alpha": alpha, **(extra or {})}
    (d / "adapter_config.json").write_text(json.dumps(cfg))
    return str(d)


def synth_sources(root: Path, rng: np.random.Generator) -> tuple[str, str]:
    """P-like (Tinker names, r=4, alpha=8, attention + all expert layouts) and v2.1-like (HF names, r=6, alpha=6)."""
    g = lambda *s: rng.standard_normal(s)  # noqa: E731
    E, D, I, L = 3, 16, 8, "language_model.layers."
    p = {f"{L}0.attn.wq_du.lora_A.weight": g(4, D), f"{L}0.attn.wq_du.lora_B.weight": g(24, 4),
         f"{L}0.attn.wr_du.lora_A.weight": g(4, D), f"{L}0.attn.wr_du.lora_B.weight": g(5, 4),
         f"{L}0.mlp.experts.w1.lora_A.weight": g(1, 4, D), f"{L}0.mlp.experts.w1.lora_B.weight": g(E, I, 4),
         f"{L}0.mlp.experts.w3.lora_A.weight": g(1, 4, D), f"{L}0.mlp.experts.w3.lora_B.weight": g(E, I, 4),
         f"{L}0.mlp.experts.w2.lora_A.weight": g(E, 4, I), f"{L}0.mlp.experts.w2.lora_B.weight": g(1, D, 4)}
    h = "base_model.model.model.language_model.layers."
    v = {f"{h}0.self_attn.q_proj.linear.lora_A.weight": g(6, D), f"{h}0.self_attn.q_proj.linear.lora_B.weight": g(24, 6),
         f"{h}1.self_attn.o_proj.gate_proj.lora_A.weight": g(6, D),
         f"{h}1.self_attn.o_proj.gate_proj.lora_B.weight": g(24, 6)}
    return (save_peft(root / "p", p, 4, 8, {"target_modules": "all-linear"}),
            save_peft(root / "v21", v, 6, 6, {"layers_to_transform": [0, 1]}))


def synth_template(root: Path, rng: np.random.Generator, rank: int) -> str:
    """A 'fresh export': random A, zero B for every module, plus one module no source has (layer 1 wk_dv)."""
    E, D, I, L = 3, 16, 8, "language_model.layers."
    spec = {"0.attn.wq_du": ((rank, D), (24, rank)), "0.attn.wr_du": ((rank, D), (5, rank)),
            "1.attn.gate_attn": ((rank, D), (24, rank)), "1.attn.wk_dv": ((rank, D), (7, rank)),
            "0.mlp.experts.w1": ((1, rank, D), (E, I, rank)), "0.mlp.experts.w3": ((1, rank, D), (E, I, rank)),
            "0.mlp.experts.w2": ((E, rank, I), (1, D, rank))}
    t = {}
    for m, (a, b) in spec.items():
        t[f"{L}{m}.lora_A.weight"] = rng.uniform(-0.25, 0.25, a)
        t[f"{L}{m}.lora_B.weight"] = np.zeros(b)
    return save_peft(root / "tpl", t, rank, 32, {"target_modules": "all-linear", "marker": "template"})


def expect_fail(fn: Callable[[], Any], needle: str) -> None:
    try:
        fn()
    except SystemExit as e:
        assert needle in str(e), f"wrong failure: {e}"
        return
    raise AssertionError(f"expected failure containing {needle!r}")


def selftest() -> None:
    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        p_dir, v_dir = synth_sources(root, rng)
        srcs = [open_adapter(v_dir), open_adapter(p_dir + "@1.0")]
        build(srcs, root / "out", 12, 4.0, 0, "random", None)
        assert verify(root / "out", srcs, 99, 0, None) < VERIFY_TOL
        out = open_adapter(str(root / "out"))
        b = out.tensor("language_model.layers.0.attn.wr_du.lora_B.weight")
        a = out.tensor("language_model.layers.0.attn.wr_du.lora_A.weight")
        assert not b[:, 4:].any() and np.abs(a[4:]).min() > 0, "padding: B zero, A nonzero"
        # Subtracting the v2.1 part (scale -1) from the combined adapter must leave P alone.
        diff = [open_adapter(str(root / "out")), open_adapter(v_dir + "@-1")]
        build(diff, root / "diff", 18, 18.0, 0, "zero", None)
        check_diff_equals(root / "diff", open_adapter(p_dir))
        expect_fail(lambda: build(srcs, root / "small", 8, 4.0, 0, "random", None), "need rank 10 > output rank 8")
        tpl = open_adapter(synth_template(root, rng, 12))
        build(srcs, root / "tout", 12, 32.0, 0, "random", tpl)
        verify(root / "tout", srcs, 99, 0, tpl)
        to, tt = open_adapter(str(root / "tout")), tpl
        k = "language_model.layers.0.attn.wq_du.lora_A.weight"
        assert np.array_equal(to.tensor(k)[10:], tt.tensor(k)[10:]), "pad rows come from the template"
        k = "language_model.layers.1.attn.wk_dv.lora_A.weight"
        assert np.array_equal(to.tensor(k), tt.tensor(k)), "template-only module copied"
        expect_fail(lambda: build(srcs, root / "t2", 16, 32.0, 0, "random", tpl), "has rank 12, expected 16")
    print("selftest OK")


def check_diff_equals(diff_dir: Path, p: Adapter) -> None:
    d = open_adapter(str(diff_dir))
    for m in p.keys:
        for e in range(max(1, experts_of(p, p.keys[m]))):
            want = module_delta(p, p.keys[m]["A"], p.keys[m]["B"], p.scale, e)
            got = module_delta(d, d.keys[m]["A"], d.keys[m]["B"], d.scale, e)
            assert np.abs(got - want).max() <= 1e-4 * np.abs(want).max(), m
    for m in d.keys:
        if m not in p.keys:
            got = module_delta(d, d.keys[m]["A"], d.keys[m]["B"], d.scale, 0)
            assert np.abs(got).max() < 1e-4, f"{m} should cancel"
    print("  combined - v2.1 == P: OK")


# ---------------------------------------------------------------- main

def parse() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--src", action="append", default=[], help="PEFT dir, optionally DIR@SCALE (repeatable)")
    ap.add_argument("--out", help="output dir (adapter_config.json + adapter_model.safetensors only)")
    ap.add_argument("--rank", type=int, default=0, help="output rank (default: max summed source rank)")
    ap.add_argument("--alpha", type=float, default=None, help=f"output lora_alpha (default {DEFAULT_ALPHA:g})")
    ap.add_argument("--template", help="fresh save_weights_external export with the target LoraConfig")
    ap.add_argument("--pad-init", choices=["random", "zero"], default="random")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--verify", action="store_true", help="after building, check sampled module deltas")
    ap.add_argument("--verify-only", action="store_true", help="check an existing --out against the sources")
    ap.add_argument("--n-verify", type=int, default=16)
    ap.add_argument("--selftest", action="store_true")
    return ap.parse_args()


def main() -> None:
    a = parse()
    if a.selftest:
        selftest()
        return
    if not a.src or not a.out:
        fail("--src (one or more) and --out are required")
    srcs = [open_adapter(s) for s in a.src]
    tpl = open_adapter(a.template) if a.template else None
    out = Path(a.out).expanduser()
    for s in srcs:
        print(f"source {s.path}: {len(s.keys)} modules, r={s.cfg['r']} alpha={s.cfg['lora_alpha']} scale={s.scale:g}")
    if not a.verify_only:
        if (out / "adapter_model.safetensors").exists():
            fail(f"{out} already holds an adapter; pick a new --out")
        rank, alpha = resolve_rank_alpha(a, tpl)
        plans = build(srcs, out, rank, alpha, a.seed, a.pad_init, tpl)
        print(f"wrote {out}: {len(plans)} modules")
    if a.verify or a.verify_only:
        verify(out, srcs, a.n_verify, a.seed, tpl)


if __name__ == "__main__":
    sys.exit(main())
