"""Offline self-test for merge_adapter.py (run through `merge_adapter.py --selftest`). Synthetic tensors only, with
the real adapter's name and shape pattern scaled down: attention lora_A [r,in] / lora_B [out,r]; experts w1/w3 with a
shared lora_A [1,r,D] and per-expert lora_B [E,I,r]; w2 with per-expert lora_A [E,r,I] and a shared lora_B [1,D,r].
Layer 0 is NVFP4 and layer 1 MXFP8, as layers 0-19 / 20-23 of the real checkpoint. Needs torch + safetensors."""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from typing import Any, Callable

import merge_adapter as M

D, I, E, R = 64, 32, 4, 4                  # d_model, expert hidden, experts, rank
ATTN_SHAPES = {"wq_du": (96, D), "wk_dv": (32, D), "wv_dv": (32, D), "gate_attn": (96, D), "wo_ud": (D, 96),
               "wr_du": (16, D)}


def check(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)
    print(f"  ok  {msg}", flush=True)


def expect_fail(fn: Callable[[], Any], msg: str) -> None:
    try:
        fn()
    except SystemExit as e:
        check("FAIL" in str(e), f"{msg} (fails loudly: {str(e)[:90]})")
        return
    raise AssertionError(f"{msg}: did not fail")


# ---------------------------------------------------------------- synthetic quantised experts

def rand_nvfp4(g: Any, n: int, rows: int, cols: int) -> tuple[Any, Any, Any, Any]:
    """Random stored NVFP4 tensors: codes (incl. -0 codes), e4m3 block scales, amax [n], denominator [1]."""
    import torch
    codes = torch.randint(0, 256, (n, rows, cols // 2), generator=g, dtype=torch.int32).to(torch.uint8)
    scale = (torch.rand(n, rows, cols // 16, generator=g) * 3 + 0.25).to(torch.float8_e4m3fn)
    amax = torch.rand(n, generator=g) * 0.5 + 0.1
    return codes, scale, amax, torch.tensor([448.0 * 6.0])


def rand_mxfp8(g: Any, n: int, rows: int, cols: int) -> tuple[Any, Any]:
    import torch
    vals = (torch.randn(n, rows, cols, generator=g) * 50).clamp(-448, 448).to(torch.float8_e4m3fn)
    scale = torch.randint(118, 124, (n, rows // 32, 1, cols // 32), generator=g).expand(-1, -1, 32, -1)
    return vals, scale.reshape(n, rows, cols // 32).to(torch.uint8).contiguous()


def test_codecs() -> None:
    import torch
    print("codecs:")
    g = torch.Generator().manual_seed(0)
    codes, scale, amax, den = rand_nvfp4(g, 3, 8, 64)
    f = amax / den
    w = M.nvfp4_dequant(codes, scale, f)
    q, s2, st = M.nvfp4_quant(w, scale, f, "keep")
    check(torch.equal(q, codes) and M.tensor_bytes(s2) == M.tensor_bytes(scale) and st.saturated == 0,
          "NVFP4 zero-delta re-encode reproduces stored bytes (incl. -0 codes)")
    vals, msc = rand_mxfp8(g, 3, 64, 64)
    w8 = M.mxfp8_dequant(vals, msc)
    q8, s8, _ = M.mxfp8_quant(w8, msc, "keep")
    check(M.tensor_bytes(q8) == M.tensor_bytes(vals) and torch.equal(s8, msc), "MXFP8 zero-delta re-encode exact")
    check(M.uniform_over_32_rows(msc), "MXFP8 32x32 scale detection")
    test_survival(w, codes, scale, f)
    test_bump(w, scale, f)
    test_bump_mxfp8(w8, msc)


def test_survival(w: Any, codes: Any, scale: Any, f: Any) -> None:
    import torch
    tiny = torch.full_like(w, 1e-7)
    q, sc, st = M.nvfp4_quant(w + tiny, scale, f, "keep")
    sv = M.Survival()
    sv.add(w, tiny, M.nvfp4_dequant(q, sc, f), M.changed_bytes(q, codes), q.numel(), st)
    check(abs(sv.summary()["survival"]) < 0.05, f"tiny delta does not survive 4-bit ({sv.summary()['survival']:.3f})")
    other = torch.randint(0, 256, codes.shape, generator=torch.Generator().manual_seed(1), dtype=torch.int32)
    grid_delta = M.nvfp4_dequant(other.to(torch.uint8), scale, f) - w
    q, sc, st = M.nvfp4_quant(w + grid_delta, scale, f, "keep")
    sv = M.Survival()
    sv.add(w, grid_delta, M.nvfp4_dequant(q, sc, f), 0, 1, st)
    check(sv.summary()["survival"] > 0.999, f"on-grid delta survives fully ({sv.summary()['survival']:.4f})")


def test_bump(w: Any, scale: Any, f: Any) -> None:
    import torch
    big = torch.zeros_like(w)
    big[0, 0, :16] = 8.0 * scale[0, 0, 0].float() * f[0]          # past 6 * scale * factor in one block
    _, _, st_keep = M.nvfp4_quant(w + big, scale, f, "keep")
    q, sc, st_bump = M.nvfp4_quant(w + big, scale, f, "bump")
    err = (M.nvfp4_dequant(q, sc, f) - (w + big))[0, 0, :16].abs().max()
    check(st_keep.saturated > 0 and st_bump.bumped_blocks >= 1 and st_bump.saturated == 0,
          f"NVFP4 out-of-range block: keep saturates ({st_keep.saturated}), bump rescales ({st_bump.bumped_blocks})")
    check(float(err) <= float(sc[0, 0, 0].float() * f[0]) * 1.0001, "bumped block error within one grid step")


def test_bump_mxfp8(w8: Any, msc: Any) -> None:
    import torch
    big = torch.zeros_like(w8)
    big[0, 0, 0] = 1000.0 * float(torch.exp2(msc[0, 0, 0].float() - 127))
    _, s_new, st = M.mxfp8_quant(w8 + big, msc, "bump")
    check(st.bumped_blocks == 32 and st.saturated == 0 and M.uniform_over_32_rows(s_new),
          "MXFP8 bump raises a whole 32x32 block and keeps scales uniform")


# ---------------------------------------------------------------- broadcasting of shared A / shared B

def rand_adapter(g: Any, scale_b: float = 1.0) -> dict[str, Any]:
    """Adapter tensors for layers 0 and 1 under the real names and the real (scaled-down) shape pattern."""
    import torch
    t: dict[str, Any] = {}
    for layer in (0, 1):
        p = f"language_model.layers.{layer}."
        for name, (o, i) in ATTN_SHAPES.items():
            t[p + f"attn.{name}.lora_A.weight"] = torch.randn(R, i, generator=g) * 0.1
            t[p + f"attn.{name}.lora_B.weight"] = torch.randn(o, R, generator=g) * 0.1 * scale_b
        for w in ("w1", "w3"):
            t[p + f"mlp.experts.{w}.lora_A.weight"] = torch.randn(1, R, D, generator=g) * 0.1
            t[p + f"mlp.experts.{w}.lora_B.weight"] = torch.randn(E, I, R, generator=g) * 0.1 * scale_b
        t[p + "mlp.experts.w2.lora_A.weight"] = torch.randn(E, R, I, generator=g) * 0.1
        t[p + "mlp.experts.w2.lora_B.weight"] = torch.randn(1, D, R, generator=g) * 0.1 * scale_b
    return t


def layer_ad(t: dict[str, Any], layer: int) -> dict[str, tuple[Any, Any]]:
    p = f"language_model.layers.{layer}.mlp.experts."
    return {w: (t[p + w + ".lora_A.weight"], t[p + w + ".lora_B.weight"]) for w in ("w1", "w2", "w3")}


def test_broadcast() -> None:
    import torch
    print("expert delta broadcasting:")
    ad = layer_ad(rand_adapter(torch.Generator().manual_seed(2)), 0)
    s = 0.7
    d13 = M.expert_delta_w13(ad, 1, 3, s, False, "cpu")
    ok = all(torch.allclose(d13[k, 0::2], s * ad["w1"][1][e] @ ad["w1"][0][0], atol=1e-6)
             and torch.allclose(d13[k, 1::2], s * ad["w3"][1][e] @ ad["w3"][0][0], atol=1e-6)
             for k, e in enumerate((1, 2)))
    check(ok and d13.shape == (2, 2 * I, D), "w13 delta = interleaved (gate=w1 even rows, up=w3 odd rows), shared A")
    sw = M.expert_delta_w13(ad, 1, 3, s, True, "cpu")
    check(torch.allclose(sw[0, 0::2], d13[0, 1::2]) and torch.allclose(sw[0, 1::2], d13[0, 0::2]), "--swap-w1-w3")
    d2 = M.expert_delta_w2(ad, 1, 3, s, "cpu")
    ok2 = all(torch.allclose(d2[k], s * ad["w2"][1][0] @ ad["w2"][0][e], atol=1e-6) for k, e in enumerate((1, 2)))
    check(ok2 and d2.shape == (2, D, I), "w2 delta = shared B @ per-expert A")


# ---------------------------------------------------------------- names

def test_names() -> None:
    print("name parsing:")
    check(M.parse_name("language_model.layers.7.attn.wo_ud.lora_B.weight") == (7, "wo_ud", "attn", "B"), "attention")
    check(M.parse_name("base_model.model.language_model.layers.0.mlp.experts.w3.lora_A.weight")
          == (0, "w3", "expert", "A"), "expert with PEFT prefix")
    expect_fail(lambda: M.group_modules(["language_model.layers.0.mlp.gate.lora_A.weight"]), "router tensor")
    expect_fail(lambda: M.group_modules(["language_model.layers.0.attn.wq_du.lora_A.weight"]), "unpaired lora_A")
    expect_fail(lambda: M.lora_scale({"r": 4, "lora_alpha": 4, "use_rslora": True}, 1.0), "rsLoRA config")


# ---------------------------------------------------------------- end to end on a tiny checkpoint

def build_base(root: str, g: Any) -> dict[str, Any]:
    import torch
    from safetensors.torch import save_file
    t: dict[str, Any] = {"model.llm.embed.weight": torch.randn(10, D, generator=g).to(torch.bfloat16)}
    for layer in (0, 1):
        p = f"model.llm.layers.{layer}."
        for name, shape in ATTN_SHAPES.items():
            t[p + f"attn.{name}.weight"] = torch.randn(*shape, generator=g).to(torch.bfloat16)
        t[p + "mlp.gate.weight"] = torch.randn(E, D, generator=g).to(torch.bfloat16)
    for which, rows, cols in (("w13", 2 * I, D), ("w2", D, I)):
        k = f"model.llm.layers.0.mlp.experts.{which}_weight"
        t[k], t[k + "_scale"], t[k + "_amax"], t[k + "_outer_scale_denominator"] = rand_nvfp4(g, E, rows, cols)
        k1 = f"model.llm.layers.1.mlp.experts.{which}_weight"
        t[k1], t[k1 + "_scale"] = rand_mxfp8(g, E, rows, cols)
    fn = "model-00001-of-00001.safetensors"
    save_file(t, os.path.join(root, fn))
    save_file({"fuse_weight": torch.randn(4, 4, generator=g)}, os.path.join(root, "mtp.safetensors"))
    json.dump({"weight_map": {k: fn for k in t}}, open(os.path.join(root, "model.safetensors.index.json"), "w"))
    json.dump({"text_config": {"hidden_size": D}}, open(os.path.join(root, "config.json"), "w"))
    groups = [{"weight_transform": "nvfp4_nearest", "weight_scale_format": "amax_denominator",
               "weight_block_shape": [1, 16], "tensors": [f"model.llm.layers.0.mlp.experts.{w}_weight" for w in ("w13", "w2")]},
              {"weight_transform": "mxfp8_nearest", "weight_block_shape": [32, 32],
               "tensors": [f"model.llm.layers.1.mlp.experts.{w}_weight" for w in ("w13", "w2")]}]
    json.dump({"model": {"parameter_groups": groups}}, open(os.path.join(root, "hf_quant_config.json"), "w"))
    return t


def write_adapter(root: str, t: dict[str, Any], alpha: int = 8) -> None:
    from safetensors.torch import save_file
    os.makedirs(root, exist_ok=True)
    save_file(t, os.path.join(root, "adapter_model.safetensors"))
    json.dump({"peft_type": "LORA", "r": R, "lora_alpha": alpha, "use_rslora": False, "use_dora": False,
               "rank_pattern": {}, "alpha_pattern": {}, "target_modules": "all-linear"},
              open(os.path.join(root, "adapter_config.json"), "w"))


def merge(base: str, adapter: str, out: str, parts: str, chunk: int = 3, mode: str = "keep",
          rounding: str = "nearest", seed: int = 0) -> dict[str, Any]:
    a = argparse.Namespace(base=base, adapter=adapter, out=out, parts=parts, scale=1.0, report="", device="cpu",
                           expert_chunk=chunk, expert_scales=mode, swap_w1_w3=False, selftest=False,
                           rounding=rounding, seed=seed, spectral=True)
    M.run_merge(a)
    return json.load(open(out + ".merge_report.json"))


def load(out: str) -> dict[str, Any]:
    from safetensors.torch import load_file
    return load_file(os.path.join(out, "model-00001-of-00001.safetensors"))


def check_attn(base_t: dict[str, Any], ad: dict[str, Any], got: dict[str, Any], s: float) -> bool:
    import torch
    ok = True
    for layer in (0, 1):
        for name in ATTN_SHAPES:
            p = f"language_model.layers.{layer}.attn.{name}"
            want = (base_t[f"model.llm.layers.{layer}.attn.{name}.weight"].double()
                    + s * ad[p + ".lora_B.weight"].double() @ ad[p + ".lora_A.weight"].double()).to(torch.bfloat16)
            ok &= torch.equal(got[f"model.llm.layers.{layer}.attn.{name}.weight"], want)
    return ok


def expected_experts(base_t: dict[str, Any], ad: dict[str, Any], s: float) -> dict[str, Any]:
    """Whole-tensor (single chunk) reference for the chunked merge: same codecs, no chunking, no file offsets."""
    out = {}
    for layer in (0, 1):
        la = layer_ad(ad, layer)
        for which in ("w13", "w2"):
            k = f"model.llm.layers.{layer}.mlp.experts.{which}_weight"
            d = M.expert_delta_w13(la, 0, E, s, False, "cpu") if which == "w13" else M.expert_delta_w2(la, 0, E, s, "cpu")
            if layer == 0:
                f = base_t[k + "_amax"] / base_t[k + "_outer_scale_denominator"]
                w = M.nvfp4_dequant(base_t[k], base_t[k + "_scale"], f)
                out[k] = M.nvfp4_quant(w + d, base_t[k + "_scale"], f, "keep")[0]
            else:
                w = M.mxfp8_dequant(base_t[k], base_t[k + "_scale"])
                out[k] = M.mxfp8_quant(w + d, base_t[k + "_scale"], "keep")[0]
    return out


def test_end_to_end() -> None:
    import torch
    print("end to end (tiny checkpoint, chunk 3 of 4 experts):")
    g = torch.Generator().manual_seed(3)
    with tempfile.TemporaryDirectory() as tmp:
        base, adp = os.path.join(tmp, "base"), os.path.join(tmp, "adapter")
        os.makedirs(base)
        base_t = build_base(base, g)
        ad = rand_adapter(g, scale_b=20.0)          # large B so part of the expert delta crosses grid steps
        write_adapter(adp, ad)
        s = 8 / R
        rep = merge(base, adp, os.path.join(tmp, "attn"), "attn")
        got = load(os.path.join(tmp, "attn"))
        check(check_attn(base_t, ad, got, s), "attn: 12 tensors = bf16(W + alpha/r * B @ A)")
        same = all(torch.equal(got[k].view(torch.uint8) if got[k].dtype != torch.bfloat16 else got[k],
                               base_t[k].view(torch.uint8) if base_t[k].dtype != torch.bfloat16 else base_t[k])
                   for k in base_t if ".attn." not in k)
        check(same and len(rep["skipped_modules"]) == 6, "attn: experts and other tensors untouched, 6 expert modules skipped")
        check(os.stat(os.path.join(base, "mtp.safetensors")).st_ino == os.stat(os.path.join(tmp, "attn", "mtp.safetensors")).st_ino,
              "attn: unchanged files hard-linked")
        rep = merge(base, adp, os.path.join(tmp, "full"), "attn_experts")
        got = load(os.path.join(tmp, "full"))
        want = expected_experts(base_t, ad, s)
        check(check_attn(base_t, ad, got, s), "full: attention identical to attn variant")
        check(all(M.tensor_bytes(got[k]) == M.tensor_bytes(v) for k, v in want.items()),
              "full: chunked in-place expert writes == whole-tensor requant (NVFP4 + MXFP8)")
        sv = rep["survival_per_layer"]
        check(set(sv) == {"0", "1"} and 0 < sv["0"]["survival"] <= 1 and sv["1"]["survival"] > sv["0"]["survival"],
              f"survival reported per layer (NVFP4 {sv['0']['survival']:.3f} < MXFP8 {sv['1']['survival']:.3f})")
        check(rep["verify"]["rewritten_verified"] == 12 + 8,
              f"full: 12 attention + 4 expert weight + 4 scale tensors read back ({rep['verify']})")
        expect_fail(lambda: merge(base, adp, os.path.join(tmp, "full"), "attn_experts"), "existing --out")
        bad = dict(ad)
        bad["language_model.layers.0.mlp.gate.lora_A.weight"] = torch.zeros(R, D)
        write_adapter(os.path.join(tmp, "bad"), bad)
        expect_fail(lambda: merge(base, os.path.join(tmp, "bad"), os.path.join(tmp, "x"), "attn"),
                    "adapter tensor with no checkpoint target")
        wrong = dict(ad)
        wrong["language_model.layers.1.attn.wk_dv.lora_B.weight"] = torch.zeros(48, R)
        write_adapter(os.path.join(tmp, "wrong"), wrong)
        expect_fail(lambda: merge(base, os.path.join(tmp, "wrong"), os.path.join(tmp, "y"), "attn"), "shape mismatch")
        check(not os.path.exists(os.path.join(tmp, "y")), "no output dir left after a failed merge")


# ---------------------------------------------------------------- stochastic rounding

def mean_error(samples: list[Any], target: Any, d: Any) -> float:
    """|mean(samples) - target| / |d|: how much of the intended delta the average misses."""
    import torch
    m = torch.stack([s.double() for s in samples]).mean(0)
    return float((m - target.double()).norm() / d.double().norm())


def test_sr_bf16() -> None:
    import torch
    g = torch.Generator().manual_seed(5)
    w = torch.randn(32, 32, generator=g).to(torch.bfloat16)
    step = w.float().abs() * 2.0 ** -7                    # bf16 spacing near |w|
    d = 0.3 * step * torch.randn(32, 32, generator=g)
    d[:, :8] = 0.0                                        # zero-delta columns must not move
    w[0, 0] = -0.0                                        # a stored -0 with zero delta must stay -0
    x = w.float() + d
    srs = [M.attn_candidates(w, d.double(), "attn", s)["stochastic"] for s in range(600)]
    near = (w.double() + d.double()).to(torch.bfloat16)
    e_sr, e_near = mean_error([s.float() for s in srs], x, d), mean_error([near.float()], x, d)
    check(e_sr < 0.15 and e_near > 0.6, f"BF16: mean of 600 stochastic merges -> W+delta (miss {e_sr:.3f} of |delta|; "
          f"nearest misses {e_near:.3f})")
    check(all(torch.equal(s[:, :8].view(torch.int16), w[:, :8].view(torch.int16)) for s in srs[:50]),
          "BF16: zero-delta elements byte-identical")
    again = M.attn_candidates(w, d.double(), "attn", 3)["stochastic"]
    check(torch.equal(again.view(torch.int16), srs[3].view(torch.int16))
          and not torch.equal(srs[3], srs[4]), "BF16: same seed reproduces bytes, another seed differs")


def nvfp4_sr_case(g: Any) -> tuple[Any, Any, Any, Any, Any, Any]:
    """Stored NVFP4 chunk + a sub-step delta (zero on the first 16 columns and on saturating +-6 elements)."""
    import torch
    codes, scale, amax, den = rand_nvfp4(g, 2, 8, 64)
    f = amax / den
    w = M.nvfp4_dequant(codes, scale, f)
    unit = (scale.float() * f[:, None, None]).repeat_interleave(16, -1)
    d = 0.2 * unit * torch.randn(w.shape, generator=g)
    d[..., :16] = 0.0
    d[(w.abs() >= 6 * unit * 0.999)] = 0.0                # +-6 cannot move outward without saturating
    return codes, scale, f, w, d, unit


def sr_outputs(fmt: str, q: Any, sc: Any, f: Any, w: Any, d: Any, seed: int) -> tuple[Any, Any]:
    """The production path (merge_adapter.requant_both) on one synthetic chunk: (stochastic codes, dequantised)."""
    t = M.ExpertTensor("model.llm.layers.0.mlp.experts.w2_weight", fmt, 0, "w2")
    a = argparse.Namespace(seed=seed, device="cpu", expert_scales="keep")
    c = M.ChunkCtx(t, 0, w.shape[0], q, sc, w, d, f)
    qn, scn, _ = M.requant_both(c, a)["stochastic"]
    return qn, M.dequant(t, qn, scn, f, 0, w.shape[0])


def test_sr_nvfp4() -> None:
    import torch
    g = torch.Generator().manual_seed(6)
    codes, scale, f, w, d, _ = nvfp4_sr_case(g)
    codes[..., 0] = 0x08                                  # a stored -0 (low nibble) with zero delta must stay -0
    w = M.nvfp4_dequant(codes, scale, f)
    x, keep = w + d, d == 0
    outs = [sr_outputs(M.NVFP4, codes, scale, f, w, d, s) for s in range(800)]
    qn = M.nvfp4_quant(x, scale, f, "keep")[0]
    e_sr = mean_error([b for _, b in outs], x, d)
    e_near = mean_error([M.nvfp4_dequant(qn, scale, f)], x, d)
    check(e_sr < 0.15 and e_near > 0.8, f"NVFP4: mean of 800 stochastic merges -> W+delta (miss {e_sr:.3f}; "
          f"nearest misses {e_near:.3f})")
    b0 = outs[0][1]
    check(torch.equal(b0[keep], w[keep]) and torch.equal(torch.signbit(b0[keep]), torch.signbit(w[keep]))
          and torch.equal(outs[0][0][..., :8], codes[..., :8]), "NVFP4: zero-delta elements keep their codes (incl. -0)")
    again = sr_outputs(M.NVFP4, codes, scale, f, w, d, 11)[0]
    check(torch.equal(again, outs[11][0]) and not torch.equal(outs[11][0], outs[12][0]),
          "NVFP4: same seed reproduces bytes, another seed differs")


def test_sr_mxfp8() -> None:
    import torch
    g = torch.Generator().manual_seed(7)
    vals, msc = rand_mxfp8(g, 2, 32, 64)
    w = M.mxfp8_dequant(vals, msc)
    d = 0.3 * w.abs() * 2.0 ** -3 * torch.randn(w.shape, generator=g)   # ~0.3 of the local e4m3 step
    d[..., :32] = 0.0
    x = w + d
    outs = [sr_outputs(M.MXFP8, vals, msc, None, w, d, s) for s in range(800)]
    e_sr = mean_error([b for _, b in outs], x, d)
    e_near = mean_error([M.mxfp8_dequant(M.mxfp8_quant(x, msc, "keep")[0], msc)], x, d)
    check(e_sr < 0.15 and e_near > 0.5, f"MXFP8: mean of 800 stochastic merges -> W+delta (miss {e_sr:.3f}; "
          f"nearest misses {e_near:.3f})")
    again = sr_outputs(M.MXFP8, vals, msc, None, w, d, 5)[0]
    check(torch.equal(outs[0][0][..., :32].view(torch.uint8), vals[..., :32].view(torch.uint8))
          and M.tensor_bytes(again) == M.tensor_bytes(outs[5][0]), "MXFP8: zero-delta bytes kept; same seed reproduces")


def test_sr_end_to_end() -> None:
    import torch
    print("stochastic merge end to end:")
    g = torch.Generator().manual_seed(8)
    with tempfile.TemporaryDirectory() as tmp:
        base, adp = os.path.join(tmp, "base"), os.path.join(tmp, "adapter")
        os.makedirs(base)
        build_base(base, g)
        write_adapter(adp, rand_adapter(g, scale_b=0.05))           # small delta: most of it under half a step
        r1 = merge(base, adp, os.path.join(tmp, "sr1"), "attn_experts", chunk=3, rounding="stochastic", seed=7)
        merge(base, adp, os.path.join(tmp, "sr2"), "attn_experts", chunk=2, rounding="stochastic", seed=7)
        merge(base, adp, os.path.join(tmp, "sr3"), "attn_experts", chunk=3, rounding="stochastic", seed=8)
        shard = "model-00001-of-00001.safetensors"
        b1, b2, b3 = (open(os.path.join(tmp, x, shard), "rb").read() for x in ("sr1", "sr2", "sr3"))
        check(b1 == b2 and b1 != b3, "same seed -> identical checkpoint bytes, also across chunk sizes; new seed differs")
        fn, fs = r1["fidelity"]["nearest"]["all"], r1["fidelity"]["stochastic"]["all"]
        check(fs["projected_survival"] > fn["projected_survival"] and fs["projected_survival"] > 0.5,
              f"report: scale along delta nearest {fn['projected_survival']:.3f} < stochastic {fs['projected_survival']:.3f}")
        groups = set(r1["fidelity"]["stochastic"])
        check({"attn", "attn.wq_du", "experts.nvfp4", "experts.mxfp8", "experts.layer0", "all"} <= groups,
              f"report groups per mode: {sorted(groups)[:6]}...")
        spec = r1["spectral"]
        check(len(spec) > 0 and {r["mode"] for r in spec} == {"nearest", "stochastic"}
              and all({"ratio_top", "ratio_rand", "noise_top_over_rand"} <= set(r) for r in spec),
              f"spectral sample: {len(spec)} rows over both modes")


def run_selftest() -> None:
    test_names()
    test_codecs()
    test_broadcast()
    print("stochastic rounding:")
    test_sr_bf16()
    test_sr_nvfp4()
    test_sr_mxfp8()
    test_end_to_end()
    test_sr_end_to_end()
    print("SELFTEST_OK", flush=True)
