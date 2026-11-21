"""Reusable Transformers loader for Inkling-2-Mini (PR #51050 @ a10ebc3c).

Pod-side usage (ink-research-*, pinned transformers installed):
    import sys; sys.path.insert(0, "/root/parity")
    from hf_load import load, render, encode
    model, proc = load()                      # /root/w/orig, experts dequantised to BF16 (~53 GB) on cuda:0
    ids = encode(proc, "Hello", effort=0)     # LongTensor [1, T] on model.device, thinking off

Thinking modes: the official chat template takes `reasoning_effort` (float in [0, 0.99] or a name).
effort=0 is "thinking off" (template prints "Thinking effort level: 0"); 0.5 and 0.9 are the other
modes used in this project. 0.9 is the template default. The template has no generation suffix:
generation starts right after the rendered prompt (the model emits <|message_model|> itself).

Layer-visit map: 24 physical layers; layers 6..17 run 3 times (passes 1..3), so 48 visits.
`output_hidden_states=True` returns one entry per visit. `loop_pass_hooks` records the output of a
physical layer on each call (3 calls per forward for a looped layer = passes 1, 2, 3).
"""
from __future__ import annotations

from typing import Callable

import torch

ORIG = "/root/w/orig"
OU = "/root/w/ou"
EFFORTS = {"off": 0, "0.5": 0.5, "0.9": 0.9}


def load(path: str = ORIG, device: str = "cuda:0", dtype: torch.dtype = torch.bfloat16,
         act_quant: bool = False):
    """Load model + processor. Transformers dequantises NVFP4/MXFP8 experts to `dtype` at load.
    act_quant=True emulates the checkpoint's required expert ACTIVATION quantisation (W4A4),
    as SGLang runs it; see enable_act_quant."""
    from transformers import AutoModelForMultimodalLM, AutoProcessor

    proc = AutoProcessor.from_pretrained(path)
    import os

    gpu_cap = os.environ.get("HF_GPU_GIB")  # e.g. "36": offload the rest to CPU when the GPU is shared
    if gpu_cap:
        model = AutoModelForMultimodalLM.from_pretrained(
            path, dtype=dtype, device_map="auto", max_memory={0: f"{gpu_cap}GiB", "cpu": "300GiB"})
    else:
        model = AutoModelForMultimodalLM.from_pretrained(path, dtype=dtype, device_map=device)
    model.eval()
    if act_quant:
        enable_act_quant(model)
    return model, proc


# ---- expert activation fake-quant, ported from SGLang inkling2_experts_sm90.py / inkling2_moe.py ----
def _e2m1_round(x: torch.Tensor) -> torch.Tensor:
    """Nearest-even onto {0,.5,1,1.5,2,3,4,6}, saturating at 6 (cvt.rn.satfinite.e2m1)."""
    a = x.abs()
    r = torch.where(a <= 0.25, 0.0, torch.where(a < 0.75, 0.5, torch.where(a <= 1.25, 1.0, torch.where(
        a < 1.75, 1.5, torch.where(a <= 2.5, 2.0, torch.where(a < 3.5, 3.0, torch.where(a <= 5.0, 4.0, 6.0)))))))
    return torch.where(x < 0, -r, r)


def nvfp4_fake(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Rows of x -> (NVFP4 values without the outer scale, exact in bf16; per-row outer scale amax/1536).
    Per-token amax, 16-wide blocks, E4M3 block scales, 'fouroversix_mae' choice between max->6 and max->4."""
    big = 3.4028234663852886e38
    x = x.float()
    rows, k = x.shape
    amax = x.abs().amax(-1)
    enc = (1536.0 / amax).clamp_max(big)
    dec = 1.0 / enc
    xb = x.view(rows, k // 16, 16)
    bmax = xb.abs().amax(-1)
    out = []
    for div in (6.0, 4.0):
        s = (bmax * (enc[:, None] * (1.0 / div))).clamp_max(448.0).to(torch.float8_e4m3fn).float()
        inv = (1.0 / (s * dec[:, None])).clamp_max(big)
        rec = _e2m1_round(xb * inv[..., None]) * s[..., None]
        err = (rec * (amax / 1536.0)[:, None, None] - xb).abs().sum(-1)
        out.append((rec, err))
    (rec6, err6), (rec4, err4) = out
    y = torch.where((err4 < err6)[..., None], rec4, rec6).view(rows, k)
    return torch.nan_to_num(y), amax / 1536.0


def mxfp8_fake(x: torch.Tensor) -> torch.Tensor:
    """Rows of x as MXFP8 (E4M3 values, power-of-two UE8M0 scale per 32, scale rounded up)."""
    x = x.float()
    rows, k = x.shape
    xb = x.view(rows, k // 32, 32)
    raw = xb.abs().amax(-1) / 448.0
    e = ((raw.view(torch.int32) + 0x007FFFFF) & 0x7F800000) >> 23
    e = torch.where(raw <= 2.0 ** -127, torch.zeros_like(e), e.clamp_max(254))
    scale = (e << 23).view(torch.float32).clamp_min(2.0 ** -127)
    v = (xb / scale[..., None]).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).float()
    return (v * scale[..., None]).view(rows, k)


def _qlinear(x: torch.Tensor, w: torch.Tensor, fmt: str) -> torch.Tensor:
    """Linear on the fake-quantised input; fp32 output (outer scale applied after the matmul, as the kernel)."""
    if fmt == "nvfp4":
        y, outer = nvfp4_fake(x)
        return torch.nn.functional.linear(y.to(w.dtype), w).float() * outer[:, None]
    return torch.nn.functional.linear(mxfp8_fake(x).to(w.dtype), w).float()


def _experts_forward_aq(self, hidden_states, top_k_index, top_k_weights):
    import torch.nn.functional as F

    fmt = self._ink_fmt
    final = torch.zeros_like(hidden_states, dtype=torch.float32)
    with torch.no_grad():
        mask = F.one_hot(top_k_index, num_classes=self.num_experts + 1).permute(2, 1, 0)
        hit = torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero()
    for e in hit:
        e = e[0]
        if e == self.num_experts:
            continue
        pos, tok = torch.where(mask[e])
        gate_up = _qlinear(hidden_states[tok], self.gate_up_proj[e], fmt)
        gate, up = gate_up.chunk(2, dim=-1)
        act = self.act_fn.gate_softcap(gate) * torch.sigmoid(gate) * self.act_fn.up_softcap(up)  # fp32
        down = _qlinear(act, self.down_proj[e], fmt)
        final.index_add_(0, tok, down * top_k_weights[tok, pos, None].float())
    return final.to(hidden_states.dtype)


def enable_act_quant(model, nvfp4_layers=range(0, 20)) -> None:
    """Patch every expert block: NVFP4 activations on layers 0-19, MXFP8 on 20-23 (hf_quant_config)."""
    import re
    import types

    n = 0
    for name, mod in model.named_modules():
        if type(mod).__name__ == "Inkling2LoopExperts":
            layer = int(re.search(r"layers\.(\d+)\.", name).group(1))
            mod._ink_fmt = "nvfp4" if layer in nvfp4_layers else "mxfp8"
            bound = types.MethodType(_experts_forward_aq, mod)
            if hasattr(mod, "_old_forward"):  # accelerate offload hook wraps forward; keep the wrapper
                mod._old_forward = bound
            else:
                mod.forward = bound
            n += 1
    assert n == 24, n


def render(proc, user: str | list[dict], effort: float | str = 0, system: str | None = None) -> str:
    """Render the official chat template as text. `user` is a string or a full messages list."""
    if isinstance(user, str):
        messages = [{"role": "user", "content": user}]
        if system is not None:
            messages.insert(0, {"role": "system", "content": system})
    else:
        messages = user
    tok = getattr(proc, "tokenizer", proc)
    return tok.apply_chat_template(messages, tokenize=False, reasoning_effort=effort)


def encode(proc, user: str | list[dict], effort: float | str = 0, device: str = "cuda:0",
           system: str | None = None) -> torch.Tensor:
    """Token ids for the rendered prompt (no BOS is added beyond what the template prints)."""
    tok = getattr(proc, "tokenizer", proc)
    text = render(proc, user, effort, system)
    ids = tok(text, add_special_tokens=False, return_tensors="pt").input_ids
    return ids.to(device)


def text_model(model):
    """The decoder stack (Inkling2LoopTextModel), whatever wrapper `model` is."""
    for name, mod in model.named_modules():
        if type(mod).__name__ == "Inkling2LoopTextModel":
            return mod
    raise RuntimeError("Inkling2LoopTextModel not found")


def loop_pass_hooks(model, layer_idx: int, store: list[torch.Tensor]) -> Callable[[], None]:
    """Record each call's output of physical layer `layer_idx` (cpu float32). Returns a remover."""
    layer = text_model(model).layers[layer_idx]

    def hook(_mod, _inp, out):
        t = out[0] if isinstance(out, tuple) else out
        store.append(t.detach().float().cpu())

    handle = layer.register_forward_hook(hook)
    return handle.remove


@torch.no_grad()
def greedy(model, ids: torch.Tensor, max_new_tokens: int = 64) -> list[int]:
    out = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), do_sample=False,
                         max_new_tokens=max_new_tokens)
    return out[0, ids.shape[1]:].tolist()


@torch.no_grad()
def teacher_forced_top1(model, prompt_ids: torch.Tensor, cont: list[int]) -> list[int]:
    """Argmax prediction at each continuation position given prompt + cont[:i]."""
    if not cont:
        return []
    full = torch.cat([prompt_ids, torch.tensor([cont], device=prompt_ids.device)], dim=1)
    logits = model(input_ids=full, attention_mask=torch.ones_like(full), use_cache=False).logits
    p = prompt_ids.shape[1]
    return logits[0, p - 1: p - 1 + len(cont)].argmax(-1).tolist()
