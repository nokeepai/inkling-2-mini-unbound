"""Faster drop-in for hf_load._experts_forward_aq (same math, fewer kernel launches).

Activation fake-quant is per row (per token), so quantising all rows once and then slicing per expert gives the
same values as quantising each expert's rows separately. Matmuls stay per expert on the same rows, so results
should be bit-identical to the reference (checked by test_fast_aq.py).
"""
from __future__ import annotations

import types

import torch

from hf_load import mxfp8_fake, nvfp4_fake


def _quant_rows(x: torch.Tensor, fmt: str):
    if fmt == "nvfp4":
        y, outer = nvfp4_fake(x)
        return y, outer
    return mxfp8_fake(x), None


def _experts_forward_fast(self, hidden_states, top_k_index, top_k_weights):
    fmt = self._ink_fmt
    w = self.gate_up_proj
    n_tok, k = top_k_index.shape
    flat_e = top_k_index.reshape(-1)
    valid = flat_e < self.num_experts
    order = torch.argsort(flat_e, stable=True)
    order = order[valid[order]]
    tok = order // k
    pos = order % k
    counts = torch.bincount(flat_e[order], minlength=self.num_experts).tolist()

    xq, xo = _quant_rows(hidden_states, fmt)          # quantise every token once
    xq = xq.to(w.dtype)
    gu_rows = []
    start = 0
    for e, c in enumerate(counts):
        if c == 0:
            continue
        t = tok[start:start + c]
        gu = torch.nn.functional.linear(xq[t], self.gate_up_proj[e]).float()
        gu_rows.append(gu)
        start += c
    if not gu_rows:
        return torch.zeros_like(hidden_states)
    gate_up = torch.cat(gu_rows)
    if xo is not None:
        gate_up = gate_up * xo[tok][:, None]
    gate, up = gate_up.chunk(2, dim=-1)
    act = self.act_fn.gate_softcap(gate) * torch.sigmoid(gate) * self.act_fn.up_softcap(up)
    aq, ao = _quant_rows(act, fmt)                    # quantise every (token, expert) row once
    aq = aq.to(w.dtype)
    outs = []
    start = 0
    for e, c in enumerate(counts):
        if c == 0:
            continue
        outs.append(torch.nn.functional.linear(aq[start:start + c], self.down_proj[e]).float())
        start += c
    down = torch.cat(outs)
    if ao is not None:
        down = down * ao[:, None]
    final = torch.zeros_like(hidden_states, dtype=torch.float32)
    # Reference adds per expert in ascending expert order; index_add_ over the expert-sorted rows keeps that order
    # per token only up to atomic-add ordering, so do it per expert to stay bit-identical.
    start = 0
    wts = top_k_weights[tok, pos, None].float()
    for e, c in enumerate(counts):
        if c == 0:
            continue
        final.index_add_(0, tok[start:start + c], down[start:start + c] * wts[start:start + c])
        start += c
    return final.to(hidden_states.dtype)


def enable_fast_aq(model) -> int:
    n = 0
    for _name, mod in model.named_modules():
        if type(mod).__name__ == "Inkling2LoopExperts":
            assert hasattr(mod, "_ink_fmt"), "call hf_load.enable_act_quant first"
            mod.forward = types.MethodType(_experts_forward_fast, mod)
            n += 1
    return n


def _experts_forward_fast16(self, hidden_states, top_k_index, top_k_weights):
    """Same ops as the PR's Inkling2LoopExperts.forward (W4A16), without the per-expert host sync."""
    n_tok, k = top_k_index.shape
    flat_e = top_k_index.reshape(-1)
    valid = flat_e < self.num_experts
    order = torch.argsort(flat_e, stable=True)
    order = order[valid[order]]
    tok = order // k
    pos = order % k
    counts = torch.bincount(flat_e[order], minlength=self.num_experts).tolist()
    final = torch.zeros_like(hidden_states, dtype=torch.float32)
    start = 0
    for e, c in enumerate(counts):
        if c == 0:
            continue
        t, p = tok[start:start + c], pos[start:start + c]
        h = self._apply_gate(torch.nn.functional.linear(hidden_states[t], self.gate_up_proj[e]))
        h = torch.nn.functional.linear(h, self.down_proj[e])
        h = h * top_k_weights[t, p, None]
        final.index_add_(0, t, h.to(final.dtype))
        start += c
    return final.to(hidden_states.dtype)


def enable_fast16(model) -> int:
    n = 0
    for _name, mod in model.named_modules():
        if type(mod).__name__ == "Inkling2LoopExperts":
            mod.forward = types.MethodType(_experts_forward_fast16, mod)
            n += 1
    return n
