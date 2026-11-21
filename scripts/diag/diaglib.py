"""Shared helpers for the Phase 2 diagnostics (agent diag). Runs on the pod."""
from __future__ import annotations

import json
import sys
import time

import torch

sys.path.insert(0, "/root/parity")
from hf_load import load, render, text_model  # noqa: E402

END_MSG, MSG_MODEL, THINK, TEXT, EOS = 200010, 200001, 200008, 200004, 200006
PREFIX = [MSG_MODEL, THINK]
NV = 48


def log(*a) -> None:
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def read_jsonl(path: str) -> list[dict]:
    return [json.loads(l) for l in open(path) if l.strip()]


def prompt_ids(proc, text: str, effort: float) -> list[int]:
    tok = getattr(proc, "tokenizer", proc)
    return tok(render(proc, text, effort), add_special_tokens=False).input_ids


def left_pad(seqs: list[list[int]], pad: int, device="cuda:0"):
    n = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), n), pad, dtype=torch.long)
    mask = torch.zeros((len(seqs), n), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, n - len(s):] = torch.tensor(s)
        mask[i, n - len(s):] = 1
    return ids.to(device), mask.to(device)


class VisitCounter:
    """Maps each call of a physical layer's submodule to its visit index (call order)."""

    def __init__(self, tm):
        self.calls: dict[int, int] = {}
        self.handle = tm.register_forward_pre_hook(self._reset)

    def _reset(self, *_a, **_k):
        self.calls = {}

    def visit(self, layer: int, key: str) -> int:
        k = (layer, key)
        c = self.calls.get(k, 0)
        self.calls[k] = c + 1
        if layer < 6:
            return layer
        if layer >= 18:
            return layer + 24
        return layer + 12 * c


def visit_layer(v: int) -> int:
    if v < 18:
        return v
    if v < 42:
        return 6 + (v - 6) % 12
    return v - 24


def project_out_hooks(model, Q: torch.Tensor, layers: list[int], which: list[str]):
    """Forward hooks removing span(Q) (Q: [2304, r], orthonormal, fp32) from writer outputs."""
    tm = text_model(model)
    handles = []
    Qd = Q.to("cuda:0", torch.float32)

    def hook(_m, _i, out):
        y = out.float()
        y = y - (y @ Qd) @ Qd.T
        return y.to(out.dtype)

    for i in layers:
        for w in which:
            handles.append(getattr(tm.layers[i], w).register_forward_hook(hook))
    return lambda: [h.remove() for h in handles]


def budget_batches(lens: list[int], max_rows: int, max_tokens: int) -> list[list[int]]:
    """Indices sorted by length, grouped so rows <= max_rows and rows * max_len <= max_tokens."""
    order = sorted(range(len(lens)), key=lambda i: lens[i])
    out, cur = [], []
    for i in order:
        if cur and (len(cur) + 1 > max_rows or (len(cur) + 1) * lens[i] > max_tokens):
            out.append(cur)
            cur = []
        cur.append(i)
    if cur:
        out.append(cur)
    return out
