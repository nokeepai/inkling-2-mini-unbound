"""Swap candidate BF16 tensors into a loaded HF Inkling-2-Mini, in place, and restore them.

Candidate format: harness/CANDIDATE_FORMAT.md. Name map: harness/hf_name_map.json
({ckpt_name: {"hf": param_name, "transform": "none", "ckpt_dtype", "hf_dtype", "ckpt_shape", "hf_shape"}}).

Usage (pod):
    from cands import load_candidate, restore
    info = load_candidate(model, "/srv/cands/c0")   # verifies sha256 and the copy, bit-exact
    ...
    restore(model)                                  # verifies every touched param is back to the original
"""
from __future__ import annotations

import hashlib
import json
import os

import torch
from safetensors.torch import load_file

HERE = os.path.dirname(os.path.abspath(__file__))
NAME_MAP_PATH = os.environ.get("HF_NAME_MAP", os.path.join(HERE, "hf_name_map.json"))
FORBIDDEN = ("experts", "w13_weight", "w2_weight", "_scale", "amax", "outer_scale")

_name_map: dict | None = None


def name_map() -> dict:
    global _name_map
    if _name_map is None:
        _name_map = json.load(open(NAME_MAP_PATH))
    return _name_map


SUPPLEMENTARY = os.environ.get("FID_SUPPLEMENTARY", "0") == "1"


def assert_w4a16(model) -> None:
    """The harness engine is W4A16 (PR loader, act_quant OFF). Refuse a model with expert activation fake-quant
    patched in (hf_load.enable_act_quant sets _ink_fmt) unless FID_SUPPLEMENTARY=1."""
    if SUPPLEMENTARY:
        return
    for _n, m in model.named_modules():
        if hasattr(m, "_ink_fmt"):
            raise RuntimeError("act_quant is ON in this model; harness requires W4A16 (set FID_SUPPLEMENTARY=1 "
                               "only for supplementary runs)")


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def _params(model) -> dict[str, torch.nn.Parameter]:
    return dict(model.named_parameters())


def _to_param_layout(t: torch.Tensor, entry: dict) -> torch.Tensor:
    if entry.get("transform", "none") == "T":
        t = t.T
    return t.reshape(entry["hf_shape"])


def _bits_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.dtype == b.dtype and a.shape == b.shape and torch.equal(a.contiguous().view(torch.uint8),
                                                                    b.contiguous().view(torch.uint8))


@torch.no_grad()
def load_candidate(model, cand_dir: str, verify_sha: bool = True) -> dict:
    """Copy every tensor of cand_dir/tensors.safetensors into the matching HF parameter.
    Originals are kept on CPU in model._fid_saved (first load only; a second load restores first)."""
    assert_w4a16(model)
    if getattr(model, "_fid_saved", None):
        restore(model)
    path = os.path.join(cand_dir, "tensors.safetensors")
    spec = json.load(open(os.path.join(cand_dir, "spec.json")))
    if verify_sha and os.path.exists(os.path.join(cand_dir, "sha256.txt")):
        want = open(os.path.join(cand_dir, "sha256.txt")).read().split()[0]
        got = _sha256(path)
        assert got == want, f"{cand_dir}: sha256 mismatch {got} != {want}"
    tensors = load_file(path)
    nm, params = name_map(), _params(model)
    saved: dict[str, torch.Tensor] = {}
    model._fid_saved = saved
    for ck, t in tensors.items():
        assert not any(f in ck for f in FORBIDDEN), f"forbidden tensor in candidate: {ck}"
        assert ck in nm, f"{ck} not in hf_name_map.json"
        e = nm[ck]
        assert t.dtype == torch.bfloat16, f"{ck}: candidate tensors must be BF16, got {t.dtype}"
        assert list(t.shape) == e["ckpt_shape"], f"{ck}: shape {list(t.shape)} != {e['ckpt_shape']}"
        p = params[e["hf"]]
        saved[e["hf"]] = p.detach().to("cpu", copy=True)
        new = _to_param_layout(t, e).to(device=p.device, dtype=p.dtype)
        p.data.copy_(new)
        # Verify: the param, cast back to the checkpoint dtype/layout, equals the candidate tensor bit-exact.
        back = p.detach().reshape(e["hf_shape"])
        if e.get("transform", "none") == "T":
            back = back.T
        back = back.reshape(e["ckpt_shape"]).to("cpu", torch.bfloat16)
        assert _bits_equal(back, t), f"{ck}: copy verification failed"
    return {"cand_id": spec.get("cand_id"), "n_tensors": len(tensors), "tensors": sorted(tensors)}


@torch.no_grad()
def restore(model) -> int:
    """Put back every original saved by load_candidate and verify bit-exact. Returns the count."""
    saved = getattr(model, "_fid_saved", None) or {}
    params = _params(model)
    for name, orig in saved.items():
        p = params[name]
        p.data.copy_(orig.to(p.device))
        assert _bits_equal(p.detach().cpu(), orig), f"restore failed for {name}"
    n = len(saved)
    model._fid_saved = {}
    return n
