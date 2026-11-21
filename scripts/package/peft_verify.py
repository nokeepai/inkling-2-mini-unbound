"""Load the exported adapters with PEFT itself (PeftModel.from_pretrained) on a stub module tree that has the real
HF module names, merge, and compare every merged weight with the canonical edited BF16 tensor.

  python3 peft_verify.py <cand_dir> <orig_dir> <export_dir> [--build-cand PATH] [--name-map PATH]

Checks key names, adapter_config (target_modules, rank/alpha patterns, use_dora) and the merge, end to end.
Writes <export_dir>/peft_verify.json.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os

import torch
from safetensors import safe_open
from safetensors.torch import load_file
from torch import nn


def stub(mods: dict[str, torch.Tensor]) -> nn.Module:
    """nn.Module tree with a bias-free Linear at each dotted path, weights copied in (base dtype kept)."""
    root = nn.Module()
    for path, W in mods.items():
        node = root
        parts = path.split(".")
        for p in parts[:-1]:
            if not hasattr(node, p):
                node.add_module(p, nn.Module())
            node = getattr(node, p)
        lin = nn.Linear(W.shape[1], W.shape[0], bias=False, dtype=W.dtype)
        with torch.no_grad():
            lin.weight.copy_(W)
        node.add_module(parts[-1], lin)
    return root


def check(adapter_dir: str, base: dict[str, torch.Tensor], want: dict[str, torch.Tensor]) -> dict:
    import peft
    model = peft.PeftModel.from_pretrained(stub({k: v.clone() for k, v in base.items()}), adapter_dir)
    loaded = sum(1 for n, _ in model.named_parameters() if "lora_" in n)
    merged = model.merge_and_unload()
    out, total = {}, 0
    for path, Wn in want.items():
        got = merged.get_submodule(path).weight.detach()
        if got.dtype != Wn.dtype:
            got = got.to(Wn.dtype)
        nd = int((got != Wn).sum())
        total += nd
        a = Wn.float().abs().clamp_min(torch.finfo(torch.bfloat16).tiny)
        ulp = torch.pow(2.0, torch.floor(torch.log2(a)) - 7)
        out[path] = {"n_diff": nd, "max_diff_ulp": float(((got.float() - Wn.float()).abs() / ulp).max())}
    return {"peft_version": peft.__version__, "lora_params_loaded": loaded, "n_diff_total": total,
            "max_diff_ulp": max(v["max_diff_ulp"] for v in out.values()),
            "elements": sum(w.numel() for w in want.values()), "n_diff_per_module": out}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cand"), ap.add_argument("orig"), ap.add_argument("export")
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--build-cand", default=os.path.join(here, "build_cand.py"))
    ap.add_argument("--name-map", default=os.path.join(here, "hf_name_map.json"))
    a = ap.parse_args()
    spec = importlib.util.spec_from_file_location("build_cand", a.build_cand)
    bc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bc)
    Wn = bc.build(bc.parse(load_file(os.path.join(a.cand, "lowrank.safetensors"))), a.orig)
    torch.set_num_threads(4)
    nm = json.load(open(a.name_map))
    files = bc.weight_files(a.orig)
    base, want = {}, {}
    for n in Wn:
        path = nm[n]["hf"].removesuffix(".weight")
        dt = getattr(torch, nm[n].get("hf_dtype", "bfloat16"))  # HF keeps the router in fp32
        with safe_open(files[n], "pt") as f:
            base[path] = f.get_tensor(n).to(dt)
        want[path] = Wn[n].to(dt)
    res = {k: check(os.path.join(a.export, k), base, want) for k in ("dora", "lora") if os.path.isdir(os.path.join(a.export, k))}
    json.dump(res, open(os.path.join(a.export, "peft_verify.json"), "w"), indent=1)
    print(json.dumps({k: {x: v[x] for x in ("peft_version", "lora_params_loaded", "n_diff_total", "max_diff_ulp", "elements")} for k, v in res.items()}))


if __name__ == "__main__":
    main()
