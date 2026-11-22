"""Merge a LoRA (train_lora.py) into the v1.0 BF16 tensors; write the edited tensors under checkpoint names.

  python3 merge_lora.py <lora.pt> <base_ckpt_dir> <hf_name_map.json> <out.safetensors> [merge_scale=1.0]

W' = bf16( W.float() + merge_scale * (alpha/r) * B @ A ), fp32 on CPU. Also writes <out>.json with per-tensor
relative delta norms.
"""
from __future__ import annotations

import json
import os
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def main() -> None:
    lora_p, base, map_p, out = sys.argv[1:5]
    ms = float(sys.argv[5]) if len(sys.argv) > 5 else 1.0
    d = torch.load(lora_p, map_location="cpu", weights_only=False)
    a = d["meta"]["args"]
    scale = a["alpha"] / a["rank"] * ms
    hf2ck = {}
    for ck, v in json.load(open(map_p)).items():
        hf = v["hf"]
        hf2ck[hf[: -len(".weight")] if hf.endswith(".weight") else hf] = ck
    wm = json.load(open(os.path.join(base, "model.safetensors.index.json")))["weight_map"]
    res, stats = {}, {}
    for hfname, (A, B) in sorted(d["lora"].items()):
        key = hfname.replace("model.language_model.", "model.language_model.", 1)
        ck = hf2ck.get(key)
        assert ck is not None, hfname
        with safe_open(os.path.join(base, wm[ck]), "pt") as f:
            W = f.get_tensor(ck)
        assert W.dtype == torch.bfloat16 and tuple(W.shape) == (B.shape[0], A.shape[1]), (ck, W.shape, A.shape, B.shape)
        delta = scale * (B.double() @ A.double())
        Wn = (W.double() + delta).to(torch.bfloat16)
        res[ck] = Wn.contiguous()
        stats[ck] = {"rel_delta": float(delta.norm() / W.double().norm()),
                     "changed_frac": float((Wn != W).float().mean())}
    save_file(res, out)
    json.dump({"lora": lora_p, "merge_scale": ms, "scale": scale, "tensors": stats}, open(out + ".json", "w"), indent=1)
    print("MERGED", len(res), "mean rel_delta", sum(s["rel_delta"] for s in stats.values()) / len(stats))


if __name__ == "__main__":
    main()
