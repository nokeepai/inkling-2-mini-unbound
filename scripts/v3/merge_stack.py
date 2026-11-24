"""Merge a STACK of LoRAs (e.g. SFT then DPO, from train_lora.py / train_dpo.py) into the base BF16 tensors in one step.

  python3 merge_stack.py <base_ckpt_dir> <hf_name_map.json> <out.safetensors> <lora1.pt>[:merge_scale] [<lora2.pt>[:ms]] ...

e.g. python3 merge_stack.py /root/w/v2 hf_name_map.json cands/v3a/tensors.safetensors sft/lora_final.pt:1.0 dpo/lora_final.pt

W' = bf16( W.double() + sum_i ms_i * (alpha_i/r_i) * B_i @ A_i ), fp64 on CPU, single rounding.
Output: the edited tensors under the base checkpoint's own names (same as merge_lora.py), plus <out>.json with
per-tensor relative delta norms (total and per LoRA). A DPO LoRA records the SFT LoRA/scale it was trained on top of;
this script warns if the stack does not contain that SFT LoRA at that scale.
"""
from __future__ import annotations

import json
import os
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def parse_spec(spec: str) -> tuple[str, float]:
    if ":" in spec and not os.path.exists(spec):
        path, ms = spec.rsplit(":", 1)
        return path, float(ms)
    return spec, 1.0


def check_stack(loras: list[dict]) -> None:
    for d in loras:
        a = d["meta"]["args"]
        if not a.get("sft_lora"):
            continue
        tail = lambda q: "/".join(os.path.abspath(q).split("/")[-2:])  # noqa: E731  run_dir/file
        want = (tail(a["sft_lora"]), float(a.get("sft_scale", 1.0)))
        have = [(tail(x["path"]), x["ms"]) for x in loras if x is not d]
        if want not in have:
            print(f"WARNING: {d['path']} was trained on top of {want}, stack has {have}", flush=True)


def main() -> None:
    base, map_p, out = sys.argv[1:4]
    loras = []
    for spec in sys.argv[4:]:
        path, ms = parse_spec(spec)
        d = torch.load(path, map_location="cpu", weights_only=False)
        a = d["meta"]["args"]
        loras.append({"path": path, "ms": ms, "scale": a["alpha"] / a["rank"] * ms, "meta": d["meta"], "lora": d["lora"]})
    assert loras, "no LoRA given"
    check_stack(loras)
    hf2ck = {}
    for ck, v in json.load(open(map_p)).items():
        hf = v["hf"]
        hf2ck[hf[: -len(".weight")] if hf.endswith(".weight") else hf] = ck
    wm = json.load(open(os.path.join(base, "model.safetensors.index.json")))["weight_map"]
    names = sorted({n for x in loras for n in x["lora"]})
    res, stats = {}, {}
    for hfname in names:
        ck = hf2ck.get(hfname)
        assert ck is not None, hfname
        key = ck if ck in wm else hfname + ".weight"  # base stored under checkpoint names, else HF names
        assert key in wm, (hfname, ck)
        with safe_open(os.path.join(base, wm[key]), "pt") as f:
            W = f.get_tensor(key)
        assert W.dtype == torch.bfloat16, (key, W.dtype)
        Wd = W.double()
        total = torch.zeros_like(Wd)
        per = {}
        for i, x in enumerate(loras):
            if hfname not in x["lora"]:
                continue
            A, B = x["lora"][hfname]
            assert tuple(W.shape) == (B.shape[0], A.shape[1]), (key, W.shape, A.shape, B.shape)
            delta = x["scale"] * (B.double() @ A.double())
            per[f"lora{i}"] = float(delta.norm() / Wd.norm())
            total += delta
        Wn = (Wd + total).to(torch.bfloat16)
        res[key] = Wn.contiguous()
        stats[key] = {"rel_delta": float(total.norm() / Wd.norm()), "per_lora": per,
                      "changed_frac": float((Wn != W).float().mean())}
    save_file(res, out)
    json.dump({"loras": [{"path": x["path"], "merge_scale": x["ms"], "scale": x["scale"]} for x in loras],
               "tensors": stats}, open(out + ".json", "w"), indent=1)
    print("MERGED", len(res), "mean rel_delta", sum(s["rel_delta"] for s in stats.values()) / len(stats))


if __name__ == "__main__":
    main()
