"""ON release pod: package v2.1's finetune/ folder (mirrors package_v2.py).

  python3 package_v21.py <v2_ckpt_dir> <hf_name_map.json> <v21_tensors.safetensors> <Q8_0.sha256> <out_dir> <lora.pt> [<lora.pt> ...]

v2.1 = v2 + a STACK of LoRAs (SFT then DPO), merged as W' = bf16(W + sum_i (alpha_i/r_i) B_i A_i).
One PEFT adapter reproduces the stack exactly by concatenating ranks with the scales folded into A:
  A = [s_1 A_1; s_2 A_2; ...]  (r_total x in),  B = [B_1 | B_2 | ...]  (out x r_total),  lora_alpha = r_total (scale 1).
Writes lora_v2.1.safetensors + adapter_config.json (applied on top of Unbound v2.0), MANIFEST.json (sha256 of the
120 tensors that differ from v2.0 + Q8_0 sha), train_config.json (both stages' hyperparameters), and checks the
reconstruction: bf16(W_v2 + B A) vs the shipped tensors (fraction of identical elements, max abs diff).
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file

SKIP_ARGS = ("data", "out", "base", "sft_lora")


def load_stack(paths: list[str]) -> list[dict]:
    out = []
    for p in paths:
        d = torch.load(p, map_location="cpu", weights_only=False)
        a = d["meta"]["args"]
        out.append({"path": p, "args": a, "scale": a["alpha"] / a["rank"], "lora": d["lora"]})
    return out


def concat(stack: list[dict], name: str) -> tuple[torch.Tensor, torch.Tensor]:
    As, Bs = [], []
    ref = next(x["lora"][name] for x in stack if name in x["lora"])
    n_in, n_out = ref[0].shape[1], ref[1].shape[0]
    for x in stack:
        r = x["args"]["rank"]
        if name in x["lora"]:
            A, B = x["lora"][name]
            As.append(A.float() * x["scale"])
            Bs.append(B.float())
        else:  # module not trained in this stage: zero block keeps ranks aligned
            As.append(torch.zeros(r, n_in))
            Bs.append(torch.zeros(n_out, r))
    return torch.cat(As, 0).contiguous(), torch.cat(Bs, 1).contiguous()


def base_tensor(base: str, wm: dict, key: str) -> torch.Tensor:
    with safe_open(os.path.join(base, wm[key]), "pt") as f:
        return f.get_tensor(key)


def main() -> None:
    base, map_p, tp, gsha, out, *lps = sys.argv[1:]
    os.makedirs(out, exist_ok=True)
    stack = load_stack(lps)
    r_total = sum(x["args"]["rank"] for x in stack)
    hf2ck = {v["hf"].removesuffix(".weight"): ck for ck, v in json.load(open(map_p)).items()}
    wm = json.load(open(os.path.join(base, "model.safetensors.index.json")))["weight_map"]
    names = sorted({n for x in stack for n in x["lora"]})
    sd, ident, maxdiff = {}, [], 0.0
    with safe_open(tp, "pt") as shipped:
        for name in names:
            A, B = concat(stack, name)
            sd[f"base_model.model.{name}.lora_A.weight"] = A
            sd[f"base_model.model.{name}.lora_B.weight"] = B
            ck = hf2ck[name]
            key = ck if ck in wm else name + ".weight"
            W = base_tensor(base, wm, key)
            rec = (W.double() + B.double() @ A.double()).to(torch.bfloat16)
            tgt = shipped.get_tensor(key)
            ident.append(float((rec == tgt).float().mean()))
            maxdiff = max(maxdiff, float((rec.float() - tgt.float()).abs().max()))
    save_file(sd, f"{out}/lora_v2.1.safetensors", metadata={"format": "pt"})
    mods = sorted({n.split(".")[-1] for n in names})
    cfg = {"peft_type": "LORA", "task_type": "CAUSAL_LM", "r": r_total, "lora_alpha": r_total, "lora_dropout": 0.0,
           "bias": "none", "fan_in_fan_out": False, "target_modules": mods,
           "base_model_name_or_path": "nokeepai/Inkling-2-Mini-Unbound@v2.0",
           "_note": f"Applied on top of Unbound v2.0, not the original. Ranks of the SFT and DPO stages concatenated "
                    f"({'+'.join(str(x['args']['rank']) for x in stack)}={r_total}) with their scales folded into A. "
                    f"Merged result = Unbound v2.1 (BF16 rounding aside)."}
    json.dump(cfg, open(f"{out}/adapter_config.json", "w"), indent=1)
    man = {}
    with safe_open(tp, "pt") as f:
        for k in sorted(f.keys()):
            t = f.get_tensor(k)
            man[k] = {"shape": list(t.shape), "dtype": str(t.dtype).replace("torch.", ""),
                      "sha256": hashlib.sha256(t.view(torch.uint16).numpy().tobytes()).hexdigest()}
    json.dump({"changed_vs_v2.0": man, "n_changed": len(man), "gguf_q8_0_sha256": open(gsha).read().split()[0]},
              open(f"{out}/MANIFEST.json", "w"), indent=1)
    stages = [{k: v for k, v in x["args"].items() if k not in SKIP_ARGS} for x in stack]
    json.dump({"stages": dict(zip(("sft", "dpo"), stages)), "merge_scales": [1.0] * len(stack)},
              open(f"{out}/train_config.json", "w"), indent=1)
    check = {"min_identical_frac": min(ident), "mean_identical_frac": sum(ident) / len(ident), "max_abs_diff": maxdiff}
    json.dump(check, open(f"{out}/../finetune_check.json", "w"), indent=1)
    print("PACKAGED", len(names), "modules; r", r_total, "; tensors", len(man), "; recon", check)


if __name__ == "__main__":
    main()
