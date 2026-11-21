"""ON POD: approximate LoRA of our edit at ranks 16 and 64 (PEFT + GGUF).

delta = W'(bf16 built edit) - W(bf16 original), in fp32. Truncated SVD: delta ~ U_r S_r V_r^T.
LoRA:  B = U_r * sqrt(S_r) [out, r],  A = sqrt(S_r) * V_r^T [r, in],  alpha = r (scale 1), so W + B A ~ W'.
Writes:
  /data/release/lora/peft-r{R}/adapter_config.json + adapter_model.safetensors  (HF module names, fp32)
  /data/release/Inkling-2-Mini-Unbound-LoRA-r{R}.gguf  (written directly with gguf-py, F32, the layout
      convert_lora_to_gguf.py emits: <name>.lora_a [r, in], <name>.lora_b [out, r])
  /data/release/lora/svd_errors.json
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch
from gguf import GGUFReader, GGUFType, GGUFWriter, Keys
from safetensors import safe_open
from safetensors.torch import load_file, save_file

RANKS = (16, 64)
OUT = "/data/release/lora"
NAME_MAP = json.load(open("/root/gguf/name_map.json"))          # ckpt -> gguf
HF_MAP = json.load(open("/root/gguf/hf_name_map.json"))        # ckpt -> {"hf": ...}


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    built = load_file("/root/gguf/candbuild/tensors.safetensors")
    orig = safe_open("/data/orig/model-00001-of-00001.safetensors", "pt")
    arch = str(bytes(GGUFReader("/data/base/Inkling-2-Mini-Q8_0.gguf").fields["general.architecture"].parts[-1]), "utf-8")
    errs: dict[str, dict] = {}
    factors: dict[int, dict[str, tuple[torch.Tensor, torch.Tensor]]] = {r: {} for r in RANKS}
    for ck in sorted(built):
        W = orig.get_tensor(ck).float()
        D = built[ck].float() - W
        U, S, Vh = (x.cpu() for x in torch.linalg.svd(D.double().cuda(), full_matrices=False))
        nD = torch.linalg.norm(D.double())
        e = {"delta_fro": float(nD), "W_fro": float(torch.linalg.norm(W)), "delta_over_W": float(nD / torch.linalg.norm(W))}
        for r in RANKS:
            rel = float(torch.sqrt((S[r:] ** 2).sum()) / nD)
            e[f"rel_err_r{r}"] = rel
            sq = S[:r].sqrt()
            B = (U[:, :r] * sq).float().contiguous()            # [out, r]
            A = (sq[:, None] * Vh[:r]).float().contiguous()     # [r, in]
            factors[r][ck] = (A, B)
        errs[ck] = e
        print(ck, {k: round(v, 4) for k, v in e.items()})
    json.dump(errs, open(f"{OUT}/svd_errors.json", "w"), indent=1)
    for r in RANKS:
        write_peft(r, factors[r])
        write_gguf(r, factors[r], arch)


def write_peft(r: int, fac: dict) -> None:
    d = f"{OUT}/peft-r{r}"
    os.makedirs(d, exist_ok=True)
    sd = {}
    for ck, (A, B) in fac.items():
        mod = HF_MAP[ck]["hf"].removesuffix(".weight")
        sd[f"base_model.model.{mod}.lora_A.weight"] = A
        sd[f"base_model.model.{mod}.lora_B.weight"] = B
    save_file(sd, f"{d}/adapter_model.safetensors")
    layers = sorted(int(ck.split(".")[3]) for ck in fac)
    cfg = {"peft_type": "LORA", "task_type": "CAUSAL_LM", "base_model_name_or_path": "thinkingmachines/Inkling-2-Mini",
           "revision": "ed40e7da", "r": r, "lora_alpha": r, "lora_dropout": 0.0, "bias": "none", "fan_in_fan_out": False,
           "target_modules": sorted({HF_MAP[ck]["hf"].removesuffix(".weight") for ck in fac}),
           "layers_to_transform": layers, "use_rslora": False, "use_dora": False, "init_lora_weights": True}
    json.dump(cfg, open(f"{d}/adapter_config.json", "w"), indent=1)


def write_gguf(r: int, fac: dict, arch: str) -> None:
    path = f"/data/release/Inkling-2-Mini-Unbound-LoRA-r{r}.gguf"
    w = GGUFWriter(path, arch)
    w.add_type(GGUFType.ADAPTER)
    w.add_string(Keys.Adapter.TYPE, "lora")
    w.add_float32(Keys.Adapter.LORA_ALPHA, float(r))
    w.add_name(f"Inkling-2-Mini-Unbound-LoRA-r{r}")
    for ck in sorted(fac, key=lambda k: int(k.split(".")[3])):
        A, B = fac[ck]
        g = NAME_MAP[ck]
        w.add_tensor(g + ".lora_a", A.numpy())
        w.add_tensor(g + ".lora_b", B.numpy())
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print("wrote", path, os.path.getsize(path))


if __name__ == "__main__":
    main()
