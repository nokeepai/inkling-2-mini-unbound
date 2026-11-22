"""ON eval pod: package v2's finetune/ folder.
  python3 package_v2.py <E.pt> <E tensors.safetensors> <E-Q8_0.gguf.sha256> <out_dir>
- lora_v2.safetensors + adapter_config.json: PEFT LoRA (keys base_model.model.<module>.lora_{A,B}.weight), applied on
  top of Unbound v1.0 (HF module names, scale alpha/r).
- MANIFEST.json: sha256 of every tensor that differs from v1.0 (the 120 merged BF16 attention tensors).
- train_config.json: hyperparameters from the LoRA meta.
"""
import hashlib, json, os, sys
import torch
from safetensors import safe_open
from safetensors.torch import save_file

lp, tp, gsha, out = sys.argv[1:5]
os.makedirs(out, exist_ok=True)
d = torch.load(lp, map_location="cpu", weights_only=False)
meta, lora = d["meta"], d["lora"]
a = meta["args"]
sd, mods = {}, set()
for name, (A, B) in lora.items():
    sd[f"base_model.model.{name}.lora_A.weight"] = A.float().contiguous()
    sd[f"base_model.model.{name}.lora_B.weight"] = B.float().contiguous()
    mods.add(name.split(".")[-2] + "." + name.split(".")[-1] if name.endswith((".linear", ".gate_proj")) else name.split(".")[-1])
save_file(sd, f"{out}/lora_v2.safetensors", metadata={"format": "pt"})
cfg = {"peft_type": "LORA", "task_type": "CAUSAL_LM", "r": a["rank"], "lora_alpha": a["alpha"], "lora_dropout": 0.0,
       "bias": "none", "fan_in_fan_out": False, "target_modules": sorted(mods),
       "base_model_name_or_path": "nokeepai/Inkling-2-Mini-Unbound@v1.0",
       "_note": "Applied on top of Unbound v1.0, not the original. Merged result = Unbound v2 (BF16 rounding aside)."}
json.dump(cfg, open(f"{out}/adapter_config.json", "w"), indent=1)
man = {}
with safe_open(tp, "pt") as f:
    for k in sorted(f.keys()):
        t = f.get_tensor(k)
        man[k] = {"shape": list(t.shape), "dtype": str(t.dtype).replace("torch.", ""),
                  "sha256": hashlib.sha256(t.view(torch.uint16).numpy().tobytes()).hexdigest()}
json.dump({"changed_vs_v1.0": man, "n_changed": len(man), "gguf_q8_0_sha256": open(gsha).read().split()[0]},
          open(f"{out}/MANIFEST.json", "w"), indent=1)
json.dump({k: v for k, v in a.items() if k not in ("data", "out", "base")} | {"trained_on": "1,186 harmful + 1,561 harmless rows"},
          open(f"{out}/train_config.json", "w"), indent=1)
print("PACKAGED", len(sd) // 2, "LoRA modules;", len(man), "tensors;", sorted(mods))
