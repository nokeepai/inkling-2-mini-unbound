"""Try float32 operation orders for W' = W - s Q Q^T W on layers 6 and 12 to find OU's exact computation; also row norms."""
import json

import torch
from safetensors import safe_open

ORIG = "/root/w/orig/model-00001-of-00001.safetensors"
OU = "/root/w/ou/model-00001-of-00001.safetensors"
e = torch.load("/root/w/ou/edit.pt", map_location="cpu", weights_only=True)
Q = e["Q"].float()
res = {}
with safe_open(ORIG, "pt") as fo, safe_open(OU, "pt") as fu:
    for layer in (4, 6, 12, 17):
        name = f"model.llm.layers.{layer}.attn.wo_ud.weight"
        W, Wou = fo.get_tensor(name), fu.get_tensor(name)
        s = float(e["s_attn"][layer])
        Wf = W.float()
        I = torch.eye(Q.shape[0])
        variants = {
            "W - s*Q@(Qt@W)": Wf - s * (Q @ (Q.T @ Wf)),
            "W - Q@(s*(Qt@W))": Wf - Q @ (s * (Q.T @ Wf)),
            "W - (s*Q)@(Qt@W)": Wf - (s * Q) @ (Q.T @ Wf),
            "W - s*(Q@Qt)@W": Wf - s * ((Q @ Q.T) @ Wf),
            "(I - s*Q@Qt)@W": (I - s * (Q @ Q.T)) @ Wf,
            "W - (s*Q@Qt)@W": Wf - (s * (Q @ Q.T)) @ Wf,
        }
        out = {}
        for k, v in variants.items():
            out[k] = int((v.to(torch.bfloat16) != Wou).sum().item())
        rn = Wf.norm(dim=1)
        rn_ou = Wou.float().norm(dim=1)
        cn = Wf.norm(dim=0)
        out["row_norm_min_max"] = [rn.min().item(), rn.max().item()]
        out["row_norm_ou_min_max"] = [rn_ou.min().item(), rn_ou.max().item()]
        out["col_norm_min_max"] = [cn.min().item(), cn.max().item()]
        res[layer] = out
        print(layer, out, flush=True)
with open("/root/logs/recon_variants.json", "w") as f:
    json.dump(res, f, indent=1)
