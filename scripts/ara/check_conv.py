import sys, json, math, torch
sys.argv = [sys.argv[0], "/dev/null", "/tmp/x"]
CAP = "/root/ara/cap"
from safetensors import safe_open
idx = json.load(open("/root/w/orig/model.safetensors.index.json"))["weight_map"]
YV = torch.load(f"{CAP}/yver.pt"); CONV = torch.load(f"{CAP}/conv.pt"); V = torch.load(f"{CAP}/valid.pt")
for l in (4, 6, 11, 17, 23):
    n = f"model.llm.layers.{l}.attn.wo_ud.weight"
    with safe_open(f"/root/w/orig/{idx[n]}", "pt") as f:
        W = f.get_tensor(n).cuda()
    X = torch.load(f"{CAP}/L{l}.pt")[:16].cuda()
    h = (X.bfloat16() @ W.T).float()
    w = CONV[l].cuda(); T = h.shape[-2]
    y = ((1 + w[:, 3]) * h[..., 3:, :] + w[:, 2] * h[..., 2:T-1, :] + w[:, 1] * h[..., 1:T-2, :] + w[:, 0] * h[..., :T-3, :]) / math.sqrt(5)
    ref = YV[l].cuda(); m = V[:16].cuda()[:, None, :, None].float()
    err = ((y - ref) * m).norm() / (ref * m).norm()
    print(l, "relerr", float(err), "valid frac", float(V.float().mean()))
