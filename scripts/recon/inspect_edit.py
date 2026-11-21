"""Inspect OpenUncensored edit.pt (weights_only load) and write a tidy summary JSON."""
import json
import sys

import torch

src = sys.argv[1]
out = sys.argv[2]
obj = torch.load(src, map_location="cpu", weights_only=True)


def describe(x, depth=0, key="root"):
    pad = "  " * depth
    if isinstance(x, torch.Tensor):
        info = {"shape": list(x.shape), "dtype": str(x.dtype)}
        if x.numel() <= 8:
            info["values"] = x.float().flatten().tolist()
        else:
            xf = x.float()
            info["norm"] = xf.norm().item()
            if xf.dim() == 2:
                info["row_norms"] = xf.norm(dim=-1).tolist()[:8]
                s = torch.linalg.svdvals(xf)
                info["svals"] = s.tolist()[:8]
                info["gram"] = (xf @ xf.T).tolist() if xf.shape[0] <= 8 else None
        print(f"{pad}{key}: {info}")
        return info
    if isinstance(x, dict):
        print(f"{pad}{key}: dict({len(x)})")
        return {str(k): describe(v, depth + 1, str(k)) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        print(f"{pad}{key}: {type(x).__name__}({len(x)})")
        if all(isinstance(v, (int, float, str, bool)) for v in x):
            print(f"{pad}  values: {x}")
            return list(x)
        return [describe(v, depth + 1, f"[{i}]") for i, v in enumerate(x)]
    print(f"{pad}{key}: {x!r}")
    return x


summary = describe(obj)
with open(out, "w") as f:
    json.dump(summary, f, indent=1)
