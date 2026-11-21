"""Write ou_edit_summary.json: structure, strengths, layer map and direction stats of OU's edit.pt (no raw tensors)."""
import json
import sys

import torch

src, out = sys.argv[1], sys.argv[2]
e = torch.load(src, map_location="cpu", weights_only=True)
Q = e["Q"].double()
s_attn = e["s_attn"].tolist()
s_mlp = e["s_mlp"].tolist()
gram = (Q.T @ Q).tolist()
loop = range(6, 18)
layer_map = []
for layer, s in enumerate(s_attn):
    if s == 0:
        continue
    visits = 3 if layer in loop else 1
    layer_map.append({
        "layer": layer,
        "tensor": f"model.llm.layers.{layer}.attn.wo_ud.weight",
        "hf_module": f"model.language_model.layers.{layer}.self_attn.o_proj.linear",
        "s": s,
        "eigen_on_span_Q": 1 - s,
        "looped": layer in loop,
        "applications_per_token": visits,
    })
summary = {
    "source": "OpenUncensored/Inkling-2-Mini-Uncensored@4e5f06d7/edit.pt",
    "keys": {k: (list(v.shape) + [str(v.dtype)] if isinstance(v, torch.Tensor) else v) for k, v in e.items()},
    "name": e.get("name"),
    "note": e.get("note"),
    "Q": {
        "shape": list(Q.shape),
        "dtype": str(e["Q"].dtype),
        "rank": int(torch.linalg.matrix_rank(Q).item()),
        "singular_values": torch.linalg.svdvals(Q).tolist(),
        "gram_QtQ": gram,
        "col_norms": Q.norm(dim=0).tolist(),
        "max_abs_entry": Q.abs().max().item(),
        "top_dims_col0": torch.topk(Q[:, 0].abs(), 8).indices.tolist(),
        "top_dims_col1": torch.topk(Q[:, 1].abs(), 8).indices.tolist(),
    },
    "s_attn": s_attn,
    "s_mlp": s_mlp,
    "layer_map": layer_map,
    "formula": "W_new = W - s * Q @ Q.T @ W  (W = wo_ud, shape [2304, 3072]; Q orthonormal [2304, 2])",
}
with open(out, "w") as f:
    json.dump(summary, f, indent=1)
print(json.dumps({k: summary[k] for k in ("s_attn", "s_mlp", "layer_map")}, indent=None))
print(summary["Q"])
