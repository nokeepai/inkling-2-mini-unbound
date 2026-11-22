"""ON eval pod: checkpoint name -> GGUF name for every tensor in a built tensors.safetensors, by suffix rule
(llama.cpp tensor_mapping for this arch); patch_v11b.py then PROVES each pair byte-for-byte against the original.
  python3 make_map_v11b.py <tensors.safetensors> <out name_map.json>"""
import json, re, sys
from safetensors import safe_open
RULE = {"attn.wo_ud.weight": "attn_output.weight", "attn.wq_du.weight": "attn_q.weight", "attn.wk_dv.weight": "attn_k.weight",
        "attn.wv_dv.weight": "attn_v.weight", "attn.gate_attn.weight": "attn_gate.weight"}
keys = list(safe_open(sys.argv[1], "pt").keys())
out = {}
for k in keys:
    m = re.match(r"model\.llm\.layers\.(\d+)\.(.+)$", k)
    out[k] = f"blk.{int(m.group(1))}.{RULE[m.group(2)]}"
json.dump(out, open(sys.argv[2], "w"), indent=0)
print(len(out))
