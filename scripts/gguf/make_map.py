"""ON POD: checkpoint name -> GGUF name for the 20 edited wo_ud tensors (tensor_mapping.py: attn.wo_ud -> attn_output)."""
import json
import re

from safetensors import safe_open

keys = sorted(safe_open("/root/gguf/candbuild/tensors.safetensors", "pt").keys())
pat = re.compile(r"model\.llm\.layers\.(\d+)\.attn\.wo_ud\.weight$")
name_map = {k: f"blk.{int(pat.match(k).group(1))}.attn_output.weight" for k in keys}
json.dump(name_map, open("/root/gguf/name_map.json", "w"), indent=1)
print(len(name_map), "mapped")
