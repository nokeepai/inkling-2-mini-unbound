"""Print dtype/shape of selected tensors from the orig safetensors header (run on pod)."""
import json
import re
import struct

PATH = "/root/w/orig/model-00001-of-00001.safetensors"
with open(PATH, "rb") as f:
    n = struct.unpack("<Q", f.read(8))[0]
    header = json.loads(f.read(n))
header.pop("__metadata__", None)
keep = re.compile(r"llm\.(layers\.(5|7|19|20)\.|embed|unembed|norm|attn_res\.output|middle_cycle)")
rows = {k: (v["dtype"], v["shape"]) for k, v in header.items() if keep.search(k)}
print(json.dumps(dict(sorted(rows.items())), indent=0))
dt = {}
for k, v in header.items():
    dt[v["dtype"]] = dt.get(v["dtype"], 0) + 1
print(dt)
