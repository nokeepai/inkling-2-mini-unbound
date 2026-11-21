#!/usr/bin/env python3
"""List which checkpoint tensor kinds any candidate edits (from lowrank/tensors safetensors headers)."""
import glob, json, struct, collections

kinds = collections.Counter()
for f in glob.glob("/Users/nikshepsvn/Code/inkling-abl/methods/*/cands/*/*.safetensors"):
    with open(f, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        header = json.loads(fh.read(n))
    for k in header:
        if k == "__metadata__":
            continue
        base = k.rsplit(".", 1)[0] if f.endswith("lowrank.safetensors") else k
        kinds[base.split("layers.")[-1].split(".", 1)[-1]] += 1
print(dict(kinds))
