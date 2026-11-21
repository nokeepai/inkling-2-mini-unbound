"""ON POD: compare tensor names/types of GGUF files: python3 list_types.py a.gguf b.gguf ..."""
import collections
import sys

from gguf import GGUFReader

for p in sys.argv[1:]:
    rd = GGUFReader(p)
    kinds = collections.Counter()
    for t in rd.tensors:
        kinds[(t.name.split(".", 2)[-1] if t.name.startswith("blk.") else t.name, t.tensor_type.name)] += 1
    print("==", p, len(rd.tensors), "tensors")
    for (n, ty), c in sorted(kinds.items()):
        print(f"  {n:45s} {ty:8s} x{c}")
    print("  scale-like:", [t.name for t in rd.tensors if "scale" in t.name][:10])
