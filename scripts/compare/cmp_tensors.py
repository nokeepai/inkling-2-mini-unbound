"""cmp_tensors.py a.gguf b.gguf: per-tensor type/shape/bytes comparison. Prints the count identical and every differing name."""
import hashlib
import sys

import numpy as np
from gguf import GGUFReader

a, b = (GGUFReader(p) for p in sys.argv[1:3])
ta, tb = {t.name: t for t in a.tensors}, {t.name: t for t in b.tensors}
same, diff = 0, []
for n in sorted(set(ta) | set(tb)):
    if n not in ta or n not in tb:
        diff.append(f"{n}: only in {'a' if n in ta else 'b'}")
        continue
    x, y = ta[n], tb[n]
    if x.tensor_type != y.tensor_type or list(x.shape) != list(y.shape):
        diff.append(f"{n}: type {x.tensor_type.name} vs {y.tensor_type.name}")
        continue
    if hashlib.sha256(np.asarray(x.data).tobytes()).digest() == hashlib.sha256(np.asarray(y.data).tobytes()).digest():
        same += 1
    else:
        diff.append(f"{n}: bytes differ ({x.tensor_type.name})")
print(f"CMP {sys.argv[1]} vs {sys.argv[2]}: {same} identical, {len(diff)} differ")
for d in diff[:60]:
    print("  ", d)
