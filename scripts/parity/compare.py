"""Compare greedy token streams.  python compare.py <a.json> <b.json> [n]
Prints exact-64 match count, first-divergence positions, and per-token agreement."""
import json
import sys

a = {x["i"]: x["tokens"] for x in json.load(open(sys.argv[1]))}
b = {x["i"]: x["tokens"] for x in json.load(open(sys.argv[2]))}
n = int(sys.argv[3]) if len(sys.argv) > 3 else len(b)
keys = sorted(set(a) & set(b))[:n]
exact, div = 0, {}
for k in keys:
    x, y = a[k][:64], b[k][:64]
    if x == y:
        exact += 1
        continue
    d = next((j for j, (p, q) in enumerate(zip(x, y)) if p != q), min(len(x), len(y)))
    div[k] = d
print(json.dumps({"n": len(keys), "exact64": exact, "divergence_pos": div}))
