"""Show where two token streams diverge.  python show_div.py <a.json> <b.json> idx [idx...]"""
import json
import sys

from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("/root/w/orig")
a = {x["i"]: x["tokens"] for x in json.load(open(sys.argv[1]))}
b = {x["i"]: x["tokens"] for x in json.load(open(sys.argv[2]))}
for k in map(int, sys.argv[3:]):
    x, y = a[k], b[k]
    d = next((j for j, (p, q) in enumerate(zip(x, y)) if p != q), None)
    print(f"--- prompt {k} diverge at {d}")
    print("common:", repr(tok.decode(x[:d])[-200:]))
    print("A next:", repr(tok.decode(x[d:d + 12])))
    print("B next:", repr(tok.decode(y[d:d + 12])))
