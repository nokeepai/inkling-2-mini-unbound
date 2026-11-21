"""Build the 50 harmless parity prompts: tatsu-lab/alpaca, shuffle(seed=1234), first 50.
Renders with the official template, thinking off (reasoning_effort=0). Writes /root/parity/prompts.json.
"""
import json
import sys

sys.path.insert(0, "/root/parity")
from datasets import load_dataset  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 50
ds = load_dataset("tatsu-lab/alpaca", split="train").shuffle(seed=1234).select(range(N))
tok = AutoTokenizer.from_pretrained("/root/w/orig")
rows = []
for i, r in enumerate(ds):
    text = r["instruction"] if not r["input"] else f"{r['instruction']}\n\n{r['input']}"
    rendered = tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False, reasoning_effort=0)
    ids = tok(rendered, add_special_tokens=False).input_ids
    rows.append({"i": i, "text": text, "rendered": rendered, "input_ids": ids})
json.dump(rows, open("/root/parity/prompts.json", "w"))
print("wrote", len(rows), "prompts; example:", repr(rows[0]["rendered"][:300]), rows[0]["input_ids"][:20])
