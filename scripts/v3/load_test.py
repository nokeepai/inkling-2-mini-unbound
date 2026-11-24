"""ON train pod: prove the v2 checkpoint loads with the pinned transformers under this pod's RAM, and time a forward."""
import sys
import time

import torch

sys.path.insert(0, "/root/parity")
from hf_load import encode, load  # noqa: E402

t = time.time()
model, proc = load("/root/w/v2")
print("loaded", round(time.time() - t), "s; gpu GB", round(torch.cuda.memory_allocated() / 2**30, 1), flush=True)
ids = encode(proc, "Write a haiku about rain.", effort=0)
with torch.no_grad():
    out = model.generate(ids, max_new_tokens=40, do_sample=False)
print(repr(proc.tokenizer.decode(out[0, ids.shape[1]:])), flush=True)
