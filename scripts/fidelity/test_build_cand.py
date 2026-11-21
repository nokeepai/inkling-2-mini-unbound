"""Pod: build c0_lr with the single-thread builder twice, count diffs vs OU, and make a mini orig dir (the 20 edited
wo_ud tensors + an index) so the Mac can reproduce the build. Output: /root/fid/build_test.json, /root/fid/mini_orig/."""
import json
import os
import shutil
import subprocess
import sys

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

LR = "/root/fid/lr_test_WtQ"
res = {}
shas = []
for k in range(2):
    d = f"/root/fid/bt{k}"
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d)
    shutil.copy(f"{LR}/lowrank.safetensors", d)
    r = subprocess.run([sys.executable, "/root/fid/build_cand.py", d, "/root/w/orig", d], capture_output=True, text=True)
    shas.append(r.stdout.strip().splitlines()[-1])
res["pod_shas"] = shas
built = load_file("/root/fid/bt0/tensors.safetensors")
with safe_open("/root/w/ou/model-00001-of-00001.safetensors", "pt") as fu, \
        safe_open("/root/w/orig/model-00001-of-00001.safetensors", "pt") as fo:
    per = {n: int((built[n].view(torch.int16) != fu.get_tensor(n).view(torch.int16)).sum()) for n in built}
    os.makedirs("/root/fid/mini_orig", exist_ok=True)
    mini = {n: fo.get_tensor(n) for n in built}
save_file(mini, "/root/fid/mini_orig/mini.safetensors")
json.dump({"metadata": {}, "weight_map": {n: "mini.safetensors" for n in mini}},
          open("/root/fid/mini_orig/model.safetensors.index.json", "w"))
res["diff_vs_ou_total"] = sum(per.values())
res["diff_vs_ou_per_tensor"] = per
res["max_abs"] = max(float((built[n].float() - mini[n].float()).abs().max()) for n in [])  if False else None
json.dump(res, open("/root/fid/build_test.json", "w"), indent=1)
print(json.dumps({k: v for k, v in res.items() if k != "diff_vs_ou_per_tensor"}))
print("per tensor:", list(per.values()))
for k in range(2):
    shutil.rmtree(f"/root/fid/bt{k}", ignore_errors=True)
