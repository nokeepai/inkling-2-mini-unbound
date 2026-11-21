import json, sys
d = json.load(open(sys.argv[1]))
pt = d["per_tensor"]
print("peft:", json.dumps(pt.get("_peft_native"), indent=0))
names = [n for n in pt if not n.startswith("_")]
ranks = sorted({int(k) for n in names for k in pt[n]["lora"]})
print("rank  n_diff_total  max_rel_fro  ")
for k in ranks:
    nd = sum(pt[n]["lora"][str(k)]["n_diff_after_bf16_merge"] for n in names)
    rf = max(pt[n]["lora"][str(k)]["rel_fro_err"] for n in names)
    mu = max(pt[n]["lora"][str(k)]["max_diff_ulp_after_merge"] for n in names)
    print(k, nd, rf, mu)
print("delta numerical ranks", [pt[n]["delta_rank_numerical"] for n in names])
