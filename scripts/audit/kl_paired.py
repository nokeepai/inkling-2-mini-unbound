"""Auditor: paired prompt-bootstrap of KL differences between candidates (same 1,000 test prompts, same cache).
Uses per_prompt_kl and n_pos weights reconstructed from kl_results.jsonl: the pooled KL is sum(kl_i*len_i)/sum(len_i);
per-prompt lengths are not stored, so the paired test uses the unweighted per-prompt mean (stated in the output).

  python3 kl_paired.py <kl_results.jsonl> <candA> <candB>
"""
import json
import random
import sys

rows = {json.loads(l)["cand"] + "|" + json.loads(l)["tag"]: json.loads(l) for l in open(sys.argv[1])}
a = rows[f"{sys.argv[2]}|baseline"]["per_prompt_kl"]
b = rows[f"{sys.argv[3]}|baseline"]["per_prompt_kl"]
d = [x - y for x, y in zip(a, b)]
rng = random.Random(1234)
k = len(d)
ms = sorted(sum(d[rng.randrange(k)] for _ in range(k)) / k for _ in range(10000))
print(json.dumps({"A": sys.argv[2], "B": sys.argv[3], "mean_per_prompt_kl_A": sum(a) / k, "mean_per_prompt_kl_B": sum(b) / k,
                  "diff_A_minus_B": sum(d) / k, "CI": [ms[250], ms[9749]], "n": k,
                  "A_lower_on_prompts": sum(x < 0 for x in d) / k}))
