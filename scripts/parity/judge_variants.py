"""Pod: judge the check set under dtype/batch variants to find the CPU-vs-GPU discrepancy.
  JUDGE_DTYPE=float32|bfloat16 python3 judge_variants.py <batch>"""
import json
import os
import sys

import torch

sys.path.insert(0, "/srv/judge/harness")
torch.cuda.set_per_process_memory_fraction(16 / (torch.cuda.get_device_properties(0).total_memory / 2**30))
import judge as J  # noqa: E402

rows = [json.loads(l) for l in open("/srv/judge_q/jc_test_src.jsonl")]
res = J.judge(rows, batch_size=int(sys.argv[1]))
print(os.environ.get("JUDGE_DTYPE"), sys.argv[1], "refused", sum(r["refused"] for r in res),
      [r["judge_score"] for r in res][:20])
