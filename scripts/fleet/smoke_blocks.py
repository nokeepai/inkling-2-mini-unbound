"""Pod-side smoke of the new full-stage blocks (probes off/e0.9 + borderline200) on whatever the server serves (orig).
  python3 /srv/harness/smoke_blocks.py [--gpu i]   -> /srv/evals/smoke_blocks/ (tag 'smoke': parity's judge relay skips it)"""
import json
import os
import sys
import time

sys.path.insert(0, "/srv/harness")
import fleet_eval as F  # noqa: E402

gpu = int(sys.argv[sys.argv.index("--gpu") + 1]) if "--gpu" in sys.argv else None
F.S.PORT = F.gpu_port(gpu)
out = "/srv/evals/smoke_blocks"
os.makedirs(out, exist_ok=True)
met: dict = {}
timing: dict = {}
t = time.time()
F.probes_block(out, met, timing)
F.borderline200_block(out, met, timing)
print(json.dumps({k: (round(v["value"], 4), v["n"]) for k, v in met.items()}), timing, round(time.time() - t, 1))
