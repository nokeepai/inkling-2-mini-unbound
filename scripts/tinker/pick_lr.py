"""Compare stage-1 runs on their last-5-step training metrics (proxy for picking the LR).

  ~/.inkling-abl/tinker_lab/.venv/bin/python scripts/tinker/pick_lr.py w2_lr5e5 w2_lr2e5
"""
from __future__ import annotations

import json
import sys

import lab_common as lc

for tag in sys.argv[1:]:
    steps = [json.loads(line) for line in (lc.LAB_DIR / "logs" / f"train_{tag}.log").read_text().splitlines()]
    steps = [s for s in steps if s["event"] == "step"]
    last = steps[-5:]
    avg = lambda k: sum(s[k] for s in last) / len(last)  # noqa: E731
    print(f"{tag}: steps={len(steps)} last5 reward={avg('reward'):.3f} quality={avg('r_quality'):.3f} "
          f"cap={avg('cap_hit'):.3f} think={avg('think_tok_mean'):.0f} "
          f"judge_err={sum(s.get('judge_errors', 0) for s in steps)} "
          f"first5 reward={sum(s['reward'] for s in steps[:5]) / 5:.3f}")
