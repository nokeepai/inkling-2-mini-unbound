"""Mac: score val screens with the compare pipeline's own code (aggregate.jev / kld / boot), so numbers match compare.

  python3 score_val.py <name> [<name> ...]     # pulls nothing; expects ~/.inkling-abl/v11b_private/val/{out,logs}
Prints one JSON line per model: refusal off/e0.9, Jev useful off/e0.9 (CI), over-refusal, MMLU, GSM8K, KLD, top-1.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.expanduser("~/Code/inkling-abl/scripts/compare"))
import aggregate  # noqa: E402

aggregate.P = os.path.expanduser("~/.inkling-abl/v11b_private/val")


def v(x):
    return None if x is None else (round(x["value"], 4) if isinstance(x, dict) else x)


def main() -> None:
    for name in sys.argv[1:]:
        mf = f"{aggregate.P}/out/{name}/metrics.json"
        met = json.load(open(mf))["metrics"] if os.path.exists(mf) else {}
        ju = {m: aggregate.jev(name, m) for m in ("off", "e09")}
        k = aggregate.kld(name) or {}
        row = {"name": name,
               "refusal_off": v(met.get("refusal_off")), "refusal_e09": v(met.get("refusal_e09")),
               "useful_off": v(ju["off"]), "useful_off_CI": ju["off"] and ju["off"]["CI"],
               "useful_e09": v(ju["e09"]), "useful_e09_CI": ju["e09"] and ju["e09"]["CI"],
               "overrefusal": v(met.get("overrefusal_off")),
               "mmlu": v(met.get("mmlu300")), "gsm8k": v(met.get("gsm8k100")), "cap_off": v(met.get("harmful_off_cap_hit")), "cap_e09": v(met.get("harmful_e09_cap_hit")),
               "kld": v(k.get("kld")), "top1": v(k.get("top1"))}
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
