"""Collect latest metric values per candidate from results/runs.jsonl.  metrics.py [cand_id...] (prefix match ok)"""
import json
import os
import sys

R = os.path.expanduser("~/Code/inkling-abl")


def collect(prefixes=None, exclude_pods=()) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for line in open(f"{R}/results/runs.jsonl"):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        c = row.get("candidate")
        if not c or (prefixes and not any(c.startswith(p) for p in prefixes)):
            continue
        if row.get("pod") in exclude_pods and row.get("agent") != "kl_worker":
            continue
        if "error" in row or str(row.get("metric", "")).endswith("error"):
            out.setdefault(c, {})[f"_error_{row.get('stage', row.get('metric', 'x'))}"] = str(row.get("error", row.get("value")))[:300]
            continue
        if row.get("split", "val") != "val" or "metric" not in row:
            continue
        key = row["metric"]
        if row.get("stage") == "screen" and not key.startswith("screen"):
            key = "screen_" + key
        out.setdefault(c, {})[key] = row.get("value")
    return out


if __name__ == "__main__":
    for c, m in collect(sys.argv[1:] or None).items():
        print(c, json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in sorted(m.items())}))
