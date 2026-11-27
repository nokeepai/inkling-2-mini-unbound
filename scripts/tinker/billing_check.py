"""Billing audit: list today's (UTC) usage buckets by model and kind, with estimated cost.

Run before and after each lab job. Any base_model other than the free ID, or any non-zero
estimated_cost_usd, is a stop-and-report signal.

  ~/.inkling-abl/tinker_lab/.venv/bin/python scripts/tinker/billing_check.py [--days 2]
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any

import lab_common as lc


def fetch_usage(days: int) -> Any:
    lc.load_key()
    import tinker

    now = datetime.now(timezone.utc)
    start = (now - timedelta(days=days - 1)).replace(hour=0, minute=0, second=0, microsecond=0)
    # The API wants hour-aligned bounds; round up so the current hour is included.
    end = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
    return tinker.AdminClient().get_billing_usage(start, end).result()


def summarise(resp: Any) -> dict[str, Any]:
    totals: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    cost_total = 0.0
    for ev in resp.data:
        info = ev.event_info
        kind = getattr(info, "type", "unknown")
        if kind == "sampling_prefill":
            kind += "_cached" if getattr(info, "cached", False) else "_full"
        if kind == "storage":
            kind += "_free" if getattr(info, "free", False) else "_paid"
        qty = getattr(info, "token_count", None) or getattr(info, "count", None) \
            or getattr(info, "gigabyte_hours", 0.0)
        key = str(ev.base_model)
        totals[key][kind] += float(qty or 0)
        totals[key]["usd_est"] += float(ev.estimated_cost_usd or 0.0)
        cost_total += float(ev.estimated_cost_usd or 0.0)
    non_free = [m for m in totals if m not in (lc.FREE_MODEL, "None")]
    return {"by_model": {k: dict(v) for k, v in totals.items()},
            "estimated_cost_usd_total": round(cost_total, 6),
            "non_free_models_seen": non_free}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=1)
    args = ap.parse_args()
    try:
        out = summarise(fetch_usage(args.days))
    except Exception as e:  # report, never leak the key
        out = {"error": lc.scrub(f"{type(e).__name__}: {e}")}
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
