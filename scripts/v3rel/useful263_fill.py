#!/usr/bin/env python3
"""Copy the usefulness re-run's numbers into package_v3/values.json (the `useful263.*` leaves the cards read).

    useful263_fill.py --out-dir DIR [--stats STATS_JSON] [--values package_v3/values.json] [--dry-run]

Two layers, because the two judges can finish at different times:
  * always, from DIR (the judge pass folder written by useful263.py): the second judge's (Bison) rates for v3 and
    for v2.1 re-judged in the same pass, their paired difference, and the judge-free counts (refusals by phrase
    list, thinking-limit hits);
  * with --stats (useful263.py stats output): the main judge's (Jev) rates and paired difference, and
    `flags.useful263_done`. Refused unless the stats are a full 263-request comparison.
Same definitions as the v2.1 release (scripts/charts/data_v21.py). Prints numbers only.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "charts"))
import data_v21 as D  # noqa: E402

BASIS_N = 263
MODES = ("off", "e09")
SOURCE = "scripts/v3rel/useful263_fill.py (v3 run of 2026-11-27, v2.1 answers re-judged in the same pass)"


def rows(path: Path) -> dict[str, dict]:
    return {r["id"]: r for r in map(json.loads, path.read_text().splitlines())}


def rate_text(r: dict) -> str:
    return f"{D.f1(r['value'])} [{r['ci'][0]:.1f}, {r['ci'][1]:.1f}]"


def bison_strings(out_dir: Path) -> dict[str, str]:
    """Bison is counted on the requests it judged for BOTH models, so the two rates and the difference agree."""
    out: dict[str, str] = {}
    counts: dict[str, dict[str, dict]] = {}
    for mode in MODES:
        gens = {s: rows(out_dir / f"{s}_{mode}.jsonl") for s in ("ref", "new")}
        judged = {s: rows(out_dir / f"bison_{s}_{mode}.jsonl") for s in ("ref", "new")}
        ids = sorted(gens["ref"])
        if len(ids) != BASIS_N or ids != sorted(gens["new"]):
            raise SystemExit(f"STOP: {mode} does not cover the same {BASIS_N} requests for both models")
        both = [i for i in ids if all(judged[s].get(i, {"error": 1})["error"] is None for s in judged)]
        if len(both) < 0.95 * BASIS_N:
            raise SystemExit(f"STOP: Bison judged only {len(both)} of {BASIS_N} requests for both models ({mode})")
        useful = {s: {i: int(judged[s][i]["useful"]) for i in both} for s in judged}
        diff = D.head_to_head(useful["new"], useful["ref"], both)
        tag = "off" if mode == "off" else "on"
        out[f"bison_v3_{tag}"] = rate_text(D.rate([useful["new"][i] for i in both]))
        out[f"bison_v21_{tag}"] = rate_text(D.rate([useful["ref"][i] for i in both]))
        out[f"bison_diff_{tag}"] = f"{diff['diff']:+.1f} pts {D.sci(diff['ci'])}"
        out[f"bison_n_{tag}"] = str(len(both))
        counts[mode] = {s: {"refused": sum(int(gens[s][i]["refusal"]) for i in ids),
                            "cap": sum(bool(gens[s][i]["cap_hit"]) for i in ids)} for s in gens}
    off, on = counts["off"], counts["e09"]
    out["refused_v3"] = (f"{off['new']['refused']} of {BASIS_N} requests with thinking off and "
                         f"{on['new']['refused']} of {BASIS_N} with thinking on")
    out["cap_v3_on"] = D.f1(D.pct(on["new"]["cap"], BASIS_N))
    out["cap_v21_on"] = D.f1(D.pct(on["ref"]["cap"], BASIS_N))
    return out


def jev_strings(stats: dict) -> dict[str, str]:
    if stats.get("mode") != "compare" or stats.get("basis_n") != BASIS_N:
        raise SystemExit(f"STOP: need a compare run on {BASIS_N} requests, got {stats.get('mode')} / {stats.get('basis_n')}")
    for side in ("new", "ref_rejudged"):
        for mode in MODES:
            n = stats["models"][side][f"useful_{mode}"]["n"]
            if n != BASIS_N:
                raise SystemExit(f"STOP: {side} {mode} has {n} judged rows, not {BASIS_N}")
    new, ref, diff = stats["paste"]["new"], stats["paste"]["ref_rejudged"], stats["paste"]["new_minus_ref"]
    return {"v3_off": f"{new['OFF']} {new['OFF_CI']}", "v3_on": f"{new['E09']} {new['E09_CI']}",
            "v21_off": f"{ref['OFF']} {ref['OFF_CI']}", "v21_on": f"{ref['E09']} {ref['E09_CI']}",
            "diff_off": diff["H2H_OFF"], "diff_on": diff["H2H_E09"]}


def write(values: Path, new: dict[str, str], done: bool) -> None:
    vals = json.loads(values.read_text())
    group = vals["useful263"]
    for k, v in new.items():
        leaf = group.setdefault(k, {"unit": "text", "note": f"usefulness re-run, {k}"})
        leaf.update(value=v, source=SOURCE)
    if done:
        vals["flags"]["useful263_done"]["value"] = True
    values.write_text(json.dumps(vals, indent=1) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--stats", type=Path)
    ap.add_argument("--values", type=Path, default=ROOT / "package_v3" / "values.json")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    new = bison_strings(a.out_dir.expanduser())
    if a.stats:
        new |= jev_strings(json.loads(a.stats.expanduser().read_text()))
    for k, v in sorted(new.items()):
        print(f"  useful263.{k} = {v}")
    if a.dry_run:
        print("DRY RUN: values.json not changed.")
        return 0
    write(a.values, new, done=bool(a.stats))
    print("FILLED", len(new), "leaves;", "flags.useful263_done = true" if a.stats else "main judge still pending")
    return 0


if __name__ == "__main__":
    sys.exit(main())
