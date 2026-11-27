#!/usr/bin/env python3
"""Useful answers on the 263-request basis: a new model vs the v2.1 reference, judged in one pass.

    useful263.py judge --ev-root ~/.inkling-abl/v3_private/ev --new v3u --ref v31C \
        --out-dir ~/.inkling-abl/v3_private/ev/useful263 [--max-usd 0.5] [--dry-run]
    useful263.py stats --out-dir ~/.inkling-abl/v3_private/ev/useful263 --json OUT.json
    useful263.py stats --selfcheck --json OUT.json        # v2.1's saved files only; must match package_v21

judge: copies the ref's and the new model's old_{off,e09}.jsonl into OUT_DIR (sources are never touched), keeping
only the ref's basis ids (the 263 = v2's 277 minus cbrn_weapons, already stripped of mc_ids), then runs
scripts/v3/judge_v3.py jev on all four files and bison on all four in the same pass, so judge version drift hits
ref and new alike.
stats: the exact definitions of scripts/charts/data_v21.py (imported, not copied): useful = Jev p_fulfils >= 0.5,
95% bootstrap (10,000, seed 1234), refusal phrase flag, think-cap flag, Bison on its judged rows only, paired
head-to-head bootstrap (10,000, seed 7).
Only ids, counts and numeric fields are printed; prompt, answer and thinking text is never printed.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(HERE.parent / "charts"))
# data_v21 has no import-time side effects: it only reads package_v2/data_test8k.json and imports agg_v3.
import data_v21 as D  # noqa: E402

JUDGE = HERE.parent / "v3" / "judge_v3.py"
PUBLISHED = REPO / "package_v21" / "values.json"
MODES: tuple[str, ...] = D.MODES
SIDES = ("ref", "new")
BACKENDS = ("jev", "bison")
DEFAULT_EV = os.path.expanduser("~/.inkling-abl/v3_private/ev")
# v2.1's bison run: ~300-410k input tokens per 263-row file at $0.10 / 1M; used for the dry-run estimate.
BISON_USD_PER_CHAR = 0.10 / 1e6 / 3.6

Row = dict[str, Any]


# ---- file helpers (ids and numeric fields only) ------------------------------
def read_lines(path: Path) -> list[str]:
    return [line for line in path.open() if line.strip()]


def line_id(line: str) -> str:
    return json.loads(line)["id"]


def latest_by_id(path: Path) -> dict[str, Row]:
    """judge_v3 appends retries, so the last row for an id is the one that counts (same as data_v21.rows)."""
    out: dict[str, Row] = {}
    if path.exists():
        for line in read_lines(path):
            r = json.loads(line)
            out[r["id"]] = r
    return out


def basis_ids(ref_gens: Path) -> list[str]:
    return sorted(line_id(line) for line in read_lines(ref_gens))


# ---- judge -------------------------------------------------------------------
def copy_basis(src: Path, dst: Path, basis: set[str]) -> dict[str, int]:
    """Copy rows verbatim, keeping one row per basis id (last wins, as in data_v21.rows); src is read only."""
    if not src.exists():
        raise SystemExit(f"missing {src} (merge shards into old_<mode>.jsonl first)")
    lines = read_lines(src)
    keep: dict[str, str] = {}
    for line in lines:
        i = line_id(line)
        if i in basis:
            keep[i] = line
    missing = len(basis - set(keep))
    if missing:
        raise SystemExit(f"{src.name}: {missing} basis ids missing (generation not finished?)")
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists() or dst.read_text() != "".join(keep[i] for i in sorted(keep)):
        dst.write_text("".join(keep[i] for i in sorted(keep)))
    return {"src_rows": len(lines), "kept": len(keep), "dropped_off_basis": len(lines) - len(keep)}


def judge_status(gens: Path, out: Path, only: set[str] | None = None) -> dict[str, Any]:
    """Judged / errored counts for one file; `only` restricts to basis ids (dry run reads the uncopied source)."""
    lines = [line for line in read_lines(gens) if only is None or line_id(line) in only]
    want = {line_id(line) for line in lines}
    got = latest_by_id(out)
    ok = {i for i, r in got.items() if i in want and r.get("error") is None}
    models = collections.Counter(str(got[i].get("judge_model")) for i in ok if not got[i].get("empty"))
    chars = sum(len(json.loads(line).get("answer") or "") + len(json.loads(line).get("prompt") or "")
                for line in lines if line_id(line) not in ok)
    return {"rows": len(want), "judged": len(ok), "errors": len(want) - len(ok), "empty": sum(
        1 for i in ok if got[i].get("empty")), "judge_model": dict(models), "todo_chars": chars}


def run_judge(gens: Path, out: Path, backend: str, max_usd: float | None, conc: int) -> str:
    cmd = [sys.executable, str(JUDGE), str(gens), str(out), "--backend", backend, "--conc", str(conc)]
    if max_usd is not None:
        cmd += ["--max-usd", str(max_usd)]
    p = subprocess.run(cmd, capture_output=True, text=True)
    # Only judge_v3's summary lines (counts and cost) are surfaced; its row output never contains request text.
    tail = [ln for ln in p.stdout.splitlines() if ln.startswith(("JUDGE_DONE", "STOPPED"))]
    if p.returncode:
        tail.append(f"exit {p.returncode}: {p.stderr.strip().splitlines()[-1][:200] if p.stderr.strip() else ''}")
    return " | ".join(tail)


def judge_jobs(out_dir: Path) -> list[tuple[str, Path, Path]]:
    """All jev jobs first, then all bison jobs, so both models see the same judge state."""
    return [(b, out_dir / f"{s}_{m}.jsonl", out_dir / f"{b}_{s}_{m}.jsonl")
            for b in BACKENDS for s in SIDES for m in MODES]


def cmd_judge(a: argparse.Namespace) -> None:
    ev, out_dir = Path(a.ev_root).expanduser(), Path(a.out_dir).expanduser()
    basis = set(basis_ids(ev / a.ref / "old_off.jsonl"))
    assert basis == set(basis_ids(ev / a.ref / "old_e09.jsonl")), "ref off/e09 ids differ"
    print(f"basis {len(basis)} ids (from {a.ref}/old_off.jsonl)")
    for side, name in (("ref", a.ref), ("new", a.new)):
        for m in MODES:
            src, dst = ev / name / f"old_{m}.jsonl", out_dir / f"{side}_{m}.jsonl"
            if a.dry_run and not src.exists():
                print(f"copy {side}_{m}: SOURCE MISSING {src}")
                continue
            info = copy_basis(src, dst, basis) if not a.dry_run else {"src_rows": len(read_lines(src))}
            print(f"copy {side}_{m}: {info}")
    est = 0.0
    sources = {f"{s}_{m}.jsonl": ev / name / f"old_{m}.jsonl" for s, name in (("ref", a.ref), ("new", a.new)) for m in MODES}
    for backend, gens, out in judge_jobs(out_dir):
        src = sources[gens.name] if a.dry_run else gens
        if not src.exists():
            print(f"{backend:5s} {gens.name:14s} no input yet")
            continue
        st = judge_status(src, out, basis)
        if backend == "bison":
            est += st["todo_chars"] * BISON_USD_PER_CHAR
        if st["errors"] == 0:
            print(f"{backend:5s} {gens.name:14s} skip: {st['judged']}/{st['rows']} judged {st['judge_model']}")
            continue
        if a.dry_run:
            print(f"{backend:5s} {gens.name:14s} would judge {st['errors']} of {st['rows']} rows")
            continue
        for _ in range(a.passes):  # extra pass retries rate-limit errors (judge_v3 resumes errored rows only)
            line = run_judge(gens, out, backend, a.max_usd, a.conc)
            st = judge_status(gens, out)
            if st["errors"] == 0:
                break
        print(f"{backend:5s} {gens.name:14s} {st['judged']}/{st['rows']} judged, {st['errors']} errors, "
              f"{st['empty']} empty, model {st['judge_model']} [{line}]")
    print(f"bison cost estimate for remaining rows: ${est:.3f} (jev: no per-call price in judge_v3)")


# ---- stats -------------------------------------------------------------------
def load_side(gens: Path, jev: Path, bison: Path) -> dict[str, dict]:
    """Same shape as data_v21.load_model: {mode: {gens, jev, bison}}; paths contain '{m}'."""
    out: dict[str, dict] = {}
    for m in MODES:
        out[m] = {"gens": D.rows(str(gens).format(m=m), D.G_KEYS), "jev": D.judged(str(jev).format(m=m))}
        if Path(str(bison).format(m=m)).exists():
            out[m]["bison"] = D.judged(str(bison).format(m=m))
    return out


def check_cover(name: str, d: dict[str, dict], basis: list[str]) -> None:
    for m in MODES:
        for part in ("gens", "jev"):
            miss = [i for i in basis if i not in d[m][part]]
            if miss:
                raise SystemExit(f"{name} {m}: {len(miss)} basis ids missing from {part} (rerun judge)")


def h2h(a: dict, b: dict, basis: list[str]) -> dict[str, dict]:
    """a minus b per mode with data_v21.head_to_head (paired bootstrap, 10,000, seed 7)."""
    return {m: D.head_to_head(a[m]["jev"], b[m]["jev"], basis) for m in MODES}


def fmt_rate(x: dict) -> str:
    return f"{D.f1(x['value'])} [{x['ci'][0]:.1f}, {x['ci'][1]:.1f}]"


def fmt_diff(h: dict) -> str:
    return f"{h['diff']:+.1f} pts {D.sci(h['ci'])} (wins {h['a_only']} / losses {h['b_only']}, n {h['n']})"


def paste_strings(s: dict) -> dict[str, str]:
    """Strings formatted exactly like package_v21/values.json (data_v21.model_values)."""
    v: dict[str, str] = {"REF": f"{D.f1(s['refusal_off']['value'])} / {D.f1(s['refusal_e09']['value'])}",
                         "REF_N": f"{s['refusal_off']['count']} / {s['refusal_e09']['count']}"}
    for m in MODES:
        t = m.upper()
        v[t] = D.f1(s[f"useful_{m}"]["value"])
        v[f"{t}_CI"] = f"[{s[f'useful_{m}']['ci'][0]:.1f}, {s[f'useful_{m}']['ci'][1]:.1f}]"
        v[f"CAP_{t}"] = D.f1(s[f"thinkcap_{m}"]["value"])
        if f"bison_useful_{m}" in s:
            v[f"BISON_{t}"], v[f"BISON_{t}_N"] = D.f1(s[f"bison_useful_{m}"]["value"]), str(s[f"bison_useful_{m}"]["n"])
    return v


def h2h_strings(h: dict[str, dict]) -> dict[str, str]:
    v: dict[str, str] = {}
    for m in MODES:
        t = m.upper()
        v[f"H2H_{t}"] = f"{h[m]['diff']:+.1f} pts {D.sci(h[m]['ci'])}"
        v[f"H2H_{t}_D"], v[f"H2H_{t}_CI"] = f"{h[m]['diff']:+.1f}", D.sci(h[m]["ci"])
        v[f"H2H_{t}_WIN"], v[f"H2H_{t}_LOSS"] = str(h[m]["a_only"]), str(h[m]["b_only"])
    return v


def print_table(models: dict[str, dict]) -> None:
    print(f"{'model':14s} {'mode':4s} {'n':>4s} {'useful (Jev)':22s} {'refused':12s} {'cap':12s} {'Bison (n)':16s}")
    for name, s in models.items():
        for m in MODES:
            r, c = s[f"refusal_{m}"], s[f"thinkcap_{m}"]
            b = s.get(f"bison_useful_{m}")
            bs = f"{D.f1(b['value'])} ({b['n']})" if b else "-"
            print(f"{name:14s} {m:4s} {s['n']:4d} {fmt_rate(s[f'useful_{m}']):22s} "
                  f"{D.f1(r['value'])} ({r['count']}){'':2s} {D.f1(c['value'])} ({c['count']}){'':2s} {bs}")


def selfcheck(models: dict[str, dict], extra_h2h: dict[str, dict] | None) -> dict[str, Any]:
    """Compare the recomputed ref strings with the published V21_* (and H2H_V2_*) values."""
    pub = json.load(PUBLISHED.open())
    got = {f"V21_{k}": x for k, x in paste_strings(models["ref"]).items() if k != "REF_N"}
    if extra_h2h:
        got |= {k.replace("H2H_", "H2H_V2_"): x for k, x in h2h_strings(extra_h2h).items() if k.count("_") > 1}
    rows = {k: {"published": pub.get(k), "recomputed": x, "match": pub.get(k) == x} for k, x in got.items()}
    print("\nSELFCHECK vs package_v21/values.json")
    for k, r in rows.items():
        print(f"  {'OK ' if r['match'] else 'GAP'} {k:18s} published {r['published']!s:16s} recomputed {r['recomputed']}")
    print(f"  {sum(r['match'] for r in rows.values())}/{len(rows)} match")
    return rows


def v2_campaign() -> dict[str, dict] | None:
    """v2 release campaign files, used only in --selfcheck to test the paired bootstrap against H2H_V2_*."""
    base = Path(D.V) / D.CAMPAIGN["v2"]
    if not (base / "jev_harmful_off.jsonl").exists():
        return None
    return {m: {"jev": D.judged(str(base / f"jev_harmful_{m}.jsonl"))} for m in MODES}


def cmd_stats(a: argparse.Namespace) -> None:
    ev, out_dir = Path(a.ev_root).expanduser(), Path(a.out_dir).expanduser()
    pub_dir = ev / a.ref
    published_ref = load_side(pub_dir / "old_{m}.jsonl", pub_dir / "jev_old_{m}.jsonl", pub_dir / "bison_old_{m}.jsonl")
    basis = sorted(published_ref["off"]["gens"])
    assert set(published_ref["e09"]["gens"]) == set(basis)
    check_cover("ref_published", published_ref, basis)
    result: dict[str, Any] = {"basis_n": len(basis), "definitions": "scripts/charts/data_v21.py (imported)"}
    if a.selfcheck:
        models = {"ref": D.summary(published_ref, basis)}
        print_table(models)
        v2 = v2_campaign()
        extra = h2h(published_ref, v2, basis) if v2 else None
        result |= {"mode": "selfcheck", "models": models, "selfcheck": selfcheck(models, extra)}
        if extra:
            result["h2h_v21_minus_v2_campaign"] = extra
    else:
        sides = {s: load_side(out_dir / f"{s}_{{m}}.jsonl", out_dir / f"jev_{s}_{{m}}.jsonl",
                              out_dir / f"bison_{s}_{{m}}.jsonl") for s in SIDES}
        for s, d in sides.items():
            check_cover(s, d, basis)
        models = {"ref_rejudged": D.summary(sides["ref"], basis), "ref_published": D.summary(published_ref, basis),
                  "new": D.summary(sides["new"], basis)}
        diff = h2h(sides["new"], sides["ref"], basis)
        drift = h2h(sides["ref"], published_ref, basis)
        print_table(models)
        print("\npaired new minus ref (same judge pass):")
        for m in MODES:
            print(f"  {m:4s} {fmt_diff(diff[m])}")
        print("judge drift, ref re-judged minus ref published:")
        for m in MODES:
            print(f"  {m:4s} {fmt_diff(drift[m])}")
        paste = {"new": paste_strings(models["new"]), "ref_rejudged": paste_strings(models["ref_rejudged"]),
                 "ref_published": paste_strings(models["ref_published"]), "new_minus_ref": h2h_strings(diff),
                 "drift_ref": h2h_strings(drift)}
        print("\nready to paste:")
        for grp, vals in paste.items():
            print(f"  [{grp}] " + "; ".join(f"{k}={x}" for k, x in vals.items()))
        result |= {"mode": "compare", "models": models, "new_minus_ref": diff, "ref_rejudged_minus_published": drift,
                   "paste": paste, "judge_models": judge_models(out_dir)}
    Path(a.json).expanduser().write_text(json.dumps(result, indent=1))
    print("wrote", a.json)


def judge_models(out_dir: Path) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for b in BACKENDS:
        for s in SIDES:
            for m in MODES:
                rows = latest_by_id(out_dir / f"{b}_{s}_{m}.jsonl").values()
                out[f"{b}_{s}_{m}"] = dict(collections.Counter(str(r.get("judge_model")) for r in rows
                                                               if r.get("error") is None and not r.get("empty")))
    return out


# ---- cli ---------------------------------------------------------------------
def parse() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    j = sub.add_parser("judge")
    s = sub.add_parser("stats")
    for p in (j, s):
        p.add_argument("--ev-root", default=DEFAULT_EV)
        p.add_argument("--ref", default="v31C")
        p.add_argument("--out-dir", default=os.path.join(DEFAULT_EV, "useful263"))
    j.add_argument("--new", required=True)
    j.add_argument("--max-usd", type=float, default=None, help="passed to every judge_v3 call (per file)")
    j.add_argument("--conc", type=int, default=16)
    j.add_argument("--passes", type=int, default=2, help="judge_v3 runs per file while errors remain")
    j.add_argument("--dry-run", action="store_true", help="copy nothing, call nothing; print what would be judged")
    s.add_argument("--json", required=True)
    s.add_argument("--selfcheck", action="store_true", help="v2.1 saved files only, compared with package_v21")
    return ap.parse_args()


def main() -> None:
    a = parse()
    (cmd_judge if a.cmd == "judge" else cmd_stats)(a)


if __name__ == "__main__":
    main()
