#!/usr/bin/env python3
"""Export per-item results for publication, with NO prompt, answer or thinking text.

  python3 -I package_v3/results/build_results.py --root ~/.inkling-abl/stack_evals/judged \
      --tags stock,PRIME_TAG,v21,V3_TAG [--extra-root LABEL=ROOT ...] [--root-e03 ROOT --e03-tags stock,v21] \
      [--ci package_v3/ci_prime.json --ci package_v3/ci_v3.json] --out package_v3/results/data

Every output row is built from a whitelist of numeric and flag fields (FIELDS); string fields are limited to short
identifiers (STRING_FIELDS, capped at 80 characters). Model text is never copied: maths "number correct" is computed
in memory by scripts/evalsuite/stats.py from the saved answer and only the boolean is written. The uncensored
benchmark keeps only ids, category, level, grades and token counts (UBENCH_FIELDS).

Output: <out>/<root label>/<tag>/<suite>.jsonl, <out>/<root label>/<tag>/settings_<suite>.json (decoding settings
and run arguments, no paths or URLs), <out>/intervals/<file>.json (the compare.py --ci --json files, numbers only),
and <out>/MANIFEST.json (row counts per file).
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "scripts" / "evalsuite"))
import stats as ST  # noqa: E402  (the maths "number" scorer; reads the answer in memory, returns a bool)

NUMERIC_OR_FLAG = ("passed", "correct", "correct_number", "maths_correct", "no_answer", "no_box", "think_tokens",
                   "answer_tokens", "think_cap_hit", "answer_runaway", "loop_any", "loop_think", "loop_answer",
                   "forced", "seconds", "judge_p", "alt_p", "words", "answer_words", "target_words",
                   # agent_hard: per-task outcome
                   "solved_at", "total_tokens", "turns")
STRING_FIELDS = ("id", "category", "subject", "set", "condition", "finish", "stop", "name", "key", "level", "split",
                 "ub_label", "ub_strict")
UBENCH_FIELDS = ("id", "key", "category", "level", "condition", "think_tokens", "answer_tokens", "think_cap_hit",
                 "answer_runaway", "ub_willing", "ub_held", "ub_complete", "ub_delivery", "ub_credit", "ub_refused",
                 "ub_lectured", "ub_label", "ub_strict")
MATHS_SUITES = {"pub_gsm8k", "pub_gsm8k_full", "hardmath", "main"}
SETTINGS_KEYS = ("effort", "temperature", "top_p", "think_cap", "answer_cap", "concurrency", "force_mode")
ARG_KEYS = ("seed", "repeats", "limit", "suites", "ubench_conditions", "judges", "effort", "temperature", "top_p")


def clean_value(k: str, v: object) -> object:
    if isinstance(v, bool) or v is None:
        return v
    if isinstance(v, (int, float)):
        return v
    if k in STRING_FIELDS and isinstance(v, (str, int)):
        return str(v)[:80]
    return None


def export_row(r: dict, suite: str) -> dict:
    if suite.startswith("ubench"):
        out = {k: clean_value(k, r.get(k)) for k in UBENCH_FIELDS if k in r}
        if isinstance(r.get("ub_probs"), dict):
            out["ub_probs"] = {str(k)[:40]: float(v) for k, v in r["ub_probs"].items() if isinstance(v, (int, float))}
        out["error"] = bool(r.get("error") or r.get("err"))
        return {k: v for k, v in out.items() if v is not None}
    out = {k: clean_value(k, r[k]) for k in (*STRING_FIELDS, *NUMERIC_OR_FLAG) if k in r}
    out["error"] = bool(r.get("error"))   # the message itself is never copied
    if isinstance(r.get("turns"), list):
        out["n_turns"] = len(r["turns"])   # agent_hard: count only, never the turns themselves
    if suite in MATHS_SUITES:
        out["strict"] = ST._strict(r)
        out["number"] = None if r.get("error") else ST._number(r)   # bool computed in memory from the answer
    return {k: v for k, v in out.items() if v is not None}


def settings_of(meta_path: Path) -> dict:
    m = json.loads(meta_path.read_text())
    c, a = m.get("client_cfg", {}), m.get("args", {})
    return {"client": {k: c[k] for k in SETTINGS_KEYS if k in c}, "args": {k: a[k] for k in ARG_KEYS if k in a},
            "started": m.get("started")}


def export_tag(root: Path, tag: str, dest: Path, manifest: dict) -> None:
    tdir = root / tag
    if not tdir.is_dir():
        print(f"  missing: {tdir}")
        return
    for sdir in sorted(p for p in tdir.iterdir() if p.is_dir()):
        src = sdir / "r0.jsonl"
        if not src.exists():
            continue
        rows = [json.loads(x) for x in src.read_text().splitlines() if x.strip()]
        out = dest / tag / f"{sdir.name}.jsonl"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("".join(json.dumps(export_row(r, sdir.name)) + "\n" for r in rows))
        manifest[str(out.relative_to(dest.parent))] = len(rows)
        meta = tdir / f"meta_{sdir.name}.json"
        if meta.exists():
            (dest / tag / f"settings_{sdir.name}.json").write_text(json.dumps(settings_of(meta), indent=1))
    print(f"  exported {root.name}/{tag}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="main results root (<tag>/<suite>/r0.jsonl)")
    ap.add_argument("--tags", required=True, help="comma list of tags under --root")
    ap.add_argument("--extra-root", action="append", default=[], metavar="LABEL=ROOT",
                    help="another results root (LABEL=ROOT[@tag1,tag2]); repeatable")
    ap.add_argument("--root-e03", help="low-effort arm results root")
    ap.add_argument("--e03-tags", default="stock,v21")
    ap.add_argument("--ci", action="append", default=[], help="compare.py --ci --json file; repeatable")
    ap.add_argument("--out", default=str(HERE / "data"))
    a = ap.parse_args()
    out = Path(a.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, int] = {}
    jobs = [("main", Path(a.root).expanduser(), a.tags)]
    for x in a.extra_root:   # LABEL=ROOT, or LABEL=ROOT@tag1,tag2 when that root uses other tag names
        label, rest = x.split("=", 1)
        root, _, tags = rest.partition("@")
        jobs.append((label, Path(root).expanduser(), tags or a.tags))
    if a.root_e03:
        jobs.append(("effort_low", Path(a.root_e03).expanduser(), a.e03_tags))
    for label, root, tags in jobs:
        for tag in [t for t in tags.split(",") if t]:
            export_tag(root, tag, out / label, manifest)
    for c in a.ci:
        p = Path(c).expanduser()
        (out / "intervals").mkdir(exist_ok=True)
        shutil.copyfile(p, out / "intervals" / p.name)   # compare.py --json holds numbers only
        manifest[f"intervals/{p.name}"] = 1
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=1, sort_keys=True))
    print(f"wrote {len(manifest)} files under {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
