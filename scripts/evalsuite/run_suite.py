"""Portable loop/overthinking + quality eval for any Inkling-2-Mini variant behind an OpenAI-compatible
endpoint (vLLM or llama.cpp server). Stdlib only; ports the Tinker lab evals (scripts/tinker/).

  python3 run_suite.py --base-url http://HOST:8000/v1 --model NAME --tag TAG \
      [--effort 0.9] [--temperature 0.6] [--top-p 1.0] [--repeats 3] [--seed 1234] \
      [--suites main,hardmath,code,agent_hard,agent_hard_noforce,rp,sysrules] \
      [--judges bison,haiku]          # paid (OpenRouter), off by default; or later: --judge-only
  python3 compare.py TAG1 TAG2 ...   # markdown table, mean ± sd over repeats
  --suites ubench_all (or all) adds the uncensoredbench suites (ubench.py); select_ckpt.py picks a checkpoint.
  Aliases: lab = the seven lab suites (default), ubench_all = the six ubench suites, all = both,
  pub = the four public benchmarks (pub.py, data from build_pub.py), pub_new = the five release suites
  (full GSM8K, MMLU 2k, MBPP, HumanEvalFix, MATH-500), pub_all = all nine. README.md has the commands.

Effort: sent per request as chat_template_kwargs {"reasoning_effort": E} (vLLM and llama-server --jinja both
accept it). For a server that ignores per-request kwargs, start it with the effort baked in (llama-server
--chat-template-kwargs '{"reasoning_effort":0.9}') and pass --effort-transport none. The preflight checks
that the effort reaches the template and aborts if it provably does not.
Caps: thinking 4,096 and answer 8,192 tokens, enforced client-side while streaming (see client.py).
Outputs: <out-root>/<tag>/meta.json and <out-root>/<tag>/<suite>/{r<i>.jsonl, r<i>.summary.json, summary.json}.
Data: --data-dir (default ~/.inkling-abl/tinker_lab) and --agent-dir (default ../tinker/agent_hard_tasks).
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import judges  # noqa: E402
import pub as P  # noqa: E402
import suites as S  # noqa: E402
import ubench as U  # noqa: E402
from client import Client, ClientCfg  # noqa: E402

S.SUITES.update(U.SUITES)  # registered here, not in suites.py, so ubench <-> suites never import in a cycle
S.SUITES.update(P.SUITES)  # public benchmarks (pub.py), registered the same way
ALL_SUITES = "main,hardmath,code,agent_hard,agent_hard_noforce,rp,sysrules"
UBENCH_SUITES = "ubench,ubench_fa,ubench_premises,ubench_rp,ubench_smarts,ubench_long"
PUB_SUITES = ",".join(P.SUITE_NAMES)
# "all" is left as it was so existing runs do not grow; the public suites have their own alias
SUITE_ALIASES = {"lab": ALL_SUITES, "ubench_all": UBENCH_SUITES, "all": ALL_SUITES + "," + UBENCH_SUITES,
                 "pub": PUB_SUITES, "pub_new": ",".join(P.NEW_SUITE_NAMES),
                 "pub_all": ",".join(P.SUITE_NAMES + P.NEW_SUITE_NAMES)}
KNOWN_JUDGES = ("bison", "haiku", "glm")
JEV_NOTE = ("note: Jev (TypeSafe) is not available on this machine (no key), so it is dropped. ubench marks with "
            "Bison, the benchmark's main judge; add 'glm' for the GLM text-judge second opinion that stands in for Jev.")


def parse() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--effort", type=float, default=0.9)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--suites", default=ALL_SUITES)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--seed", type=int, default=None, help="per-request seeds derived from this (reproducible)")
    ap.add_argument("--think-cap", type=int, default=4096)
    ap.add_argument("--answer-cap", type=int, default=8192)
    ap.add_argument("--concurrency", type=int, default=32, help="max in-flight HTTP requests")
    ap.add_argument("--limit", type=int, default=0, help="first N items per suite (smoke runs)")
    ap.add_argument("--judges", default="", help="comma list of bison,haiku,glm (paid; spend-capped); "
                    "ubench suites use bison (benchmark rubric) and glm")
    ap.add_argument("--ubench-dir", default="~/Code/uncensoredbench", help="benchmark data root (questions/, prompts/)")
    ap.add_argument("--ubench-conditions", default="command",
                    help="ubench / ubench_fa conditions: command, permissive (prompts/system_permissive.txt)")
    ap.add_argument("--rp-chats", type=int, default=30, help="ubench_rp chats, balanced by theme")
    ap.add_argument("--glm-sample", type=int, default=150, help="ubench answers the glm judge reads per repeat")
    ap.add_argument("--judge-only", action="store_true", help="judge saved outputs of --tag; no sampling")
    ap.add_argument("--effort-transport", choices=["kwargs", "none"], default="kwargs")
    ap.add_argument("--raw-markers", action="store_true",
                    help="server has no reasoning parser: request special tokens and split channels here")
    ap.add_argument("--force-mode", choices=["completions", "none"], default="completions",
                    help="answer after a thinking-cap hit via /v1/completions (lab default) or not at all")
    ap.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", ""))
    ap.add_argument("--extra-body", default="{}", help="JSON merged into every request body")
    ap.add_argument("--data-dir", default="~/.inkling-abl/tinker_lab")
    ap.add_argument("--agent-dir", default=str(HERE.parent / "tinker" / "agent_hard_tasks"))
    ap.add_argument("--out-root", default="~/.inkling-abl/evalsuite")
    ap.add_argument("--skip-probe", action="store_true")
    ap.add_argument("--allow-effort-unverified", action="store_true")
    return ap.parse_args()


def flatten(d: Any, prefix: str = "") -> dict[str, float]:
    out: dict[str, float] = {}
    if isinstance(d, dict):
        for k, v in d.items():
            out.update(flatten(v, f"{prefix}{k}."))
    elif isinstance(d, (int, float)) and not isinstance(d, bool):
        out[prefix[:-1]] = float(d)
    return out


def aggregate(runs: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    flat = [flatten(r) for r in runs]
    keys = sorted(set().union(*flat)) if flat else []
    agg = {}
    for k in keys:
        xs = [f[k] for f in flat if k in f]
        agg[k] = {"mean": statistics.mean(xs), "sd": statistics.stdev(xs) if len(xs) > 1 else 0.0,
                  "min": min(xs), "max": max(xs), "n": len(xs)}
    return agg


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False, default=str) + "\n" for r in rows))


def write_suite_summary(sdir: Path, tag: str, suite: str, runs: list[dict[str, Any]]) -> None:
    (sdir / "summary.json").write_text(json.dumps(
        {"tag": tag, "suite": suite, "repeats": len(runs), "runs": runs, "agg": aggregate(runs)},
        indent=1, default=str))


def make_ctx(a: argparse.Namespace, client: Client, repeat: int, sdir: Path | None = None) -> S.Ctx:
    return S.Ctx(client=client, data_dir=Path(a.data_dir).expanduser(), agent_dir=Path(a.agent_dir).expanduser(),
                 judges=tuple(j for j in a.judges.split(",") if j), limit=a.limit, seed=a.seed, repeat=repeat,
                 out_dir=sdir, ubench_dir=Path(a.ubench_dir).expanduser(),
                 ubench_conditions=tuple(c for c in a.ubench_conditions.split(",") if c),
                 rp_chats=a.rp_chats, glm_sample=a.glm_sample)


def normalise(a: argparse.Namespace) -> None:
    """Expands suite aliases and checks judge names; a Jev request is dropped with a printed note."""
    a.suites = ",".join(SUITE_ALIASES.get(s, s) for s in a.suites.split(","))
    req = [j for j in a.judges.split(",") if j]
    if "jev" in req:
        print(JEV_NOTE, flush=True)
    bad = [j for j in req if j not in KNOWN_JUDGES + ("jev",)]
    if bad:
        raise SystemExit(f"unknown judges: {bad}")
    a.judges = ",".join(j for j in req if j != "jev")


def preflight(a: argparse.Namespace, client: Client, tag_dir: Path) -> dict[str, Any]:
    probe = {} if a.skip_probe else client.probe()
    meta = {"tag": a.tag, "args": {k: v for k, v in vars(a).items() if k != "api_key"},
            "client_cfg": {k: v for k, v in vars(client.cfg).items() if k != "api_key"},
            "probe": probe, "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "python": platform.python_version()}
    tag_dir.mkdir(parents=True, exist_ok=True)
    (tag_dir / "meta.json").write_text(json.dumps(meta, indent=1, default=str))
    print(json.dumps({"probe": probe}, default=str)[:1500], flush=True)
    eff = probe.get("effort_check", {})
    if a.effort_transport == "kwargs" and eff.get("differs") is False and not a.allow_effort_unverified:
        raise SystemExit("abort: reasoning_effort did not change the rendered prompt (see meta.json)")
    return meta


def judge_only(a: argparse.Namespace, client: Client, tag_dir: Path) -> None:
    for suite in a.suites.split(","):
        sdir = tag_dir / suite
        _gen, judge, summarise = S.SUITES[suite]
        runs = []
        for i in range(a.repeats):
            path = sdir / f"r{i}.jsonl"
            if not path.exists():
                break
            recs = S.read_jsonl(path)
            ctx = make_ctx(a, client, i, sdir)
            errs = judge(ctx, recs)
            old = json.loads((sdir / f"r{i}.summary.json").read_text())
            summ = {**old, **summarise(recs), "judges": list(ctx.judges), "judge_errors": errs}
            write_jsonl(path, recs)
            (sdir / f"r{i}.summary.json").write_text(json.dumps(summ, indent=1, default=str))
            runs.append(summ)
        if runs:
            write_suite_summary(sdir, a.tag, suite, runs)
            print(f"[{suite}] rejudged {len(runs)} run(s)", flush=True)


def main() -> None:
    a = parse()
    normalise(a)
    out_root = Path(a.out_root).expanduser()
    judges.set_spend_dir(out_root)
    tag_dir = out_root / a.tag
    cfg = ClientCfg(base_url=a.base_url, model=a.model, effort=a.effort, temperature=a.temperature,
                    top_p=a.top_p, think_cap=a.think_cap, answer_cap=a.answer_cap,
                    effort_transport=a.effort_transport, raw_markers=a.raw_markers, force_mode=a.force_mode,
                    concurrency=a.concurrency, api_key=a.api_key, extra_body=json.loads(a.extra_body))
    client = Client(cfg)
    unknown = [s for s in a.suites.split(",") if s not in S.SUITES]
    if unknown:
        raise SystemExit(f"unknown suites: {unknown}")
    if a.judge_only:
        judge_only(a, client, tag_dir)
        return
    preflight(a, client, tag_dir)
    for suite in a.suites.split(","):
        sdir = tag_dir / suite
        runs = []
        for i in range(a.repeats):
            recs, summ = S.run_one(make_ctx(a, client, i, sdir), suite)
            summ.update({"tag": a.tag, "suite": suite, "repeat": i})
            write_jsonl(sdir / f"r{i}.jsonl", recs)
            (sdir / f"r{i}.summary.json").write_text(json.dumps(summ, indent=1, default=str))
            runs.append(summ)
            brief = {k: v for k, v in summ.items() if not isinstance(v, (dict, list))}
            print(f"[{suite} r{i}] {json.dumps(brief, default=str)[:600]}", flush=True)
        write_suite_summary(sdir, a.tag, suite, runs)
    print(f"done: {tag_dir}", flush=True)


if __name__ == "__main__":
    main()
