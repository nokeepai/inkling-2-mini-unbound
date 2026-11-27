"""Hard agentic debugging eval: 15 multi-function modules (100-160 lines, 3 interacting bugs),
hidden tests, up to 12 fix-test turns, ~48K context cap.

Each turn the user message carries the full failing-test output plus the current file, so
context grows like a real agent session. Tests stop at the first failing assert, so the line
of the first failure measures progress: a turn whose first failure is EARLIER than the best
reached so far has undone an earlier fix ("regression").
Metrics: solve rate, turns to solve, identical-edit repeats, regressions, per-turn cap hits,
thinking length by turn (does it grow?), total tokens.

  PY=~/.inkling-abl/tinker_lab/.venv/bin/python
  $PY scripts/tinker/agent_eval_hard.py --tag base_agenthard_e0.9 [--effort 0.3] [--model-path ...]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import code_eval as ce
import lab_common as lc
import rollout as ro

TASK_DIR = Path(__file__).parent / "agent_hard_tasks"
MAX_TURNS = 12
CONTEXT_BUDGET = 48000
LINE_RE = re.compile(r'tests\.py", line (\d+)')


def tasks() -> list[dict[str, Any]]:
    out = []
    for d in sorted(TASK_DIR.glob("t[0-9][0-9]_*")):
        meta = json.loads((d / "meta.json").read_text())
        out.append({"dir": d, "name": d.name, "description": meta["description"],
                    "buggy": (d / "buggy.py").read_text()})
    return out


def run_hidden(code: str, task_dir: Path) -> tuple[bool, str, int | None]:
    """(passed, test output, line of first failure in tests.py)."""
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "solution.py").write_text(code)
        shutil.copy(task_dir / "tests.py", Path(d) / "tests.py")
        try:
            res = subprocess.run([sys.executable, "-I", "tests.py"], cwd=d, capture_output=True,
                                 text=True, timeout=10)
        except subprocess.TimeoutExpired:
            return False, "Timed out after 10 seconds (possible infinite loop).", None
    out = (res.stdout + res.stderr).replace(d, "<tmp>")
    lines = [int(x) for x in LINE_RE.findall(out)]
    return res.returncode == 0, out, (lines[-1] if lines else None)


def first_prompt(t: dict[str, Any]) -> str:
    return (f"This module ({t['description']}) has bugs. I have a hidden unit-test suite and will "
            f"show you its output after each attempt.\n\n```python\n{t['buggy']}```\n\n"
            "Fix the bugs. Reply with the COMPLETE corrected module in a single ```python code block.")


def feedback(out: str, current: str) -> str:
    return (f"I ran the hidden tests on your version. Output:\n\n```\n{out.strip()[-3000:]}\n```\n\n"
            f"Current file:\n\n```python\n{current}```\n\n"
            "Fix the remaining problems. Reply with the COMPLETE corrected module in a single ```python code block.")


def extract(answer: str, fallback: str) -> str:
    blocks = ce.CODE_RE.findall(answer)
    return blocks[-1] if blocks else fallback


def norm(code: str) -> str:
    return "\n".join(line.rstrip() for line in code.strip().splitlines() if line.strip())


async def solve(sc: Any, renderer: Any, tok: Any, t: dict[str, Any], cfg: ro.SampleCfg,
                sem: asyncio.Semaphore) -> dict[str, Any]:
    from tinker_cookbook.renderers import Message

    msgs = [Message(role="user", content=first_prompt(t))]
    current, seen, turns = t["buggy"], [norm(t["buggy"])], []
    best_line, solved_at, stop = 0, None, "max_turns"
    for turn in range(MAX_TURNS):
        prompt = renderer.build_generation_prompt(msgs, effort=cfg.effort).to_ints()
        if len(prompt) + cfg.think_cap + cfg.answer_cap > CONTEXT_BUDGET:
            stop = "context"
            break
        r = (await ro.rollout_group(sc, prompt, 1, cfg, sem))[0]
        s = ro.to_sample(r, tok, {"category": "code"}, cfg)
        code = extract(s.answer, current)
        ok, out, line = await asyncio.to_thread(run_hidden, code, t["dir"])
        regress = line is not None and best_line and line < best_line
        turns.append({"turn": turn, "passed": ok, "repeat_edit": norm(code) in seen,
                      "no_code": not ce.CODE_RE.findall(s.answer), "regression": bool(regress),
                      "fail_line": line, "cap_hit": s.think_cap_hit, "runaway": s.answer_runaway,
                      "think_tokens": s.think_tokens, "answer_tokens": s.answer_tokens,
                      "prompt_tokens": len(prompt)})
        seen.append(norm(code))
        best_line = max(best_line, line or 0)
        current = code
        if ok:
            solved_at, stop = turn + 1, "solved"
            break
        msgs += [Message(role="assistant", content=s.answer[:20000]),
                 Message(role="user", content=feedback(out, current))]
    return {"name": t["name"], "solved_at": solved_at, "stop": stop, "turns": turns,
            "total_tokens": sum(x["think_tokens"] + x["answer_tokens"] for x in turns)}


def summarise(res: list[dict[str, Any]], args: argparse.Namespace, wall: float) -> dict[str, Any]:
    allt = [x for r in res for x in r["turns"]]
    later = [x for x in allt if x["turn"] > 0]
    solved = [r for r in res if r["solved_at"]]
    by_turn = {}
    for k in range(MAX_TURNS):
        xs = [x["think_tokens"] for x in allt if x["turn"] == k]
        if xs:
            by_turn[k] = {"n": len(xs), "median_think": statistics.median(xs),
                          "cap_hit": sum(x["cap_hit"] for x in allt if x["turn"] == k) / len(xs)}
    return {"tag": args.tag, "model_path": args.model_path, "effort": args.effort,
            "temperature": args.temperature, "no_force": args.no_force, "tasks": len(res),
            "solve_rate": len(solved) / len(res),
            "mean_turns_to_solve": statistics.mean(r["solved_at"] for r in solved) if solved else None,
            "repeat_edit_rate": sum(x["repeat_edit"] for x in later) / max(1, len(later)),
            "regression_rate": sum(x["regression"] for x in later) / max(1, len(later)),
            "no_code_rate": sum(x["no_code"] for x in allt) / len(allt),
            "cap_hit_per_turn": sum(x["cap_hit"] for x in allt) / len(allt),
            "runaway_per_turn": sum(x["runaway"] for x in allt) / len(allt),
            "median_think_turn1": by_turn.get(0, {}).get("median_think"),
            "median_think_turn4plus": statistics.median([x["think_tokens"] for x in allt if x["turn"] >= 3])
            if any(x["turn"] >= 3 for x in allt) else None,
            "mean_total_tokens": statistics.mean(r["total_tokens"] for r in res),
            "stops": {k: sum(r["stop"] == k for r in res) for k in ("solved", "max_turns", "context")},
            "by_turn": by_turn, "turns_total": len(allt), "wall_s": round(wall)}


async def run(args: argparse.Namespace) -> None:
    # --no-force: a thinking-cap hit ends the turn with no answer (models never-ending thinking).
    cfg = ro.SampleCfg(temperature=args.temperature, effort=args.effort,
                       force_answer_after_cap=not args.no_force)
    renderer, tok = lc.renderer_and_tokenizer()
    svc = lc.service_client()
    if args.model_path:
        sc = await svc.create_sampling_client(model_path=args.model_path)
        lc.assert_free(await sc.get_base_model_async())
    else:
        sc = await lc.sampling_client(svc)
    sem = asyncio.Semaphore(32)
    t0 = time.time()
    res = list(await asyncio.gather(*(solve(sc, renderer, tok, t, cfg, sem) for t in tasks())))
    lc.write_jsonl(lc.LAB_DIR / "eval" / f"{args.tag}.jsonl", res)
    summary = summarise(res, args, time.time() - t0)
    (lc.LAB_DIR / "eval" / f"{args.tag}.summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != "by_turn"}))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--model-path", default=None)
    ap.add_argument("--effort", type=float, default=0.9)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--no-force", action="store_true")
    asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    main()
