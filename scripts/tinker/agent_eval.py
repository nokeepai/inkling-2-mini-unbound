"""Agentic debugging eval (harmless): 20 buggy functions, up to 8 fix-test turns each.

Turn 1 shows the buggy function and its tests. Each reply's last ```python block runs with
the tests locally (subprocess, 10 s timeout); failures go back as the next user turn.
Metrics: solve rate, turns to solve, identical-fix repeat rate (a turn whose normalised code
equals an earlier attempt), thinking-cap hits per turn, total tokens per task.

  PY=~/.inkling-abl/tinker_lab/.venv/bin/python
  $PY scripts/tinker/agent_eval.py --tag base_agent_e09 [--effort 0.3] [--model-path tinker://...]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from typing import Any

import code_eval as ce
import lab_common as lc
import rollout as ro
from agent_tasks import TASKS

MAX_TURNS = 8
CONTEXT_BUDGET = 48000  # stop a task rather than exceed ~48K prompt tokens


def first_prompt(t: dict[str, str], show_tests: bool) -> str:
    # Hidden-tests mode (default): with the tests in view every task is a one-shot fix, so
    # nothing exercises the feedback loop; hidden, the failures arrive only as feedback.
    tests = (f"It must pass these tests:\n\n```python\n{t['tests']}\n```\n\n" if show_tests
             else "I have a hidden unit-test suite for it and will tell you what fails.\n\n")
    return (f"This Python function `{t['name']}` has a bug:\n\n```python\n{t['buggy']}```\n\n{tests}"
            "Fix the function. Reply with the complete corrected function in a single ```python code block.")


def feedback(err: str) -> str:
    return (f"I ran the tests with your version and they failed:\n\n```\n{err.strip()[-1200:]}\n```\n\n"
            "Please fix it and reply with the complete corrected function in a ```python block.")


def normalise(answer: str) -> str:
    blocks = ce.CODE_RE.findall(answer)
    code = blocks[-1] if blocks else answer
    return "\n".join(line.rstrip() for line in code.strip().splitlines() if line.strip())


async def solve(sc: Any, renderer: Any, tok: Any, t: dict[str, str], cfg: ro.SampleCfg, show_tests: bool,
                sem: asyncio.Semaphore) -> dict[str, Any]:
    from tinker_cookbook.renderers import Message

    msgs = [Message(role="user", content=first_prompt(t, show_tests))]
    seen: list[str] = []
    turns = []
    solved_at = None
    for turn in range(MAX_TURNS):
        prompt = renderer.build_generation_prompt(msgs, effort=cfg.effort).to_ints()
        if len(prompt) + cfg.think_cap + cfg.answer_cap > CONTEXT_BUDGET:
            break
        r = (await ro.rollout_group(sc, prompt, 1, cfg, sem))[0]
        s = ro.to_sample(r, tok, {"category": "code"}, cfg)
        ok, err = await asyncio.to_thread(ce.run_tests, s.answer, t["tests"].splitlines())
        norm = normalise(s.answer)
        turns.append({"turn": turn, "passed": ok, "repeat_fix": norm in seen, "cap_hit": s.think_cap_hit,
                      "runaway": s.answer_runaway, "think_tokens": s.think_tokens,
                      "answer_tokens": s.answer_tokens, "err": err[-300:]})
        seen.append(norm)
        if ok:
            solved_at = turn + 1
            break
        msgs += [Message(role="assistant", content=s.answer[:6000]), Message(role="user", content=feedback(err))]
    return {"name": t["name"], "solved_at": solved_at, "turns": turns,
            "total_tokens": sum(x["think_tokens"] + x["answer_tokens"] for x in turns)}


async def run(args: argparse.Namespace) -> None:
    cfg = ro.SampleCfg(temperature=args.temperature, effort=args.effort, force_answer_after_cap=True)
    renderer, tok = lc.renderer_and_tokenizer()
    svc = lc.service_client()
    if args.model_path:
        sc = await svc.create_sampling_client(model_path=args.model_path)
        lc.assert_free(await sc.get_base_model_async())
    else:
        sc = await lc.sampling_client(svc)
    sem = asyncio.Semaphore(32)
    t0 = time.time()
    res = await asyncio.gather(*(solve(sc, renderer, tok, t, cfg, args.show_tests, sem) for t in TASKS))
    lc.write_jsonl(lc.LAB_DIR / "eval" / f"{args.tag}.jsonl", list(res))
    all_turns = [x for r in res for x in r["turns"]]
    solved = [r for r in res if r["solved_at"]]
    later = [x for x in all_turns if x["turn"] > 0]
    summary = {"tag": args.tag, "model_path": args.model_path, "effort": args.effort,
               "temperature": args.temperature, "tasks": len(res),
               "solve_rate": len(solved) / len(res),
               "solved_first_turn": sum(r["solved_at"] == 1 for r in res) / len(res),
               "mean_turns_to_solve": statistics.mean(r["solved_at"] for r in solved) if solved else None,
               "repeat_fix_rate": (sum(x["repeat_fix"] for x in later) / len(later)) if later else 0.0,
               "cap_hit_per_turn": sum(x["cap_hit"] for x in all_turns) / len(all_turns),
               "median_think_per_turn": statistics.median(x["think_tokens"] for x in all_turns),
               "mean_total_tokens": statistics.mean(r["total_tokens"] for r in res),
               "turns_total": len(all_turns), "wall_s": round(time.time() - t0)}
    (lc.LAB_DIR / "eval" / f"{args.tag}.summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--model-path", default=None)
    ap.add_argument("--effort", type=float, default=0.9)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--show-tests", action="store_true")
    asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    main()
