"""The seven suites, each split into generate / judge / summarise so judging can be redone offline
(run_suite.py --judge-only) from the saved JSONL without touching the GPU endpoint.

Ports: main + hardmath = scripts/tinker/eval.py; code = code_eval.py; agent_hard(_noforce) =
agent_eval_hard.py; rp = rp.py eval; sysrules = sysrules_eval.py run.
"""
from __future__ import annotations

import json
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
import zlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import judges
import metrics as M
import sysrules as SR
from client import Client, Gen

Rec = dict[str, Any]
JUDGE_PREFIX = {"bison": "judge", "haiku": "alt"}  # lab field names: judge_p (Bison), alt_p (Haiku)


@dataclass
class Ctx:
    client: Client
    data_dir: Path
    agent_dir: Path
    judges: tuple[str, ...]
    limit: int
    seed: int | None
    repeat: int
    workers: int = 64
    out_dir: Path | None = None        # this suite's output folder (ubench resume files live here)
    ubench_dir: Path = Path("~/Code/uncensoredbench").expanduser()
    ubench_conditions: tuple[str, ...] = ("command",)
    rp_chats: int = 30                 # benchmark roleplay sample size (runs/v2.json)
    glm_sample: int = 150              # judge3's glm_per_model


def read_jsonl(path: Path) -> list[Rec]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def pmap(fn: Callable[[Any], Any], items: list[Any], workers: int) -> list[Any]:
    """Thread fan-out; the client's semaphore bounds concurrent HTTP requests."""
    if not items:
        return []
    with ThreadPoolExecutor(min(len(items), workers)) as ex:
        return list(ex.map(fn, items))


def seed_for(ctx: Ctx, key: str) -> int | None:
    if ctx.seed is None:
        return None
    return (ctx.seed + 1_000_003 * ctx.repeat + zlib.crc32(key.encode())) % 2**31


def limited(ctx: Ctx, rows: list[Any]) -> list[Any]:
    return rows[: ctx.limit] if ctx.limit else rows


def gen_fields(g: Gen) -> Rec:
    flags = M.loop_flags(g.thinking, g.answer, g.think_cap_hit)
    return {"think_tokens": g.think_tokens, "answer_tokens": g.answer_tokens,
            "think_cap_hit": g.think_cap_hit, "answer_runaway": g.answer_runaway,
            "loop_any": any(flags[k] for k in M.LAB_LOOP_KEYS), "loop_think": flags["loop_think"],
            "loop_answer": flags["loop_answer"], "loop_flags": flags, "forced": g.forced,
            "finish": g.finish, "split": g.split, "count_source": g.count_source,
            "seconds": g.seconds, "error": g.error}


def ok_rows(recs: list[Rec]) -> list[Rec]:
    return [r for r in recs if not r.get("error")]


def apply_judges(ctx: Ctx, items: list[Rec], request_of: Callable[[Rec], str]) -> dict[str, int]:
    """Writes <prefix>_p / <prefix>_err into each item; returns error counts per backend."""
    errors: dict[str, int] = {}
    for backend in ctx.judges:
        if backend not in JUDGE_PREFIX:  # glm is a ubench-only judge
            continue
        prefix = JUDGE_PREFIX[backend]
        res = judges.score_many(backend, [(request_of(r), r["answer"]) for r in items])
        for r, (p, err) in zip(items, res):
            r[f"{prefix}_p"], r[f"{prefix}_err"] = p, err
        errors[backend] = sum(err is not None for _, err in res)
    return errors

# ---------- main / hardmath (eval.py) ----------


def gen_single(ctx: Ctx, prompts_file: str) -> list[Rec]:
    rows = limited(ctx, read_jsonl(ctx.data_dir / prompts_file))

    def one(row: Rec) -> Rec:
        g = ctx.client.generate([{"role": "user", "content": row["prompt"]}], seed_for(ctx, row["id"]))
        return {"id": row["id"], "category": row["category"], "target_words": row.get("target_words"),
                "gold": row.get("gold"), "prompt": row["prompt"],
                "answer_words": len(M.WORD_RE.findall(g.answer)),
                "maths_correct": M.maths_quality(g.answer, row["gold"]) if row.get("gold") else None,
                **gen_fields(g), "thinking": g.thinking, "answer": g.answer}
    return pmap(one, rows, ctx.workers)


def judge_single(ctx: Ctx, recs: list[Rec]) -> dict[str, int]:
    return apply_judges(ctx, [r for r in ok_rows(recs) if r["gold"] is None], lambda r: r["prompt"])


def sum_single(recs: list[Rec]) -> Rec:
    ok = ok_rows(recs)
    gen_tokens = sum(r["think_tokens"] + r["answer_tokens"] for r in ok)
    return {"n_ok": len(ok), "n_failed": len(recs) - len(ok), "gen_tokens": gen_tokens,
            "all": M.category_metrics(ok) if ok else {},
            "by_category": {c: M.category_metrics([r for r in ok if r["category"] == c])
                            for c in sorted({r["category"] for r in ok})}}

# ---------- code (code_eval.py) ----------


def gen_code(ctx: Ctx) -> list[Rec]:
    rows = limited(ctx, read_jsonl(ctx.data_dir / "prompts_code.jsonl"))

    def one(row: Rec) -> Rec:
        g = ctx.client.generate([{"role": "user", "content": row["prompt"]}], seed_for(ctx, row["id"]))
        ok, err = M.run_tests(g.answer, row["tests"])
        return {"id": row["id"], "passed": ok, "err": err, **gen_fields(g),
                "thinking": g.thinking, "answer": g.answer}
    return pmap(one, rows, ctx.workers)


def sum_code(recs: list[Rec]) -> Rec:
    ok = ok_rows(recs)
    if not ok:
        return {"n": 0, "n_failed": len(recs)}
    return {"n": len(ok), "n_failed": len(recs) - len(ok), "pass_at_1": M.rate(ok, "passed"),
            "median_think_tokens": M.median([r["think_tokens"] for r in ok]),
            "median_answer_tokens": M.median([r["answer_tokens"] for r in ok]),
            "cap_hit": M.rate(ok, "think_cap_hit"), "runaway": M.rate(ok, "answer_runaway"),
            "loop_any": M.rate(ok, "loop_any"), "loop_think": M.rate(ok, "loop_think"),
            "loop_answer": M.rate(ok, "loop_answer")}

# ---------- agent_hard (agent_eval_hard.py) ----------


MAX_TURNS = 12
CONTEXT_BUDGET = 48000
LINE_RE = re.compile(r'tests\.py", line (\d+)')


def agent_tasks(agent_dir: Path) -> list[Rec]:
    out = []
    for d in sorted(agent_dir.glob("t[0-9][0-9]_*")):
        meta = json.loads((d / "meta.json").read_text())
        out.append({"dir": d, "name": d.name, "description": meta["description"],
                    "buggy": (d / "buggy.py").read_text()})
    return out


def run_hidden(code: str, task_dir: Path) -> tuple[bool, str, int | None]:
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


def first_prompt(t: Rec) -> str:
    return (f"This module ({t['description']}) has bugs. I have a hidden unit-test suite and will "
            f"show you its output after each attempt.\n\n```python\n{t['buggy']}```\n\n"
            "Fix the bugs. Reply with the COMPLETE corrected module in a single ```python code block.")


def feedback(out: str, current: str) -> str:
    return (f"I ran the hidden tests on your version. Output:\n\n```\n{out.strip()[-3000:]}\n```\n\n"
            f"Current file:\n\n```python\n{current}```\n\n"
            "Fix the remaining problems. Reply with the COMPLETE corrected module in a single ```python code block.")


def norm(code: str) -> str:
    return "\n".join(line.rstrip() for line in code.strip().splitlines() if line.strip())


def agent_turn(ctx: Ctx, t: Rec, msgs: list[Rec], turn: int, force: bool) -> tuple[Gen, int, str] | None:
    n_prompt, src = ctx.client.count_prompt_tokens(msgs)
    cfg = ctx.client.cfg
    if n_prompt + cfg.think_cap + cfg.answer_cap > CONTEXT_BUDGET:
        return None
    return ctx.client.generate(msgs, seed_for(ctx, f"{t['name']}/{turn}"), force=force), n_prompt, src


def solve(ctx: Ctx, t: Rec, force: bool) -> Rec:
    msgs: list[Rec] = [{"role": "user", "content": first_prompt(t)}]
    current, seen, turns = t["buggy"], [norm(t["buggy"])], []
    best_line, solved_at, stop = 0, None, "max_turns"
    for turn in range(MAX_TURNS):
        step = agent_turn(ctx, t, msgs, turn, force)
        if step is None:
            stop = "context"
            break
        g, n_prompt, src = step
        if g.error:
            stop = "error"
            break
        blocks = M.CODE_RE.findall(g.answer)
        code = blocks[-1] if blocks else current
        ok, out, line = run_hidden(code, t["dir"])
        regress = line is not None and best_line and line < best_line
        turns.append({"turn": turn, "passed": ok, "repeat_edit": norm(code) in seen, "no_code": not blocks,
                      "regression": bool(regress), "fail_line": line, "cap_hit": g.think_cap_hit,
                      "runaway": g.answer_runaway, "think_tokens": g.think_tokens,
                      "answer_tokens": g.answer_tokens, "prompt_tokens": n_prompt, "prompt_tokens_src": src,
                      "forced": g.forced, "loop_think": M.text_loops(g.thinking),
                      "loop_answer": M.text_loops(g.answer)})
        seen.append(norm(code))
        best_line = max(best_line, line or 0)
        current = code
        if ok:
            solved_at, stop = turn + 1, "solved"
            break
        msgs += [{"role": "assistant", "content": g.answer[:20000]},
                 {"role": "user", "content": feedback(out, current)}]
    return {"name": t["name"], "solved_at": solved_at, "stop": stop, "turns": turns,
            "total_tokens": sum(x["think_tokens"] + x["answer_tokens"] for x in turns)}


def gen_agent(ctx: Ctx, force: bool) -> list[Rec]:
    return pmap(lambda t: solve(ctx, t, force), limited(ctx, agent_tasks(ctx.agent_dir)), ctx.workers)


def sum_agent(res: list[Rec]) -> Rec:
    allt = [x for r in res for x in r["turns"]]
    if not allt:
        return {"tasks": len(res), "turns_total": 0}
    later = [x for x in allt if x["turn"] > 0]
    solved = [r for r in res if r["solved_at"]]
    by_turn = {}
    for k in range(MAX_TURNS):
        xs = [x for x in allt if x["turn"] == k]
        if xs:
            by_turn[k] = {"n": len(xs), "median_think": statistics.median(x["think_tokens"] for x in xs),
                          "cap_hit": sum(x["cap_hit"] for x in xs) / len(xs)}
    late = [x["think_tokens"] for x in allt if x["turn"] >= 3]
    return {"tasks": len(res), "solve_rate": len(solved) / len(res),
            "mean_turns_to_solve": statistics.mean(r["solved_at"] for r in solved) if solved else None,
            "repeat_edit_rate": sum(x["repeat_edit"] for x in later) / max(1, len(later)),
            "regression_rate": sum(x["regression"] for x in later) / max(1, len(later)),
            "no_code_rate": sum(x["no_code"] for x in allt) / len(allt),
            "cap_hit_per_turn": sum(x["cap_hit"] for x in allt) / len(allt),
            "runaway_per_turn": sum(x["runaway"] for x in allt) / len(allt),
            "loop_think_per_turn": sum(x["loop_think"] for x in allt) / len(allt),
            "loop_answer_per_turn": sum(x["loop_answer"] for x in allt) / len(allt),
            "median_think_turn1": by_turn.get(0, {}).get("median_think"),
            "median_think_turn4plus": statistics.median(late) if late else None,
            "mean_total_tokens": statistics.mean(r["total_tokens"] for r in res),
            "stops": {k: sum(r["stop"] == k for r in res) for k in ("solved", "max_turns", "context", "error")},
            "by_turn": by_turn, "turns_total": len(allt),
            "prompt_tokens_estimated": any(x["prompt_tokens_src"] == "estimate" for x in allt)}

# ---------- rp (rp.py eval) ----------


def chat_messages(users: list[str], replies: list[str]) -> list[Rec]:
    msgs: list[Rec] = []
    for i, u in enumerate(users):
        msgs.append({"role": "user", "content": u})
        if i < len(replies):
            msgs.append({"role": "assistant", "content": replies[i]})
    return msgs


def transcript(users: list[str], replies: list[str]) -> str:
    lines = []
    for i, u in enumerate(users):
        lines.append(f"USER: {u}")
        if i < len(replies):
            lines.append(f"AI: {replies[i][:1500]}")
    return "Continue this roleplay conversation in character with the next AI reply.\n\n" + "\n\n".join(lines)


def rp_chat(ctx: Ctx, row: Rec) -> list[Rec]:
    replies: list[str] = []
    out = []
    for t in range(len(row["turns"])):
        users = row["turns"][: t + 1]
        g = ctx.client.generate(chat_messages(users, replies), seed_for(ctx, f"{row['id']}/{t}"))
        flags = M.loop_flags(g.thinking, g.answer, g.think_cap_hit)
        out.append({"id": row["id"], "turn": t, **gen_fields(g),
                    "in_reply_loop": flags["span32x3"] or flags["gram30x20"],
                    "cross_turn_repeat": M.cross_turn_repeat(g.answer, replies),
                    "request": transcript(users, replies), "answer": g.answer, "thinking": g.thinking})
        if g.error:
            break
        replies.append(g.answer.strip()[:6000])
    return out


def gen_rp(ctx: Ctx) -> list[Rec]:
    chats = limited(ctx, read_jsonl(ctx.data_dir / "rp_eval.jsonl"))
    return [t for c in pmap(lambda r: rp_chat(ctx, r), chats, ctx.workers) for t in c]


def judge_rp(ctx: Ctx, recs: list[Rec]) -> dict[str, int]:
    return apply_judges(ctx, ok_rows(recs), lambda r: r["request"])


def sum_rp(recs: list[Rec]) -> Rec:
    turns = ok_rows(recs)
    n = len(turns)
    if not n:
        return {"turns": 0, "n_failed": len(recs)}
    bad = [t["think_cap_hit"] or t["answer_runaway"] or t["in_reply_loop"] or t["cross_turn_repeat"] for t in turns]

    def good(k: str) -> float | None:  # rp.py divides by all turns
        vals = [t[k] for t in turns if t.get(k) is not None]
        return sum(p >= 0.5 for p in vals) / n if vals else None
    return {"turns": n, "n_failed": len(recs) - n, "per_turn_loop_rate": sum(bad) / n,
            "cap_hit": M.rate(turns, "think_cap_hit"), "runaway": M.rate(turns, "answer_runaway"),
            "in_reply_loop": M.rate(turns, "in_reply_loop"), "loop_think": M.rate(turns, "loop_think"),
            "loop_answer": M.rate(turns, "loop_answer"), "cross_turn_repeat": M.rate(turns, "cross_turn_repeat"),
            "median_think": statistics.median(t["think_tokens"] for t in turns),
            "median_answer": statistics.median(t["answer_tokens"] for t in turns),
            "bison_good": good("judge_p"), "haiku_good": good("alt_p"),
            "loop_by_turn": {i: sum(b for t, b in zip(turns, bad) if t["turn"] == i)
                             / max(1, sum(t["turn"] == i for t in turns)) for i in range(6)}}

# ---------- sysrules (sysrules_eval.py) ----------


def sysrules_tasks(ctx: Ctx) -> list[Rec]:
    return limited(ctx, read_jsonl(ctx.data_dir / "sysrules_tasks.jsonl"))


def gen_sysrules(ctx: Ctx) -> list[Rec]:
    def one(t: Rec) -> Rec:
        msgs = [{"role": "system", "content": t["system"]}, {"role": "user", "content": t["user"]}]
        g = ctx.client.generate(msgs, seed_for(ctx, t["id"]))
        return {"id": t["id"], **SR.score_task(t, g.answer, use_judge=False), **gen_fields(g),
                "thinking": g.thinking, "answer": g.answer}
    return pmap(one, sysrules_tasks(ctx), ctx.workers)


def judge_sysrules(ctx: Ctx, recs: list[Rec]) -> dict[str, int]:
    if "haiku" not in ctx.judges:
        return {}
    tasks = {t["id"]: t for t in sysrules_tasks(ctx)}
    todo = [r for r in ok_rows(recs) if r["id"] in tasks]
    scored = pmap(lambda r: SR.score_task(tasks[r["id"]], r["answer"], use_judge=True), todo, 16)
    for r, s in zip(todo, scored):
        r.update(s)
    return {"haiku": sum(s["judge_errors"] for s in scored)}


def sum_sysrules(recs: list[Rec]) -> Rec:
    ok = ok_rows(recs)
    if not ok:
        return {"tasks": 0, "n_failed": len(recs)}
    tempted = [x for r in ok for x in r["tempted_kept"] if x is not None]
    by_rule: dict[str, list[bool]] = {}
    for r in ok:
        for k, v in r["rules"].items():
            if v is not None:
                by_rule.setdefault(k, []).append(v)
    return {"tasks": len(ok), "n_failed": len(recs) - len(ok),
            "rule_compliance": sum(r["kept"] for r in ok) / max(1, sum(r["n"] for r in ok)),
            "all_rules_kept": M.rate(ok, "all_kept"),
            "tempted_rule_compliance": sum(tempted) / max(1, len(tempted)),
            "judged_rules_scored": any(r["rules"].get(k) is not None for r in ok
                                       for k in ("question_end", "no_dates", "tone", "bullets")),
            "median_think": statistics.median(r["think_tokens"] for r in ok),
            "cap_hit": M.rate(ok, "think_cap_hit"), "runaway": M.rate(ok, "answer_runaway"),
            "loop_any": M.rate(ok, "loop_any"),
            "by_rule": {k: round(sum(v) / len(v), 3) for k, v in sorted(by_rule.items())}}

# ---------- registry ----------


def _no_judge(ctx: Ctx, recs: list[Rec]) -> dict[str, int]:
    return {}


SUITES: dict[str, tuple[Callable[[Ctx], list[Rec]], Callable[[Ctx, list[Rec]], dict[str, int]],
                        Callable[[list[Rec]], Rec]]] = {
    "main": (lambda c: gen_single(c, "prompts_eval.jsonl"), judge_single, sum_single),
    "hardmath": (lambda c: gen_single(c, "prompts_hardmath.jsonl"), judge_single, sum_single),
    "code": (gen_code, _no_judge, sum_code),
    "agent_hard": (lambda c: gen_agent(c, True), _no_judge, sum_agent),
    "agent_hard_noforce": (lambda c: gen_agent(c, False), _no_judge, sum_agent),
    "rp": (gen_rp, judge_rp, sum_rp),
    "sysrules": (gen_sysrules, judge_sysrules, sum_sysrules),
}


def run_one(ctx: Ctx, suite: str) -> tuple[list[Rec], Rec]:
    gen, judge, summarise = SUITES[suite]
    t0 = time.time()
    recs = gen(ctx)
    wall = time.time() - t0
    judge_errors = judge(ctx, recs) if ctx.judges else {}
    return recs, {**summarise(recs), "wall_seconds": round(wall, 1), "judges": list(ctx.judges),
                  "judge_errors": judge_errors}
