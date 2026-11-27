"""Self-distillation anchors: the stock model's own good answers to non-debug prompts.

Why: past step ~20 the RL keeps shortening plain Q&A answers (stock median 312 answer tokens
-> 228 at step 30) and the held-out grader drifts down; length penalties and wider batches did
not stop it. train_rl.py --anchors mixes cross-entropy on these rows into every optimizer step
so non-debug behaviour stays pinned to stock while RL changes debugging.

Rows with `system`/`messages` render their whole conversation (rollout.build_chat_prompt) and are
judged in context; scripted `turns` rows are still skipped (they need an on-policy prefix).
Per prompt: sample n rollouts from the BASE model with train_rl's renderer/rollout code, judge
them with the training judge, keep one (--select best|median) and store its exact token
sequence (prompt + sampled tokens, thinking included) so training reproduces what was sampled.

  PY=~/.inkling-abl/tinker_lab/.venv/bin/python
  $PY build_anchors.py --dry-run                     # plan only, no network
  $PY build_anchors.py --out anchors/stock_best.jsonl --max-prompts 250 --max-usd 1.0
  $PY build_anchors.py --selftest                    # offline checks
"""
from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import os
import random
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import lab_common as lc

DEFAULT_CATS = "qa,chat,creative,roleplay"
NEVER_CATS = frozenset({"debug"})


# ---------- selection (pure) ----------

@dataclasses.dataclass(frozen=True)
class Cand:
    """One scored rollout: what the selection rule needs, plus its index in the group."""
    idx: int
    answer_tokens: int
    judge_p: float | None
    clean: bool  # no loop flags, no runaway, no thinking-cap hit


def select(cands: list[Cand], rule: str) -> Cand | None:
    """`best`: highest judge score among clean rollouts at or above the clean median answer
    length (the floor stops "best" from picking short answers, the drift we fight).
    `median`: the clean rollout at the (upper) median answer length. None if nothing clean."""
    clean = [c for c in cands if c.clean and c.judge_p is not None]
    if not clean:
        return None
    lengths = sorted(c.answer_tokens for c in clean)
    if rule == "median":
        target = lengths[len(lengths) // 2]
        return min(clean, key=lambda c: (abs(c.answer_tokens - target), c.idx))
    floor = statistics.median(lengths)
    pool = [c for c in clean if c.answer_tokens >= floor]
    return max(pool, key=lambda c: (c.judge_p, -c.idx))


# ---------- anchor rows -> cross-entropy datums (used by train_rl.py) ----------

def anchor_weights(n_prompt: int, trainable: list[int], weight: float) -> list[float]:
    """Per-target-position weights: 0 on prompt tokens and forced tokens, `weight` on sampled."""
    full = [0.0] * n_prompt + [weight * m for m in trainable]
    return full[1:]  # target i predicts token i+1


def answer_only(tokens: list[int], n_prompt: int, trainable: list[int]) -> list[int]:
    """Trainable mask with the thinking segment switched off.

    Why: anchoring stock's thinking also pins when it stops thinking, which is the stall the RL is
    there to fix. Everything up to and including the first text-channel marker gets no loss, so only
    the answer is pinned. A completion with no marker (already in the text channel) is unchanged."""
    import rollout as ro

    comp = tokens[n_prompt:]
    cut = next((i for i, t in enumerate(comp) if t == ro.CONTENT_TEXT), -1)
    return [0 if i <= cut else m for i, m in enumerate(trainable)]


def anchor_datum(row: dict[str, Any], weight: float, part: str = "all") -> Any:
    import tinker
    import torch
    from tinker import TensorData

    tokens = row["tokens"]
    mask = answer_only(tokens, row["n_prompt"], row["trainable"]) if part == "answer" else row["trainable"]
    w = anchor_weights(row["n_prompt"], mask, weight)
    return tinker.Datum(
        model_input=tinker.ModelInput.from_ints(tokens[:-1]),
        loss_fn_inputs={
            "target_tokens": TensorData.from_torch(torch.tensor(tokens[1:], dtype=torch.int64)),
            "weights": TensorData.from_torch(torch.tensor(w, dtype=torch.float32)),
        },
    )


def validate_row(row: dict[str, Any]) -> str | None:
    """Reason a row is unusable, or None. Cheap structural checks only."""
    toks, n_p, tr = row.get("tokens"), row.get("n_prompt"), row.get("trainable")
    if not isinstance(toks, list) or not isinstance(n_p, int) or not isinstance(tr, list):
        return "missing fields"
    if n_p < 1 or len(toks) <= n_p or len(tr) != len(toks) - n_p:
        return "length mismatch"
    if not any(tr):
        return "no sampled tokens"
    if len(toks) > lc.CONTEXT_LIMIT:
        return "over context"
    return None


def load_anchors(path: Path, cats: frozenset[str]) -> list[dict[str, Any]]:
    rows = lc.read_jsonl(path)
    bad = [(r.get("id"), why) for r in rows if (why := validate_row(r))]
    if bad:
        raise SystemExit(f"{path}: {len(bad)} invalid anchor rows, first {bad[:3]}")
    keep = [r for r in rows if not cats or r["category"] in cats]
    if not keep:
        raise SystemExit(f"{path}: no anchor rows left after --anchor-cats {sorted(cats)}")
    return keep


def draw_anchors(rng: random.Random, pool: list[dict[str, Any]], n: int) -> list[dict[str, Any]]:
    """n rows without replacement within a step; n <= 0 touches neither rng nor pool."""
    if n <= 0:
        return []
    return rng.sample(pool, min(n, len(pool)))


# ---------- plan (offline) ----------

def plan_rows(rows: list[dict[str, Any]], cats: frozenset[str], efforts: tuple[float, ...],
              effort: float, off_frac: float, rng: random.Random,
              max_prompts: int) -> tuple[list[dict[str, Any]], Counter]:
    """Pick eligible single-turn rows and fix each one's effort/thinking like train_rl.collect:
    one effort per group from --efforts, thinking off for a share of groups."""
    skipped: Counter = Counter()
    eligible = []
    for r in rows:
        if r["category"] not in cats:
            continue
        why = "multi_turn" if "turns" in r else "has_tests" if r.get("tests") else \
            "has_gold" if r.get("gold") is not None else None
        if why:
            skipped[why] += 1
        else:
            eligible.append(r)
    rng.shuffle(eligible)
    eligible = eligible[:max_prompts] if max_prompts > 0 else eligible
    out = []
    for r in eligible:
        e = rng.choice(efforts) if efforts else effort
        thinking = not rng.random() < off_frac
        out.append({**r, "effort": e if thinking else 0.0, "_thinking": thinking})
    return out, skipped


# ---------- networked build ----------

def cands_for(samples: list[Any], rolls: list[Any], judged: list[tuple[float | None, str | None]]
              ) -> list[Cand]:
    import rewards as rw

    out = []
    for i, (s, r, (p, _e)) in enumerate(zip(samples, rolls, judged)):
        flags = rw.loop_flags(s)
        clean = not any(flags.values()) and not r.answer_runaway and not r.think_cap_hit
        out.append(Cand(i, s.answer_tokens, p, clean))
    return out


def anchor_prompt(renderer: Any, row: dict[str, Any], cfg: Any) -> tuple[list[int], str]:
    """(prompt tokens, judge request), the same render and judge context train_rl.collect uses.
    A system/messages row renders its whole conversation; its earlier assistant turns sit in the
    prompt, so anchor_weights never puts loss on them (only the sampled final reply is trainable)."""
    import rollout as ro

    if not ro.has_context(row):
        return ro.build_prompt(renderer, row["prompt"], cfg), row["prompt"]
    import rp

    chat = ro.row_chat(row)
    return ro.build_chat_prompt(renderer, chat, cfg), rp.chat_request(chat, row["category"])


def context_fields(row: dict[str, Any]) -> dict[str, Any]:
    return {k: row[k] for k in ("system", "messages") if row.get(k)}


async def build_one(sc: Any, renderer: Any, tok: Any, row: dict[str, Any], args: argparse.Namespace,
                    sem: asyncio.Semaphore, budget: dict[str, float]) -> tuple[dict | None, str]:
    """(anchor row, "") or (None, skip reason)."""
    import judge
    import rollout as ro

    cfg = ro.SampleCfg(think_cap=args.think_cap, answer_cap=args.answer_cap,
                       temperature=args.temperature, effort=row["effort"] or args.effort,
                       thinking=row["_thinking"])
    prompt, request = anchor_prompt(renderer, row, cfg)
    ro.assert_context(prompt, cfg)
    rolls = await ro.rollout_group(sc, prompt, args.n, cfg, sem)
    samples = [ro.to_sample(r, tok, row, cfg) for r in rolls]
    if judge.spent() - budget["start"] >= args.max_usd:
        return None, "budget"
    scoreable = [i for i, r in enumerate(rolls) if not r.answer_runaway and not r.think_cap_hit]
    res = await judge.score_many([(request, samples[i].answer) for i in scoreable])
    if any(err is not None for _p, err in res):
        return None, "budget" if any(judge.is_budget_error(e) for _p, e in res) else "judge_error"
    judged: list[tuple[float | None, str | None]] = [(None, None)] * len(rolls)
    for i, r in zip(scoreable, res):
        judged[i] = r
    pick = select(cands_for(samples, rolls, judged), args.select)
    if pick is None:
        return None, "no_clean"
    r, s = rolls[pick.idx], samples[pick.idx]
    return {"id": row["id"], "category": row["category"], "prompt": row["prompt"], **context_fields(row),
            "effort": row["effort"], "thinking": row["_thinking"], "select": args.select,
            "judge_p": pick.judge_p, "answer_tokens": s.answer_tokens, "think_tokens": s.think_tokens,
            "n_prompt": len(r.prompt_tokens), "tokens": r.prompt_tokens + r.completion,
            "trainable": r.trainable, "group_answer_tokens": [x.answer_tokens for x in samples],
            "answer_head": s.answer[:300]}, ""


async def build(args: argparse.Namespace, plan: list[dict[str, Any]]) -> tuple[list[dict], Counter, float]:
    import judge

    judge.configure(concurrency=args.judge_concurrency, rps=args.judge_rps)
    renderer, tok = lc.renderer_and_tokenizer()
    sc = await lc.sampling_client(lc.service_client())  # base model: assert_free inside
    sem = asyncio.Semaphore(args.concurrency)
    budget = {"start": judge.spent()}
    # Hard stop at call granularity: the judge refuses every call once this ceiling is hit.
    judge.MAX_USD = min(judge.MAX_USD, budget["start"] + args.max_usd)
    results = await asyncio.gather(*(build_one(sc, renderer, tok, r, args, sem, budget) for r in plan))
    rows = [r for r, _why in results if r is not None]
    skipped = Counter(why for r, why in results if r is None)
    return rows, skipped, judge.spent() - budget["start"]


def summary(rows: list[dict[str, Any]], skipped: Counter, usd: float, **extra: Any) -> str:
    by_cat: dict[str, list[int]] = {}
    for r in rows:
        by_cat.setdefault(r["category"], []).append(r["answer_tokens"])
    return json.dumps({"rows": len(rows), "skipped": dict(skipped), "usd": round(usd, 4),
                       "median_answer_tokens": {c: statistics.median(v) for c, v in sorted(by_cat.items())},
                       **extra})


def write_private(path: Path, rows: list[dict[str, Any]]) -> None:
    lc.write_jsonl(path, rows)
    os.chmod(path, 0o600)


# ---------- offline checks ----------

def _checks() -> None:
    row = {"id": "x", "category": "qa", "n_prompt": 4, "tokens": [10, 11, 12, 13, 20, 21, 22],
           "trainable": [1, 0, 1]}
    assert validate_row(row) is None
    w = anchor_weights(row["n_prompt"], row["trainable"], 1.0)
    assert len(w) == len(row["tokens"]) - 1 and w == [0.0, 0.0, 0.0, 1.0, 0.0, 1.0], w
    assert anchor_weights(4, [1, 1, 1], 0.5)[3:] == [0.5, 0.5, 0.5]
    d = anchor_datum(row, 1.0)
    tgt = d.loss_fn_inputs["target_tokens"].to_torch().tolist()
    wt = d.loss_fn_inputs["weights"].to_torch().tolist()
    assert d.model_input.to_ints() == row["tokens"][:-1] and tgt == row["tokens"][1:]
    assert len(wt) == len(tgt) == d.model_input.length and wt[:3] == [0.0] * 3 and sum(wt) == 2.0
    import rollout as ro
    toks = [1, 2, ro.CONTENT_THINKING, 30, 31, ro.CONTENT_TEXT, 40, 41]
    assert answer_only(toks, 2, [1] * 6) == [0, 0, 0, 0, 1, 1]       # thinking and the marker: no loss
    assert answer_only([1, 2, 40, 41], 2, [1, 1]) == [1, 1]           # no marker: unchanged
    wa = anchor_datum({"tokens": toks, "n_prompt": 2, "trainable": [1] * 6}, 1.0, "answer")
    assert wa.loss_fn_inputs["weights"].to_torch().tolist() == [0.0] * 5 + [1.0, 1.0]
    assert validate_row({**row, "trainable": [1, 1]}) == "length mismatch"
    assert validate_row({**row, "trainable": [0, 0, 0]}) == "no sampled tokens"
    g = [Cand(0, 100, 0.99, True), Cand(1, 300, 0.80, True), Cand(2, 320, 0.90, True),
         Cand(3, 500, 0.95, False), Cand(4, 280, 0.70, True)]
    assert select(g, "best").idx == 2       # 0.99 is below the median length (290)
    assert select(g, "median").idx == 1     # clean lengths 100,280,300,320 -> upper median 300
    assert select([Cand(0, 50, 0.9, False)], "best") is None
    assert select([Cand(0, 50, None, True)], "median") is None
    tie = [Cand(0, 300, 0.8, True), Cand(1, 300, 0.8, True)]
    assert select(tie, "best").idx == 0
    pool = [{"id": i} for i in range(40)]
    a = [draw_anchors(random.Random(5), pool, 16) for _ in range(2)]
    assert a[0] == a[1] and len({r["id"] for r in a[0]}) == 16
    assert draw_anchors(random.Random(6), pool, 16) != a[0]
    r0 = random.Random(9)
    before = r0.getstate()
    assert draw_anchors(r0, pool, 0) == [] and r0.getstate() == before  # n=0 consumes nothing
    rows = [{"id": f"q{i}", "category": c, "prompt": "p", "gold": None}
            for i, c in enumerate(["qa", "debug", "creative", "maths", "qa"])]
    rows.append({"id": "rp", "category": "roleplay", "turns": ["hi"], "gold": None})
    plan, skipped = plan_rows(rows, frozenset({"qa", "creative", "roleplay"}), (0.5, 0.9), 0.9,
                              0.25, random.Random(0), 0)
    assert {r["id"] for r in plan} == {"q0", "q2", "q4"} and skipped == Counter(multi_turn=1)
    plan2, _ = plan_rows(rows, frozenset({"qa", "creative", "roleplay"}), (0.5, 0.9), 0.9,
                         0.25, random.Random(0), 0)
    assert plan == plan2  # seeded plan
    assert all((r["effort"] == 0.0) == (not r["_thinking"]) for r in plan)
    print("build_anchors.py checks passed")


def parse() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--prompts", default="prompts_train_code.jsonl", help="lab-relative or absolute")
    ap.add_argument("--cats", default=DEFAULT_CATS, help="categories to anchor (debug never)")
    ap.add_argument("--efforts", default="0.5,0.6,0.7,0.8,0.9,0.9",
                    help="comma list; each prompt draws one (train_rl --efforts convention)")
    ap.add_argument("--effort", type=float, default=0.9, help="used when --efforts is empty")
    ap.add_argument("--off-frac", type=float, default=0.25, help="share of prompts sampled thinking-off")
    ap.add_argument("--n", type=int, default=8, help="rollouts per prompt")
    ap.add_argument("--select", choices=["best", "median"], default="best")
    ap.add_argument("--temperature", type=float, default=1.0, help="train_rl's default sampling temp")
    ap.add_argument("--think-cap", type=int, default=4096)
    ap.add_argument("--answer-cap", type=int, default=8192)
    ap.add_argument("--max-prompts", type=int, default=0, help="0 = all eligible")
    ap.add_argument("--max-usd", type=float, default=1.0, help="judge spend cap for this build (hard stop)")
    ap.add_argument("--out", default="anchors/stock_best.jsonl", help="lab-relative or absolute")
    ap.add_argument("--concurrency", type=int, default=64, help="sampling requests in flight")
    ap.add_argument("--judge-concurrency", type=int, default=8)
    ap.add_argument("--judge-rps", type=float, default=6.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true", help="plan and print counts; no network")
    ap.add_argument("--selftest", action="store_true", help="offline checks, then exit")
    return ap.parse_args()


def main() -> None:
    args = parse()
    if args.selftest:
        _checks()
        return
    cats = frozenset(c for c in args.cats.split(",") if c)
    if cats & NEVER_CATS:
        raise SystemExit(f"refusing: {sorted(cats & NEVER_CATS)} must not be anchored")
    efforts = tuple(float(x) for x in args.efforts.split(",")) if args.efforts else ()
    rows = lc.read_jsonl(lc.LAB_DIR / args.prompts)
    plan, skipped = plan_rows(rows, cats, efforts, args.effort, args.off_frac,
                              random.Random(args.seed), args.max_prompts)
    if args.dry_run:
        rollouts = len(plan) * args.n
        print(json.dumps({"dry_run": True, "prompts": len(plan), "rollouts": rollouts,
                          "by_cat": dict(Counter(r["category"] for r in plan)),
                          "thinking_off": sum(not r["_thinking"] for r in plan),
                          "by_effort": dict(Counter(r["effort"] for r in plan)),
                          "skipped": dict(skipped), "judge_calls_max": rollouts,
                          "max_sampled_tokens": rollouts * (args.think_cap + args.answer_cap)}))
        return
    out_rows, skipped2, usd = asyncio.run(build(args, plan))
    out = lc.LAB_DIR / args.out
    write_private(out, out_rows)
    print(summary(out_rows, skipped + skipped2, usd, out=str(out), prompts=len(plan)))


if __name__ == "__main__":
    sys.exit(main())
