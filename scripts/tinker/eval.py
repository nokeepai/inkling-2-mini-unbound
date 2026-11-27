"""Rule-based eval of the free model (or a Tinker sampler checkpoint) on prompts_eval.jsonl.

Metrics per category: thinking-cap-hit rate, answer-runaway rate, loop rate, median
think/answer tokens, creative length within +-30%, maths accuracy, top character names.

  PY=~/.inkling-abl/tinker_lab/.venv/bin/python
  $PY scripts/tinker/eval.py --tag base_t06_think                       # thinking on, t0.6
  $PY scripts/tinker/eval.py --tag base_greedy_think --temperature 0
  $PY scripts/tinker/eval.py --tag base_t06_off --thinking off
  $PY scripts/tinker/eval.py --tag ck --model-path tinker://...           # a trained sampler
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import statistics
import time
from collections import Counter
from dataclasses import asdict
from typing import Any

import lab_common as lc
import rewards as rw
import rollout as ro

NAME_RE = re.compile(r"(?<=[a-z,;] )([A-Z][a-z]{2,})\b")
NOT_NAMES = {"The", "And", "But", "She", "His", "Her", "They", "This", "That", "When", "Then",
             "Its", "Our", "You", "Your", "There", "What", "Mars", "Earth", "Moon", "Sun",
             "Monday", "Sunday", "God", "Mr", "Mrs", "Dear", "I'm", "English", "French"}


def creative_names(answers: list[str], top: int = 10) -> list[tuple[str, int]]:
    """Capitalised words mid-sentence, counted once per story: a cheap proxy for
    'every story stars Elara' style name collapse."""
    c: Counter[str] = Counter()
    for a in answers:
        c.update({m for m in NAME_RE.findall(a) if m not in NOT_NAMES})
    return c.most_common(top)


def median(xs: list[float]) -> float:
    return float(statistics.median(xs)) if xs else 0.0


def category_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    m: dict[str, Any] = {
        "n": n,
        "think_cap_hit": sum(r["think_cap_hit"] for r in rows) / n,
        "answer_runaway": sum(r["answer_runaway"] for r in rows) / n,
        "loop_any": sum(r["loop_any"] for r in rows) / n,
        "median_think_tokens": median([r["think_tokens"] for r in rows]),
        "median_answer_tokens": median([r["answer_tokens"] for r in rows]),
    }
    cr = [r for r in rows if r["target_words"]]
    if cr:
        m["length_within_30pct"] = sum(
            abs(r["answer_words"] - r["target_words"]) <= 0.3 * r["target_words"]
            and not r["answer_runaway"] for r in cr) / len(cr)
        m["median_words_over_target"] = median([r["answer_words"] / r["target_words"] for r in cr])
        m["top_names"] = creative_names([r["answer"] for r in cr])
    mx = [r for r in rows if r["gold"] is not None]
    if mx:
        m["maths_acc"] = sum(r["maths_correct"] for r in mx) / len(mx)
    for prefix in ("judge", "jev", "alt"):
        jq = [r[f"{prefix}_p"] for r in rows if r.get(f"{prefix}_p") is not None]
        if jq:
            m[f"{prefix}_mean_p"] = sum(jq) / len(jq)
            m[f"{prefix}_good_rate"] = sum(p >= 0.5 for p in jq) / len(jq)
            m[f"{prefix}_n"] = len(jq)
    return m


async def add_judge(recs: list[dict[str, Any]], prompts: dict[str, str],
                    backend: str = "bison") -> int:
    """Judge non-maths rows in place (Bison -> judge_p, Jev -> jev_p); returns error count."""
    import judge

    prefix = "judge" if backend == "bison" else backend
    todo = [r for r in recs if r["gold"] is None]
    res = await judge.score_many([(prompts[r["id"]], r["answer"]) for r in todo], backend=backend)
    for r, (p, err) in zip(todo, res):
        r[f"{prefix}_p"], r[f"{prefix}_err"] = p, err
    return sum(err is not None for _, err in res)


def write_summary(tag: str, recs: list[dict[str, Any]], extra: dict[str, Any]) -> dict[str, Any]:
    out_dir = lc.LAB_DIR / "eval"
    lc.write_jsonl(out_dir / f"{tag}.jsonl", recs)
    summary = {**extra, "all": category_metrics(recs),
               "by_category": {c: category_metrics([r for r in recs if r["category"] == c])
                               for c in sorted({r["category"] for r in recs})}}
    (out_dir / f"{tag}.summary.json").write_text(json.dumps(summary, indent=1))
    return summary


async def rejudge(tag: str, prompts_file: str, backend: str = "bison") -> None:
    """Add judge scores to an existing eval run without resampling."""
    import judge

    out_dir = lc.LAB_DIR / "eval"
    recs = lc.read_jsonl(out_dir / f"{tag}.jsonl")
    prompts = {r["id"]: r["prompt"] for r in lc.read_jsonl(lc.LAB_DIR / prompts_file)}
    errors = await add_judge(recs, prompts, backend)
    old = json.loads((out_dir / f"{tag}.summary.json").read_text())
    extra = {k: v for k, v in old.items() if k not in ("all", "by_category")}
    extra.update({f"{backend}_errors": errors, "judge_spent_usd": round(judge.spent(), 5)})
    summary = write_summary(tag, recs, extra)
    print(json.dumps({c: {k: v for k, v in m.items() if k.startswith(("judge", "jev", "alt"))}
                      for c, m in summary["by_category"].items()}, indent=1))


def row_record(row: dict[str, Any], s: rw.Sample, r: ro.Rollout) -> dict[str, Any]:
    flags = rw.loop_flags(s)
    return {
        "id": row["id"], "category": row["category"], "target_words": row.get("target_words"),
        "gold": row.get("gold"), "think_tokens": s.think_tokens, "answer_tokens": s.answer_tokens,
        "answer_words": len(rw.WORD_RE.findall(s.answer)),
        "think_cap_hit": s.think_cap_hit, "answer_runaway": s.answer_runaway,
        "loop_any": any(flags.values()), "loop_flags": flags,
        "maths_correct": rw.maths_quality(s.answer, row["gold"]) if row.get("gold") else None,
        "seconds": round(r.seconds, 1), "thinking": s.thinking, "answer": s.answer,
    }


async def run(args: argparse.Namespace) -> None:
    cfg = ro.SampleCfg(think_cap=args.think_cap, answer_cap=args.answer_cap,
                       temperature=args.temperature, effort=args.effort,
                       thinking=args.thinking == "on", force_answer_after_cap=True)
    rows = lc.read_jsonl(lc.LAB_DIR / args.prompts)[: args.limit or None]
    renderer, tok = lc.renderer_and_tokenizer()
    if args.paid:
        lc.enable_paid(args.tag)
    svc = lc.service_client()
    if args.model_path:
        # A sampler checkpoint from train_rl.py; its base model is checked server-side below.
        sc = await svc.create_sampling_client(model_path=args.model_path)
        lc.assert_free(await sc.get_base_model_async())
    else:
        sc = await lc.sampling_client(svc)
    sem = asyncio.Semaphore(args.concurrency)
    log = lc.Log(lc.LAB_DIR / "logs" / f"eval_{args.tag}.log")
    log("start", cfg=asdict(cfg), n=len(rows), model_path=args.model_path)
    t0 = time.time()

    async def one(row: dict[str, Any]) -> dict[str, Any] | None:
        prompt = ro.build_prompt(renderer, row["prompt"], cfg)
        ro.assert_context(prompt, cfg)
        try:
            r = (await ro.rollout_group(sc, prompt, 1, cfg, sem))[0]
        except Exception as e:
            log("error", id=row["id"], err=lc.scrub(f"{type(e).__name__}: {e}")[:2000])
            return None
        rec = row_record(row, ro.to_sample(r, tok, row, cfg), r)
        log("done", id=row["id"], think=rec["think_tokens"], ans=rec["answer_tokens"],
            cap=rec["think_cap_hit"], run=rec["answer_runaway"], s=rec["seconds"])
        return rec

    recs = [x for x in await asyncio.gather(*(one(r) for r in rows)) if x is not None]
    wall = time.time() - t0
    gen_tokens = sum(r["think_tokens"] + r["answer_tokens"] for r in recs)
    extra: dict[str, Any] = {"tag": args.tag, "cfg": asdict(cfg), "model_path": args.model_path,
                             "n_ok": len(recs), "n_failed": len(rows) - len(recs),
                             "wall_seconds": round(wall, 1), "gen_tokens": gen_tokens,
                             "gen_tok_per_s": round(gen_tokens / wall, 1)}
    if args.judge:
        extra["judge_errors"] = await add_judge(recs, {r["id"]: r["prompt"] for r in rows})
    summary = write_summary(args.tag, recs, extra)
    log("summary", **{k: v for k, v in summary.items() if k != "by_category"})
    print(json.dumps(summary["by_category"], indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--prompts", default="prompts_eval.jsonl")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--thinking", choices=["on", "off"], default="on")
    ap.add_argument("--effort", type=float, default=0.6)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--think-cap", type=int, default=4096)
    ap.add_argument("--answer-cap", type=int, default=8192)
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--model-path", default=None)
    ap.add_argument("--judge", action="store_true", help="Bison quality for non-maths rows")
    ap.add_argument("--paid", action="store_true",
                    help="use the paid 64K model; needs LAB_PAID_MAX_USD > 0 (ledger-capped)")
    ap.add_argument("--rejudge", action="store_true", help="judge an existing --tag, no sampling")
    ap.add_argument("--rejudge-jev", action="store_true", help="Jev (eval-only) on an existing --tag")
    ap.add_argument("--rejudge-alt", action="store_true",
                    help="Haiku via OpenRouter (eval-only, $2 cap) on an existing --tag")
    args = ap.parse_args()
    if args.rejudge or args.rejudge_jev or args.rejudge_alt:
        backend = "jev" if args.rejudge_jev else "alt" if args.rejudge_alt else "bison"
        asyncio.run(rejudge(args.tag, args.prompts, backend))
    else:
        asyncio.run(run(args))


if __name__ == "__main__":
    main()
