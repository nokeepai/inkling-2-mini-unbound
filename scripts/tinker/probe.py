"""Throughput probe: sample 16 prompts (thinking on, full caps) and run one tiny train step.

  ~/.inkling-abl/tinker_lab/.venv/bin/python scripts/tinker/probe.py
"""
from __future__ import annotations

import asyncio
import random
import time

import tinker

import lab_common as lc
import rewards as rw
import rollout as ro
import train_rl as tr


async def sample_probe(log: lc.Log, renderer, tok, svc) -> None:
    rows = random.Random(1).sample(lc.read_jsonl(lc.LAB_DIR / "prompts_train.jsonl"), 16)
    cfg = ro.SampleCfg(temperature=0.6, effort=0.9, force_answer_after_cap=True)
    sc = await lc.sampling_client(svc)
    sem = asyncio.Semaphore(64)
    t0 = time.time()

    async def one(row):
        try:
            r = (await ro.rollout_group(sc, ro.build_prompt(renderer, row["prompt"], cfg), 1, cfg, sem))[0]
            return row, r
        except Exception as e:
            log("sample_error", id=row["id"], err=lc.scrub(f"{type(e).__name__}: {e}")[:2000])
            return row, None

    res = await asyncio.gather(*(one(r) for r in rows))
    wall = time.time() - t0
    toks, per = 0, []
    for row, r in res:
        if r is None:
            continue
        s = ro.to_sample(r, tok, row, cfg)
        toks += s.think_tokens + s.answer_tokens
        per.append((row["category"], s.think_tokens, s.answer_tokens, s.think_cap_hit,
                    s.answer_runaway, round(r.seconds, 1)))
    log("sample_probe", wall_s=round(wall, 1), gen_tokens=toks, tok_per_s=round(toks / wall, 1),
        n_ok=len(per), per_sample=per)


async def train_probe(log: lc.Log, renderer, tok, svc) -> None:
    rows = random.Random(2).sample(
        [r for r in lc.read_jsonl(lc.LAB_DIR / "prompts_train.jsonl") if r["category"] == "maths"], 2)
    scfg = ro.SampleCfg(think_cap=1024, answer_cap=1024, temperature=1.0, effort=0.9)
    rcfg = rw.RewardConfig(think_cap=1024, answer_cap=1024)
    t0 = time.time()
    tc = await lc.training_client(svc, 32, train_attn=True, train_mlp=True, train_unembed=False)
    t_create = time.time() - t0
    t1 = time.time()
    sc = await tc.save_weights_and_get_sampling_client()
    t_snap = time.time() - t1
    t2 = time.time()
    groups = await tr.collect(sc, renderer, tok, rows, scfg, rcfg, 4, asyncio.Semaphore(32))
    t_sample = time.time() - t2
    data, kept = tr.make_data(groups, std_norm=False)
    if not data:  # constant groups would skip training; force a datum so the step is timed
        data = [tr.build_datum(r, 0.1) for g in groups for r in g[1] if r.completion]
    t3 = time.time()
    fb = tc.forward_backward(data, loss_fn="importance_sampling")
    opt = tc.optim_step(tinker.AdamParams(learning_rate=1e-5, beta1=0.9, beta2=0.95, eps=1e-8))
    fbr = await fb
    await opt
    t_train = time.time() - t3
    log("train_probe", create_s=round(t_create, 1), snapshot_s=round(t_snap, 1),
        sample_s=round(t_sample, 1), train_s=round(t_train, 1),
        train_tokens=sum(d.model_input.length for d in data), datums=len(data), groups_kept=kept,
        metrics=getattr(fbr, "metrics", None), step=tr.step_metrics(groups, kept))


async def main() -> None:
    log = lc.Log(lc.LAB_DIR / "logs" / "probe.log")
    renderer, tok = lc.renderer_and_tokenizer()
    svc = lc.service_client()
    for fn in (sample_probe, train_probe):
        try:
            await fn(log, renderer, tok, svc)
        except Exception as e:
            log("probe_error", fn=fn.__name__, err=lc.scrub(f"{type(e).__name__}: {e}")[:3000])


if __name__ == "__main__":
    asyncio.run(main())
