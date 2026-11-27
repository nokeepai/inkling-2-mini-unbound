"""Paid parallel-limit probe: open N paid LoRA training clients at once and run one minimal
step each (a ~40-token cross-entropy forward_backward + optim_step). Reports which clients were
created and stepped, timings, and exact error text (key scrubbed). Estimated cost: N x 40
training tokens x $0.75/1M, about $0.0001, plus nothing for sampling.

DO NOT RUN without the owner's go. It refuses unless both --go is passed and
LAB_PAID_MAX_USD > 0, and every step is pre-checked against the paid ledger.

  LAB_PAID_MAX_USD=0.05 ~/.inkling-abl/tinker_lab/.venv/bin/python \
      scripts/tinker/probe_paid_concurrency.py --go [--n 4]
"""
from __future__ import annotations

import argparse
import asyncio
import time
from typing import Any

import lab_common as lc


def tiny_datum(tok: Any) -> Any:
    import tinker
    import torch
    from tinker import TensorData

    ids = tok.encode("The quick brown fox jumps over the lazy dog. " * 4)[:40]
    return tinker.Datum(
        model_input=tinker.ModelInput.from_ints(ids[:-1]),
        loss_fn_inputs={"target_tokens": TensorData.from_torch(torch.tensor(ids[1:], dtype=torch.int64)),
                        "weights": TensorData.from_torch(torch.ones(len(ids) - 1))})


async def one(svc: Any, i: int, datum: Any, log: lc.Log) -> dict[str, Any]:
    import tinker

    t0 = time.time()
    out: dict[str, Any] = {"client": i}
    try:
        tc = await lc.training_client(svc, 8, train_attn=True, train_mlp=False, train_unembed=False)
        out["create_s"] = round(time.time() - t0, 1)
        n_tok = datum.model_input.length
        lc.paid_precheck(lc.paid_cost(train=n_tok))
        fb = tc.forward_backward([datum], loss_fn="cross_entropy")
        opt = tc.optim_step(tinker.AdamParams(learning_rate=1e-6))
        await fb
        await opt
        lc.paid_record(lc.paid_cost(train=n_tok), "train")
        out.update(ok=True, total_s=round(time.time() - t0, 1))
    except BaseException as e:  # record throttles/limits verbatim (scrubbed); SystemExit = cap
        out.update(ok=False, err=lc.scrub(f"{type(e).__name__}: {e}")[:1500])
    log("client", **out)
    return out


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--go", action="store_true")
    ap.add_argument("--n", type=int, default=4)
    args = ap.parse_args()
    if not args.go:
        raise SystemExit("refusing: pass --go only after the owner approves paid spend")
    lc.enable_paid("probe_paid_concurrency")
    log = lc.Log(lc.LAB_DIR / "logs" / "probe_paid_concurrency.log")
    _, tok = lc.renderer_and_tokenizer()
    svc = lc.service_client()
    res = await asyncio.gather(*(one(svc, i, tiny_datum(tok), log) for i in range(args.n)))
    log("summary", ok=sum(r.get("ok", False) for r in res), n=args.n,
        paid_spent_usd=round(lc.paid_spent(), 6))


if __name__ == "__main__":
    asyncio.run(main())
