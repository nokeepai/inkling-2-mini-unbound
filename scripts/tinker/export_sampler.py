"""Turn a save_state checkpoint into sampler weights so eval.py can sample it.

save_state paths (".../weights/<name>") hold trainer state; SamplingClient needs
".../sampler_weights/<name>". This loads the state into a fresh free-model LoRA client
and saves sampler weights with a TTL.

  ~/.inkling-abl/tinker_lab/.venv/bin/python scripts/tinker/export_sampler.py \
      --state tinker://.../weights/smoke5-step5 --rank 32 --name smoke5-step5
"""
from __future__ import annotations

import argparse
import asyncio

import lab_common as lc


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--attn-only", action="store_true")
    ap.add_argument("--mlp-only", action="store_true")
    ap.add_argument("--train-unembed", action="store_true")
    ap.add_argument("--ttl", type=int, default=7 * 24 * 3600)
    a = ap.parse_args()
    svc = lc.service_client()
    tc = await lc.training_client(svc, a.rank, train_attn=not a.mlp_only,
                                  train_mlp=not a.attn_only, train_unembed=a.train_unembed)
    await tc.load_state(a.state)
    res = await tc.save_weights_for_sampler(a.name, ttl_seconds=a.ttl)
    print(res.path)


if __name__ == "__main__":
    asyncio.run(main())
