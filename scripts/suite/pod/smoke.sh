#!/usr/bin/env bash
# Pod: smoke every core bench on 6 items against orig on GPU 0 (port 4000); model name "smoke".
cd /srv/suite
rm -rf /srv/suite/out/smoke
for b in mmlu_pro gsm8k math500 gpqa_diamond ifeval humaneval_plus; do
  echo "== $b"
  SUITE_LIMIT=6 python3 bench_core.py $b gen --port 4000 --model smoke 2>&1 | tail -2
done
for b in mmlu_pro gsm8k math500 gpqa_diamond ifeval humaneval_plus; do
  echo "== score $b"
  SUITE_LIMIT=6 python3 bench_core.py $b score --model smoke 2>&1 | tail -3
done
echo SMOKE_DONE
