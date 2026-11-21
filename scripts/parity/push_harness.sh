#!/usr/bin/env bash
# Upload harness code + val data to /srv/harness and /srv/data/val on both eval pods.
export LIUM_NONINTERACTIVE=1
R=/Users/nikshepsvn/Code/inkling-abl
for pod in ink-eval-1 ink-eval-2; do
  lium exec $pod "mkdir -p /srv/harness/cap /srv/data/val /srv/evals" >/dev/null
  for f in sgl_eval.py sgl_merge.py refusal_phrases_seed.txt cap/mmlu300.jsonl cap/gsm8k100.jsonl; do
    lium scp $pod "$R/harness/$f" "/srv/harness/$f"
  done
  [ -f $R/harness/refusal_phrases.txt ] && lium scp $pod "$R/harness/refusal_phrases.txt" /srv/harness/refusal_phrases.txt
  for f in harmful borderline probes; do
    lium scp $pod "$R/data/val/$f.jsonl" "/srv/data/val/$f.jsonl"
  done
done
echo HARNESS_PUSHED
