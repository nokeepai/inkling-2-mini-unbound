#!/bin/bash
R=~/Code/inkling-abl
S="bash $R/scripts/ink_lium.sh"
P=${POD:-ink-research-3}
$S exec $P "mkdir -p /root/exp/data /root/exp/out /root/diag /root/logs" >/dev/null
for f in localise.py analyse_loc.py hooks_eval.py chain.sh; do $S scp $P $R/scripts/experts/$f /root/exp/$f >/dev/null; done
if [ "$1" = "data" ]; then
  $S scp $P $R/scripts/diag/diaglib.py /root/diag/diaglib.py >/dev/null
  $S scp $P $R/methods/directions/directions.pt /root/exp/directions.pt >/dev/null
  $S scp $P $R/data/train/harmful.jsonl /root/exp/data/train_harmful.jsonl >/dev/null
  $S scp $P $R/data/train/harmless.jsonl /root/exp/data/train_harmless.jsonl >/dev/null
  $S scp $P $R/data/val/harmless.jsonl /root/exp/data/val_harmless.jsonl >/dev/null
  $S scp $P $R/harness/refusal_phrases_seed.txt /root/exp/data/refusal_phrases_seed.txt >/dev/null
  $S scp $P $R/harness/refusal_phrases.txt /root/exp/data/refusal_phrases.txt >/dev/null
fi
$S exec $P "ls -la /root/exp /root/exp/data; tail -c 300 /root/fetch.log; tail -2 /root/logs/install.log"
