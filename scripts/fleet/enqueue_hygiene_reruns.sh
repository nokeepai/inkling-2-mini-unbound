#!/usr/bin/env bash
# Owner-approved hygiene validation (lead 15:0xZ): full stage, think cap 4096, node fleet, top priority.
# Extra finalists named later by mpoa/ara: bash enqueue_hygiene_reruns.sh <cand_id>=<path> ...
set -euo pipefail
E="python3 /Users/nikshepsvn/Code/inkling-abl/scripts/fleet/enqueue.py"
A="--stage full --think-cap 4096 --node fleet --priority 100 --note hygiene-validation-cap4096"
if [ $# -eq 0 ]; then
  $E orig - $A
  $E ou methods/baselines/cands/ou $A
  for p in mpoa/cands/c1h_svd1_s2 mpoa/cands/c1t013; do  # trimmed by lead: orig, ou, finalists A and B only
    $E "$(basename $p)" "methods/$p" $A
  done
else
  for kv in "$@"; do $E "${kv%%=*}" "${kv#*=}" $A; done
fi
