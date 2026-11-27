#!/bin/bash
# On the pod: merge one adapter into the bases, then run the checks, without ever touching another
# candidate's server. Everything that uses the GPU (a cuda merge, serving) runs under one lock.
#
#   bash /srv/stack/scripts/v3stack/cand.sh <name> ["<variants>"] [cuda|cpu]
#     name      adapter at /srv/stack/adapter_<name>; weights /srv/stack_w_<name> (bases hard-linked there by
#               the lead's push_adapter.sh); results /root/stack_evals_<name>; log /root/logs/stack_<name>.log
#     variants  default "v21_full_sr"; any of v21_full_sr stock_full_sr v21 stock
#     env       PASS1 / PASS2 as in stack_test.sh; SEED [0] the stochastic-rounding seed
#
# Why not stack_test.sh alone: it builds on CPU in the background and its EXIT trap removes the shared
# server container, so two instances at once kill each other's suites.
set -u
NAME="$1"; VARIANTS="${2:-v21_full_sr}"; DEV="${3:-cuda}"
IMG="lmsysorg/sglang@sha256:421aed3ac11741381eac350bf7b1db28a0cafb6ae49b57ae78fba86a16a32306"
W="/srv/stack_w_$NAME"; A="/srv/stack/adapter_$NAME"; ST="/root/.stack_state_$NAME"; OUT="/root/stack_evals_$NAME"
LOG="/root/logs/stack_$NAME.log"
mkdir -p /root/logs "$ST" "$OUT"
exec >> "$LOG" 2>&1
say() { echo "$(date -u +%T) $*"; }

merge_one() {   # $1 variant; base name is the part before the first underscore
  local v="$1" base="${1%%_*}" gpus=() cpus=4
  case "$v" in v21|stock) return 0 ;; esac
  if [ -d "$W/$v" ] && [ -f "$W/$v.merge_report.json" ]; then say "cand merge $v: done earlier"; return 0; fi
  if [ "$DEV" = cuda ]; then gpus=(--gpus "device=0"); else cpus="$(( $(nproc) - 1 ))"; fi
  say "cand merge $v on $DEV"
  docker rm -f "candmerge_${NAME}_$v" > /dev/null 2>&1 || true
  if docker run --rm --name "candmerge_${NAME}_$v" --network none ${gpus[@]+"${gpus[@]}"} --cpus "$cpus" \
      -e OMP_NUM_THREADS="$cpus" -v "$W:/w" -v /srv/stack/scripts/v3stack:/s:ro -v "$A:/adapter:ro" \
      --entrypoint python3 "$IMG" /s/merge_adapter.py --base "/w/$base" --adapter /adapter --out "/w/$v" \
      --parts attn_experts --device "$DEV" --rounding stochastic --seed "${SEED:-0}" \
      > "/root/logs/candmerge_${NAME}_$v.log" 2>&1; then
    say "PASS candmerge_$v ($(grep -m1 MERGE_OK "/root/logs/candmerge_${NAME}_$v.log" | cut -c1-120))"
  else
    say "FAIL candmerge_$v ($(tail -n 1 "/root/logs/candmerge_${NAME}_$v.log" | cut -c1-160))"; return 1
  fi
}

merge_all() { local v; for v in $VARIANTS; do merge_one "$v" || true; done; }

[ "$DEV" = cpu ] && merge_all          # CPU merges need no GPU: do them before queueing for it
(
  flock 9
  # Runs started before this lock existed still own the server: wait them out.
  while pgrep -f 'v3stack/stack_test.sh' > /dev/null; do sleep 15; done
  [ "$DEV" = cuda ] && merge_all
  STACK=/srv/stack ADAPTER="$A" W="$W" STATE="$ST" OUT="$OUT" VARIANTS="$VARIANTS" \
    PASS1="${PASS1-agent_hard_noforce ubench}" \
    PASS2="${PASS2-main hardmath code rp pub_humaneval pub_quixbugs pub_gsm8k pub_mmlu}" \
    bash /srv/stack/scripts/v3stack/stack_test.sh
) 9> /root/gpu.lock
say "CAND_DONE $NAME"
