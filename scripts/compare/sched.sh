#!/bin/bash
# ON pod: shard the borderline set (784) of the slow models over GPUs as they free up. Identical server flags
# (run_model2.sh == run_model.sh + --shard). When a main run finishes harmful, its own eval process (started by
# run_model.sh) is stopped so it skips borderline; run_model.sh then does its KLD as usual.
set -u
LOG=/root/compare/logs; M=/data/base/Inkling-2-Mini-Q8_0.gguf
declare -A MODEL=( [unbound_q8]=/data/release/Inkling-2-Mini-Unbound-Q8_0.gguf [ou_q8]=/data/compare/ou/ou-Q8_0.gguf
  [atomic_lora_q8]=$M [atomic_adq4]=/data/compare/models/atomic/Inkling-2-Mini-Uncensored-AD-Q4_K.gguf )
declare -A LORA=( [unbound_q8]=- [ou_q8]=- [atomic_lora_q8]=/data/compare/models/atomic/Inkling-2-Mini-Uncensored-LoRA.gguf [atomic_adq4]=- )
declare -A MAINGPU=( [orig]=0 [atomic_lora_q8]=1 [unbound_q8]=2 [ou_q8]=3 [xq]=4 [atomic_adq4]=5 )
QUEUE=(atomic_lora_q8:1 unbound_q8:1 ou_q8:1 atomic_adq4:1 atomic_lora_q8:0 unbound_q8:0 ou_q8:0 atomic_adq4:0)
declare -A SHARDGPU=()   # gpu -> shard tag running there
log() { echo "$(date -u +%FT%TZ) $*" >> $LOG/sched.log; }

gpu_free() {  # main run on this gpu finished and no shard of ours is running there
  local g=$1 n
  for n in "${!MAINGPU[@]}"; do
    [ "${MAINGPU[$n]}" = "$g" ] && ! grep -q ALL_DONE $LOG/$n.status 2>/dev/null && return 1
  done
  if [ -n "${SHARDGPU[$g]:-}" ]; then
    grep -q ALL_DONE $LOG/${SHARDGPU[$g]}.status 2>/dev/null || return 1
  fi
  return 0
}

while true; do
  # stop main evals that reached borderline (harmful done) so they go on to KLD
  for n in unbound_q8 ou_q8 atomic_lora_q8 atomic_adq4; do
    if grep -q "^harmful done" $LOG/$n.eval.log 2>/dev/null && [ ! -f $LOG/$n.bl_stopped ]; then
      pid=$(pgrep -f "llama_eval.py --port $((8100 + ${MAINGPU[$n]})) --out /root/compare/out/$n " | head -n 1)
      [ -n "$pid" ] && kill $pid && log "stopped main eval of $n (pid $pid) before borderline"
      touch $LOG/$n.bl_stopped
    fi
  done
  for g in 0 1 2 3 4 5; do
    [ ${#QUEUE[@]} -eq 0 ] && break
    gpu_free $g || continue
    job=${QUEUE[0]}; QUEUE=("${QUEUE[@]:1}")
    n=${job%%:*}; k=${job##*:}; tag=${n}__bl$k
    CMP_METRICS=borderline CMP_SKIP_KLD=1 CMP_SHARD=$k/2 nohup bash /root/compare/bin/run_model2.sh $tag $g $((8100 + g)) \
      ${MODEL[$n]} ${LORA[$n]} > $LOG/$tag.run.log 2>&1 &
    SHARDGPU[$g]=$tag
    log "launched $tag on gpu $g"
    sleep 20
  done
  if [ ${#QUEUE[@]} -eq 0 ]; then
    all=1; for n in unbound_q8 ou_q8 atomic_lora_q8 atomic_adq4; do [ -f $LOG/$n.bl_stopped ] || all=0; done
    for g in "${!SHARDGPU[@]}"; do grep -q ALL_DONE $LOG/${SHARDGPU[$g]}.status 2>/dev/null || all=0; done
    [ $all = 1 ] && { log "SCHED_DONE"; exit 0; }
  fi
  sleep 30
done
