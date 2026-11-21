#!/bin/bash
# Emit new milestone lines from a pod log; exit on "peak mem" or Traceback.
export LIUM_NONINTERACTIVE=1
LOG=${1:-/root/logs/fid_kl.log}
seen=0
while true; do
  out=$(lium exec ink-research-2 "grep -E 'gen batch|GEN_DONE|CACHE_DONE|RESULT|Traceback|Error|OOM|peak mem' $LOG" 2>/dev/null | grep -v Executing || true)
  n=$(printf '%s\n' "$out" | grep -c . || true)
  if [ "$n" -gt "$seen" ]; then printf '%s\n' "$out" | tail -n $((n-seen)); seen=$n; fi
  if printf '%s' "$out" | grep -qE 'peak mem|Traceback'; then exit 0; fi
  sleep 45
done
