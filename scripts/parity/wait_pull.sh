#!/usr/bin/env bash
# Wait (locally) until the sglang image exists on pod $1, up to ~40 min.
export LIUM_NONINTERACTIVE=1
for i in $(seq 1 80); do
  out=$(lium exec "$1" "docker images -q lmsysorg/sglang:inkling-2-mini; grep -c 'Pull complete' /root/pull.log; grep -E '^[0-9]+ ' /root/logs/hf_greedy.log 2>/dev/null | tail -1" 2>/dev/null)
  id=$(echo "$out" | grep -E "^[0-9a-f]{12}$" | head -1)
  if echo "$id" | grep -qE '^[0-9a-f]{12}$'; then echo "IMAGE READY on $1 at $(date -u +%H:%M:%S)"; echo "$out"; exit 0; fi
  sleep 30
done
echo "timeout"; echo "$out"
