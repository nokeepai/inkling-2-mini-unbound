#!/bin/bash
# usage: waitfor.sh <remote log> <pattern> [max_secs]
export LIUM_NONINTERACTIVE=1
end=$(( $(date +%s) + ${3:-1800} ))
while [ $(date +%s) -lt $end ]; do
  out=$(lium exec ink-research-1 "grep -c '$2' $1 2>/dev/null; tail -n 2 $1" 2>/dev/null)
  n=$(echo "$out" | sed -n 2p)
  if [ -n "$n" ] && [ "$n" -gt 0 ] 2>/dev/null; then echo "$out"; exit 0; fi
  if echo "$out" | grep -q "Traceback\|Error"; then echo "$out"; exit 1; fi
  sleep 20
done
echo TIMEOUT; echo "$out"
