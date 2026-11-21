#!/bin/bash
# Poll nokeep.ai/api/health until the build matches the merged commit (max ~20 min).
WANT=4e07b4464d08
for i in $(seq 1 40); do
  B=$(curl -s -m 10 https://nokeep.ai/api/health)
  case "$B" in *"$WANT"*) echo "DEPLOYED $(date -u +%H:%M) $B"; exit 0 ;; esac
  sleep 30
done
echo "NOT DEPLOYED after 20 min; last: $B"; exit 1
