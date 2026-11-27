#!/bin/bash
# Usage: wait_train_done.sh <tag> [<tag> ...] — exits when every tag's train log has a "done" event, or after a 20-min stall.
LAB="$HOME/.inkling-abl/tinker_lab"
while true; do
  left=""
  for t in "$@"; do grep -q '"event": "done"' "$LAB/logs/train_$t.stdout" 2>/dev/null || left="$left $t"; done
  if [ -z "$left" ]; then echo "TRAIN DONE: $* at $(date '+%H:%M')"; exit 0; fi
  if [ -z "$(find "$LAB" -type f -newermt '-20 minutes' -print -quit)" ]; then echo "STALL at $(date '+%H:%M'); waiting on:$left"; exit 1; fi
  sleep 60
done
