#!/bin/bash
# Usage: wait_files.sh <glob> <count> — exits when <count> files match <glob> under the lab dir, or after a 20-min stall.
# Why: subagents go idle between stages; the orchestrator wakes on files and nudges them.
LAB="$HOME/.inkling-abl/tinker_lab"
PAT="$1"; WANT="$2"
while true; do
  n=$(ls $LAB/$PAT 2>/dev/null | wc -l | tr -d ' ')
  if [ "$n" -ge "$WANT" ]; then echo "READY: $n x $PAT at $(date '+%H:%M')"; ls $LAB/$PAT; exit 0; fi
  if [ -z "$(find "$LAB" -type f -newermt '-20 minutes' -print -quit)" ]; then echo "STALL at $(date '+%H:%M'): $n/$WANT"; exit 1; fi
  sleep 90
done
