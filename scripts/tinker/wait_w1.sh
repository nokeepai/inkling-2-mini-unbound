#!/bin/bash
# Exits when all 8 W1 eval summaries exist, or when nothing in the lab has changed for 20 min.
# Why: subagents go idle waiting for notifications that never arrive, so the orchestrator waits on files.
LAB="$HOME/.inkling-abl/tinker_lab"
while true; do
  n=$(ls "$LAB"/eval/w1?_step20.summary.json "$LAB"/eval/w1?_step40.summary.json 2>/dev/null | wc -l | tr -d ' ')
  if [ "$n" -ge 8 ]; then echo "W1 DONE: $n summaries at $(date '+%H:%M')"; exit 0; fi
  if [ -z "$(find "$LAB" -type f -newermt '-20 minutes' -print -quit)" ]; then
    echo "STALL at $(date '+%H:%M'): $n/8 summaries"; tail -n 2 "$LAB/status.txt"; exit 1
  fi
  sleep 120
done
