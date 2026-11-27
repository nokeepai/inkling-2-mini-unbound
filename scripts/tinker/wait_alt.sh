#!/bin/bash
# Exits once every main W1 eval (and base) carries the Haiku alt_p field, or after a 20-min stall.
LAB="$HOME/.inkling-abl/tinker_lab"
TAGS="base_t06_think_e09 w1a_step20 w1a_step40 w1b_step20 w1b_step40 w1c_step20 w1c_step40 w1d_step20 w1d_step40"
while true; do
  missing=""
  for t in $TAGS; do
    grep -q '"alt_p"' "$LAB/eval/$t.jsonl" 2>/dev/null || missing="$missing $t"
  done
  if [ -z "$missing" ]; then echo "ALT DONE at $(date '+%H:%M')"; exit 0; fi
  if [ -z "$(find "$LAB" -type f -newermt '-20 minutes' -print -quit)" ]; then echo "STALL at $(date '+%H:%M'); missing:$missing"; exit 1; fi
  sleep 90
done
