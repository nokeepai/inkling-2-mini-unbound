#!/bin/bash
# Delete Claude Code temp/scratch dirs under /private/tmp/claude-501 that have had NO file modified in 24h.
# Never touches this session's dir or anything active in the last 24h.
set -u
ME=140f1e4d-5898-4fb1-9611-0b847d426684
before=$(df -k /Users/nikshepsvn | tail -1 | awk '{print $4}')
for d in /private/tmp/claude-501/-Users-nikshepsvn-Code-innervale/*/ /private/tmp/claude-501/-Users-nikshepsvn-Code-innervale--claude-worktrees-*/; do
  case "$d" in *"$ME"*) continue ;; esac
  if [ -z "$(find "$d" -newermt '-24 hours' -print -quit 2>/dev/null)" ]; then
    rm -rf "$d"
  fi
done
after=$(df -k /Users/nikshepsvn | tail -1 | awk '{print $4}')
echo "freed MB: $(( (after - before) / 1024 ))"
df -h /Users/nikshepsvn | tail -1
