#!/bin/bash
# Count finished generations per comparison job on the pod (expected: harmful 277, borderline shards sum 784).
bash /Users/nikshepsvn/Code/inkling-abl/scripts/ink_lium.sh exec ink-gguf-1 '
cd /root/compare/out 2>/dev/null || exit 0
for d in */; do
  for f in "$d"*.jsonl; do [ -f "$f" ] && printf "%-45s %6s\n" "$f" "$(wc -l < "$f")"; done
done | sort' 2>&1 | grep -v Executing
