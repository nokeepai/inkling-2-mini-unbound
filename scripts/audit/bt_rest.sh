#!/bin/bash
# Pod-side: back-translate remaining files in /srv/audit/bt (no .en.jsonl yet) with a large batch.
cd /srv/audit/scripts
todo=()
for f in /srv/audit/bt/*.jsonl; do
  case "$f" in *.en.jsonl) continue ;; esac
  [ -f "${f%.jsonl}.en.jsonl" ] || todo+=("$f")
done
echo "todo ${#todo[@]}"
MT_BS=48 python3 mt.py back "${todo[@]}" 2>&1 | grep MT_BACK
echo BT_REST_DONE
