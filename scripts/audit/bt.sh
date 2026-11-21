#!/bin/bash
# Back-translate the lang_* robustness answers on ink-research-1 (MT_BS=4) and bring the .en files back.
set -euo pipefail
R=/Users/nikshepsvn/Code/inkling-abl
PRIV=/Users/nikshepsvn/.inkling-abl/audit_private/test
L="bash $R/scripts/ink_lium.sh"
$L exec ink-research-1 "mkdir -p /srv/audit/bt && chmod 700 /srv/audit/bt" >/dev/null
for f in "$PRIV"/*_r*/gens_robust_lang_*_off.jsonl; do
  case "$f" in *.en.jsonl) continue ;; esac
  d=$(basename "$(dirname "$f")")
  $L scp ink-research-1 "$f" "/srv/audit/bt/${d}__$(basename "$f")" >/dev/null
done
$L exec ink-research-1 "cd /srv/audit/scripts && MT_BS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 mt.py back /srv/audit/bt/*.jsonl 2>&1 | grep -c MT_BACK"
for f in "$PRIV"/*_r*/gens_robust_lang_*_off.jsonl; do
  case "$f" in *.en.jsonl) continue ;; esac
  d=$(basename "$(dirname "$f")"); b=$(basename "$f" .jsonl)
  $L scp ink-research-1 "/srv/audit/bt/${d}__${b}.en.jsonl" "$(dirname "$f")/" -d >/dev/null
  mv "$(dirname "$f")/${d}__${b}.en.jsonl" "$(dirname "$f")/${b}.en.jsonl"
done
echo BT_DONE
