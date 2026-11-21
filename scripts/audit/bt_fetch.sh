#!/bin/bash
# Fetch back-translated language answers from ink-research-1 into the private dirs, then run Jev on those dirs.
set -euo pipefail
R=/Users/nikshepsvn/Code/inkling-abl
PRIV=/Users/nikshepsvn/.inkling-abl/audit_private/test
L="bash $R/scripts/ink_lium.sh"
for f in "$PRIV"/*_r*/gens_robust_lang_*_off.jsonl; do
  case "$f" in *.en.jsonl) continue ;; esac
  d=$(basename "$(dirname "$f")"); b=$(basename "$f" .jsonl)
  $L scp ink-research-1 "/srv/audit/bt/${d}__${b}.en.jsonl" "$(dirname "$f")/" -d >/dev/null
  mv "$(dirname "$f")/${d}__${b}.en.jsonl" "$(dirname "$f")/${b}.en.jsonl"
done
for d in ou_rL c1h_r1_svd1_s3_L4_rA c1h_r1_svd1_s3_L4_rB c1dt001_rA c1dt001_rB; do
  python3 $R/scripts/audit/jev_sealed.py "$PRIV/$d" | grep -v JEV_DONE
done
python3 $R/scripts/audit/robust_table.py
