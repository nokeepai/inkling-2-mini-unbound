#!/bin/bash
# Pod-side: translations (forward) and the robustness set for one split.
# Usage: bash prep_robust.sh <data_dir> <split> [limit]     -> /srv/audit/robust_<split>/
set -euo pipefail
DATA=$1; SPLIT=$2; LIM=${3:-0}
cd /srv/audit/scripts
OUT=/srv/audit/robust_$SPLIT
mkdir -p "$OUT"
LARG=""; [ "$LIM" != "0" ] && LARG="--limit $LIM"
python3 mt.py fwd "$DATA/$SPLIT/harmful.jsonl" /srv/audit/mt_fwd_$SPLIT.jsonl $LARG
python3 robust_build.py --data "$DATA" --split "$SPLIT" --out "$OUT" --translations /srv/audit/mt_fwd_$SPLIT.jsonl $LARG
ls -la "$OUT"
echo PREP_ROBUST_DONE
