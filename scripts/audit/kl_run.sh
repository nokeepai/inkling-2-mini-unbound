#!/bin/bash
# Pod-side (ink-research-1): KL/top-1 for candidates on a split's harmless prompts, one model load (kl_eval "all":
# gen -> cache -> orig self-check -> noise bs1 -> noise shuffle -> candidates -> routediag).
# Usage: bash kl_run.sh <harmless.jsonl> <split> <n> <cand_dir> [cand_dir ...]
set -euo pipefail
SRC=$1; SPLIT=$2; N=$3; shift 3
export FID_DIR=/srv/audit/kl_$SPLIT FID_BIG=/srv/audit/klbig_$SPLIT AQ=0
cd /srv/audit/scripts
for c in "$@"; do
  [ -f "$c/tensors.safetensors" ] || python3 /srv/audit/harness/build_cand.py "$c" /root/w/orig "$c"
done
python3 kl_test.py "$SRC" "$N" all "$@"
echo KL_RUN_DONE
