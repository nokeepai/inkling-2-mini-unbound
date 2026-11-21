#!/bin/bash
# Pod-side: build (if compact) and merge one finalist.  bash pod_prep_cand.sh <id>
set -uo pipefail
ID=$1; C=/srv/cands/$ID
echo "PREP $ID $(date -u +%FT%TZ)"
ls -la $C
if [ ! -f $C/tensors.safetensors ] && [ -f $C/lowrank.safetensors ]; then
  python3 /srv/suite/build_cand.py $C /srv/w/orig $C; rc=$?
  echo "build_cand rc=$rc"
  [ $rc -eq 0 ] || { echo "BUILD_FAILED $ID rc=$rc"; exit 1; }
  cat $C/sha256.txt $C/built_sha256.txt
fi
[ -f $C/tensors.safetensors ] || { echo "NO_TENSORS $ID"; exit 1; }
if [ -d /srv/w/merged_$ID ] && [ -f /srv/w/merged_$ID/.merged_ok ]; then echo "merged exists"; else
  python3 /srv/suite/sgl_merge.py $C /srv/w/merged_$ID || { echo "MERGE_FAILED $ID"; exit 1; }
  touch /srv/w/merged_$ID/.merged_ok
fi
echo "PREP_DONE $ID $(date -u +%FT%TZ)"
