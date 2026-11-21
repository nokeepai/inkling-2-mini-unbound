#!/usr/bin/env bash
# Pod: build a finalist's tensors (build_cand.py, sha-checked) and merge into /srv/w/merged_<id>. Usage: prep_cand.sh <id>
set -euo pipefail
ID=${1:?id}
C=/srv/cands/$ID
if [ ! -f $C/tensors.safetensors ]; then
  python3 /srv/suite/build_cand.py $C /srv/w/orig $C   # exit 3 = sha mismatch vs canonical sha256.txt
fi
[ -f /srv/w/merged_$ID/.done ] || { python3 /srv/suite/sgl_merge.py $C /srv/w/merged_$ID; touch /srv/w/merged_$ID/.done; }
echo "PREP_DONE $ID"
