#!/bin/bash
# Round 2 C4 candidates built on the Mac (research-3 is gone): build, build_cand check, enqueue fleet/screen/p7.
R=~/Code/inkling-abl
PY=/opt/homebrew/bin/python3.13
export RANKING=$R/methods/experts/pod/ranking.json DIRECTIONS=$R/methods/directions/directions.pt
export ORIG_FILE=$R/methods/experts/origdir/model.safetensors
C=$R/methods/experts/cands
OUP=$R/methods/mpoa/cands/c1h_x4_ouprofile
T12=$R/methods/mpoa/cands/c1t002
set -e
$PY $R/scripts/experts/build_router_cand.py $OUP $C/c4cd6_oupx4 32 6 cond
$PY $R/scripts/experts/build_router_cand.py $OUP $C/c4cd3_oupx4 32 3 cond
$PY $R/scripts/experts/build_router_cand.py $OUP $C/c4rp8_oupx4 8 0 rproj
$PY $R/scripts/experts/build_router_cand.py $T12 $C/c4cd6k8_c1t002 8 6 cond
$PY $R/scripts/experts/build_router_cand.py $T12 $C/c4cd6k16_c1t002 16 6 cond
mkdir -p /tmp/c4built
for c in c4cd6_oupx4 c4cd3_oupx4 c4rp8_oupx4 c4cd6k8_c1t002 c4cd6k16_c1t002; do
  rm -rf /tmp/c4built/$c
  $PY $R/harness/build_cand.py $C/$c $R/methods/experts/origdir /tmp/c4built/$c | tail -1
  cp /tmp/c4built/$c/sha256.txt $C/$c/sha256.txt
  python3 $R/scripts/fleet/enqueue.py $c methods/experts/cands/$c --stage screen --priority 7 --node fleet --family experts --by experts --note "C4 router edit (gate.weight BF16) on parent; see spec"
done
