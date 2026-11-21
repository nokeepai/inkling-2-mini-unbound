#!/bin/bash
# Round 2 C4 candidates (owner: drop-in only): build on ink-research-3, pull back, enqueue fleet/screen/p7.
R=~/Code/inkling-abl
S="bash $R/scripts/ink_lium.sh"
P=ink-research-3
CANDS="c4cd6_oupx4 c4cd3_oupx4 c4rp8_oupx4 c4cd6k8_c1t002 c4cd6k16_c1t002"
$S exec $P "mkdir -p /root/exp/parents/c1h_x4_ouprofile" >/dev/null
$S scp $P $R/methods/mpoa/cands/c1h_x4_ouprofile/lowrank.safetensors /root/exp/parents/c1h_x4_ouprofile/lowrank.safetensors >/dev/null
$S scp $P $R/scripts/experts/make_cands2_remote.sh /root/exp/make_cands2_remote.sh >/dev/null
$S exec $P "bash /root/exp/make_cands2_remote.sh 2>&1 | grep -v -i deprecat | grep -v created"
for c in $CANDS; do
  mkdir -p $R/methods/experts/cands/$c
  for f in lowrank.safetensors spec.json sha256.txt; do
    $S scp $P /root/exp/cands/$c/$f $R/methods/experts/cands/$c/$f --download >/dev/null
  done
  [ -s $R/methods/experts/cands/$c/sha256.txt ] || { echo "missing $c"; continue; }
  python3 $R/scripts/fleet/enqueue.py $c methods/experts/cands/$c --stage screen --priority 7 --node fleet --family experts --by experts --note "C4 router edit (gate.weight BF16) on parent; see spec"
done
