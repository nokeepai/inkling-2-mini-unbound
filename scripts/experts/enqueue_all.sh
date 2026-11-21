#!/bin/bash
R=~/Code/inkling-abl
for c in c4rp8_c1t002 c4cd6_c1t002 c4cd3_c1t002 c4rp8_c0lr c4cd6_c0lr; do
  python3 $R/scripts/fleet/enqueue.py $c methods/experts/cands/$c --stage screen --priority 7 --node fleet --family experts --by experts --note "C4 router edit (gate.weight BF16) stacked on parent; see spec"
done
