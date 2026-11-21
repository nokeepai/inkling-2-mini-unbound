#!/bin/bash
# Push chain.sh + diag_math.py and start the chain on POD (default ink-research-1).
R=~/Code/inkling-abl
S="bash $R/scripts/ink_lium.sh"
P=${POD:-ink-research-1}
$S scp $P $R/scripts/experts/chain.sh /root/exp/chain.sh >/dev/null
$S scp $P $R/scripts/experts/diag_math.py /root/exp/diag_math.py >/dev/null
$S exec $P "cd /root/exp && python diag_math.py; NOWAIT=1 nohup bash /root/exp/chain.sh > /root/logs/exp_chain.log 2>&1 & sleep 2; nvidia-smi --query-gpu=memory.used --format=csv,noheader"
