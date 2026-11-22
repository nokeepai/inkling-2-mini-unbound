#!/bin/bash
# ON eval pod: wait for <cand>'s GGUF, then run its 8192-cap harmful val split 4 ways (shard k = rows k::4).
#   bash shard8k.sh <cand> <gpu0,gpu1,gpu2,gpu3>
set -u
C=$1; IFS=, read -ra G <<< "$2"
until [ -f /data/v11/$C-Q8_0.gguf ] && grep -q "GGUF_DONE $C" /root/logs/build_$C.log 2>/dev/null; do sleep 15; done
for k in 0 1 2 3; do nohup bash /root/v11b/harm8k.sh val8k_${C}_s$k ${G[$k]} /data/v11/$C-Q8_0.gguf /root/v11b/shard$k > /root/logs/h8k_${C}_s$k.log 2>&1 & done
wait; echo SHARDS_DONE
