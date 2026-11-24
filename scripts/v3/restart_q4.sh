#!/bin/bash
# ON a gen pod: replace Q8_0 sampling servers by v2 Q4_K_M with more slots (sampling throughput; evals stay Q8_0).
#   bash restart_q4.sh <gpu list> <np> <ctx>
set -u
GPUS=$1; NP=${2:-16}; CTX=${3:-98304}
pkill -f gen_samples.py; pkill -f gen_v3.py; pkill -f rp_play.py; pkill -f llama-server; sleep 5
bash /root/v11b/servers.sh /data/v2/Inkling-2-Mini-Unbound-v2-Q4_K_M.gguf v2q4 $GPUS $NP $CTX
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
