#!/bin/bash
# ON eval pod: start one llama-server per GPU for a model. bash servers.sh <model.gguf> <tag> <gpu list e.g. 0,1,2> [np] [ctx]
# Ports 8100+gpu. Same flags as compare/run_model.sh except -np/-c (throughput for sampling).
set -u
M=$1; TAG=$2; GPUS=$3; NP=${4:-12}; CTX=${5:-49152}
B=/data/llama.cpp/build/bin; mkdir -p /root/logs
for g in ${GPUS//,/ }; do
  P=$((8100+g))
  CUDA_VISIBLE_DEVICES=$g nohup $B/llama-server -m "$M" -ngl 99 -c $CTX -np $NP --kv-unified -ub 128 --port $P --host 127.0.0.1 \
    --special --jinja --seed 1234 --no-webui > /root/logs/srv_${TAG}_$g.log 2>&1 &
  echo $! > /root/logs/srv_$g.pid
done
for g in ${GPUS//,/ }; do
  P=$((8100+g))
  for i in $(seq 1 120); do curl -sf http://127.0.0.1:$P/health >/dev/null && break; sleep 5; done
  curl -sf http://127.0.0.1:$P/health >/dev/null && echo "UP $TAG gpu$g port$P" || echo "FAIL $TAG gpu$g"
done
