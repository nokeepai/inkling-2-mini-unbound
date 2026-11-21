#!/bin/bash
# ON pod: start a server on GPU $1 for model $2 (optional lora $3), run smoke.py, stop it.
GPU=$1; MODEL=$2; LORA=${3:-}
PORT=$((8200 + GPU))
LARGS=(); [ -n "$LORA" ] && LARGS=(--lora "$LORA")
CUDA_VISIBLE_DEVICES=$GPU /data/llama.cpp/build/bin/llama-server -m "$MODEL" "${LARGS[@]}" -ngl 99 -c ${CMP_CTX:-49152} -np 8 --kv-unified \
  --port $PORT --host 127.0.0.1 --special --jinja --seed 1234 --no-webui ${CMP_EXTRA:-} > /root/compare/logs/smoke_$GPU.server.log 2>&1 &
SPID=$!
for i in $(seq 1 120); do curl -sf http://127.0.0.1:$PORT/health > /dev/null && break; kill -0 $SPID 2>/dev/null || break; sleep 3; done
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader -i $GPU
python3 /root/compare/bin/smoke.py $PORT
grep -iE "lora|error|KV self size|kv_unified|n_ctx|CUDA0 KV|compute buffer|failed" /root/compare/logs/smoke_$GPU.server.log | head -20
kill $SPID
