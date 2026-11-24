#!/bin/bash
# ON ink-v21-serve: (re)start vLLM + tunnel. SPEC=1 enables MTP speculative decoding. EXTRA_ARGS passes more vLLM flags.
set -u
L=/root/logs/serve.log; log() { echo "$(date -u +%T) $*" >> $L; }
IMG=$(cat /root/serve/img); KEY=$(cat /root/serve/key)
docker rm -f v21 > /dev/null 2>&1
SPECARG=()
[ "${SPEC:-0}" = 1 ] && SPECARG=(--speculative-config '{"method":"mtp","num_speculative_tokens":4}')
docker run -d --name v21 --gpus all --ipc=host --network host -v /data/srv/v21ck:/model:ro $IMG \
  --model /model --served-model-name inkling-2-mini-unbound-v2.1 --host 0.0.0.0 --port 8000 --api-key "$KEY" \
  --data-parallel-size 2 --max-num-seqs 64 --gpu-memory-utilization 0.92 --max-model-len 32768 \
  --trust-remote-code --reasoning-parser inkling --tool-call-parser inkling --enable-auto-tool-choice \
  --override-generation-config '{"temperature":0.6,"top_p":0.95}' "${SPECARG[@]}" ${EXTRA_ARGS:-} > /root/logs/docker_run.log 2>&1
log VLLM_START rc=$? spec=${SPEC:-0}
for i in $(seq 1 120); do curl -sf localhost:8000/health > /dev/null && break; sleep 10; done
curl -sf localhost:8000/health > /dev/null && log VLLM_UP || { log VLLM_FAIL; docker logs v21 2>&1 | tail -40 > /root/logs/vllm_fail.log; exit 1; }
if ! pgrep -f "cloudflared tunnel" > /dev/null; then
  setsid nohup cloudflared tunnel --no-autoupdate --url http://localhost:8000 > /root/logs/cloudflared.log 2>&1 < /dev/null &
  for i in $(seq 1 30); do grep -qoE 'https://[a-z0-9-]+\.trycloudflare\.com' /root/logs/cloudflared.log && break; sleep 2; done
fi
log URL "$(grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' /root/logs/cloudflared.log | head -1)"
log SERVE_READY
