#!/bin/bash
# ON ink-v21-serve: two INDEPENDENT vLLM servers (one per GPU, ports 8001/8002, localhost only) behind lb_proxy.py on
# :8000 (API-key gate, least-busy routing, streaming pass-through). The Cloudflare tunnel already points at :8000.
# vLLM DP=2 on a MoE couples both GPUs every step over PCIe (expert all2all); independent replicas avoid that.
set -u
L=/root/logs/serve.log; log() { echo "$(date -u +%T) $*" >> $L; }
IMG=$(cat /root/serve/img)
docker rm -f v21 v21a v21b > /dev/null 2>&1
SPECARG=(); [ "${SPEC:-0}" = 1 ] && SPECARG=(--speculative-config '{"method":"mtp","num_speculative_tokens":3}')
for g in 0 1; do
  P=$((8001+g)); N=v21$([ $g = 0 ] && echo a || echo b)
  docker run -d --name $N --restart unless-stopped --gpus "\"device=$g\"" --ipc=host --network host -v /data/srv/v21ck:/model:ro \
    $IMG /model --served-model-name inkling-2-mini-unbound-v2.1 --host 127.0.0.1 --port $P \
    --max-num-seqs 64 --gpu-memory-utilization 0.92 --max-model-len 32768 --trust-remote-code \
    --reasoning-parser inkling --tool-call-parser inkling --enable-auto-tool-choice \
    --override-generation-config '{"temperature":0.6,"top_p":0.95}' "${SPECARG[@]}" > /dev/null
done
log VLLM2_START spec=${SPEC:-0}
for P in 8001 8002; do
  for i in $(seq 1 120); do curl -sf localhost:$P/health > /dev/null && break; sleep 10; done
  curl -sf localhost:$P/health > /dev/null && log UP $P || { log FAIL $P; exit 1; }
done
[ -f /root/serve/lb.pid ] && kill "$(cat /root/serve/lb.pid)" 2>/dev/null; sleep 1
setsid nohup python3 /root/stage/v11b/lb_proxy.py 8000 /root/serve/key 8001,8002 > /root/logs/lb.log 2>&1 < /dev/null &
echo $! > /root/serve/lb.pid
sleep 2; log LB_UP "$(curl -s -o /dev/null -w '%{http_code}' localhost:8000/health)"
