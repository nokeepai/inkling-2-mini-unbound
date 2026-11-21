#!/usr/bin/env bash
# Per-GPU variant of scripts/parity/sglang_up.sh for multi-GPU pods (agent `fleet`). Identical image, serve command and
# ENGINE.md flags; the only differences are the container name (sgl<gpu>), `--gpus device=<gpu>` instead of all, and
# the host port. Weights must already be in /srv/w (pod_bringup_remote.sh copies them once for all GPUs).
#   bash sglang_up_gpu.sh <gpu> <host_port>          (idempotent; waits for /health)
# Stop one: docker rm -f sgl<gpu>
set -euo pipefail
GPU=${1:?gpu index}
HPORT=${2:?host port}
MODEL=/w/orig
EXTRA="--mem-fraction-static 0.73815 --chunked-prefill-size 4096 --max-running-requests 31 --random-seed 1234"
IMAGE=lmsysorg/sglang:inkling-2-mini
NAME=sgl$GPU
SIG="$MODEL|$HPORT|$EXTRA|gpu$GPU"
[ -f /srv/w/orig/config.json ] || { echo "no /srv/w/orig"; exit 1; }
if docker ps --format '{{.Names}}' | grep -qx "$NAME"; then
  if [ "$(docker inspect -f '{{ index .Config.Labels "sig" }}' $NAME)" = "$SIG" ] && curl -sf localhost:$HPORT/health >/dev/null; then
    echo "already up: $SIG"; exit 0
  fi
fi
docker rm -f $NAME >/dev/null 2>&1 || true
[ -f /srv/admin_key ] || head -c 24 /dev/urandom | base64 | tr -d "/+=" > /srv/admin_key
docker run -d --name $NAME --label "sig=$SIG" --gpus "device=$GPU" --ipc=host \
  -v /srv/w:/w -v /srv/parity:/parity -p $HPORT:30000 -e SGLANG_ADMIN_API_KEY="$(cat /srv/admin_key)" \
  $IMAGE \
  sglang serve --model-path $MODEL --tp 1 --attention-backend fa4 --dtype bfloat16 --page-size 128 \
    --enable-multimodal --reasoning-parser inkling --tool-call-parser inkling2 \
    --host 0.0.0.0 --port 30000 $EXTRA >/dev/null
for i in $(seq 1 360); do
  if curl -sf localhost:$HPORT/health >/dev/null; then echo "UP after $((i*5))s: $SIG"; exit 0; fi
  if ! docker ps --format '{{.Names}}' | grep -qx "$NAME"; then echo "container died"; docker logs --tail 80 $NAME; exit 1; fi
  sleep 5
done
echo "timeout waiting for health: $NAME"; docker logs --tail 80 $NAME; exit 1
