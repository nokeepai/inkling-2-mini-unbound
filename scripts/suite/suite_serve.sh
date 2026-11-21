#!/usr/bin/env bash
# Suite launcher (agent `suite`): ENGINE.md image + serve command, any model dir, one container per GPU.
#   bash suite_serve.sh <gpu> <host_port> <model=/w/orig> [engine flag overrides...]
# 5090: no overrides (frozen ENGINE.md flags). PRO 6000 long-context: explicit overrides (recorded in suite/PLAN.md).
set -euo pipefail
GPU=${1:?gpu}; HPORT=${2:?port}; MODEL=${3:-/w/orig}; shift 3 || true
EXTRA="${*:---mem-fraction-static 0.73815 --chunked-prefill-size 4096 --max-running-requests 31 --random-seed 1234}"
IMAGE=lmsysorg/sglang:inkling-2-mini
NAME=s$GPU
SIG="$MODEL|$HPORT|$EXTRA|gpu$GPU"
if docker ps --format '{{.Names}}' | grep -qx "$NAME"; then
  if [ "$(docker inspect -f '{{ index .Config.Labels "sig" }}' $NAME)" = "$SIG" ] && curl -sf localhost:$HPORT/health >/dev/null; then
    echo "already up: $SIG"; exit 0; fi
fi
docker rm -f $NAME >/dev/null 2>&1 || true
[ -f /srv/admin_key ] || head -c 24 /dev/urandom | base64 | tr -d "/+=" > /srv/admin_key
docker run -d --name $NAME --label "sig=$SIG" --gpus "device=$GPU" --ipc=host \
  -v /srv/w:/w -v /srv/suite:/suite -p $HPORT:30000 -e SGLANG_ADMIN_API_KEY="$(cat /srv/admin_key)" \
  $IMAGE \
  sglang serve --model-path $MODEL --tp 1 --attention-backend fa4 --dtype bfloat16 --page-size 128 \
    --enable-multimodal --reasoning-parser inkling --tool-call-parser inkling2 \
    --host 0.0.0.0 --port 30000 $EXTRA >/dev/null
for i in $(seq 1 360); do
  if curl -sf localhost:$HPORT/health >/dev/null; then echo "UP after $((i*5))s: $SIG"; exit 0; fi
  if ! docker ps --format '{{.Names}}' | grep -qx "$NAME"; then echo "container died"; docker logs --tail 80 $NAME; exit 1; fi
  sleep 5
done
echo "timeout: $NAME"; docker logs --tail 80 $NAME; exit 1
