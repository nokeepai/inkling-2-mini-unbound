#!/usr/bin/env bash
# Pod-side SGLang launcher for Inkling-2-Mini (image lmsysorg/sglang:inkling-2-mini, pinned digest
# amd64 sha256:421aed3a...). Idempotent: if container `sgl` already runs with the same settings and
# /health answers, it does nothing; otherwise it (re)starts it and waits for /health.
#
# Usage (on the pod):  bash /root/parity/sglang_up.sh [model=/w/orig] [host_port=41000] [extra flags...]
#   extra flags ONLY for memory on small GPUs (e.g. --mem-fraction-static 0.85 --context-length 32768)
#   INK_DUMP_DIR=/parity/dump bash sglang_up.sh ...   -> debug mode: per-visit residual dumps on prefill
#                                                        (scripts/parity/sgl_dump/sitecustomize.py)
# IMPORTANT: /root on Lium pods is a gocryptfs FUSE mount that the DinD (sysbox) daemon cannot
#   bind-mount. Weights must be copied to /srv/w (overlay root) first; this script does it if missing.
#   Inside the container: /srv/w -> /w, /srv/parity -> /parity.
# Stop:                docker rm -f sgl
# Query from the pod:  curl -s localhost:41000/generate -H 'content-type: application/json' \
#                        -d '{"input_ids":[...],"sampling_params":{"temperature":0,"max_new_tokens":64}}'
#   OpenAI API:        curl -s localhost:41000/v1/chat/completions -H 'content-type: application/json' \
#                        -d '{"model":"default","messages":[{"role":"user","content":"hi"}],
#                             "chat_template_kwargs":{"reasoning_effort":0}}'
# Logs:                docker logs --tail 50 sgl
set -euo pipefail
MODEL=${1:-/w/orig}
HPORT=${2:-41000}
shift $(( $# > 2 ? 2 : $# )) || true
EXTRA="$*"
IMAGE=lmsysorg/sglang:inkling-2-mini
NAME=sgl
DUMP=${INK_DUMP_DIR:-}
SIG="$MODEL|$HPORT|$EXTRA|$DUMP"
mkdir -p /root/logs /srv/w /srv/parity/sgl_dump
SRC_NAME=$(basename "$MODEL")
if [ ! -f "/srv/w/$SRC_NAME/config.json" ]; then
  echo "copying /root/w/$SRC_NAME -> /srv/w/$SRC_NAME"; cp -r "/root/w/$SRC_NAME" "/srv/w/$SRC_NAME"
fi
[ -f /root/parity/sgl_dump/sitecustomize.py ] && cp /root/parity/sgl_dump/sitecustomize.py /srv/parity/sgl_dump/

if docker ps --format '{{.Names}}' | grep -qx "$NAME"; then
  if [ "$(docker inspect -f '{{ index .Config.Labels "sig" }}' $NAME)" = "$SIG" ] && curl -sf localhost:$HPORT/health >/dev/null; then
    echo "already up: $SIG"; exit 0
  fi
fi
docker rm -f $NAME >/dev/null 2>&1 || true

[ -f /srv/admin_key ] || head -c 24 /dev/urandom | base64 | tr -d "/+=" > /srv/admin_key
DUMP_ARGS=(-e SGLANG_ADMIN_API_KEY="$(cat /srv/admin_key)")
if [ -n "$DUMP" ]; then DUMP_ARGS+=(-e INK_DUMP_DIR="$DUMP" -e PYTHONPATH=/parity/sgl_dump); fi

docker run -d --name $NAME --label "sig=$SIG" --gpus all --ipc=host \
  -v /srv/w:/w -v /srv/parity:/parity -p $HPORT:30000 "${DUMP_ARGS[@]}" \
  $IMAGE \
  sglang serve --model-path $MODEL --tp 1 --attention-backend fa4 --dtype bfloat16 --page-size 128 \
    --enable-multimodal --reasoning-parser inkling --tool-call-parser inkling2 \
    --host 0.0.0.0 --port 30000 $EXTRA >/dev/null

echo "image: $(docker inspect -f '{{index .RepoDigests 0}}' $IMAGE)"
for i in $(seq 1 240); do
  if curl -sf localhost:$HPORT/health >/dev/null; then echo "UP after $((i*5))s: $SIG"; exit 0; fi
  if ! docker ps --format '{{.Names}}' | grep -qx "$NAME"; then echo "container died"; docker logs --tail 80 $NAME; exit 1; fi
  sleep 5
done
echo "timeout waiting for health"; docker logs --tail 80 $NAME; exit 1
