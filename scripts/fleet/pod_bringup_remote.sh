#!/usr/bin/env bash
# Pod side of pod_bringup.sh (agent `fleet`). Idempotent; every step checks before acting. Log: /root/logs/bringup.log
# Ends with a line BRINGUP_DONE or BRINGUP_FAIL <reason>.
set -uo pipefail
mkdir -p /root/logs /srv/w /srv/fleet /srv/harness/cap /srv/data/val /srv/evals /srv/cands
exec >> /root/logs/bringup.log 2>&1
echo "=== bringup $(date -u +%FT%TZ)"
fail() { echo "BRINGUP_FAIL $*"; exit 1; }
IMAGE=lmsysorg/sglang:inkling-2-mini
INDEX_DIGEST=sha256:efac853b46811ccb760545b7b29e7119bdf907537ccf6e863fac3011fe8b98de
IMAGE_ID=aff54906ae2e
PORT=${FLEET_PORT:-3001}            # single-GPU pods (parity's convention)
BASE=${FLEET_BASE_PORT:-$PORT}      # multi-GPU pods: GPU i serves on BASE+i (first ports of the pod's Lium port map)

# 1. orig weights only. Reference hashes: /root/fleet/orig_hashes.txt (orig lines of another pod's hashes.txt, pushed by
#    pod_bringup.sh), so we start as soon as orig is complete, without waiting for pod_fetch.sh's OU download/FETCH_DONE.
REF=/root/fleet/orig_hashes.txt
[ -s $REF ] || fail "no reference hashes $REF"
orig_present() { while read -r _ f; do [ -f "/root/w/$f" ] || return 1; done < $REF; }
if ! orig_present; then
  if ! pgrep -f "hf download|pod_fetch.sh" >/dev/null; then
    echo "[1] no fetch running; starting pod_fetch.sh"
    nohup bash /root/fleet/pod_fetch.sh > /root/fetch.log 2>&1 < /dev/null &
  fi
  until orig_present; do echo "[1] waiting for /root/w/orig ($(du -sh /root/w/orig 2>/dev/null | cut -f1))"; sleep 30; done
fi
echo "[1] /root/w/orig complete (all $(wc -l < $REF) files present)"

# 2. copy orig to /srv/w (docker cannot bind-mount the /root FUSE mount); verify every file against the reference
verify() { ( cd "$1" && sed 's# orig/# ./#' $REF | sha256sum -c --quiet - ); }
if [ ! -f /srv/w/orig/.fleet_copied ]; then
  if [ -f /srv/w/orig/config.json ] && verify /srv/w/orig; then
    echo "[2] existing /srv/w/orig verified"
  else
    echo "[2] copying /root/w/orig -> /srv/w/orig"
    rm -rf /srv/w/orig.tmp
    cp -r /root/w/orig /srv/w/orig.tmp || fail "copy failed"
    verify /srv/w/orig.tmp || fail "orig copy hash mismatch"
    docker rm -f sgl >/dev/null 2>&1 || true
    rm -rf /srv/w/orig && mv /srv/w/orig.tmp /srv/w/orig
  fi
  touch /srv/w/orig/.fleet_copied
fi
echo "[2] /srv/w/orig ok ($(du -sh /srv/w/orig | cut -f1))"

# 3. pinned image
if [ "$(docker image inspect -f '{{.Id}}' $IMAGE 2>/dev/null | cut -c8-19)" != "$IMAGE_ID" ]; then
  echo "[3] pulling $IMAGE@$INDEX_DIGEST"
  docker pull "lmsysorg/sglang@$INDEX_DIGEST" || fail "docker pull"
  docker tag "lmsysorg/sglang@$INDEX_DIGEST" $IMAGE
fi
[ "$(docker image inspect -f '{{.Id}}' $IMAGE | cut -c8-19)" = "$IMAGE_ID" ] || fail "image id != $IMAGE_ID"
echo "[3] image ok: $(docker image inspect -f '{{.Id}} {{index .RepoDigests 0}}' $IMAGE)"

# 4. host python deps for the scorer (tokenizer + safetensors merge)
if ! python3 -c "import torch, transformers, safetensors" 2>/dev/null; then
  echo "[4] installing python deps"
  # archive URL (one HTTP GET) instead of git clone: the clone failed mid-transfer from HK (13:00Z)
  pip install -q --retries 5 "https://github.com/huggingface/transformers/archive/a10ebc3c1c44b01d0ee825d48b8d0de3253e4539.zip" \
    safetensors==0.8.0 sentencepiece protobuf || fail "pip install"
  python3 -c "import torch" 2>/dev/null || pip install -q torch || fail "torch install"
fi
echo "[4] python ok: $(python3 -c 'import torch,transformers;print(torch.__version__, transformers.__version__)')"

if [ -n "${FLEET_PREP_ONLY:-}" ]; then echo "BRINGUP_DONE prep-only $(date -u +%FT%TZ)"; exit 0; fi

# 5. admin key + frozen SGLang server (ENGINE.md flags; sglang_up.sh is idempotent)
[ -f /srv/admin_key ] || head -c 24 /dev/urandom | base64 | tr -d "/+=" > /srv/admin_key
mkdir -p /srv/parity
NGPU=$(nvidia-smi -L | grep -c '^GPU ')
echo "[5] $NGPU GPU(s): $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | sort | uniq -c | tr '\n' ' ')"
# Single-GPU pods: parity's launcher (container sgl, port $PORT). Multi-GPU: container sgl<i> on GPU i, port PORT+i.
if [ "$NGPU" = 1 ]; then
  mkdir -p /root/parity && cp /root/fleet/sglang_up.sh /root/parity/sglang_up.sh
  bash /root/parity/sglang_up.sh /w/orig $PORT \
    --mem-fraction-static 0.73815 --chunked-prefill-size 4096 --max-running-requests 31 --random-seed 1234 \
    || fail "sglang_up.sh"
  GPUS=("-")
  echo "- $PORT" > /srv/fleet/gpus.txt
else
  GPUS=()
  : > /srv/fleet/gpus.txt.tmp
  for i in $(seq 0 $((NGPU - 1))); do
    bash /root/fleet/sglang_up_gpu.sh $i $((BASE + i)) > /root/logs/sglang_up_gpu$i.log 2>&1 &
    GPUS+=("$i")
    echo "$i $((BASE + i))" >> /srv/fleet/gpus.txt.tmp
  done
  wait
  mv /srv/fleet/gpus.txt.tmp /srv/fleet/gpus.txt
  for i in "${GPUS[@]}"; do echo "  gpu$i: $(tail -n 1 /root/logs/sglang_up_gpu$i.log)"; done
fi

# 6. per server: health, pinned orig probe baseline, and a short greedy reply
cd /srv/harness || fail "no /srv/harness (Mac side pushes it first)"
for g in "${GPUS[@]}"; do
  if [ "$g" = "-" ]; then P=$PORT; GA="--port $P"; else P=$((BASE + g)); GA="--gpu $g"; fi
  curl -sf localhost:$P/health >/dev/null || fail "health :$P"
  python3 fleet_eval.py --probe-only $GA || fail "probe-only gpu $g"
  python3 - <<EOF || fail "probe generate gpu $g"
import sys; sys.path.insert(0, "/srv/harness"); import sgl_eval as S
S.PORT = $P
r = S.capped(S.render(S.PROBE, 0), 256, 128)
print("probe answer:", repr(r["answer"][:120]))
assert len(r["answer"].strip()) > 20, "empty probe answer"
EOF
done
echo "BRINGUP_DONE $(date -u +%FT%TZ) gpus=${GPUS[*]}"
