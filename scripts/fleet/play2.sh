#!/usr/bin/env bash
# Pod side, ink-play-1 (38 GB overlay): serve ONE merged model, never a copy of orig on the overlay.
# Build tensors reading orig from /root/w/orig (FUSE), merge straight into /srv/w/unbound, wait for the pinned image,
# serve on host port 30000 with the ENGINE.md flags, verify one chat completion. Log: /root/logs/play2.log
set -uo pipefail
exec >> /root/logs/play2.log 2>&1
CID=c1h_r1_svd1_s3_L4
D=/srv/cands/$CID
OUT=/srv/w/unbound
IMAGE=lmsysorg/sglang:inkling-2-mini
IMAGE_ID=aff54906ae2e
fail() { echo "PLAY_FAIL $*"; df -h / | tail -1; exit 1; }
echo "=== play2 $(date -u +%FT%TZ)"
python3 -c "import safetensors" 2>/dev/null || pip install -q --retries 5 safetensors==0.8.0 || fail "pip safetensors"
python3 -c "import torch" || fail "no torch"
# 1. build (CPU; tensors land next to the lowrank, 280 MB)
if [ "$(sha256sum $D/tensors.safetensors 2>/dev/null | cut -d' ' -f1)" != "$(cut -d' ' -f1 $D/sha256.txt)" ]; then
  python3 /srv/harness/build_cand.py $D /root/w/orig $D || fail "build_cand exit $? (3 = sha mismatch vs canonical)"
fi
[ "$(sha256sum $D/tensors.safetensors | cut -d' ' -f1)" = "$(cut -d' ' -f1 $D/sha256.txt)" ] || fail "built sha != canonical"
echo "[1] built $CID, sha $(cut -c1-16 $D/sha256.txt) matches canonical"
# 2. image first (it is the larger write), then check space before writing 19 GB
until [ "$(docker image inspect -f '{{.Id}}' lmsysorg/sglang@sha256:421aed3ac11741381eac350bf7b1db28a0cafb6ae49b57ae78fba86a16a32306 2>/dev/null | cut -c8-19)" = "$IMAGE_ID" ]; do
  pgrep -f "docker pull" >/dev/null || fail "image missing and no pull running"
  echo "[2] waiting for image pull ($(df -h / | tail -1 | awk '{print $4}') free)"; sleep 30
done
docker tag lmsysorg/sglang@sha256:421aed3ac11741381eac350bf7b1db28a0cafb6ae49b57ae78fba86a16a32306 $IMAGE
echo "[2] image ok; free $(df -h / | tail -1 | awk '{print $4}')"
# 3. merged model straight onto the overlay
if [ ! -f $OUT/.merged_ok ]; then
  FREE=$(df -BG / | tail -1 | awk '{print $4}' | tr -d G)
  [ "$FREE" -ge 20 ] || fail "only ${FREE}G free on / (need 20G for the merged model)"
  rm -rf $OUT
  python3 /srv/harness/sgl_merge.py $D $OUT --base /root/w/orig || fail "sgl_merge"
  touch $OUT/.merged_ok
fi
echo "[3] merged -> $OUT ($(du -sh $OUT | cut -f1)); free $(df -h / | tail -1 | awk '{print $4}')"
# 4. serve (parity's launcher; frozen flags)
mkdir -p /root/parity
bash /root/fleet/sglang_up.sh /w/unbound 30000 \
  --mem-fraction-static 0.73815 --chunked-prefill-size 4096 --max-running-requests 31 --random-seed 1234 || fail "sglang_up"
# 5. one chat
R=$(curl -s localhost:30000/v1/chat/completions -H 'content-type: application/json' -d '{"model":"default","messages":[{"role":"user","content":"In one sentence, who are you?"}],"max_tokens":300,"temperature":0,"chat_template_kwargs":{"reasoning_effort":0}}')
echo "chat: ${R:0:800}"
echo "$R" | grep -q '"content"' || fail "chat completion"
echo "PLAY_DONE $(date -u +%FT%TZ)"
