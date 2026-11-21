#!/usr/bin/env bash
# Pod side bring-up for the suite (agent `suite`). Log /root/logs/suite_bringup.log, ends SUITE_BRINGUP_DONE/FAIL.
# Steps: python deps (background) ; orig weights -> /srv/w/orig (hash-verified) ; pinned image ; merged_ou. No servers.
set -uo pipefail
mkdir -p /root/logs /srv/w /srv/suite /srv/cands
exec >> /root/logs/suite_bringup.log 2>&1
echo "=== suite bringup $(date -u +%FT%TZ)"
fail() { echo "SUITE_BRINGUP_FAIL $*"; exit 1; }
IMAGE=lmsysorg/sglang:inkling-2-mini
INDEX_DIGEST=sha256:efac853b46811ccb760545b7b29e7119bdf907537ccf6e863fac3011fe8b98de
AMD_DIGEST=sha256:421aed3ac11741381eac350bf7b1db28a0cafb6ae49b57ae78fba86a16a32306
IMAGE_ID=aff54906ae2e
REF=/root/suite/orig_hashes.txt
(
  pip install -q --retries 5 "https://github.com/huggingface/transformers/archive/a10ebc3c1c44b01d0ee825d48b8d0de3253e4539.zip" \
    safetensors==0.8.0 sentencepiece protobuf > /root/logs/suite_pip.log 2>&1
  pip install -q --retries 5 "lm-eval[ifeval]==0.4.9.1" evalplus==0.3.1 "math-verify[antlr4_13_2]==0.8.0" \
    pandas pyarrow pillow langdetect immutabledict nltk >> /root/logs/suite_pip.log 2>&1
  python3 -c "import torch" 2>/dev/null || pip install -q torch >> /root/logs/suite_pip.log 2>&1
  python3 -c "import nltk; nltk.download('punkt_tab', quiet=True); nltk.download('punkt', quiet=True)" >> /root/logs/suite_pip.log 2>&1
  echo PIP_DONE >> /root/logs/suite_pip.log
) &
PIPPID=$!
orig_present() { while read -r _ f; do [ -f "/root/w/$f" ] || return 1; done < $REF; }
until orig_present; do echo "[1] waiting for /root/w/orig ($(du -sh /root/w/orig 2>/dev/null | cut -f1))"; sleep 30; done
echo "[1] orig present"
verify() { ( cd "$1" && sed 's# orig/# ./#' $REF | sha256sum -c --quiet - ); }
if [ ! -f /srv/w/orig/.fleet_copied ]; then
  rm -rf /srv/w/orig.tmp; cp -r /root/w/orig /srv/w/orig.tmp || fail copy
  verify /srv/w/orig.tmp || fail "orig hash mismatch"
  rm -rf /srv/w/orig; mv /srv/w/orig.tmp /srv/w/orig; touch /srv/w/orig/.fleet_copied
fi
echo "[2] /srv/w/orig ok"
until grep -q 'Status: \|Digest:' /root/pull.log 2>/dev/null || ! pgrep -f "docker pull" >/dev/null; do echo "[3] waiting for pull"; sleep 20; done
if [ "$(docker image inspect -f '{{.Id}}' $IMAGE 2>/dev/null | cut -c8-19)" != "$IMAGE_ID" ]; then
  docker pull "lmsysorg/sglang@$AMD_DIGEST" || fail pull
  docker tag "lmsysorg/sglang@$AMD_DIGEST" $IMAGE
fi
[ "$(docker image inspect -f '{{.Id}}' $IMAGE | cut -c8-19)" = "$IMAGE_ID" ] || fail "image id mismatch: $(docker image inspect -f '{{.Id}}' $IMAGE)"
echo "[3] image ok"
wait $PIPPID
grep -q PIP_DONE /root/logs/suite_pip.log || fail pip
python3 -c "import transformers, lm_eval, evalplus, math_verify; print('[4]', transformers.__version__, lm_eval.__version__)" || fail "imports"
if [ ! -f /srv/w/merged_ou/.done ]; then
  python3 /srv/suite/sgl_merge.py /srv/cands/ou /srv/w/merged_ou || fail "merge ou"
  touch /srv/w/merged_ou/.done
fi
echo "[5] merged_ou ok"
echo "SUITE_BRINGUP_DONE $(date -u +%FT%TZ)"
