#!/bin/bash
# ON POD, GPUs 6-7 only. KL divergence of every GGUF vs the base BF16 GGUF on a fixed slice of wikitext-2 test.
# Usage: kld_all.sh base | kld_all.sh <tag>=<path> [...]
set -u
export CUDA_VISIBLE_DEVICES=6,7
P=/data/llama.cpp/build-m/bin/llama-perplexity
TXT=/data/work/wiki.test.raw
LOGITS=/data/work/base_bf16.kld
ARGS="-f $TXT -c 512 -b 512 -ub 16 --chunks 40"
mkdir -p /data/release/kld
if [ "$1" = "base" ]; then
  if [ ! -f $TXT ]; then
    curl -sL -o /data/work/wikitext-2-raw-v1.zip https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip
    python3 -c "import zipfile;z=zipfile.ZipFile('/data/work/wikitext-2-raw-v1.zip');open('$TXT','wb').write(z.read('wikitext-2-raw/wiki.test.raw'))"
  fi
  sha256sum $TXT > /data/release/kld/neutral_text.sha256
  $P -m /data/base/Inkling-2-Mini-BF16.gguf $ARGS --kl-divergence-base $LOGITS > /data/release/kld/base_bf16.log 2>&1
  echo "BASE rc=$? $(grep -E 'Final estimate' /data/release/kld/base_bf16.log)"
  exit 0
fi
for tp in "$@"; do
  tag=${tp%%=*}; path=${tp#*=}
  $P -m $path $ARGS --kl-divergence-base $LOGITS --kl-divergence > /data/release/kld/$tag.log 2>&1
  echo "$tag rc=$? $(grep -E '^Mean +KLD' /data/release/kld/$tag.log | head -1)"
done
echo KLD_DONE
