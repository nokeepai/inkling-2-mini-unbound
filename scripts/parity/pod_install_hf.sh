#!/usr/bin/env bash
# Install pinned Transformers (PR #51050) into the research pod's python. Idempotent.
set -euo pipefail
mkdir -p /root/logs /root/parity
COMMIT=a10ebc3c1c44b01d0ee825d48b8d0de3253e4539
have=$(python -c "import transformers,os;print(getattr(transformers,'__version__',''))" 2>/dev/null || true)
pip install -q "git+https://github.com/huggingface/transformers@${COMMIT}" accelerate datasets safetensors sentencepiece protobuf
# keep the source tree too, to read the PR's docs and modeling code
if [ ! -d /root/tf-src ]; then
  git clone -q --filter=blob:none https://github.com/huggingface/transformers /root/tf-src
  git -C /root/tf-src checkout -q ${COMMIT}
fi
pip freeze > /root/parity/pod_freeze_research.txt
python -c "import transformers;print('TF', transformers.__version__, transformers.__file__)"
echo INSTALL_DONE
