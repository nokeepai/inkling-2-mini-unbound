#!/usr/bin/env bash
# Install pinned transformers (PR #51050) + deps at the versions in methods/parity/pod_freeze_research.txt.
set -euo pipefail
mkdir -p /root/logs /root/fid /root/parity
pip install -q "git+https://github.com/huggingface/transformers@a10ebc3c1c44b01d0ee825d48b8d0de3253e4539" \
  accelerate==1.16.0 datasets==5.2.0 safetensors==0.8.0 sentencepiece==0.2.3 protobuf==7.36.2 \
  tokenizers==0.23.3 huggingface_hub==2.4.0 numpy==2.5.2
python3 -c "import transformers,torch;print('TF', transformers.__version__, torch.__version__)"
pip freeze > /root/fid/pod_freeze.txt
echo INSTALL_DONE
