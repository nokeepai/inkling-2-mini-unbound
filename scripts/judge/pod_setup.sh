#!/usr/bin/env bash
# Runs ON the pod (ink-eval-2). Installs judge deps into system python and
# downloads base + LoRA weights to /srv/judge. HF_TOKEN is passed via env.
set -euo pipefail
mkdir -p /srv/judge /root/logs
pip install --break-system-packages -q "transformers>=4.44" "peft>=0.11" accelerate sentencepiece "huggingface_hub[cli]" 2>&1 | tail -n 5
python3 -c 'import transformers, peft, accelerate; print("deps", transformers.__version__, peft.__version__, accelerate.__version__)'
python3 - <<'EOF'
import os
from huggingface_hub import snapshot_download
tok = os.environ["HF_TOKEN"]
snapshot_download("google/gemma-2b", local_dir="/srv/judge/gemma-2b", token=tok,
                  allow_patterns=["*.json", "*.safetensors", "tokenizer.model"])
snapshot_download("qylu4156/strongreject-15k-v1", local_dir="/srv/judge/strongreject-15k-v1", token=tok)
print("WEIGHTS_DONE")
EOF
ls -la /srv/judge/gemma-2b /srv/judge/strongreject-15k-v1
