#!/usr/bin/env bash
# Local: set up the judge worker on ink-research-2 without touching fidelity's pinned transformers.
set -euo pipefail
export LIUM_NONINTERACTIVE=1
POD=ink-research-2
R=/Users/nikshepsvn/Code/inkling-abl
lium exec $POD "mkdir -p /srv/judge/harness /srv/judge_q/in /srv/judge_q/out /root/logs"
lium scp $POD $R/harness/judge.py /srv/judge/harness/judge.py
lium scp $POD $R/harness/judge_worker.py /srv/judge/harness/judge_worker.py
cat > /private/tmp/judge_r2_pod.sh <<'EOF'
set -e
pip install -q --no-deps "peft>=0.11" 2>&1 | tail -n 2
python3 -c 'import transformers, peft; print("deps", transformers.__version__, peft.__version__)'
python3 - <<'PY'
import os
from huggingface_hub import snapshot_download
tok = os.environ["HF_TOKEN"]
snapshot_download("google/gemma-2b", local_dir="/srv/judge/gemma-2b", token=tok,
                  allow_patterns=["*.json", "*.safetensors", "tokenizer.model"])
snapshot_download("qylu4156/strongreject-15k-v1", local_dir="/srv/judge/strongreject-15k-v1", token=tok)
print("WEIGHTS_DONE")
PY
EOF
lium scp $POD /private/tmp/judge_r2_pod.sh /srv/judge/setup.sh
HF_TOKEN_VAL="$(cat "$HOME/.cache/huggingface/token")"
lium exec $POD -e "HF_TOKEN=$HF_TOKEN_VAL" "bash /srv/judge/setup.sh > /root/logs/judge_setup.log 2>&1; tail -n 4 /root/logs/judge_setup.log"
