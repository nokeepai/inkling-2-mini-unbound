#!/usr/bin/env bash
# Set up the HarmBench judge worker on a research pod (default ink-research-2b). Leaves the pod's transformers alone
# (fidelity's pin): installs only bitsandbytes --no-deps. Downloads HB weights to /srv/judge2/hb13.
set -euo pipefail
POD=${1:-ink-research-2b}
R=/Users/nikshepsvn/Code/inkling-abl
W="bash $R/scripts/ink_lium.sh"
$W exec $POD "mkdir -p /srv/judge/harness /srv/judge2 /srv/judge_q/in /srv/judge_q/out /root/logs"
for f in judge.py judge_alt.py judge_worker.py; do $W scp $POD $R/harness/$f /srv/judge/harness/$f; done
cat > /private/tmp/judge_pod_setup.sh <<'EOF'
set -e
pip install -q --no-deps "bitsandbytes==0.50.2" 2>&1 | tail -n 1
python3 - <<'PY'
import os
from huggingface_hub import snapshot_download
snapshot_download("cais/HarmBench-Llama-2-13b-cls", local_dir="/srv/judge2/hb13", token=os.environ.get("HF_TOKEN"),
                  allow_patterns=["*.json", "*.safetensors", "tokenizer.model"])
print("HB_DONE", flush=True)
PY
python3 -c 'import transformers, bitsandbytes; print("deps", transformers.__version__, bitsandbytes.__version__)'
EOF
$W scp $POD /private/tmp/judge_pod_setup.sh /srv/judge/setup_hb.sh
$W exec $POD -e "HF_TOKEN=$(cat "$HOME/.cache/huggingface/token")" "bash /srv/judge/setup_hb.sh > /root/logs/judge_setup.log 2>&1; tail -n 3 /root/logs/judge_setup.log"
