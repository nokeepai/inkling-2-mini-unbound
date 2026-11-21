#!/bin/bash
# Runs on a pod: fetch both checkpoints at pinned revisions and record file hashes.
set -euo pipefail
mkdir -p /root/w /root/logs
pip install -q -U "huggingface_hub[hf_transfer]" >/dev/null 2>&1 || pip install -q -U huggingface_hub
export HF_HUB_ENABLE_HF_TRANSFER=1
hf download thinkingmachines/Inkling-2-Mini --revision ed40e7da625f594fc9c4990f5f66be24873c22d7 --local-dir /root/w/orig
hf download OpenUncensored/Inkling-2-Mini-Uncensored --revision 4e5f06d72b5a70c9dcd5e8046aea576eb3ee8487 --local-dir /root/w/ou
cd /root/w
find orig ou -type f ! -path '*/.cache/*' -print0 | sort -z | xargs -0 sha256sum > /root/w/hashes.txt
echo FETCH_DONE
