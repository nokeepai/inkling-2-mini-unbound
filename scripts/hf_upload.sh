#!/bin/bash
# Owner-approved (16:35Z) PRIVATE upload of the baked winner to flashvenom/Inkling-2-Mini-Unbound.
# The token never appears on a command line or in logs: it is copied to the pod's tmpfs (/dev/shm, mode 600),
# read into the upload process's environment, and deleted right after.
# Usage: hf_upload.sh <cand_id>
set -euo pipefail
R=/Users/nikshepsvn/Code/inkling-abl
POD=ink-research-2b
REPO=flashvenom/Inkling-2-Mini-Unbound
CAND="$1"
TOKFILE=$(mktemp)
chmod 600 "$TOKFILE"
/opt/homebrew/opt/python@3.13/bin/python3.13 -c "from huggingface_hub import get_token; t=get_token(); assert t; open('$TOKFILE','w').write(t)"
# 1. create the repo as PRIVATE from the Mac (no data leaves the Mac except the repo name)
/opt/homebrew/opt/python@3.13/bin/python3.13 -c "from huggingface_hub import create_repo; create_repo('$REPO', private=True, repo_type='model', exist_ok=True); print('repo ready (private)')"
/opt/homebrew/opt/python@3.13/bin/python3.13 -c "from huggingface_hub import HfApi; i=HfApi().repo_info('$REPO'); assert i.private, 'repo is not private'; print('verified private')"
# 2. ship the token to tmpfs on the pod, upload, delete
bash $R/scripts/ink_lium.sh scp $POD "$TOKFILE" /dev/shm/hft > /dev/null
rm -f "$TOKFILE"
bash $R/scripts/ink_lium.sh exec $POD "chmod 600 /dev/shm/hft; cd /root/pkg/out/$CAND && HF_HUB_ENABLE_HF_TRANSFER=1 HF_TOKEN=\$(cat /dev/shm/hft) hf upload $REPO . . --repo-type model > /root/logs/hf_upload.log 2>&1; echo UPLOAD_EXIT=\$?; rm -f /dev/shm/hft; tail -3 /root/logs/hf_upload.log | grep -v -i token"
bash $R/scripts/ink_lium.sh exec $POD "ls /dev/shm/hft 2>&1 | head -1"
/opt/homebrew/opt/python@3.13/bin/python3.13 -c "from huggingface_hub import HfApi; i=HfApi().repo_info('$REPO', files_metadata=True); print('private:', i.private, 'files:', len(i.siblings), 'GB:', round(sum((s.size or 0) for s in i.siblings)/1e9,2))"
