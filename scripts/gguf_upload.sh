#!/bin/bash
# Owner-approved (18:55Z) upload of the GGUF release to a NEW public repo nokeepai/Inkling-2-Mini-Unbound-GGUF.
# Org token: copied to the pod's tmpfs only, used by the upload process, deleted after.
set -euo pipefail
R=/Users/nikshepsvn/Code/inkling-abl
L="bash $R/scripts/ink_lium.sh"
POD=ink-gguf-1
REPO=nokeepai/Inkling-2-Mini-Unbound-GGUF
: "${HF_ORG_TOKEN:?set HF_ORG_TOKEN}"
TOK=$(mktemp); chmod 600 "$TOK"; printf '%s' "$HF_ORG_TOKEN" > "$TOK"
$L scp $POD "$TOK" /dev/shm/hft > /dev/null; rm -f "$TOK"
$L scp $POD "$R/package/gguf/README.md" /data/release/README.md > /dev/null
cat > /tmp/gguf_up_pod.sh <<'EOF'
#!/bin/bash
set -e
chmod 600 /dev/shm/hft; export HF_TOKEN=$(cat /dev/shm/hft); export HF_HUB_ENABLE_HF_TRANSFER=1
cd /data/release
# vision projector from the official repo (unchanged)
for f in mmproj-Inkling-2-Mini-BF16.gguf mmproj-Inkling-2-Mini-Q8_0.gguf; do
  [ -f "$f" ] || hf download ggml-org/Inkling-2-Mini-GGUF "$f" --local-dir . > /dev/null
done
python3 -c "from huggingface_hub import create_repo; create_repo('nokeepai/Inkling-2-Mini-Unbound-GGUF', repo_type='model', private=False, exist_ok=True); print('repo ready')"
mkdir -p /data/up && rm -rf /data/up/*
for f in *.gguf README.md MANIFEST.tsv; do ln -f "$f" /data/up/"$f"; done
python3 -c "from huggingface_hub import HfApi; HfApi().upload_folder(repo_id='nokeepai/Inkling-2-Mini-Unbound-GGUF', folder_path='/data/up', repo_type='model', commit_message='Inkling-2-Mini-Unbound GGUF v1.0')" > /root/logs/gguf_upload.log 2>&1 && echo UPLOAD_OK || { echo UPLOAD_FAIL; tail -5 /root/logs/gguf_upload.log; }
rm -f /dev/shm/hft
echo GGUF_UPLOAD_DONE
EOF
$L scp $POD /tmp/gguf_up_pod.sh /root/gguf_up_pod.sh > /dev/null
$L exec $POD "nohup bash /root/gguf_up_pod.sh > /root/logs/gguf_up_driver.log 2>&1 &"
echo started
