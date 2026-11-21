#!/bin/bash
# ON POD. Rebuild llama.cpp at origin/master (has #32704 loop-slot fix, not yet in a tag), build our 20 tensors,
# patch the BF16 GGUF and verify.
set -u
cd /data/llama.cpp
git checkout -q 19bc41dbb
{ git rev-parse HEAD; echo "origin/master (after b12608; includes #32704 690b87d88 and #32669)"; git log -1 --format='%H %cI %s'; } > /data/release/llama_cpp_commit.txt
cat /data/release/llama_cpp_commit.txt
cmake --build build -j 200 --target llama-cli llama-server llama-perplexity llama-quantize llama-completion > /data/logs/build2.log 2>&1
echo "BUILD2 rc=$?"
pip install -q -e /data/llama.cpp/gguf-py
python3 -c 'import gguf;print("gguf-py", gguf.__file__)'
until grep -q DL_BASE_DONE /data/logs/dl_base.log && [ -f /data/orig/model-00001-of-00001.safetensors ] && ! pgrep -f "hf download thinkingmachines" >/dev/null; do sleep 20; done
grep DL_BASE_DONE /data/logs/dl_base.log
ls -la /data/base /data/orig
python3 /root/gguf/build_cand.py /root/gguf/cand /data/orig /root/gguf/candbuild
echo "BUILD_CAND rc=$?"
python3 /root/gguf/make_map.py
python3 /root/gguf/patch_gguf.py /data/base/Inkling-2-Mini-BF16.gguf /data/orig /root/gguf/candbuild/tensors.safetensors /root/gguf/name_map.json /data/release/Inkling-2-Mini-Unbound-BF16.gguf
echo "PATCH rc=$?"
mkdir -p /data/release/proof
python3 /root/gguf/verify_patch.py /data/base/Inkling-2-Mini-BF16.gguf /data/release/Inkling-2-Mini-Unbound-BF16.gguf /root/gguf/candbuild/tensors.safetensors /root/gguf/name_map.json /data/release/proof
echo "STEP1_DONE rc=$?"
