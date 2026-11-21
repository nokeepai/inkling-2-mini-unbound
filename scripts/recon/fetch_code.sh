#!/bin/bash
# Fetch inkling model code from transformers at the pinned PR commit, plus checkpoint small files.
set -u
R=/Users/nikshepsvn/Code/inkling-abl
C=a10ebc3c1c44b01d0ee825d48b8d0de3253e4539
D=$R/vendor/transformers-pr51050
BASE=https://raw.githubusercontent.com/huggingface/transformers/$C
FILES="
docs/source/en/model_doc/inkling2.md
docs/source/en/model_doc/inkling2_loop.md
src/transformers/conversion_mapping.py
src/transformers/core_model_loading.py
src/transformers/quantizers/quantizer_finegrained_fp8.py
src/transformers/models/inkling2_loop/__init__.py
src/transformers/models/inkling2_loop/configuration_inkling2_loop.py
src/transformers/models/inkling2_loop/modeling_inkling2_loop.py
src/transformers/models/inkling2_loop/modular_inkling2_loop.py
tests/models/inkling2_loop/test_modeling_inkling2_loop.py
"
for f in $FILES; do mkdir -p "$D/$(dirname $f)"; curl -sfL "$BASE/$f" -o "$D/$f" || echo "FAIL $f"; done
# whole inkling2 folder
for f in $(gh api "repos/huggingface/transformers/contents/src/transformers/models/inkling2?ref=$C" --jq '.[].name'); do
  mkdir -p $D/src/transformers/models/inkling2
  curl -sfL "$BASE/src/transformers/models/inkling2/$f" -o "$D/src/transformers/models/inkling2/$f" || echo "FAIL inkling2/$f"
done
# also quantizers that might be used for nvfp4
for f in $(gh api "repos/huggingface/transformers/contents/src/transformers/quantizers?ref=$C" --jq '.[].name' | grep -iE 'fp4|modelopt|mxfp|compressed|auto'); do
  curl -sfL "$BASE/src/transformers/quantizers/$f" -o "$D/src/transformers/quantizers/$f" || echo "FAIL q/$f"
done
echo CODE_DONE
# checkpoints
HF=https://huggingface.co
for pair in "orig thinkingmachines/Inkling-2-Mini ed40e7da625f594fc9c4990f5f66be24873c22d7" "ou OpenUncensored/Inkling-2-Mini-Uncensored 4e5f06d72b5a70c9dcd5e8046aea576eb3ee8487"; do
  set -- $pair
  for f in config.json generation_config.json hf_quant_config.json model.safetensors.index.json chat_template.jinja tokenizer_config.json special_tokens_map.json README.md preprocessor_config.json processor_config.json; do
    curl -sfL "$HF/$2/resolve/$3/$f" -o "$R/vendor/ckpt/$1/$f" || { echo "MISS $1/$f"; rm -f "$R/vendor/ckpt/$1/$f"; }
  done
done
curl -sfL "$HF/OpenUncensored/Inkling-2-Mini-Uncensored/resolve/4e5f06d72b5a70c9dcd5e8046aea576eb3ee8487/edit.pt" -o $R/vendor/ckpt/ou/edit.pt || echo "MISS edit.pt"
curl -sfL "$HF/api/models/OpenUncensored/Inkling-2-Mini-Uncensored/tree/4e5f06d72b5a70c9dcd5e8046aea576eb3ee8487?recursive=1" -o $R/vendor/ckpt/ou/tree.json
curl -sfL "$HF/api/models/thinkingmachines/Inkling-2-Mini/tree/ed40e7da625f594fc9c4990f5f66be24873c22d7?recursive=1" -o $R/vendor/ckpt/orig/tree.json
echo ALL_DONE
