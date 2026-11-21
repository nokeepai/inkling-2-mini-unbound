#!/bin/bash
# Assemble R/out/final/ from the winner's small artifacts (lead 2026-11-21: no multi-GB files on the Mac).
# The full baked checkpoint stays on the pod at /root/pkg/out/<cand> until the owner decides.
# Usage: assemble_final.sh <pod> <cand_id>   (run after run_bake.sh on the pod; then run make_report.py)
set -eu
P=${1:?pod}; C=${2:?cand_id}
R=$HOME/Code/inkling-abl; F=$R/out/final
bash $R/scripts/package/fetch_small.sh "$P" "$C"
S=$R/package/$C
mkdir -p $F/dora $F/configs
cp $S/edit.pt $S/exactness.json $S/peft_verify.json $F/
cp $S/$C.MANIFEST.json $F/MANIFEST.json
cp $S/$C.MANIFEST.md $F/MANIFEST.md
cp $S/$C.bake_log.json $F/bake_log.json
cp $S/dora/adapter_model.safetensors $S/dora/adapter_config.json $F/dora/
cp $R/scripts/package/apply_edit.py $F/
for f in config.json generation_config.json hf_quant_config.json model.safetensors.index.json chat_template.jinja \
         processor_config.json special_tokens_map.json tokenizer_config.json tokenizer.json; do
  cp $R/vendor/ckpt/orig/$f $F/configs/
done
cp $R/package/MODEL_CARD_DRAFT.md $F/MODEL_CARD.md
du -sh $F; ls -la $F
echo "checkpoint stays on $P:/root/pkg/out/$C (not copied)"
