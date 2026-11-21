#!/bin/bash
# Pull a candidate's small packaging artifacts (not the 20 GB checkpoint, not the full-rank LoRA) to R/package/<cand>/.
P=${1:?pod}; C=${2:?cand_id}
L="bash $HOME/Code/inkling-abl/scripts/ink_lium.sh"
D=$HOME/Code/inkling-abl/package/$C; mkdir -p $D/dora
for f in export/edit.pt export/peft_verify.json export/exactness.json MANIFEST.json MANIFEST.md bake_log.json; do
  case $f in export/*) src=/root/pkg/out/$C.export/${f#export/} ;; *) src=/root/pkg/out/$C.$f ;; esac
  $L scp "$P" "$src" "$D/" -d >/dev/null 2>&1 || echo "failed $src"
done
for f in adapter_model.safetensors adapter_config.json; do
  $L scp "$P" /root/pkg/out/$C.export/dora/$f "$D/dora/" -d >/dev/null 2>&1 || echo "failed dora/$f"
done
ls -la $D $D/dora
