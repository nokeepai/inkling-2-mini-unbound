#!/bin/bash
# Finalise the release dir on the pod (no uploads anywhere else):
#   README.md      <- R/out/final/MODEL_CARD.md
#   charts/*.png   <- R/package/charts/*.png
#   abliteration/  <- edit.pt, apply_edit.py, dora/ (from the pod's export), then MANIFEST.json/.md
# Re-runs the manifest (edited tensors + README + added files) and refreshes R/out/final/. Rerun after every card edit.
set -eu
P=${1:?pod}; C=${2:?cand_id}
R=$HOME/Code/inkling-abl; L="bash $R/scripts/ink_lium.sh"; D=/root/pkg/out/$C
if grep -q "{{" $R/out/final/MODEL_CARD.md; then echo "WARNING: card still has {{placeholders}}" >&2; fi
$L exec "$P" "mkdir -p $D/charts $D/abliteration/dora"
$L scp "$P" $R/out/final/MODEL_CARD.md $D/README.md >/dev/null
for f in $R/package/charts/*.png; do $L scp "$P" "$f" $D/charts/ >/dev/null; done
mkdir -p $R/out/final/charts; cp $R/package/charts/*.png $R/out/final/charts/
$L exec "$P" "cp $D.export/edit.pt $D/abliteration/ && cp /root/pkg/scripts/apply_edit.py $D/abliteration/ && cp $D.export/dora/adapter_model.safetensors $D.export/dora/adapter_config.json $D/abliteration/dora/ && cd /root/pkg/scripts && nice -n 10 python3 manifest.py /root/w/orig $D $D.MANIFEST --expect $D.bake_log.json --workers 4 && cp $D.MANIFEST.json $D/abliteration/MANIFEST.json && cp $D.MANIFEST.md $D/abliteration/MANIFEST.md && sha256sum $D/README.md && ls -la $D/charts $D/abliteration $D/abliteration/dora"
$L scp "$P" $D.MANIFEST.json $R/out/final/MANIFEST.json -d >/dev/null
$L scp "$P" $D.MANIFEST.md $R/out/final/MANIFEST.md -d >/dev/null
shasum -a 256 $R/out/final/MODEL_CARD.md
