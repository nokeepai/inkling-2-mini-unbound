#!/bin/bash
# On-pod: bake <cand_id> from /root/w/orig, build its manifest, then export edit.pt/LoRA/DoRA. CPU+disk only, low priority.
C=${1:?cand_id}; O=/root/pkg/out/$C
export OMP_NUM_THREADS=4 PYTHONPATH=/root/pkg/pylib
cd /root/pkg/scripts
rm -rf "$O" "$O.export"
T=$(date +%s); nice -n 10 python3 bake.py /root/pkg/cands/$C /root/w/orig "$O"; echo BAKE_RC=$? secs=$(( $(date +%s)-T ))
T=$(date +%s); nice -n 10 python3 manifest.py /root/w/orig "$O" /root/pkg/out/$C.MANIFEST --expect /root/pkg/out/$C.bake_log.json --workers 4; echo MANIFEST_RC=$? secs=$(( $(date +%s)-T ))
du -sh "$O"
T=$(date +%s); nice -n 10 python3 export_edit.py /root/pkg/cands/$C /root/w/orig "$O.export"; echo EXPORT_RC=$? secs=$(( $(date +%s)-T ))
T=$(date +%s); nice -n 10 python3 peft_verify.py /root/pkg/cands/$C /root/w/orig "$O.export"; echo PEFT_RC=$? secs=$(( $(date +%s)-T ))
du -sh "$O.export"/*; echo BAKE_ALL_DONE
