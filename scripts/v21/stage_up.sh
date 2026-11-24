#!/bin/bash
# ON release pod: stage release files (hard links) in repo layout and upload each repo's set to branch v2.1. HF token in $HF_HOME.
set -u
R=/data/rel; U=/data/up; N=Inkling-2-Mini-Unbound-v2.1; L=/root/logs/rel21.log; log() { echo "$(date -u +%T) $*" >> $L; }
rm -rf $U; mkdir -p $U/gguf $U/basalt $U/main/lora_v2.1 $U/main/finetune_v2.1
for T in Q8_0 Q6_K Q5_K_M Q4_K_M IQ3_M IQ2_M; do ln $R/gguf/$N-$T.gguf $U/gguf/; done
ln $R/lora_v21/$N-LoRA-r16.gguf $R/lora_v21/$N-LoRA-r64.gguf $U/gguf/
ln $R/$N-NVFP4-q8.basalt $U/basalt/
ln $R/v21ck/model-00001-of-00001.safetensors $U/main/
cp -r $R/lora_v21/peft-r16 $R/lora_v21/peft-r64 $U/main/lora_v2.1/
cp $R/finetune/* $U/main/finetune_v2.1/
python3 - <<'PY'
from huggingface_hub import HfApi
a = HfApi()
for r in ("Inkling-2-Mini-Unbound", "Inkling-2-Mini-Unbound-GGUF", "Inkling-2-Mini-Unbound-Basalt"):
    a.create_branch(f"nokeepai/{r}", branch="v2.1", exist_ok=True)
    print("branch ok", r)
PY
log BRANCHES_OK
cd /root/v11b
( python3 /root/v21/up21.py nokeepai/Inkling-2-Mini-Unbound $U/main v2.1 > $R/logs/up_main.log 2>&1; log UP_MAIN rc=$? "$(tail -n1 $R/logs/up_main.log)" ) &
( python3 /root/v21/up21.py nokeepai/Inkling-2-Mini-Unbound-Basalt $U/basalt v2.1 > $R/logs/up_basalt.log 2>&1; log UP_BASALT rc=$? "$(tail -n1 $R/logs/up_basalt.log)" ) &
( python3 /root/v21/up21.py nokeepai/Inkling-2-Mini-Unbound-GGUF $U/gguf v2.1 > $R/logs/up_gguf.log 2>&1; log UP_GGUF rc=$? "$(tail -n1 $R/logs/up_gguf.log)" ) &
wait
log UPLOAD_DONE
