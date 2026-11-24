#!/bin/bash
# Mac: LoRA(s) on ink-v3-train -> merged tensors -> Q8_0 GGUF -> full eval + KLD on ink-v3-eval -> pull + judge.
#   bash cand_pipeline.sh <name> <train_path1.pt>[:scale] [<train_path2.pt>[:scale]]
# e.g. bash cand_pipeline.sh sft1 /root/v3/runs/sft1/lora_final.pt:1.0
#      bash cand_pipeline.sh dpo1 /root/v3/runs/sft1/lora_final.pt:1.0 /root/v3/runs/dpo1/lora_final.pt
set -u
N=$1; shift
R=~/Code/inkling-abl; P=~/.inkling-abl/v3_private; L="bash $R/scripts/ink_lium.sh"
mkdir -p $P/loras/$N $P/ev/$N
$L exec ink-v3-eval "mkdir -p /root/v3/loras/$N /data/cands/$N" > /dev/null 2>&1
$L scp ink-v3-eval $R/scripts/v3/merge_stack.py /root/v11b/merge_stack.py > /dev/null 2>&1
STACK=""; i=0
for spec in "$@"; do
  src=${spec%%:*}; sc=""; [ "$spec" != "$src" ] && sc=":${spec#*:}"
  loc=$P/loras/$N/l$i.pt
  $L scp ink-v3-train "$src" "$loc" --download > /dev/null 2>&1 || { echo "download failed $src"; exit 1; }
  $L scp ink-v3-eval "$loc" /root/v3/loras/$N/l$i.pt > /dev/null 2>&1 || { echo "upload failed"; exit 1; }
  STACK="$STACK /root/v3/loras/$N/l$i.pt$sc"; i=$((i+1))
done
echo "$(date -u +%T) loras staged:$STACK"
$L exec ink-v3-eval "cd /root/v11b && python3 merge_stack.py /data/v2hf hf_name_map.json /data/cands/$N/tensors.safetensors $STACK > /data/logs/merge_$N.log 2>&1 && bash build_gguf.sh $N /data/cands/$N/tensors.safetensors > /data/logs/gguf_$N.log 2>&1; tail -2 /data/logs/gguf_$N.log" 2>&1 | tail -2
$L exec ink-v3-eval "test -f /data/v11/$N-Q8_0.gguf" > /dev/null 2>&1 || { echo "GGUF missing"; exit 1; }
echo "$(date -u +%T) gguf built; eval starting"
$L exec ink-v3-eval "cd /root/v11b && setsid nohup bash -c 'bash eval_cand.sh $N /data/v11/$N-Q8_0.gguf 0,1,2,3,4,5 all > /root/logs/ev_$N.log 2>&1; bash kld.sh $N /data/v11/$N-Q8_0.gguf 0 > /root/logs/kld_$N.log 2>&1; echo KLD_FIN > /root/v3/ev/$N/KLD_FIN' > /dev/null 2>&1 < /dev/null &" > /dev/null 2>&1
until $L exec ink-v3-eval "cat /root/v3/ev/$N/KLD_FIN 2>/dev/null" 2>/dev/null | grep -q KLD_FIN; do sleep 60; done
echo "$(date -u +%T) eval + kld done"
bash $R/scripts/v3/pull_ev.sh $N
