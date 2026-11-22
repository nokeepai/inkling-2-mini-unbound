#!/bin/bash
# Mac: pull one val screen (outputs + logs) from the eval pod into the private dir (chmod 700). bash pull_val.sh <name>
set -u
N=$1; P=~/.inkling-abl/v11b_private/val; mkdir -p $P/out/$N $P/logs; chmod 700 ~/.inkling-abl/v11b_private
L="bash $HOME/Code/inkling-abl/scripts/ink_lium.sh"
for f in metrics.json gens_harmful_off.jsonl gens_harmful_e09.jsonl gens_borderline.jsonl gens_mmlu.jsonl gens_gsm8k.jsonl; do
  $L scp ${EVAL_POD:-ink-v11-eval2} -d /root/compare/out/$N/$f $P/out/$N/$f > /dev/null 2>&1
done
for f in kld.log status eval.log; do $L scp ${EVAL_POD:-ink-v11-eval2} -d /root/compare/logs/$N.$f $P/logs/$N.$f > /dev/null 2>&1; done
ls $P/out/$N
