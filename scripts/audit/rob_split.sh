#!/bin/bash
# Pod-side (ink-fleet-2): split the remaining robustness kinds (image already done in *_tr) into group dirs and
# run them across all 8 GPUs, effort 0. Kill the earlier serial jobs first (by PID, separately).
set -uo pipefail
cd /srv/audit
mkdir -p rob_g/A rob_g/B rob_g/C rob_g/L
for k in lang_de lang_es; do ln -sf /srv/audit/robust_test/$k.jsonl rob_g/A/$k.jsonl; done
for k in lang_fr lang_zh; do ln -sf /srv/audit/robust_test/$k.jsonl rob_g/B/$k.jsonl; done
for k in longctx multiturn sysprompt; do ln -sf /srv/audit/robust_test/$k.jsonl rob_g/C/$k.jsonl; done
for k in lang_de lang_es lang_fr lang_zh; do ln -sf /srv/audit/robust_test/$k.jsonl rob_g/L/$k.jsonl; done
cd /srv/audit/scripts
run() { # gpu cand group
  rm -rf /srv/cands/$2__g$1; cp -r /srv/audit/cands/$2 /srv/cands/$2__g$1
  nohup python3 sealed_run.py --cand $2 --data /srv/audit/data --split test --gpu $1 --blocks robust \
    --robust /srv/audit/rob_g/$3 --think-cap 4096 --tag r$3 > /root/logs/audit_$2_r$3.log 2>&1 &
  echo "started $2 r$3 gpu $1"; sleep 15
}
run 0 c1h_r1_svd1_s3_L4 A
run 1 c1h_r1_svd1_s3_L4 B
run 2 c1h_r1_svd1_s3_L4 C
run 3 c1dt001 A
run 4 c1dt001 B
run 5 c1dt001 C
run 6 ou L
run 7 ou C
wait
echo ROB_SPLIT_DONE
