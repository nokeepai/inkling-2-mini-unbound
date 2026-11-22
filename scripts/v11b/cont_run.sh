#!/bin/bash
# ON eval2: continue answer-capped samples to 8192 answer tokens. v1.0 on GPUs 4,5,6 (harmful), orig on GPU 7 (anchors).
set -u
cd /root/v11b; export CMP_TOK=/data/compare/models/tok; mkdir -p gens cont
hf download $SCRATCH_REPO --include 'cont_in/*' --repo-type dataset --local-dir /root/v11b > /dev/null 2>&1; ls cont_in | wc -l
bash servers.sh /data/v10/Inkling-2-Mini-Unbound-Q8_0.gguf v10 4,5,6 16 49152
bash servers.sh /data/base/Inkling-2-Mini-Q8_0.gguf orig 7 16 49152
( for x in anchor_off anchor_e09; do python3 cont_samples.py --ports 8107 --inp cont_in/gens_$x.jsonl --out cont/gens_$x.cont.jsonl --greedy --conc 16; done; echo ANCH_DONE ) > /root/logs/cont_anchor.log 2>&1 &
for x in harm_off_sys harm_e09_sys harm_off_nosys harm_e09_nosys; do
  python3 cont_samples.py --ports 8104,8105,8106 --inp cont_in/gens_$x.jsonl --out cont/gens_$x.cont.jsonl --conc 16 >> /root/logs/cont_harm.log 2>&1
done
wait; echo CONT_ALL_DONE
