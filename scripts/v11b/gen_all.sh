#!/bin/bash
# ON eval pod: round-1 data generation. GPUs 0-5 = v1.0 Q8_0 (harmful train samples), GPUs 6-7 = original Q8_0 (anchors).
# Outputs in /root/v11b/gens (private: contains prompts + model text).
set -u
cd /root/v11b; mkdir -p gens
V10=$(ls /data/v10/*Q8_0*.gguf | grep -v mmproj | head -1)
ORIG=/data/base/Inkling-2-Mini-Q8_0.gguf
bash servers.sh "$V10" v10 0,1,2,3,4,5 12 49152
bash servers.sh "$ORIG" orig 6,7 12 49152
HP=8100,8101,8102,8103,8104,8105; OP=8106,8107
export CMP_TOK=/data/compare/models/tok
# anchors on the original (greedy), in the background
( python3 gen_samples.py --ports $OP --inp anchor_off.jsonl --out gens/gens_anchor_off.jsonl --effort 0 --n 1 --greedy --think-cap 1536 --ans-cap 768
  python3 gen_samples.py --ports $OP --inp anchor_e09.jsonl --out gens/gens_anchor_e09.jsonl --effort 0.9 --n 1 --greedy --think-cap 2048 --ans-cap 768
  echo ANCHORS_DONE ) > /root/logs/gen_anchor.log 2>&1 &
# harmful train on v1.0: off (no sys, sys) x 4, then e0.9 (no sys, sys) x 2
python3 gen_samples.py --ports $HP --inp harm_train.jsonl --out gens/gens_harm_off_nosys.jsonl --effort 0 --n 4 --think-cap 1536 --ans-cap 768 > /root/logs/gen_off_nosys.log 2>&1
python3 gen_samples.py --ports $HP --inp harm_train.jsonl --out gens/gens_harm_off_sys.jsonl --effort 0 --n 4 --system sys_uncensored.txt --think-cap 1536 --ans-cap 768 > /root/logs/gen_off_sys.log 2>&1
echo OFF_DONE
python3 gen_samples.py --ports $HP --inp harm_train.jsonl --out gens/gens_harm_e09_nosys.jsonl --effort 0.9 --n 2 --think-cap 2048 --ans-cap 768 > /root/logs/gen_e09_nosys.log 2>&1
python3 gen_samples.py --ports $HP --inp harm_train.jsonl --out gens/gens_harm_e09_sys.jsonl --effort 0.9 --n 2 --system sys_uncensored.txt --think-cap 2048 --ans-cap 768 > /root/logs/gen_e09_sys.log 2>&1
wait
echo GEN_ALL_DONE
