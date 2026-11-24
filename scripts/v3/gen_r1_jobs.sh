#!/bin/bash
# ON ink-v3-eval: the sampling jobs of gen_r1.sh only (servers already up on 8100-8105).
set -u
cd /root/v11b; export CMP_TOK=/data/compare/models/tok
P=8100,8101,8102,8103,8104,8105; I=/root/v3/harm_train_v3.jsonl; G=/root/v3/gens; mkdir -p $G
python3 gen_samples.py --ports $P --inp $I --out $G/gens_harm_off_nosys.jsonl --effort 0 --n 2 --think-cap 1536 --ans-cap 8192 --conc 4 > /root/logs/g_off_nosys.log 2>&1 &
python3 gen_samples.py --ports $P --inp $I --out $G/gens_harm_off_sys.jsonl --effort 0 --n 2 --system sys_uncensored.txt --think-cap 1536 --ans-cap 8192 --conc 4 > /root/logs/g_off_sys.log 2>&1 &
python3 gen_samples.py --ports $P --inp $I --out $G/gens_harm_e09_nosys.jsonl --effort 0.9 --n 1 --think-cap 4096 --ans-cap 8192 --conc 2 > /root/logs/g_e09_nosys.log 2>&1 &
python3 gen_samples.py --ports $P --inp $I --out $G/gens_harm_e09_sys.jsonl --effort 0.9 --n 1 --system sys_uncensored.txt --think-cap 4096 --ans-cap 8192 --conc 2 > /root/logs/g_e09_sys.log 2>&1 &
wait; echo GEN_R1_DONE
