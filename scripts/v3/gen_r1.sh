#!/bin/bash
# ON ink-v3-eval: v2 servers on all 6 GPUs; round-1 on-policy sampling of v2 on the v3 train harmful prompts
# (778 = 832 minus 54 mass-casualty-keyword rows). Outputs /root/v3/gens (private).
set -u
cd /root/v11b; mkdir -p /root/v3/gens
V2=/data/v2/Inkling-2-Mini-Unbound-v2-Q8_0.gguf
bash servers.sh "$V2" v2 0,1,2,3,4,5 8 65536
P=8100,8101,8102,8103,8104,8105
export CMP_TOK=/data/compare/models/tok
I=/root/v3/harm_train_v3.jsonl; G=/root/v3/gens
python3 gen_samples.py --ports $P --inp $I --out $G/gens_harm_off_nosys.jsonl --effort 0 --n 2 --think-cap 1536 --ans-cap 8192 --conc 4 > /root/logs/g_off_nosys.log 2>&1 &
python3 gen_samples.py --ports $P --inp $I --out $G/gens_harm_off_sys.jsonl --effort 0 --n 2 --system sys_uncensored.txt --think-cap 1536 --ans-cap 8192 --conc 4 > /root/logs/g_off_sys.log 2>&1 &
python3 gen_samples.py --ports $P --inp $I --out $G/gens_harm_e09_nosys.jsonl --effort 0.9 --n 1 --think-cap 4096 --ans-cap 8192 --conc 2 > /root/logs/g_e09_nosys.log 2>&1 &
python3 gen_samples.py --ports $P --inp $I --out $G/gens_harm_e09_sys.jsonl --effort 0.9 --n 1 --system sys_uncensored.txt --think-cap 4096 --ans-cap 8192 --conc 2 > /root/logs/g_e09_sys.log 2>&1 &
wait
echo GEN_R1_DONE
