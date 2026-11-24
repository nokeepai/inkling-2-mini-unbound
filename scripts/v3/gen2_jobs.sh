#!/bin/bash
# ON ink-v3-gen2. GPUs 0-3: v2 Q8_0 (np 8) for the v2 BASELINE on the new v3 test (greedy, think 4096, answer 8192).
# GPUs 4-7: v2 Q4_K_M (np 16) for pool sampling (training candidates). Outputs /root/v3/{eval_v2,gens}.
set -u
cd /root/v11b; export CMP_TOK=/data/compare/models/tok
for g in 0 1 2 3; do kill $(cat /root/logs/srv_$g.pid) 2>/dev/null; done; sleep 8
bash servers.sh /data/v2/Inkling-2-Mini-Unbound-v2-Q8_0.gguf v2q8 0,1,2,3 8 65536
E=8100,8101,8102,8103; S=8104,8105,8106,8107
V=/root/v3; mkdir -p $V/eval_v2 $V/gens
cat $V/test_harm.jsonl $V/test_creative.jsonl > $V/test_single.jsonl
# --- baseline evals (greedy) ---
python3 gen_v3.py --ports $E --inp $V/test_single.jsonl --out $V/eval_v2/test_off.jsonl --effort 0 --greedy --conc 6 > /root/logs/ev_off.log 2>&1 &
python3 gen_v3.py --ports $E --inp $V/test_single.jsonl --out $V/eval_v2/test_e09.jsonl --effort 0.9 --greedy --conc 4 > /root/logs/ev_e09.log 2>&1 &
python3 rp_play.py --ports $E --inp $V/test_rp.jsonl --out $V/eval_v2/rp_off.jsonl --effort 0 --greedy --conc 2 > /root/logs/ev_rp.log 2>&1 &
# --- pool sampling (Q4) ---
python3 gen_v3.py --ports $S --inp $V/pool_creative.jsonl --out $V/gens/cre_off_nosys.jsonl --effort 0 --n 1 --think-cap 1536 --ans-cap 6144 --conc 5 > /root/logs/g_cre_off_nosys.log 2>&1 &
python3 gen_v3.py --ports $S --inp $V/pool_creative.jsonl --out $V/gens/cre_off_sys.jsonl --effort 0 --n 1 --system sys_uncensored.txt --sys-tag answer_fully --think-cap 1536 --ans-cap 6144 --conc 5 > /root/logs/g_cre_off_sys.log 2>&1 &
python3 rp_play.py --ports $S --inp $V/pool_rp.jsonl --out $V/gens/rp_off.jsonl --effort 0 --n 2 --ans-cap 2048 --conc 3 > /root/logs/g_rp_off.log 2>&1 &
python3 gen_v3.py --ports $S --inp $V/pool_harm.jsonl --out $V/gens/ph_off_sys.jsonl --effort 0 --n 1 --system sys_uncensored.txt --sys-tag answer_fully --think-cap 1536 --ans-cap 6144 --conc 4 > /root/logs/g_ph_off_sys.log 2>&1 &
python3 gen_v3.py --ports $S --inp $V/pool_creative.jsonl --out $V/gens/cre_e09_sys.jsonl --effort 0.9 --n 1 --system sys_uncensored.txt --sys-tag answer_fully --think-cap 4096 --ans-cap 6144 --conc 3 > /root/logs/g_cre_e09_sys.log 2>&1 &
wait
echo GEN2_JOBS_DONE
