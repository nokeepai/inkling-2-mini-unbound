#!/bin/bash
# ON gen2: v2 (Q4) continues teacher-opened creative answers (prefill = first sentences of the Venice answer), off, no sys.
cd /root/v11b; export CMP_TOK=/data/compare/models/tok
setsid nohup python3 gen_v3.py --ports 8104,8105,8106,8107 --inp /root/v3/pool_creative_prefill.jsonl --out /root/v3/gens/cre_off_prefill.jsonl --effort 0 --n 1 --sys-tag prefill --think-cap 1536 --ans-cap 6144 --conc 4 > /root/logs/g_cre_prefill.log 2>&1 < /dev/null &
sleep 5; head -1 /root/logs/g_cre_prefill.log
