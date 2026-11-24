#!/bin/bash
# ON gen2: restart the v2 e0.9 baseline on the 240-row subset (even rows), resuming the rows already done.
set -u
cd /root/v3
python3 -c "
import json
rows=[json.loads(l) for l in open('test_single.jsonl')]
open('test_single_e09sub.jsonl','w').writelines(json.dumps(r)+'\n' for r in rows[::2])"
cd /root/v11b; export CMP_TOK=/data/compare/models/tok
setsid nohup python3 gen_v3.py --ports 8100,8101,8102,8103 --inp /root/v3/test_single_e09sub.jsonl --out /root/v3/eval_v2/test_e09.jsonl --effort 0.9 --greedy --conc 4 > /root/logs/ev_e09.log 2>&1 < /dev/null &
sleep 5; head -1 /root/logs/ev_e09.log
