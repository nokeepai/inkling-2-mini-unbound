#!/bin/bash
# ON an eval pod: full v3 eval of one Q8_0 GGUF on the given GPUs (servers np 8, ctx 65536, ports 8100+gpu).
#   bash eval_cand.sh <name> <model.gguf> <gpu list e.g. 0,1,2,3,4,5> [what=all|new|old]
# Outputs /root/v3/ev/<name>/: new test (gen_v3 greedy: off all 480, e0.9 the 240 even rows), RP (30 chats, off),
# old sealed test via the compare pipeline llama_eval.py (harmful off+e0.9 at answer cap 8192, sharded per GPU;
# borderline sharded; MMLU-300 + GSM8K-100 on the first GPU). KLD is run separately (kld.sh) after servers stop.
set -u
N=$1; M=$2; GPUS=$3; WHAT=${4:-all}
O=/root/v3/ev/$N; mkdir -p $O; cd /root/v11b
export CMP_TOK=/data/compare/models/tok
for g in ${GPUS//,/ }; do kill $(cat /root/logs/srv_$g.pid 2>/dev/null) 2>/dev/null; done; sleep 6
bash servers.sh "$M" $N $GPUS 8 65536
IFS=, read -ra G <<< "$GPUS"; NG=${#G[@]}; PORTS=$(for g in "${G[@]}"; do printf "%s," $((8100+g)); done); PORTS=${PORTS%,}
if [ "$WHAT" != old ]; then
  python3 - <<'EOF'
import json
rows = [json.loads(l) for l in open("/root/v3/test_single.jsonl")]
with open("/root/v3/test_single_e09sub.jsonl", "w") as f:
    for r in rows[::2]:
        f.write(json.dumps(r) + "\n")
EOF
  python3 gen_v3.py --ports $PORTS --inp /root/v3/test_single.jsonl --out $O/test_off.jsonl --effort 0 --greedy --conc 4 > $O/log_off.txt 2>&1 &
  python3 gen_v3.py --ports $PORTS --inp /root/v3/test_single_e09sub.jsonl --out $O/test_e09.jsonl --effort 0.9 --greedy --conc 3 > $O/log_e09.txt 2>&1 &
  python3 rp_play.py --ports $PORTS --inp /root/v3/test_rp.jsonl --out $O/rp_off.jsonl --effort 0 --greedy --system sys_rp.txt --conc 2 > $O/log_rp.txt 2>&1 &
fi
if [ "$WHAT" != new ]; then
  cd /root/compare/bin
  for i in "${!G[@]}"; do
    P=$((8100+${G[$i]}))
    ( CMP_DATA=/root/v3/oldtest/s${i}of${NG} CMP_ANS_CAP=8192 python3 llama_eval.py --port $P --out $O/old_s$i --metrics harmful --conc 4 > $O/log_old_s$i.txt 2>&1
      CMP_DATA=/root/v3/oldtest python3 llama_eval.py --port $P --out $O/bl_s$i --metrics borderline --shard $i/$NG --conc 4 > $O/log_bl_s$i.txt 2>&1 ) &
  done
  CMP_DATA=/root/v3/oldtest python3 llama_eval.py --port $((8100+${G[0]})) --out $O/cap --metrics mmlu,gsm8k --conc 4 > $O/log_cap.txt 2>&1 &
  cd /root/v11b
fi
wait
echo "EVAL_DONE $N $(date -u +%FT%TZ)" | tee $O/DONE
