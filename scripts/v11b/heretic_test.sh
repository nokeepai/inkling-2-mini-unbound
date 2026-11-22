#!/bin/bash
# ON eval pod: sealed-test harmful run for JoaoZaokk's Heretic (UD-Q4_K_XL) on GPU 7, served by the 0xE2E Inkling
# engine (llama.cpp + Unsloth inkling2 layout; stock llama.cpp can't load this file). Same flags/caps/scorer as harm8k.sh.
set -u
N=test8k_heretic; G=7; PORT=8207; OUT=/root/compare/out/$N; LOG=/root/compare/logs; mkdir -p $OUT
CUDA_VISIBLE_DEVICES=$G /data/e2e_engine/build/bin/llama-server -m /data/compare/models/heretic/Inkling-2-Mini-Heretic-UD-Q4_K_XL.gguf \
  -ngl 99 -c 49152 -np 6 --kv-unified -ub 128 --port $PORT --host 127.0.0.1 --special --jinja --seed 1234 > $LOG/$N.server.log 2>&1 &
SP=$!
for i in $(seq 1 120); do curl -sf http://127.0.0.1:$PORT/health > /dev/null && break; sleep 5; done
echo "$(date -u +%FT%TZ) SERVER_UP" >> $LOG/$N.status
cd /root/compare/bin && CMP_DATA=/root/v11b/test CMP_TOK=/data/compare/models/tok CMP_ANS_CAP=8192 \
  python3 llama_eval.py --port $PORT --out $OUT --conc 6 --metrics harmful > $LOG/$N.eval.log 2>&1
echo "$(date -u +%FT%TZ) EVAL rc=$?" >> $LOG/$N.status
kill $SP; echo "$(date -u +%FT%TZ) ALL_DONE" >> $LOG/$N.status
