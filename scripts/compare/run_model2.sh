#!/bin/bash
# Runs ON ink-gguf-1. One model on one GPU: llama-server + llama_eval.py, then llama-perplexity KLD vs the original Q8_0.
#   bash run_model.sh <name> <gpu> <port> <model.gguf> [lora.gguf] [--base-kld-first]
# Identical flags for every model. Logs: /root/compare/logs/<name>.*  Outputs: /root/compare/out/<name>/
set -u
NAME=$1; GPU=$2; PORT=$3; MODEL=$4; LORA=${5:-}; MODE=${6:-}
B=/data/llama.cpp/build/bin
OUT=/root/compare/out/$NAME; LOG=/root/compare/logs
KLD=/data/compare/kld; WIKI=/data/compare/wiki.test.raw
mkdir -p $OUT $KLD
export CUDA_VISIBLE_DEVICES=$GPU
LARGS=(); [ -n "$LORA" ] && [ "$LORA" != "-" ] && LARGS=(--lora "$LORA")
CTX=${CMP_CTX:-40960}; NP=${CMP_NP:-8}
echo "$(date -u +%FT%TZ) START $NAME gpu=$GPU model=$MODEL lora=$LORA ctx=$CTX np=$NP" >> $LOG/$NAME.status

kld() {  # $1 = base|cand
  local PPL_ARGS=(-m "$MODEL" "${LARGS[@]}" -f $WIKI -c 4096 --chunks 40 -ngl 99 -b 4096 -ub 512 --seed 1234)
  if [ "$1" = base ]; then
    $B/llama-perplexity "${PPL_ARGS[@]}" --kl-divergence-base $KLD/base.kld > $LOG/$NAME.kldbase.log 2>&1
    echo "$(date -u +%FT%TZ) KLD_BASE rc=$?" >> $LOG/$NAME.status
  else
    while [ ! -f $KLD/base.done ]; do sleep 30; done
    $B/llama-perplexity "${PPL_ARGS[@]}" --kl-divergence-base $KLD/base.kld --kl-divergence > $LOG/$NAME.kld.log 2>&1
    echo "$(date -u +%FT%TZ) KLD rc=$?" >> $LOG/$NAME.status
  fi
}

if [ "$MODE" = "--base-kld-first" ]; then kld base; touch $KLD/base.done; fi

$B/llama-server -m "$MODEL" "${LARGS[@]}" -ngl 99 -c $CTX -np $NP --kv-unified -ub 128 --port $PORT --host 127.0.0.1 \
  --special --jinja --seed 1234 --no-webui > $LOG/$NAME.server.log 2>&1 &
SPID=$!
echo $SPID > $LOG/$NAME.server.pid
for i in $(seq 1 180); do
  curl -sf http://127.0.0.1:$PORT/health > /dev/null && break
  kill -0 $SPID 2>/dev/null || { echo "$(date -u +%FT%TZ) SERVER_DIED" >> $LOG/$NAME.status; exit 1; }
  sleep 5
done
echo "$(date -u +%FT%TZ) SERVER_UP" >> $LOG/$NAME.status; curl -s http://127.0.0.1:$PORT/lora-adapters > $OUT/lora_adapters.json; curl -s http://127.0.0.1:$PORT/props > $OUT/props.json
cd /root/compare/bin
python3 llama_eval.py --port $PORT --out $OUT --conc $NP --metrics ${CMP_METRICS:-mmlu,gsm8k,harmful,borderline} \
  ${CMP_LIMIT:+--limit $CMP_LIMIT} ${CMP_SHARD:+--shard $CMP_SHARD} > $LOG/$NAME.eval.log 2>&1
echo "$(date -u +%FT%TZ) EVAL rc=$?" >> $LOG/$NAME.status
kill $SPID; wait $SPID 2>/dev/null
[ "${CMP_SKIP_KLD:-}" = 1 ] || { [ "$MODE" = "--base-kld-first" ] || kld cand; }
echo "$(date -u +%FT%TZ) ALL_DONE" >> $LOG/$NAME.status
