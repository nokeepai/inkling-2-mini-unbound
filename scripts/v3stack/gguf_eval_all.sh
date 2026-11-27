#!/usr/bin/env bash
# ON THE POD. The GGUF comparison set, one model on the GPU at a time (gguf_eval.sh), then a table:
#   prime_q4km -> stock_q4km -> prime_iq2m -> prime_q8   (STOCK_ALL=1 adds stock_iq2m, stock_q8)
#
#   nohup bash /srv/stack/scripts/v3stack/gguf_eval_all.sh > /root/logs/gguf_eval_all.out 2>&1 &
#   grep -E ' (PASS|FAIL|WARN) |GGUF_EVAL_DONE' /root/logs/gguf_eval.log
#
# Stock baselines are quantised here from the official BF16 GGUF. ggml-org/Inkling-2-Mini-GGUF publishes BF16,
# Q8_0, IQ4_NL, imatrix.gguf and mmproj, but no Q4_K_M or IQ2_M (this repo's scripts only ever fetched those
# files), and its Q8_0 may come from a different llama-quantize build. So every stock quant is made with the
# Prime files' own recipe (gguf_v3.sh: same imatrix, attention overrides read from the Prime file), then its
# per-tensor type map is compared with the Prime file of the same type; a mismatch stops that comparison.
# The stock Q4_K_M build runs on the CPU while Prime Q4_K_M is on the GPU. Idempotent: built files and finished
# suites are kept.
#
# Env (defaults): DATA [/root/data]; PRIME_DIR [the one dir under $DATA/v3gguf holding <dir>/<NAME>-Q4_K_M.gguf,
# not stock_base]; PRIME_NAME [from PRIME_DIR's Q4_K_M file]; BASE_BF16 [$DATA/base/Inkling-2-Mini-BF16.gguf];
# IM [$DATA/base/imatrix.gguf]; STOCK_DIR [$DATA/v3gguf/stock_base]; STOCK_ALL [0]; QUANT_THREADS [nproc/2];
# SUITES [pub_gsm8k pub_humaneval agent_hard_noforce]; plus everything gguf_eval.sh reads (OUT, LOG, PORT, ...).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA="${DATA:-/root/data}"
LLAMA_BIN="${LLAMA_BIN:-$DATA/llama.cpp/build/bin}"; export LLAMA_BIN
BASE_BF16="${BASE_BF16:-$DATA/base/Inkling-2-Mini-BF16.gguf}"
IM="${IM:-$DATA/base/imatrix.gguf}"
STOCK_DIR="${STOCK_DIR:-$DATA/v3gguf/stock_base}"
OUT="${OUT:-/root/gguf_evals}"; export OUT
LOG="${LOG:-/root/logs/gguf_eval.log}"; export LOG
QUANT_THREADS="${QUANT_THREADS:-$(( $(nproc) / 2 > 0 ? $(nproc) / 2 : 1 ))}"
read -r -a SUITES <<< "${SUITES:-pub_gsm8k pub_humaneval agent_hard_noforce}"
LIB=(python3 "$HERE/gguf_eval_lib.py")
mkdir -p "$OUT" "$STOCK_DIR/logs" "$(dirname "$LOG")"

say() { echo "$(date -u +%FT%TZ) $*" | tee -a "$LOG"; }

find_prime() {
  local d n hits=()
  for d in "$DATA"/v3gguf/*/; do
    n="$(basename "$d")"
    [ "$n" = stock_base ] || [ "$n" = proof ] && continue
    compgen -G "$d*-Q4_K_M.gguf" > /dev/null && hits+=("${d%/}")
  done
  [ ${#hits[@]} = 1 ] || { say "FAIL all: set PRIME_DIR; candidates under $DATA/v3gguf: ${hits[*]:-none}"; exit 1; }
  echo "${hits[0]}"
}

PRIME_DIR="${PRIME_DIR:-$(find_prime)}"
if [ -z "${PRIME_NAME:-}" ]; then
  q4="$(compgen -G "$PRIME_DIR/*-Q4_K_M.gguf" | head -n 1)"
  PRIME_NAME="$(basename "${q4%-Q4_K_M.gguf}")"
fi
prime_file() { echo "$PRIME_DIR/$PRIME_NAME-$1.gguf"; }
stock_file() { echo "$STOCK_DIR/Inkling-2-Mini-$1.gguf"; }

# ---------------------------------------------------------------- stock quants (CPU only)

ensure_imatrix() {
  [ -f "$IM" ] && return 0
  say "INFO all: $IM missing; fetching ggml-org/Inkling-2-Mini-GGUF imatrix.gguf (public)"
  hf download ggml-org/Inkling-2-Mini-GGUF imatrix.gguf --local-dir "$(dirname "$IM")" > "$STOCK_DIR/logs/dl_imatrix.log" 2>&1
  [ -f "$IM" ]
}

build_stock() {   # $1 type. Same commands as gguf_v3.sh step 3; overrides read from the Prime file of that type.
  local t="$1" out keep ref
  out="$(stock_file "$t")"; ref="$(prime_file "$t")"
  if [ -f "$out" ]; then echo "stock $t: built earlier"; return 0; fi
  [ -f "$BASE_BF16" ] || { echo "missing $BASE_BF16"; return 1; }
  local free_gb need_gb
  free_gb="$(df -P -BG "$STOCK_DIR" | awk 'NR==2 {gsub("G","",$4); print $4}')"
  need_gb=$(( $(stat -c %s "$ref" 2>/dev/null || echo 30000000000) / 1000000000 + 5 ))
  [ "$free_gb" -ge "$need_gb" ] || { echo "only ${free_gb} GB free under $STOCK_DIR, need $need_gb"; return 1; }
  if [ "$t" = Q8_0 ]; then
    CUDA_VISIBLE_DEVICES="" "$LLAMA_BIN/llama-quantize" "$BASE_BF16" "$out.part" Q8_0 "$QUANT_THREADS"
  else
    ensure_imatrix
    [ -f "$ref" ] || { echo "missing $ref (needed to read its attention overrides)"; return 1; }
    read -r -a keep <<< "$("${LIB[@]}" keep-flags --gguf "$ref")"
    echo "overrides from $(basename "$ref"): ${keep[*]:-none}"
    CUDA_VISIBLE_DEVICES="" "$LLAMA_BIN/llama-quantize" --imatrix "$IM" ${keep[@]+"${keep[@]}"} \
      "$BASE_BF16" "$out.part" "$t" "$QUANT_THREADS"
  fi
  mv "$out.part" "$out"
  echo "stock $t built: $(du -h "$out" | cut -f1)"
}

stage_stock() {   # $1 type: build + type-map check against Prime; one PASS/FAIL line
  local t="$1" log="$STOCK_DIR/logs/q_$1.log" ok="$STOCK_DIR/logs/ok_$1"
  rm -f "$ok"
  if ! build_stock "$t" > "$log" 2>&1; then
    say "FAIL stock build $t: $(tail -n 2 "$log" | tr '\n' ' ' | head -c 300)"; return 1
  fi
  if [ -f "$(prime_file "$t")" ] && ! "${LIB[@]}" types-diff --a "$(prime_file "$t")" --b "$(stock_file "$t")" >> "$log" 2>&1; then
    say "FAIL stock build $t: tensor types differ from $(basename "$(prime_file "$t")") (see $log)"; return 1
  fi
  touch "$ok"   # wait_stock reads this, not the shared log (old FAIL lines would mislead a rerun)
  say "PASS stock build $t ($(tail -n 1 "$log"))"
}

# ---------------------------------------------------------------- evals

eval_one() {   # $1 gguf, $2 tag
  if [ ! -f "$1" ]; then say "FAIL $2: missing $1"; return 1; fi
  bash "$HERE/gguf_eval.sh" "$1" "$2" "${SUITES[@]}" || true
}

say "INFO all: prime $PRIME_DIR/$PRIME_NAME-*, stock $STOCK_DIR, suites [${SUITES[*]}], results $OUT"
STOCK_TYPES="Q4_K_M"; [ "${STOCK_ALL:-0}" = 1 ] && STOCK_TYPES="Q4_K_M IQ2_M Q8_0"
( for t in $STOCK_TYPES; do stage_stock "$t" || true; done ) &
BUILD_PID=$!

eval_one "$(prime_file Q4_K_M)" prime_q4km
wait_stock() {   # $1 type: 0 when the stock file is built and its types match
  while [ ! -f "$STOCK_DIR/logs/ok_$1" ] && kill -0 "$BUILD_PID" 2>/dev/null; do sleep 20; done
  [ -f "$STOCK_DIR/logs/ok_$1" ]
}
if wait_stock Q4_K_M; then eval_one "$(stock_file Q4_K_M)" stock_q4km; else say "FAIL stock_q4km: no usable stock Q4_K_M"; fi
eval_one "$(prime_file IQ2_M)" prime_iq2m
eval_one "$(prime_file Q8_0)" prime_q8
TAGS="prime_q4km,stock_q4km,prime_iq2m,prime_q8"
if [ "${STOCK_ALL:-0}" = 1 ]; then
  wait "$BUILD_PID" || true
  wait_stock IQ2_M && eval_one "$(stock_file IQ2_M)" stock_iq2m
  wait_stock Q8_0 && eval_one "$(stock_file Q8_0)" stock_q8
  TAGS="prime_q8,stock_q8,prime_q4km,stock_q4km,prime_iq2m,stock_iq2m"
fi
wait "$BUILD_PID" 2>/dev/null || true
"${LIB[@]}" table --root "$OUT" --tags "$TAGS" --tsv "$OUT/gguf_table.tsv" | tee -a "$LOG"
say "GGUF_EVAL_ALL_DONE table $OUT/gguf_table.tsv"
