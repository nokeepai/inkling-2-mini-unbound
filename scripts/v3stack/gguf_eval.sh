#!/usr/bin/env bash
# ON THE POD (one RTX 5090). Does a fine-tune's effect survive GGUF quantisation? Serves ONE GGUF with native
# llama-server (llama.cpp @19bc41dbb), runs the evalsuite against it with the pods' settings, stops the server.
#
#   bash gguf_eval.sh <gguf_path> <tag> [suites...]     default suites: pub_gsm8k pub_humaneval agent_hard_noforce
#   grep -E ' (PASS|FAIL) ' /root/logs/gguf_eval.log
#
# Results: $OUT/<tag>/<suite>/summary.json (+ meta_<suite>.json, probe.json, server.log). A suite whose summary.json
# exists is skipped, so rerunning the same command resumes. One GPU user at a time: flock on $OUT/.gpu.lock (a
# second invocation waits), and the GPU must be idle before the server starts.
# Sanity suites run first (pub_gsm8k --limit 100, pub_humaneval --limit 50): a build that cannot do arithmetic
# gives meaningless stall numbers, so the agent suite is skipped when gsm8k accuracy < SANITY_MIN (FORCE_AGENT=1
# runs it anyway).
#
# Settings = the full-precision pods' (stack_test.sh run_one_suite): effort 0.9, temperature 0.6, top_p 1.0, seed
# 1234, repeats 1, thinking cap 4,096, answer cap 8,192. Engine-specific choices, and why:
#   --jinja --reasoning-format auto   llama.cpp's Inkling PEG parser returns thinking in reasoning_content (split=field,
#                                     as on SGLang); no --special / --raw-markers needed
#   --top-k 0 --min-p 0 (+ per request) llama-server defaults are top_k 40 / min_p 0.05; SGLang applied neither
#   --skip-probe                      the client's /tokenize probe mis-detects llama-server as vLLM (llama-server
#                                     answers {"tokens": []} to a messages body), which would abort the run and zero
#                                     the agent prompt-token counts; skipping it leaves the chars/3.5 estimate the
#                                     SGLang pods used (their probe said "unavailable"). gguf_eval_lib.py probe checks
#                                     effort, the thinking split and the forced-answer path instead.
#   -ub 128 --seed 1234 --kv-unified  as scripts/v11b/servers.sh and scripts/compare/run_model2.sh
#   --fit off, -ngl 999               full offload or a clear failure, never a silent CPU split
#
# Env (defaults): LLAMA_BIN [/root/data/llama.cpp/build/bin]; STACK [/srv/stack] holding scripts/ and eval_data/;
# OUT [/root/gguf_evals]; LOG [/root/logs/gguf_eval.log]; PORT [8500]; GPU [0]; NP / CTX (both or neither: override
# the VRAM plan); SERVE_TIMEOUT [900]; SUITE_TIMEOUT [10800]; SANITY_MIN [0.5]; FORCE_AGENT [0]; EXTRA_SERVER_ARGS;
# DRY_BASE_URL (offline test: skip the GPU/server steps and run against this URL, e.g. mock_server.py).
set -euo pipefail

GGUF="${1:?usage: gguf_eval.sh <gguf_path> <tag> [suites...]}"
TAG="${2:?usage: gguf_eval.sh <gguf_path> <tag> [suites...]}"
shift 2
SUITES=("$@")
[ ${#SUITES[@]} -gt 0 ] || SUITES=(pub_gsm8k pub_humaneval agent_hard_noforce)

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLAMA_BIN="${LLAMA_BIN:-/root/data/llama.cpp/build/bin}"
STACK="${STACK:-/srv/stack}"
SCRIPTS="${SCRIPTS:-$STACK/scripts}"
EVAL_DATA="${EVAL_DATA:-$STACK/eval_data}"
AGENT_DIR="${AGENT_DIR:-$SCRIPTS/tinker/agent_hard_tasks}"
OUT="${OUT:-/root/gguf_evals}"
LOG="${LOG:-/root/logs/gguf_eval.log}"
PORT="${PORT:-8500}"
GPU="${GPU:-0}"
SERVE_TIMEOUT="${SERVE_TIMEOUT:-900}"
SUITE_TIMEOUT="${SUITE_TIMEOUT:-10800}"
SANITY_MIN="${SANITY_MIN:-0.5}"
EFFORT=0.9; TEMP=0.6; TOP_P=1.0; SEED=1234; THINK_CAP=4096; ANSWER_CAP=8192
LIB=(python3 "$HERE/gguf_eval_lib.py")
TAG_DIR="$OUT/$TAG"
SERVER_PID=""
mkdir -p "$TAG_DIR" "$(dirname "$LOG")"

say() { echo "$(date -u +%FT%TZ) $*" | tee -a "$LOG"; }
die() { say "FAIL $TAG: $*"; echo "GGUF_EVAL_DONE $TAG" | tee -a "$LOG"; exit 1; }

suite_limit() {   # the sanity suites are subsets; the agent suite always runs all 15 tasks
  case "$1" in pub_gsm8k) echo 100 ;; pub_humaneval) echo 50 ;; *) echo 0 ;; esac
}
suite_done() { [ -f "$TAG_DIR/$1/summary.json" ]; }

todo_suites() {
  local s
  for s in "${SUITES[@]}"; do suite_done "$s" || echo "$s"; done
}

# ---------------------------------------------------------------- preflight (no GPU touched)

preflight() {
  [ -f "$SCRIPTS/evalsuite/run_suite.py" ] || die "missing $SCRIPTS/evalsuite/run_suite.py"
  local s
  for s in "${SUITES[@]}"; do
    case "$s" in
      agent_hard_noforce|agent_hard) [ -d "$AGENT_DIR/t01_inventory_ledger" ] || die "missing agent tasks under $AGENT_DIR" ;;
      pub_*) [ -f "$EVAL_DATA/pub/$s.jsonl" ] || die "missing $EVAL_DATA/pub/$s.jsonl" ;;
    esac
  done
  [ -n "${DRY_BASE_URL:-}" ] && return 0
  [ -x "$LLAMA_BIN/llama-server" ] || die "missing $LLAMA_BIN/llama-server"
  [ -f "$GGUF" ] || die "missing $GGUF"
  # gguf_v3.sh writes Q8_0 straight to its final name, so a running llama-quantize may still be filling it
  if pgrep -af llama-quantize 2>/dev/null | grep -qF -- "$GGUF"; then die "$GGUF is still being written by llama-quantize"; fi
  local head
  head="$(git -C "$LLAMA_BIN/../.." rev-parse HEAD 2>/dev/null || echo unknown)"
  case "$head" in 19bc41dbb*) ;; *) say "WARN $TAG: llama.cpp HEAD is $head, not 19bc41dbb (loop-slot fix)" ;; esac
}

# ---------------------------------------------------------------- server

gpu_free_mib() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$GPU" | tr -d ' '; }
gpu_used_mib() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU" | tr -d ' '; }

stop_server() {
  [ -n "$SERVER_PID" ] || return 0
  kill "$SERVER_PID" 2>/dev/null || true
  local i
  for i in $(seq 1 30); do kill -0 "$SERVER_PID" 2>/dev/null || break; sleep 1; done
  kill -9 "$SERVER_PID" 2>/dev/null || true
  wait "$SERVER_PID" 2>/dev/null || true
  SERVER_PID=""
}

start_server() {
  local used plan np ctx need
  used="$(gpu_used_mib)"
  # Another job (stack_test's docker server, a stray llama-server) would skew both speed and fit
  [ "$used" -lt 2000 ] || die "GPU $GPU already has ${used} MiB in use; stop the other job first (nvidia-smi)"
  ! curl -s -o /dev/null "http://127.0.0.1:$PORT/health" || die "port $PORT already answers; set PORT or stop that server"
  if ! plan="$("${LIB[@]}" plan --gguf "$GGUF" --free-mib "$(gpu_free_mib)" --np "${NP:-0}" --ctx "${CTX:-0}" 2>&1)"; then
    die "$plan"
  fi
  read -r np ctx need <<< "$plan"
  CONCURRENCY="$np"
  say "INFO $TAG: $(basename "$GGUF") $(( $(stat -c %s "$GGUF") >> 20 )) MiB, np $np ctx $ctx (unified), est. ${need} MiB"
  # shellcheck disable=SC2086  # EXTRA_SERVER_ARGS is a word list on purpose
  CUDA_VISIBLE_DEVICES="$GPU" "$LLAMA_BIN/llama-server" -m "$GGUF" --alias "$TAG" -ngl 999 --fit off \
    -c "$ctx" -np "$np" --kv-unified -ub 128 --host 127.0.0.1 --port "$PORT" \
    --jinja --reasoning-format auto --temp "$TEMP" --top-p "$TOP_P" --top-k 0 --min-p 0 --seed "$SEED" \
    --no-webui ${EXTRA_SERVER_ARGS:-} > "$TAG_DIR/server.log" 2>&1 &
  SERVER_PID=$!
  local t=0
  while [ "$t" -lt "$SERVE_TIMEOUT" ]; do
    curl -sf "http://127.0.0.1:$PORT/health" > /dev/null 2>&1 && break
    kill -0 "$SERVER_PID" 2>/dev/null || die "llama-server exited while loading: $(grep -iE 'error|out of memory|failed' "$TAG_DIR/server.log" | tail -n 3 | tr '\n' ' ' | head -c 400) (lower NP/CTX?)"
    sleep 5; t=$((t + 5))
  done
  [ "$t" -lt "$SERVE_TIMEOUT" ] || die "llama-server not healthy after ${SERVE_TIMEOUT}s (see $TAG_DIR/server.log)"
  local off
  off="$(grep -oE 'offloaded [0-9]+/[0-9]+ layers to GPU' "$TAG_DIR/server.log" | tail -n 1 || true)"
  if [ -n "$off" ]; then
    read -r a b <<< "$(echo "$off" | grep -oE '[0-9]+/[0-9]+' | tr '/' ' ')"
    [ "$a" = "$b" ] || die "only $off; the file does not fit fully on the GPU"
  else
    say "WARN $TAG: no 'offloaded N/N layers' line in server.log; check full offload by hand"
  fi
  say "PASS $TAG serve: up after ${t}s, $off, GPU used $(gpu_used_mib) MiB"
}

# ---------------------------------------------------------------- suites

run_suite() {   # $1 suite
  local s="$1" limit rc=0 extra=()
  limit="$(suite_limit "$s")"
  [ "$limit" = 0 ] || extra=(--limit "$limit")
  # coreutils timeout on the pod; absent on macOS, where only DRY runs happen
  local tmo=(); command -v timeout > /dev/null && tmo=(timeout "$SUITE_TIMEOUT")
  ${tmo[@]+"${tmo[@]}"} python3 "$SCRIPTS/evalsuite/run_suite.py" \
    --base-url "$BASE_URL/v1" --model "$TAG" --tag "$TAG" --out-root "$OUT" --suites "$s" \
    --effort "$EFFORT" --temperature "$TEMP" --top-p "$TOP_P" --seed "$SEED" --repeats 1 \
    --think-cap "$THINK_CAP" --answer-cap "$ANSWER_CAP" --concurrency "$CONCURRENCY" \
    --effort-transport kwargs --force-mode completions --skip-probe \
    --extra-body '{"top_k": 0, "min_p": 0.0}' \
    --data-dir "$EVAL_DATA" --agent-dir "$AGENT_DIR" ${extra[@]+"${extra[@]}"} > "$TAG_DIR/run_$s.log" 2>&1 || rc=$?
  # run_suite.py rewrites <tag>/meta.json on every call; keep each suite's copy
  [ -f "$TAG_DIR/meta.json" ] && cp "$TAG_DIR/meta.json" "$TAG_DIR/meta_$s.json"
  if [ "$rc" = 0 ] && suite_done "$s"; then
    say "PASS $TAG/$s $("${LIB[@]}" brief --summary "$TAG_DIR/$s/summary.json")"
    return 0
  fi
  # the run log holds only numbers and errors (run_suite prints summaries, never answers; --skip-probe)
  say "FAIL $TAG/$s rc=$rc (see $TAG_DIR/run_$s.log): $(tail -n 2 "$TAG_DIR/run_$s.log" | tr '\n' ' ' | head -c 300)"
  return 1
}

sanity_blocks_agent() {   # 0 = skip the agent suite: the build fails the gsm8k floor
  [ "${FORCE_AGENT:-0}" = 1 ] && return 1
  suite_done pub_gsm8k || return 1
  local r
  if r="$("${LIB[@]}" sanity --summary "$TAG_DIR/pub_gsm8k/summary.json" --min "$SANITY_MIN")"; then return 1; fi
  say "FAIL $TAG sanity: gsm8k $r < $SANITY_MIN; skipping agent suites (FORCE_AGENT=1 to run them)"
  return 0
}

# ---------------------------------------------------------------- main

main() {
  preflight
  local todo
  todo="$(todo_suites)"
  if [ -z "$todo" ]; then
    say "PASS $TAG: [${SUITES[*]}] done earlier"
    local s; for s in "${SUITES[@]}"; do say "PASS $TAG/$s $("${LIB[@]}" brief --summary "$TAG_DIR/$s/summary.json")"; done
    echo "GGUF_EVAL_DONE $TAG" | tee -a "$LOG"; return 0
  fi
  if [ -n "${DRY_BASE_URL:-}" ]; then
    BASE_URL="$DRY_BASE_URL"; CONCURRENCY="${NP:-8}"
    say "INFO $TAG: DRY run against $BASE_URL (no GPU, no server)"
  else
    exec 9> "$OUT/.gpu.lock"
    flock -n 9 || { say "INFO $TAG: waiting for the GPU lock ($OUT/.gpu.lock)"; flock 9; }
    trap stop_server EXIT
    trap 'exit 130' INT TERM
    BASE_URL="http://127.0.0.1:$PORT"
    start_server
  fi
  # strict on the pod: a server that ignores the effort or leaks thinking into the answer would make every
  # no-code number wrong. Not strict in a DRY run (the mock has no /apply-template in field style).
  local probe_args=(probe --base "$BASE_URL" --out "$TAG_DIR/probe.json")
  [ -n "${DRY_BASE_URL:-}" ] || probe_args+=(--strict)
  if "${LIB[@]}" "${probe_args[@]}" > "$TAG_DIR/probe.log" 2>&1; then
    say "PASS $TAG probe: $(head -c 600 "$TAG_DIR/probe.log")"
  else
    die "probe: $(head -c 600 "$TAG_DIR/probe.log")"
  fi
  local s fails=0
  # sanity suites before the agent suite, whatever order they were given in
  for s in $(printf '%s\n' $todo | grep '^pub_' || true) $(printf '%s\n' $todo | grep -v '^pub_' || true); do
    case "$s" in agent_*) if sanity_blocks_agent; then fails=$((fails + 1)); continue; fi ;; esac
    run_suite "$s" || fails=$((fails + 1))
  done
  stop_server
  say "$([ "$fails" = 0 ] && echo PASS || echo FAIL) $TAG: $fails suite(s) failed; results $TAG_DIR"
  echo "GGUF_EVAL_DONE $TAG" | tee -a "$LOG"
  [ "$fails" = 0 ]
}

main
