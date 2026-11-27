#!/usr/bin/env bash
# Stacking test, ON THE POD (one GPU). Does Unbound v2.1 + the stock-trained anti-stall adapter keep v2.1's
# behaviour and gain the debugging fix? Nothing here rents, extends or stops a pod.
#
#   nohup bash /root/stack/scripts/v3stack/stack_test.sh > /root/logs/stack_test.log 2>&1 &
#   grep -E ' (PASS|FAIL|SKIP) ' /root/logs/stack_test.log
#
# Variants (weights under $W, results under $OUT/<variant>/<suite>/):
#   v21       Unbound v2.1 from Hugging Face, untouched (the reference)
#   v21_attn  v2.1 + the adapter's attention part merged (BF16)          merge_adapter.py --parts attn
#   v21_full  v2.1 + attention + expert part (NVFP4/MXFP8 requantised)    merge_adapter.py --parts attn_experts
#   v21_attn_sr / v21_full_sr   the same two merges with --rounding stochastic (E[stored] = W + delta)
#   stock / stock_full_sr       (STOCK=1) the untouched stock checkpoint, and stock + full adapter, stochastic
# All of them are full checkpoints served by the same engine: the frozen SGLang image pinned by digest
# (harness/ENGINE.md, state/pins.json), one container, --tp 1. No variant needs runtime adapter loading.
#
# Order: pass 1 = agent_hard_noforce, ubench (the two answers) for every variant; pass 2 = main, hardmath, code, rp.
# So a run cut short after pass 1 still answers the question. Generation only; judging is a separate command
# (README.md). Idempotent: every stage has a marker in $STATE, a suite with summary.json is skipped, a merge whose
# output dir exists is skipped. Rerun the same command after a crash or preemption.
#
# Env (defaults in brackets): STACK [/root/stack] holding scripts/, adapter/, eval_data/, uncensoredbench/;
# W [/srv/stack_w] weights (the pod's docker cannot bind-mount the /root FUSE mount: ENGINE.md); OUT
# [/root/stack_evals]; VARIANTS ["v21 v21_attn v21_full"]; STOCK=1 appends "stock stock_full_sr"; PASS1 / PASS2
# suite lists (PASS2="" = pass 1 only: agent_hard_noforce then ubench); MERGE_SEED [0]; UBENCH_LIMIT [220 = 5 per
# category x level cell]; GPU [0]; SERVER [sglang | vllm]; FULL_MIN_SURVIVAL [0: always serve v21_full];
# HF_TOKEN (optional, read by huggingface_hub, never printed or stored); FORCE=<stage> redoes one stage.
set -euo pipefail

STACK="${STACK:-/srv/stack}"   # not /root: the pod's /root is a FUSE mount docker cannot bind (seen on ink-v3stack)
SCRIPTS="${SCRIPTS:-$STACK/scripts}"
ADAPTER="${ADAPTER:-$STACK/adapter}"
EVAL_DATA="${EVAL_DATA:-$STACK/eval_data}"
UBENCH="${UBENCH:-$STACK/uncensoredbench}"
W="${W:-/srv/stack_w}"   # UNVERIFIED that this pod has the eval fleet's /root FUSE quirk; /srv works either way
OUT="${OUT:-/root/stack_evals}"
LOGS=/root/logs
STATE="${STATE:-/root/.stack_state}"
VARIANTS="${VARIANTS:-v21 v21_attn v21_full}"
if [ "${STOCK:-0}" = 1 ]; then
  for _v in stock stock_full_sr; do case " $VARIANTS " in *" $_v "*) ;; *) VARIANTS="$VARIANTS $_v" ;; esac; done
fi
PASS1="${PASS1-agent_hard_noforce ubench}"    # '-' not ':-': an explicitly empty PASS2="" skips pass 2
PASS2="${PASS2-main hardmath code rp}"
MERGE_SEED="${MERGE_SEED:-0}"
UBENCH_LIMIT="${UBENCH_LIMIT:-220}"           # ubench --limit spreads round-robin over 11 categories x 4 levels
EFFORT="${EFFORT:-0.9}"; TEMP="${TEMP:-0.6}"; SEED="${SEED:-1234}"
CONCURRENCY="${CONCURRENCY:-31}"              # = the frozen engine's --max-running-requests
GPU="${GPU:-0}"
PORT="${PORT:-8400}"
SERVER="${SERVER:-sglang}"
FULL_MIN_SURVIVAL="${FULL_MIN_SURVIVAL:-0}"
MERGE_DEVICE="${MERGE_DEVICE:-cpu}"           # cpu lets the merges run while the v21 server holds the GPU
MERGE_CPUS="${MERGE_CPUS:-}"                  # default: half the cores, so the server's CPU side is not starved
SERVE_TIMEOUT="${SERVE_TIMEOUT:-1800}"
SUITE_TIMEOUT="${SUITE_TIMEOUT:-7200}"
EVAL_EXTRA="${EVAL_EXTRA:-}"                  # extra run_suite.py flags, e.g. --raw-markers
MIN_FREE_GB="${MIN_FREE_GB:-25}"               # preflight floor; each fetch/merge checks its own ~21 GB before starting
VARIANT_GB="${VARIANT_GB:-21}"                # one 19.5 GB shard (+ mtp/tokenizer for a fetch) + slack
HF_REPO="nokeepai/Inkling-2-Mini-Unbound"
HF_REV="${HF_REV:-v2.1}"
# Stock checkpoint at the revision this repo pins everywhere (scripts/pod_fetch.sh, state/pins.json orig_rev).
# Its file list is HF_FILES exactly (vendor/ckpt/orig/tree.json).
STOCK_REPO="thinkingmachines/Inkling-2-Mini"
STOCK_REV="${STOCK_REV:-ed40e7da625f594fc9c4990f5f66be24873c22d7}"
# Stock-layout files, as scripts/v21/rel21.sh and scripts/v3rl/pod_setup.sh download them.
# UNVERIFIED for the v2.1 revision (v2.0 had exactly these). Fallback: HF_FILES="" downloads the whole revision.
HF_FILES="${HF_FILES-chat_template.jinja config.json generation_config.json hf_quant_config.json \
model-00001-of-00001.safetensors model.safetensors.index.json mtp.safetensors processor_config.json \
special_tokens_map.json tiktoken/tokenizer.model tokenizer.json tokenizer_config.json}"
# Frozen scoring engine (harness/ENGINE.md, state/pins.json): linux/amd64 manifest of lmsysorg/sglang:inkling-2-mini.
SGL_IMG="lmsysorg/sglang@sha256:421aed3ac11741381eac350bf7b1db28a0cafb6ae49b57ae78fba86a16a32306"
SGL_FLAGS="${SGL_FLAGS:---mem-fraction-static 0.73815 --chunked-prefill-size 4096 --max-running-requests 31 --random-seed 1234}"
# Fallback engine: the vLLM image that served the v2.1 checkpoint publicly (scripts/v21/serve_run.sh), on 2x PRO 6000.
# UNVERIFIED: tag not pinned by digest; fit on a 32 GB card. Fallback: SERVER=sglang (default).
VLLM_IMG="${VLLM_IMG:-vllm/vllm-openai:inkling-2-mini}"
# Any image with torch (float8 dtypes) + safetensors. UNVERIFIED for the SGLang image's versions; the selftest stage
# runs inside it before any merge. Fallback: MERGE_IMG=<a pytorch image> or pip install torch safetensors on the host.
MERGE_IMG="${MERGE_IMG:-$SGL_IMG}"
CNAME=stackserve
mkdir -p "$LOGS" "$STATE" "$OUT"

say() { echo "$(date -u +%H:%M:%S) $*"; }

# A stage runs in a subshell with errexit on, is skipped when its marker exists, and prints one PASS/FAIL line.
stage() {
  local name="$1"; shift
  if [ -f "$STATE/$name.ok" ] && [ "${FORCE:-}" != "$name" ]; then
    say "PASS $name (done earlier: $(head -c 200 "$STATE/$name.ok" | tr '\n' ' '))"; return 0
  fi
  local rc=0
  set +e
  ( set -euo pipefail; "$@" ) > "$LOGS/stack_$name.log" 2>&1
  rc=$?
  set -e
  if [ "$rc" = 0 ]; then
    tail -n 1 "$LOGS/stack_$name.log" > "$STATE/$name.ok"
    say "PASS $name ($(head -c 200 "$STATE/$name.ok"))"
    return 0
  fi
  say "FAIL $name rc=$rc (see $LOGS/stack_$name.log): $(tail -n 3 "$LOGS/stack_$name.log" | tr '\n' ' ' | head -c 300)"
  return 1
}

# ---------------------------------------------------------------- setup stages

st_preflight() {
  command -v nvidia-smi >/dev/null && command -v docker >/dev/null && command -v python3 >/dev/null
  nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
  local mem
  mem="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i "$GPU" | tr -d ' ')"
  [ "$mem" -ge 30000 ] || { echo "GPU $GPU has ${mem} MiB; the 26B NVFP4 checkpoint needs a 32 GB card"; exit 1; }
  python3 -c 'import sys; assert sys.version_info >= (3, 9), sys.version'   # evalsuite uses dict[...] aliases
  local f
  for f in "$ADAPTER/adapter_config.json" "$ADAPTER/adapter_model.safetensors" \
           "$SCRIPTS/v3stack/merge_adapter.py" "$SCRIPTS/v3stack/selftest_merge.py" "$SCRIPTS/evalsuite/run_suite.py" \
           "$EVAL_DATA/prompts_eval.jsonl" "$EVAL_DATA/prompts_hardmath.jsonl" "$EVAL_DATA/prompts_code.jsonl" \
           "$EVAL_DATA/rp_eval.jsonl" "$UBENCH/tools/judge_v2.py" "$UBENCH/questions/drugs.jsonl"; do
    [ -f "$f" ] || { echo "missing $f (README.md step 2 rsyncs it)"; exit 1; }
  done
  [ -d "$SCRIPTS/tinker/agent_hard_tasks/t01_inventory_ledger" ] || { echo "missing agent_hard_tasks"; exit 1; }
  mkdir -p "$W"
  local free
  free="$(df -P -BG "$W" | awk 'NR==2 {gsub("G","",$4); print $4}')"
  [ "$free" -ge "$MIN_FREE_GB" ] || { echo "only ${free} GB free under $W, need $MIN_FREE_GB"; exit 1; }
  local v missing=0
  for v in $VARIANTS; do
    variant_spec "$v" > /dev/null || { echo "unknown variant $v"; exit 1; }
    [ -d "$W/$v" ] || missing=$((missing + 1))
  done
  echo "preflight ok: GPU $GPU ${mem} MiB, ${free} GB free under $W; $missing variant(s) to fetch/build need" \
       "~$((missing * VARIANT_GB)) GB (checked again before each one)"
}

st_image() {
  if [ "$SERVER" = vllm ]; then docker pull -q "$VLLM_IMG"; fi
  docker pull -q "$SGL_IMG"
  docker image inspect "$SGL_IMG" --format '{{.Id}}'
}

st_hf_cli() {
  command -v hf >/dev/null 2>&1 || pip install -q -U "huggingface_hub[hf_transfer]" >/dev/null 2>&1 \
    || pip install -q -U huggingface_hub
  hf version 2>/dev/null || hf --help | head -n 1
}

st_fetch() {   # $1 name (v21 | stock), $2 repo, $3 revision
  local name="$1" repo="$2" rev="$3" tmp="$W/$1.download"
  if [ -f "$W/$name/model.safetensors.index.json" ]; then echo "$name already under $W"; return 0; fi
  disk_ok "fetch $name" "$VARIANT_GB"
  export HF_HUB_ENABLE_HF_TRANSFER=1 HF_HUB_DOWNLOAD_TIMEOUT=60
  mkdir -p "$tmp"
  # shellcheck disable=SC2086  # HF_FILES is a word list on purpose
  hf download "$repo" $HF_FILES --revision "$rev" --local-dir "$tmp" 2>&1 | tail -n 3
  [ -f "$tmp/model.safetensors.index.json" ] && [ -f "$tmp/hf_quant_config.json" ]
  rm -rf "$tmp/.cache"
  (cd "$tmp" && find . -type f -print0 | sort -z | xargs -0 sha256sum) > "$W/$name.sha256"
  rm -rf "${W:?}/$name"
  mv "$tmp" "$W/$name"
  echo "$name fetched from $repo@$rev: $(du -sh "$W/$name" | cut -f1)"
}

free_gb() { df -P -BG "$W" | awk 'NR==2 {gsub("G","",$4); print $4}'; }

# Before a fetch or merge: room for one more ~21 GB checkpoint? If not, name what could go (variant dirs under $W
# that this run does not use) instead of failing halfway through a 19.5 GB copy.
disk_ok() {   # $1 what, $2 GB needed
  local free d n listed=0
  free="$(free_gb)"
  [ "$free" -ge "$2" ] && return 0
  echo "not enough disk for $1: ${free} GB free under $W, need $2 GB. Delete a variant this run does not use, e.g.:"
  for d in "$W"/*/; do
    [ -d "$d" ] || continue
    n="$(basename "$d")"
    case " $VARIANTS v21 stock " in *" $n "*) continue ;; esac
    echo "  rm -rf $W/$n $W/$n.merge_report.json    # $(du -sh "$d" 2>/dev/null | cut -f1)"
    listed=1
  done
  [ "$listed" = 1 ] || echo "  (nothing unused under $W; drop a variant from VARIANTS, e.g. v21_full, and rm -rf $W/v21_full)"
  return 1
}

# variant -> "base parts rounding" for merged variants, "fetch" for downloaded ones
variant_spec() {
  case "$1" in
    v21|stock) echo fetch ;;
    v21_attn) echo "v21 attn nearest" ;;
    v21_full) echo "v21 attn_experts nearest" ;;
    v21_attn_sr) echo "v21 attn stochastic" ;;
    v21_full_sr) echo "v21 attn_experts stochastic" ;;
    stock_full_sr) echo "stock attn_experts stochastic" ;;
    *) return 1 ;;
  esac
}

needs_base() {   # $1 v21 | stock: does any variant in VARIANTS need that checkpoint?
  local v spec
  for v in $VARIANTS; do
    [ "$v" = "$1" ] && return 0
    spec="$(variant_spec "$v")" || continue
    [ "${spec%% *}" = "$1" ] && return 0
  done
  return 1
}

st_selftest() {
  docker run --rm --network none -v "$SCRIPTS/v3stack:/s:ro" --entrypoint python3 "$MERGE_IMG" \
    /s/merge_adapter.py --selftest
}

# ---------------------------------------------------------------- merges (CPU, in the image, in the background)

merge_variant() {   # $1 variant; output dir existing = merged and verified (merge_adapter.py renames last)
  local v="$1" gpus=() spec base parts rounding
  spec="$(variant_spec "$v")"
  base="${spec%% *}"; parts="$(echo "$spec" | cut -d' ' -f2)"; rounding="${spec##* }"
  if [ -d "$W/$v" ] && [ -f "$W/$v.merge_report.json" ]; then echo "merge $v: done earlier"; return 0; fi
  [ -f "$W/$base/model.safetensors.index.json" ] || { echo "base $base not fetched under $W"; return 1; }
  disk_ok "merge $v" "$VARIANT_GB"
  local cpus="${MERGE_CPUS:-$(( $(nproc) / 2 ))}"
  [ "$cpus" -ge 1 ] || cpus=1
  [ "$MERGE_DEVICE" = cpu ] || gpus=(--gpus "device=$GPU")
  docker rm -f "stackmerge_$v" > /dev/null 2>&1 || true
  docker run --rm --name "stackmerge_$v" --network none ${gpus[@]+"${gpus[@]}"} --cpus "$cpus" \
    -e OMP_NUM_THREADS="$cpus" -v "$W:/w" \
    -v "$SCRIPTS/v3stack:/s:ro" -v "$ADAPTER:/adapter:ro" --entrypoint python3 "$MERGE_IMG" \
    /s/merge_adapter.py --base "/w/$base" --adapter /adapter --out "/w/$v" --parts "$parts" --device "$MERGE_DEVICE" \
    --rounding "$rounding" --seed "$MERGE_SEED"
}

build_all() {   # sequential: one shard copy at a time keeps disk and RAM use flat
  local v
  trap - EXIT INT TERM   # never stop the GPU server from this background job
  for v in $VARIANTS; do
    [ "$(variant_spec "$v")" = fetch ] && continue
    [ -d "$W/$v" ] || rm -f "$STATE/merge_$v.ok"     # dir deleted to free disk: the old marker is stale
    stage "merge_$v" merge_variant "$v" || true
  done
}

wait_build() {   # $1 variant; 0 when its weights are ready
  local v="$1"
  if [ "$(variant_spec "$v")" = fetch ]; then [ -f "$W/$v/model.safetensors.index.json" ]; return; fi
  while [ -n "${BUILD_PID:-}" ] && kill -0 "$BUILD_PID" 2>/dev/null && [ ! -f "$STATE/merge_$v.ok" ]; do sleep 20; done
  if [ -f "$STATE/merge_$v.ok" ] && [ -d "$W/$v" ]; then
    say "PASS merge_$v ($(head -c 200 "$STATE/merge_$v.ok"))"; return 0
  fi
  return 1
}

full_gate() {   # 0 = serve v21_full. Optional skip when almost none of the expert delta survives 4-bit storage.
  local r="$W/v21_full.merge_report.json" s
  [ "$FULL_MIN_SURVIVAL" = 0 ] && return 0
  [ -f "$r" ] || return 0
  s="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["survival_all"]["survival"])' "$r")"
  say "v21_full expert delta survival (all experts): $s (FULL_MIN_SURVIVAL=$FULL_MIN_SURVIVAL)"
  python3 -c 'import sys; sys.exit(0 if float(sys.argv[1]) >= float(sys.argv[2]) else 1)' "$s" "$FULL_MIN_SURVIVAL"
}

# ---------------------------------------------------------------- serving

stop_server() { docker rm -f "$CNAME" > /dev/null 2>&1 || true; }

start_server() {   # $1 variant
  local v="$1" log="$LOGS/stack_serve_$1.log"
  stop_server
  if [ "$SERVER" = vllm ]; then
    # Flags from scripts/v21/serve_run.sh minus data-parallel and API key. UNVERIFIED on one 32 GB GPU.
    docker run -d --name "$CNAME" --gpus "device=$GPU" --ipc=host -p "127.0.0.1:$PORT:8000" -v "$W:/w" "$VLLM_IMG" \
      --model "/w/$v" --served-model-name "$v" --host 0.0.0.0 --port 8000 --max-num-seqs "$CONCURRENCY" \
      --gpu-memory-utilization 0.92 --max-model-len 65536 --trust-remote-code --reasoning-parser inkling \
      --tool-call-parser inkling --enable-auto-tool-choice > /dev/null
  else
    # The frozen engine command (harness/ENGINE.md, scripts/suite/suite_serve.sh), model dir swapped.
    # shellcheck disable=SC2086  # SGL_FLAGS is a word list on purpose
    docker run -d --name "$CNAME" --gpus "device=$GPU" --ipc=host -p "127.0.0.1:$PORT:30000" -v "$W:/w" "$SGL_IMG" \
      sglang serve --model-path "/w/$v" --tp 1 --attention-backend fa4 --dtype bfloat16 --page-size 128 \
      --enable-multimodal --reasoning-parser inkling --tool-call-parser inkling2 --host 0.0.0.0 --port 30000 \
      $SGL_FLAGS > /dev/null
  fi
  local t=0
  while [ "$t" -lt "$SERVE_TIMEOUT" ]; do
    if curl -sf "http://127.0.0.1:$PORT/health" > /dev/null 2>&1; then
      docker logs "$CNAME" > "$log" 2>&1 || true
      say "PASS serve $v ($SERVER, up after ${t}s)"; return 0
    fi
    if ! docker ps --format '{{.Names}}' | grep -qx "$CNAME"; then break; fi
    sleep 10; t=$((t + 10))
  done
  docker logs "$CNAME" > "$log" 2>&1 || true
  say "FAIL serve $v ($SERVER): not healthy after ${t}s; tail of $log: $(tail -n 5 "$log" | tr '\n' ' ' | head -c 400)"
  stop_server
  return 1
}

model_id() {
  curl -sf "http://127.0.0.1:$PORT/v1/models" | python3 -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])'
}

# Server-agnostic check that reasoning_effort reaches the chat template: the rendered prompt prints
# "Thinking effort level: <x>", so prompt_tokens must differ between efforts. run_suite.py's own probe needs a
# vLLM /tokenize or llama.cpp /apply-template route; UNVERIFIED that SGLang offers either, so after this check
# passes the suite runs with --allow-effort-unverified (it still aborts if its probe proves the effort is ignored).
effort_check() {   # $1 model id
  python3 - "http://127.0.0.1:$PORT" "$1" <<'PY'
import json, sys, urllib.request
base, model = sys.argv[1], sys.argv[2]
def prompt_tokens(effort):
    body = {"model": model, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 1, "temperature": 0,
            "chat_template_kwargs": {"reasoning_effort": effort}}
    req = urllib.request.Request(base + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=120))["usage"]["prompt_tokens"]
n = {e: prompt_tokens(e) for e in (0, 0.9, 0.12345)}
print("prompt_tokens by effort:", n)
sys.exit(0 if len(set(n.values())) > 1 else 1)
PY
}

# ---------------------------------------------------------------- suites

suite_done() { [ -f "$OUT/$1/$2/summary.json" ]; }

run_one_suite() {   # $1 variant, $2 suite, $3 model id
  local v="$1" s="$2" mid="$3" rc=0 extra=()
  if suite_done "$v" "$s"; then say "PASS $v/$s (done earlier)"; return 0; fi
  [ "$s" = ubench ] && extra=(--limit "$UBENCH_LIMIT")
  # shellcheck disable=SC2206  # EVAL_EXTRA is a word list on purpose
  [ -n "$EVAL_EXTRA" ] && extra+=($EVAL_EXTRA)
  EVALSUITE_UBENCH_TOOLS="$UBENCH/tools" timeout "$SUITE_TIMEOUT" python3 "$SCRIPTS/evalsuite/run_suite.py" \
    --base-url "http://127.0.0.1:$PORT/v1" --model "$mid" --tag "$v" --out-root "$OUT" --suites "$s" \
    --effort "$EFFORT" --temperature "$TEMP" --seed "$SEED" --repeats 1 --concurrency "$CONCURRENCY" \
    --data-dir "$EVAL_DATA" --agent-dir "$SCRIPTS/tinker/agent_hard_tasks" --ubench-dir "$UBENCH" \
    --allow-effort-unverified ${extra[@]+"${extra[@]}"} > "$LOGS/stack_${v}_$s.log" 2>&1 || rc=$?
  # run_suite.py rewrites <tag>/meta.json on every call; keep each suite's copy (server probe, args)
  [ -f "$OUT/$v/meta.json" ] && cp "$OUT/$v/meta.json" "$OUT/$v/meta_$s.json"
  if [ "$rc" = 0 ] && suite_done "$v" "$s"; then
    say "PASS $v/$s $(brief "$v" "$s")"; return 0
  fi
  say "FAIL $v/$s rc=$rc (see $LOGS/stack_${v}_$s.log): $(tail -n 2 "$LOGS/stack_${v}_$s.log" | tr '\n' ' ' | head -c 300)"
  return 1
}

brief() {   # one-line headline numbers of a finished suite
  python3 - "$OUT/$1/$2/summary.json" <<'PY' 2>/dev/null || true
import json, sys
agg = json.load(open(sys.argv[1]))["agg"]
keys = ("solve_rate", "no_code_rate", "pass_at_1", "n_ok", "n", "n_failed", "wall_seconds")
print(" ".join(f"{k}={agg[k]['mean']:.3g}" for k in keys if k in agg))
PY
}

# UNVERIFIED: that SGLang accepts the suite client's request body (stream_options.continuous_usage_stats, seed,
# chat_template_kwargs) and returns thinking in reasoning_content. This smoke proves it per variant before the suites;
# fallbacks: EVAL_EXTRA=--raw-markers (thinking split client-side), or SERVER=vllm.
smoke_client() {   # $1 variant, $2 model id: 2 main prompts end to end through the suite client (split thinking)
  local v="$1" mid="$2" d="$OUT/_smoke"
  rm -rf "${d:?}/$v"
  python3 "$SCRIPTS/evalsuite/run_suite.py" --base-url "http://127.0.0.1:$PORT/v1" --model "$mid" --tag "$v" \
    --out-root "$d" --suites main --limit 2 --effort "$EFFORT" --temperature "$TEMP" --seed "$SEED" \
    --data-dir "$EVAL_DATA" --allow-effort-unverified ${EVAL_EXTRA:+$EVAL_EXTRA} > "$LOGS/stack_smoke_$v.log" 2>&1
  python3 - "$d/$v" "$EVAL_EXTRA" <<'PY'
import json, sys
from pathlib import Path
d = Path(sys.argv[1])
probe = json.loads((d / "meta.json").read_text())["probe"]
summ = json.loads((d / "main" / "summary.json").read_text())["agg"]
split = probe.get("split_check", {})
print("probe:", json.dumps({"effort": probe.get("effort_check"), "split": split.get("split"),
                            "count_source": split.get("count_source"), "error": split.get("error")}))
assert summ["n_ok"]["mean"] == 2, f"n_ok {summ['n_ok']['mean']} of 2"
assert split.get("split") == "field" or "--raw-markers" in sys.argv[2], \
    "thinking did not come back in reasoning_content (try EVAL_EXTRA=--raw-markers)"
PY
}

serve_and_run() {   # $1 variant, $2 suite list
  local v="$1" s mid todo=""
  for s in $2; do suite_done "$v" "$s" || todo="$todo $s"; done
  if [ -z "$todo" ]; then say "PASS $v: [$2] done earlier"; return 0; fi
  if ! wait_build "$v"; then
    say "FAIL $v: weights not built: $(tail -n 4 "$LOGS/stack_merge_$v.log" 2>/dev/null | tr '\n' ' ' | head -c 500)"
    return 1
  fi
  if [ "$v" = v21_full ] && ! full_gate; then
    say "SKIP $v: expert delta survival below FULL_MIN_SURVIVAL; v21_full would be ~= v21_attn"; return 0
  fi
  start_server "$v" || return 1
  mid="$(model_id)"
  if ! effort_check "$mid" > "$LOGS/stack_effort_$v.log" 2>&1; then
    say "FAIL $v effort check: reasoning_effort does not reach the template ($(tail -n 2 "$LOGS/stack_effort_$v.log" | tr '\n' ' '))"
    stop_server; return 1
  fi
  say "PASS $v effort check ($(tail -n 1 "$LOGS/stack_effort_$v.log"))"
  if ! smoke_client "$v" "$mid" > "$LOGS/stack_smoke_check_$v.log" 2>&1; then
    say "FAIL $v client smoke: $(tail -n 2 "$LOGS/stack_smoke_check_$v.log" | tr '\n' ' ' | head -c 300)"
    stop_server; return 1
  fi
  say "PASS $v client smoke ($(grep '^probe:' "$LOGS/stack_smoke_check_$v.log" | head -c 250))"
  for s in $todo; do run_one_suite "$v" "$s" "$mid" || true; done
  stop_server
}

final_table() {
  python3 - "$OUT" "$W" $VARIANTS <<'PY' | tee "$OUT/stack_table.tsv"
import json, sys
from pathlib import Path
out, w, variants = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3:]
rows = [("agent_hard_noforce", (k,)) for k in ("solve_rate", "no_code_rate", "mean_turns_to_solve", "cap_hit_per_turn")]
rows += [("code", ("pass_at_1",))] + [(s, ("n_ok", "n", "turns")) for s in ("ubench", "main", "hardmath", "code", "rp")]
print("metric\t" + "\t".join(variants))
for suite, keys in rows:
    vals, label = [], keys[0]
    for v in variants:
        f = out / v / suite / "summary.json"
        agg = json.loads(f.read_text())["agg"] if f.exists() else {}
        hit = next((k for k in keys if k in agg), None)
        label = hit or label
        vals.append(f"{agg[hit]['mean']:.3g}" if hit else "-")
    print(f"{suite}.{label}\t" + "\t".join(vals))
surv = []
for v in variants:
    r = w / f"{v}.merge_report.json"
    s = json.loads(r.read_text()).get("survival_all", {}).get("survival") if r.exists() else None
    surv.append("-" if s is None else f"{s:.3f}")
print("expert_delta_survival\t" + "\t".join(surv))
PY
}

# ---------------------------------------------------------------- main

trap stop_server EXIT
trap 'exit 130' INT TERM
say "stack test: variants [$VARIANTS], server $SERVER, GPU $GPU, weights $W, results $OUT"
stage preflight st_preflight
stage image st_image
stage hf_cli st_hf_cli
stage selftest st_selftest
for b in v21 stock; do [ -d "$W/$b" ] || rm -f "$STATE/fetch_$b.ok"; done   # stale after a manual delete
if needs_base v21; then stage fetch_v21 st_fetch v21 "$HF_REPO" "$HF_REV"; fi
if needs_base stock; then stage fetch_stock st_fetch stock "$STOCK_REPO" "$STOCK_REV"; fi
build_all > "$LOGS/stack_build_all.log" 2>&1 &
BUILD_PID=$!
for pass_suites in "$PASS1" "$PASS2"; do
  [ -n "$pass_suites" ] || continue
  say "pass: [$pass_suites]"
  for v in $VARIANTS; do serve_and_run "$v" "$pass_suites" || true; done
done
wait "$BUILD_PID" || true
grep -E ' (PASS|FAIL) ' "$LOGS/stack_build_all.log" || true
final_table
say "STACK_DONE results $OUT; pull them back and judge locally (README.md)"
