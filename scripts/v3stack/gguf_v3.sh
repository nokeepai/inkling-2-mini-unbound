#!/bin/bash
# ON POD, CPU only. v3 GGUF ladder for one merged checkpoint (attention + experts changed), same recipe as rel21.sh:
#   prove (once per official GGUF, shared by every variant) -> patch BF16 -> verify -> Q8_0 + Q6_K/Q5_K_M/Q4_K_M/IQ3_M/IQ2_M
#   (imatrix, attention kept q8_0) -> SHA256SUMS / SIZES / MANIFEST.tsv
#
#   bash gguf_v3.sh <stock_ckpt_dir> <merged_ckpt_dir> <out_dir> <name>
#   e.g. bash gguf_v3.sh /data/orig /srv/stack_w/v21_full_sr /data/v3gguf/unbound Inkling-2-Mini-Unbound-v3
#
# <stock_ckpt_dir> must be the checkpoint the official GGUF was converted from (thinkingmachines @ed40e7da), also for
# a merge built on v2.1. Rerunning resumes: finished steps (their output exists) are skipped.
# Env: OFF (official BF16 GGUF), IM (imatrix), B (llama.cpp bin dir), PROOF (shared proof file).
set -u
S=$(cd "$(dirname "$0")" && pwd)
STOCK=$1; MERGED=$2; R=$3; N=$4
OFF=${OFF:-/data/base/Inkling-2-Mini-BF16.gguf}; IM=${IM:-/data/base/imatrix.gguf}
B=${B:-/data/llama.cpp/build/bin}; PROOF=${PROOF:-/data/v3gguf/proof/proof.json}
mkdir -p "$R/logs" "$R/verify" "$(dirname "$PROOF")"; L=$R/logs/gguf_v3.log
log() { echo "$(date -u +%T) $*" | tee -a "$L"; }
step() {  # step TAG cmd...: run, log rc + last line, stop the ladder on failure
  local tag=$1; shift
  "$@" > "$R/logs/$tag.log" 2>&1; local rc=$?
  log "$tag" rc=$rc "$(tail -n1 "$R/logs/$tag.log" | cut -c1-120)"
  [ $rc -eq 0 ] || exit $rc
}
P="python3 $S/gguf_v3.py"
log START "$N" "$(df -h "$R" | tail -n1 | tr -s ' ' | cut -d' ' -f4) free"
(cd "$B/../.." && git rev-parse HEAD 2>/dev/null | grep -q '^19bc41dbb') || log "WARNING llama.cpp is not at 19bc41dbb"

# ---- 1. map + prove (stock checkpoint vs official BF16 GGUF; once) ----
step MAP $P map --ckpt "$STOCK" --gguf "$OFF" --out "$R/map.json"
if [ ! -f "$PROOF" ]; then
  step PROVE_SAMPLE $P prove --base-ckpt "$STOCK" --gguf "$OFF" --out "${PROOF%.json}_sample.json"
  step PROVE_FULL $P prove --base-ckpt "$STOCK" --gguf "$OFF" --out "$PROOF" --full
fi

# ---- 2. patch + verify ----
[ -f "$R/$N-BF16.gguf" ] || step PATCH $P patch --base-ckpt "$STOCK" --merged-ckpt "$MERGED" --gguf "$OFF" \
  --proof "$PROOF" --out "$R/$N-BF16.gguf" --name "$N" --report "$R/patch_report.json"
[ -f "$R/verify/verify.json" ] || step VERIFY $P verify --gguf "$R/$N-BF16.gguf" --official "$OFF" \
  --merged-ckpt "$MERGED" --proof "$PROOF" --out-dir "$R/verify"

# ---- 3. quantise: exactly rel21.sh's flags ----
[ -f "$R/$N-Q8_0.gguf" ] || step QUANT_Q8_0 "$B/llama-quantize" "$R/$N-BF16.gguf" "$R/$N-Q8_0.gguf" Q8_0
KEEP="--tensor-type attn_q=q8_0 --tensor-type attn_k=q8_0 --tensor-type attn_v=q8_0 --tensor-type attn_output=q8_0 --tensor-type attn_gate=q8_0"
for T in Q6_K Q5_K_M Q4_K_M IQ3_M IQ2_M; do
  [ -f "$R/$N-$T.gguf" ] && continue
  ( CUDA_VISIBLE_DEVICES="" "$B/llama-quantize" --imatrix "$IM" $KEEP "$R/$N-BF16.gguf" "$R/$N-$T.gguf.part" $T 6 \
      > "$R/logs/q_$T.log" 2>&1 && mv "$R/$N-$T.gguf.part" "$R/$N-$T.gguf"; log QUANT $T rc=$? ) &
done
wait
for T in Q8_0 Q6_K Q5_K_M Q4_K_M IQ3_M IQ2_M; do [ -f "$R/$N-$T.gguf" ] || { log "MISSING $T"; exit 1; }; done

# ---- 4. checksums ----
(cd "$R" && sha256sum "$N"-*.gguf > SHA256SUMS && ls -l "$N"-*.gguf > SIZES \
  && { printf 'file\tbytes\tsha256\n'; while read -r h f; do printf '%s\t%s\t%s\n' "$f" "$(stat -c %s "$f")" "$h"; done < SHA256SUMS; } > MANIFEST.tsv)
log GGUF_V3_DONE "$(wc -l < "$R/SHA256SUMS") files"
