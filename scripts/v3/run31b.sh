#!/bin/bash
# ON ink-v31-eval: v3.1 = full-strength merges of the saved v3 LoRAs + loop decoding test. Unattended.
#   1. after FAST_DONE: decoding test on v2 Q8 (e0.9, 240 test rows): DRY, temp 0.6, temp 0.6 + DRY  (8 GPUs)
#   2. after V31_SETUP_DONE (in parallel, CPU): merge + GGUF for A/B/C
#   3. KLD for A/B/C (3 GPUs), then full eval of C, B, A in turn (8 GPUs each)
set -u
export V3_REFUSALS=/root/harness/refusal_phrases.txt CMP_TOK=/data/compare/models/tok
cd /root/v11b
V=/root/v3; O=$V/ev; mkdir -p $O /data/v11 /data/cands /data/logs
log() { echo "$(date -u +%T) $*" >> /root/logs/run31.log; }

# ---- 2. merges in the background (needs BF16 GGUF + v2 HF ckpt) ----
build() {  # name stack...
  local N=$1; shift; mkdir -p /data/cands/$N
  python3 merge_stack.py /data/v2hf hf_name_map.json /data/cands/$N/tensors.safetensors "$@" > /data/logs/merge_$N.log 2>&1 \
    && bash build_gguf.sh $N /data/cands/$N/tensors.safetensors > /data/logs/gguf_$N.log 2>&1
  log "built $N: $(tail -1 /data/logs/gguf_$N.log)"
}
(
  until [ -f /root/big/bf16.done ]; do sleep 15; done
  log "bf16 ready"
  build v31C $V/loras/sft1.pt:1.0 $V/loras/dpo2.pt:1.0
  build v31B $V/loras/sft1.pt:1.0
  build v31A $V/loras/sft1.pt:0.6 $V/loras/dpo2.pt:1.0
  touch /root/logs/builds.done
) &

# ---- 3. KLD then evals ----
until [ -f /root/logs/builds.done ]; do sleep 15; done
bash kld.sh base_warm /data/base/Inkling-2-Mini-Q8_0.gguf 0 > /dev/null 2>&1   # builds base.kld once
g=0; for N in v31C v31B v31A; do bash kld.sh $N /data/v11/$N-Q8_0.gguf $g > /dev/null 2>&1 & g=$((g+1)); done; wait
for N in v31C v31B v31A; do log "kld $N $(grep -a 'Mean    KLD' $O/$N/kld.txt)"; done
for N in v31C v31B v31A; do
  bash eval_cand.sh $N /data/v11/$N-Q8_0.gguf 0,1,2,3,4,5,6,7 all > /root/logs/ev_$N.log 2>&1
  for f in /root/logs/srv_*.pid; do kill $(cat $f) 2>/dev/null; done
  for p in $(pgrep -x llama-server); do kill $p; done; sleep 8
  log "eval $N done"
done
log "RUN31_DONE"
