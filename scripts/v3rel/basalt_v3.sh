#!/bin/bash
# ON a Linux pod (CPU only, no GPU needed): build the v3 Basalt pack from a merged v3 checkpoint and verify it.
#
#   bash basalt_v3.sh CKPT_DIR OUT_DIR [FILE_NAME] [PACK_NAME] [STOCK_CKPT_DIR]
#   e.g. bash basalt_v3.sh /srv/stack_w/unbound /srv/basalt_v3 \
#          Inkling-2-Mini-Unbound-v3-NVFP4-q8.basalt "Inkling 2 Mini Unbound v3"
#
# Prerequisites (checked below, the script stops at the first one missing):
#   - Linux, python3 >= 3.10 with the venv module, git, curl, sha256sum, the `hf` CLI (huggingface_hub; the official
#     pack is public, no token is used or needed).
#   - Converter: github.com/jesdga95/basalt tag v0.5.5 = commit ff0b978d2ee9ff195d113b6feb80961b2d2f99ef, the same
#     tool and recipe as v1.0/v2/v2.1 (tools/convert/convert.py --checkpoint; inkling2 path checkpoint_inkling2.py).
#     Its pinned requirements.txt (numpy 2.5.3 on python >= 3.12) go into $B/venv.
#   - Official pack to diff against: jesdga/Inkling-2-Mini-Basalt Inkling-2-Mini-NVFP4-DQ8.basalt at revision
#     6c9fbf0730a6d2e8813a1b262c2ab3548cf0239e (19.73 GB, sha256 f7b4e5f8...3787; the 0.5.5 pack, earlier named -q8).
#     It is NOT on main any more (main now has the 0.5.6 -MX8 pack), so rel21.sh's download line would fail today.
#   - CKPT_DIR: the merged checkpoint in the stock layout (config.json, hf_quant_config.json, model-00001-of-00001,
#     mtp.safetensors, processor_config.json, generation_config.json, tokenizer files) AND a README.md whose front
#     matter says `license: apache-2.0` (the converter copies it into gguf_kv general.license; merge_adapter.py
#     hard-links the base's README, stock or v2.1, which both have it).
#   - Disk: about 45 GB free under OUT_DIR (pack 19.7 GB + official 19.7 GB + temp); the checkpoint is read in place.
#   - CPU/RAM/time: v1.0's convert took 38 s with --workers 24 (logged in out/basalt/logs/convert_run.log); verify
#     re-derives every expert and tensor (~15 GB read twice) in a few minutes on 24 cores; the official download is
#     the long pole (minutes, bandwidth-bound). Plan for 15-30 min end to end and ~16 GB RAM.
# Engine-level checks (`basalt serve` on a Blackwell GPU, CUDA 13, driver R580+) are separate and not done here.
set -euo pipefail
CK=${1:?CKPT_DIR}; OUT=${2:?OUT_DIR}
FILE=${3:-Inkling-2-Mini-Unbound-v3-NVFP4-q8.basalt}
NAME=${4:-Inkling 2 Mini Unbound v3}
STOCK=${5:-}
B=${BASALT_HOME:-/root/b}
WORKERS=${WORKERS:-24}
TAG=v0.5.5; COMMIT=ff0b978d2ee9ff195d113b6feb80961b2d2f99ef
OFF_REPO=jesdga/Inkling-2-Mini-Basalt; OFF_FILE=Inkling-2-Mini-NVFP4-DQ8.basalt
OFF_REV=6c9fbf0730a6d2e8813a1b262c2ab3548cf0239e
HERE=$(cd "$(dirname "$0")" && pwd)
[ "$(uname -s)" = Linux ] || { echo "STOP: Linux pods only (paths and multiprocessing are pod-tested)"; exit 1; }
mkdir -p "$OUT" "$B"; L=$OUT/basalt_v3.log
log() { echo "$(date -u +%T) $*" | tee -a "$L"; }
die() { log "STOP: $*"; exit 1; }

# ---- prerequisites
for t in python3 git curl sha256sum hf df; do command -v $t > /dev/null || die "missing tool: $t"; done
python3 -c 'import sys, venv; assert sys.version_info >= (3, 10)' || die "need python3 >= 3.10 with venv"
for f in config.json hf_quant_config.json model-00001-of-00001.safetensors model.safetensors.index.json \
         mtp.safetensors processor_config.json generation_config.json tokenizer.json tokenizer_config.json README.md; do
  [ -f "$CK/$f" ] || die "checkpoint lacks $f"
done
grep -qE '^license: *apache-2.0' "$CK/README.md" || die "$CK/README.md has no 'license: apache-2.0' front matter"
FREE=$(df -Pk "$OUT" | awk 'NR==2 {print int($4/1048576)}')
[ "$FREE" -ge 45 ] || die "only ${FREE} GB free under $OUT, need 45"
log "START ck=$CK out=$OUT file=$FILE name='$NAME' free=${FREE}GB cpus=$(nproc) workers=$WORKERS"

# ---- converter at the pinned commit
if [ ! -d "$B/src/.git" ]; then git clone -q --depth 1 --branch $TAG https://github.com/jesdga95/basalt "$B/src"; fi
[ "$(git -C "$B/src" rev-parse HEAD)" = $COMMIT ] || die "basalt src is not $TAG ($COMMIT)"
[ -x "$B/venv/bin/python" ] || { python3 -m venv "$B/venv" && "$B/venv/bin/pip" install -q -r "$B/src/requirements.txt"; }
"$B/venv/bin/python" -c 'import numpy' || die "venv lacks numpy"
log "TOOLS basalt $(git -C "$B/src" rev-parse --short HEAD), $("$B/venv/bin/python" -c 'import numpy; print("numpy", numpy.__version__)')"

# ---- official 0.5.5 pack (by revision), in the background while converting
( [ -f "$B/official/$OFF_FILE" ] || hf download $OFF_REPO $OFF_FILE --revision $OFF_REV --local-dir "$B/official" \
    > "$OUT/dl_official.log" 2>&1; echo "rc=$?" >> "$OUT/dl_official.log" ) &
DL=$!

# ---- convert
T0=$(date +%s)
( cd "$B/src" && "$B/venv/bin/python" tools/convert/convert.py --checkpoint "$CK" --out "$OUT/$FILE" --name "$NAME" \
    --workers "$WORKERS" --force ) > "$OUT/convert.log" 2>&1 || die "convert failed: $(tail -n2 "$OUT/convert.log")"
log "CONVERT $(( $(date +%s) - T0 ))s $(tail -n1 "$OUT/convert.log" | cut -c1-120)"
(cd "$OUT" && sha256sum "$FILE" > "$FILE.sha256"); log "SHA $(cut -c1-64 "$OUT/$FILE.sha256") $(stat -c %s "$OUT/$FILE") bytes"

wait $DL || true
grep -q 'rc=0' "$OUT/dl_official.log" || [ -f "$B/official/$OFF_FILE" ] || die "official pack download failed (see $OUT/dl_official.log)"

# ---- verify (v3 rules: every byte official or converter(merged); see verify_basalt_v3.py)
T0=$(date +%s)
"$B/venv/bin/python" "$HERE/verify_basalt_v3.py" "$OUT/$FILE" "$B/official/$OFF_FILE" "$CK" --basalt-src "$B/src" \
    --workers "$WORKERS" ${STOCK:+--stock-checkpoint "$STOCK"} > "$OUT/basalt_verify.log" 2>&1 && RC=0 || RC=$?
log "VERIFY rc=$RC $(( $(date +%s) - T0 ))s $(grep -E '^RESULT' "$OUT/basalt_verify.log" | cut -c1-160)"
grep -E '^  (experts layer|gguf:|independent)' "$OUT/basalt_verify.log" | tail -n 8 | tee -a "$L"
[ $RC -eq 0 ] || die "verify failed: see $OUT/basalt_verify.log"
log "ALL_DONE $OUT/$FILE (verify log: $OUT/basalt_verify.log, for golive_v3.py basalt-upload --verify-log)"
