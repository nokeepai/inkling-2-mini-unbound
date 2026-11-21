#!/bin/bash
# Auditor: stage harness (unchanged copies), audit scripts, val data and test candidates, then rsync to a pod's /srv/audit.
# Usage: bash push.sh <ink-pod> [cand_dir ...]      (candidates land in /srv/audit/cands/<basename>/ and /srv/cands/<basename>/)
set -euo pipefail
R=/Users/nikshepsvn/Code/inkling-abl
POD=$1; shift
L="bash $R/scripts/ink_lium.sh"
ST=$(mktemp -d /private/tmp/claude-501/audit_stage.XXXX)
mkdir -p "$ST/harness/cap" "$ST/scripts" "$ST/data/val"
for f in sgl_eval.py fleet_eval.py cands.py build_cand.py sgl_merge.py kl_eval.py judge_alt.py judge.py fast_aq.py \
         hf_name_map.json refusal_phrases.txt refusal_phrases_seed.txt screen_ids.json borderline200_ids.json; do
  cp "$R/harness/$f" "$ST/harness/"
done
cp "$R"/harness/cap/*.jsonl "$ST/harness/cap/"
cp "$R"/scripts/audit/*.py "$R"/scripts/audit/*.sh "$ST/scripts/"
cp "$R"/data/val/*.jsonl "$ST/data/val/"
( cd "$ST" && shasum -a 256 harness/*.py harness/*.json harness/*.txt harness/cap/*.jsonl > harness_sha256.txt )
# every staged harness file must match its line in harness/SEALED (the seal is the reference, not the working tree)
( cd "$ST" && grep -E '^[0-9a-f]{64}  harness/' "$R/harness/SEALED" | while read -r h f; do
    [ -f "$f" ] && echo "$h  $f"; done > sealed_subset.txt && shasum -a 256 -c sealed_subset.txt | grep -v ': OK$' && { echo "HARNESS != SEALED"; exit 1; } || true )
echo "staged harness files matching SEALED: $(wc -l < "$ST/sealed_subset.txt")"
$L rsync "$POD" "$ST/" /srv/audit/ >/dev/null
$L exec "$POD" "mkdir -p /srv/audit/cands /srv/cands /srv/audit/evals /srv/audit/out /root/logs" >/dev/null
for c in "$@"; do
  n=$(basename "$c")
  $L rsync "$POD" "$c/" "/srv/audit/cands/$n/" >/dev/null
done
rm -rf "$ST"
$L exec "$POD" "mkdir -p /srv/cands /srv/audit/evals /srv/audit/out /root/logs; for d in /srv/audit/cands/*/; do n=\$(basename \$d); [ -e /srv/cands/\$n ] || cp -r \$d /srv/cands/\$n; done; ls /srv/audit /srv/audit/cands; cd /srv/audit && sha256sum -c harness_sha256.txt | grep -vc ': OK' || true"
