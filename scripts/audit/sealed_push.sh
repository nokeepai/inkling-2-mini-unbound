#!/bin/bash
# Auditor (Mac): decrypt R/sealed.tar.enc into a private dir OUTSIDE the repo and push the test split to the audit pods.
# Usage: SEALED_PASS_FILE=<file holding the passphrase> bash sealed_push.sh <ink-pod> [<ink-pod> ...]
# Never prints test text. Plain files: ~/.inkling-abl/audit_plain/sealed/test/*.jsonl (dir 700, files 600);
# on pods: /srv/audit/data/test/ (dir 700). Verifies the per-file sha256 against harness/DATA_CARD.md.
set -euo pipefail
R=/Users/nikshepsvn/Code/inkling-abl
P=/Users/nikshepsvn/.inkling-abl/audit_plain
L="bash $R/scripts/ink_lium.sh"
: "${SEALED_PASS_FILE:?set SEALED_PASS_FILE}"
umask 077
mkdir -p "$P"
chmod 700 "$P"
if [ ! -f "$P/sealed/test/harmful.jsonl" ]; then
  openssl enc -d -aes-256-cbc -pbkdf2 -iter 200000 -in "$R/sealed.tar.enc" -pass "file:$SEALED_PASS_FILE" | tar -C "$P" -xf -
fi
echo "plaintext tar sha256: $(openssl enc -d -aes-256-cbc -pbkdf2 -iter 200000 -in "$R/sealed.tar.enc" -pass "file:$SEALED_PASS_FILE" | shasum -a 256 | cut -d' ' -f1)"
for f in borderline harmful harmless probes; do
  got=$(shasum -a 256 "$P/sealed/test/$f.jsonl" | cut -d' ' -f1)
  want=$(grep "sealed/test/$f.jsonl" "$R/harness/DATA_CARD.md" | grep -oE '[0-9a-f]{64}')
  [ "$got" = "$want" ] && echo "test/$f.jsonl sha OK ($(wc -l < "$P/sealed/test/$f.jsonl") rows)" || { echo "test/$f.jsonl SHA MISMATCH"; exit 1; }
done
for POD in "$@"; do
  $L exec "$POD" "mkdir -p /srv/audit/data/test && chmod 700 /srv/audit/data /srv/audit/data/test" >/dev/null
  $L rsync "$POD" "$P/sealed/test/" /srv/audit/data/test/ >/dev/null
  $L exec "$POD" "chmod 600 /srv/audit/data/test/*.jsonl; wc -l /srv/audit/data/test/*.jsonl | tail -1"
done
echo SEALED_PUSH_DONE
