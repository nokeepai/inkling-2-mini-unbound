#!/bin/bash
# Encrypt R/sealed/ into R/sealed.tar.enc with a passphrase stored OUTSIDE the repo.
# Verifies a full decrypt round-trip (byte-identical tar) before removing the plaintext.
set -euo pipefail
R=/Users/nikshepsvn/Code/inkling-abl
PASS=/Users/nikshepsvn/.inkling-abl/sealed.pass
mkdir -p /Users/nikshepsvn/.inkling-abl
chmod 700 /Users/nikshepsvn/.inkling-abl
if [ ! -f "$PASS" ]; then
  openssl rand -base64 48 > "$PASS"
  chmod 600 "$PASS"
fi
TMP=$(mktemp -d)
tar -C "$R" -cf "$TMP/sealed.tar" sealed
openssl enc -aes-256-cbc -pbkdf2 -iter 200000 -salt -in "$TMP/sealed.tar" -out "$R/sealed.tar.enc" -pass "file:$PASS"
openssl enc -d -aes-256-cbc -pbkdf2 -iter 200000 -in "$R/sealed.tar.enc" -out "$TMP/check.tar" -pass "file:$PASS"
A=$(shasum -a 256 "$TMP/sealed.tar" | cut -d' ' -f1)
B=$(shasum -a 256 "$TMP/check.tar" | cut -d' ' -f1)
if [ "$A" != "$B" ]; then echo "ROUND-TRIP MISMATCH; plaintext kept" >&2; exit 1; fi
tar -tf "$TMP/check.tar" | grep -c '\.jsonl$' | sed 's/^/jsonl files in archive: /'
rm -P "$TMP/sealed.tar" "$TMP/check.tar" 2>/dev/null || rm -f "$TMP/sealed.tar" "$TMP/check.tar"
rmdir "$TMP"
rm -rf "$R/sealed"
chmod 600 "$R/sealed.tar.enc"
echo "plaintext_tar_sha256=$A"
echo "ciphertext_sha256=$(shasum -a 256 "$R/sealed.tar.enc" | cut -d' ' -f1)"
echo "passphrase at $PASS (outside repo)"
