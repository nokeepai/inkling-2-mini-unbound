#!/bin/bash
set -euo pipefail
P=/Users/nikshepsvn/.inkling-abl/compare_private
mkdir -p $P; chmod 700 $P
openssl enc -d -aes-256-cbc -pbkdf2 -iter 200000 -pass file:/Users/nikshepsvn/.inkling-abl/sealed.pass -in /Users/nikshepsvn/Code/inkling-abl/sealed.tar.enc -out $P/sealed.tar
tar -tf $P/sealed.tar
mkdir -p $P/plain; tar -xf $P/sealed.tar -C $P/plain; rm $P/sealed.tar
find $P/plain -name '*.jsonl' -exec wc -l {} \;
