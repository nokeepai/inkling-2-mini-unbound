#!/bin/bash
set -eu
L="bash /Users/nikshepsvn/Code/inkling-abl/scripts/ink_lium.sh"
P=/Users/nikshepsvn/.inkling-abl/compare_private/data
H=/Users/nikshepsvn/Code/inkling-abl/harness
$L exec ink-gguf-1 "mkdir -p /root/compare/data /root/compare/bin; chmod 700 /root/compare/data"
$L scp ink-gguf-1 $P/harmful.jsonl /root/compare/data/harmful.jsonl
$L scp ink-gguf-1 $P/borderline.jsonl /root/compare/data/borderline.jsonl
for f in /Users/nikshepsvn/Code/inkling-abl/scripts/compare/llama_eval.py $H/refusal_phrases.txt $H/cap/mmlu300.jsonl $H/cap/gsm8k100.jsonl; do
  $L scp ink-gguf-1 $f /root/compare/bin/$(basename $f)
done
$L exec ink-gguf-1 "wc -l /root/compare/data/*.jsonl; ls -la /root/compare/bin"
