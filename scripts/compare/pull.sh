#!/bin/bash
# Mac-side: pull all outputs + logs (they contain test prompts) into the private dir only.
P=/Users/nikshepsvn/.inkling-abl/compare_private
L="bash /Users/nikshepsvn/Code/inkling-abl/scripts/ink_lium.sh"
mkdir -p $P/pull; chmod 700 $P
$L exec ink-gguf-1 "cd /root/compare && tar czf /root/compare/pull.tgz out logs" > /dev/null
$L scp ink-gguf-1 /root/compare/pull.tgz $P/pull/ -d > /dev/null
tar xzf $P/pull/pull.tgz -C $P
rm $P/pull/pull.tgz
ls $P/out
