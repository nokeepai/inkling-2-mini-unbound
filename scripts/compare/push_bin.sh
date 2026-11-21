#!/bin/bash
L="bash /Users/nikshepsvn/Code/inkling-abl/scripts/ink_lium.sh"
for f in "$@"; do $L scp ink-gguf-1 /Users/nikshepsvn/Code/inkling-abl/scripts/compare/$f /root/compare/bin/$f > /dev/null; done
$L exec ink-gguf-1 "ls -la /root/compare/bin"
