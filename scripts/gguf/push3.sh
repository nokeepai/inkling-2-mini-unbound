#!/bin/bash
# Upload named files from R/scripts/gguf/ to /root/gguf/ on ink-gguf-1.
R=/Users/nikshepsvn/Code/inkling-abl
for f in "$@"; do bash $R/scripts/ink_lium.sh scp ink-gguf-1 $R/scripts/gguf/$f /root/gguf/$f 2>&1 | grep -i -E "fail|error"; done
echo pushed "$@"
