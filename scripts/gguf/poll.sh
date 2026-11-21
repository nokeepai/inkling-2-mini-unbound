#!/bin/bash
# poll.sh <seconds> "<remote cmd>": wait, then run a command on ink-gguf-1.
sleep "$1"
bash /Users/nikshepsvn/Code/inkling-abl/scripts/ink_lium.sh exec ink-gguf-1 "$2" 2>&1 | grep -v '^Executing on'
