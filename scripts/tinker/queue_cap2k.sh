#!/bin/bash
# Start w4_cap2k when w2_long's free training slot frees.
LAB="$HOME/.inkling-abl/tinker_lab"
HERE="$(cd "$(dirname "$0")" && pwd)"
until grep -q "train exit" "$LAB/logs/train_w2_long.stdout" 2>/dev/null; do sleep 15; done
"$HERE/launch_w4.sh" w4_cap2k
