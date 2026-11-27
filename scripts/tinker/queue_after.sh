#!/bin/bash
# Launch <arm> (see launch_w4.sh) when <wait_tag>'s training exits and frees its slot.
#   scripts/tinker/queue_after.sh <wait_tag> <arm>
LAB="$HOME/.inkling-abl/tinker_lab"
HERE="$(cd "$(dirname "$0")" && pwd)"
until grep -q "train exit" "$LAB/logs/train_$1.stdout" 2>/dev/null; do sleep 15; done
sleep 10
"$HERE/launch_w4.sh" "$2"
