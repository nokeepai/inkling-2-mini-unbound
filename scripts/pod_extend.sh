#!/bin/bash
# Reschedule an ink-* pod's removal to <duration> from now (max 2h per the brief). Usage: pod_extend.sh <ink-pod> [2h]
set -u
export LIUM_NONINTERACTIVE=1
export LIUM_NO_POD_INDEX=1
R=~/Code/inkling-abl
source "$R/scripts/ink_guard.sh"
NAME="${1:-}"; DUR="${2:-2h}"
ink_guard "$NAME" || exit 2
case "$DUR" in 30m|45m|60m|90m|1h|2h) ;; *) echo "pod_extend: duration must be ≤ 2h (30m|45m|60m|90m|1h|2h)" >&2; exit 2 ;; esac
lium rm "$NAME" --name-only --in "$DUR" -y 2>&1 | tail -1
