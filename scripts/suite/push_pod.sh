#!/usr/bin/env bash
# Mac: push suite pod scripts (scripts/suite/pod/*) to ink-suite-1:/srv/suite/. Extra args: commands to run after.
set -euo pipefail
R=/Users/nikshepsvn/Code/inkling-abl
POD=${POD:-ink-suite-1}
ink() { bash "$R/scripts/ink_lium.sh" "$@"; }
for f in $R/scripts/suite/pod/*; do ink scp "$POD" "$f" "/srv/suite/$(basename $f)" >/dev/null; done
echo "pushed $(ls $R/scripts/suite/pod | tr '\n' ' ')"
