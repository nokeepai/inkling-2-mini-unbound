#!/bin/bash
# Pull finished sealed eval dirs from a pod: full outputs to the private dir (outside git, chmod 700),
# metrics only into R/audit/test/evals/. Usage: bash pull.sh <ink-pod> <dir name> [...]
set -euo pipefail
R=/Users/nikshepsvn/Code/inkling-abl
PRIV=/Users/nikshepsvn/.inkling-abl/audit_private/test
L="bash $R/scripts/ink_lium.sh"
mkdir -p "$PRIV"; chmod 700 /Users/nikshepsvn/.inkling-abl/audit_private "$PRIV"
POD=$1; shift
for n in "$@"; do
  $L exec "$POD" "cd /srv/audit/evals && tar czf /srv/audit/pull_$n.tgz $n" >/dev/null
  $L scp "$POD" "/srv/audit/pull_$n.tgz" "$PRIV/" -d >/dev/null
  tar -C "$PRIV" -xzf "$PRIV/pull_$n.tgz" && rm "$PRIV/pull_$n.tgz"
  mkdir -p "$R/audit/test/evals/$n"
  cp "$PRIV/$n/metrics.json" "$R/audit/test/evals/$n/" 2>/dev/null || cp "$PRIV/$n/error.json" "$R/audit/test/evals/$n/"
  echo "pulled $n"
done
