#!/bin/bash
# Pull one finished gens file early (private dir). Usage: bash pull_partial.sh <pod> <evaldir> <file>
set -euo pipefail
R=/Users/nikshepsvn/Code/inkling-abl
PRIV=/Users/nikshepsvn/.inkling-abl/audit_private/test
mkdir -p "$PRIV/$2"
bash $R/scripts/ink_lium.sh scp "$1" "/srv/audit/evals/$2/$3" "$PRIV/$2/" -d >/dev/null
wc -l < "$PRIV/$2/$3"
