#!/bin/bash
# Exercise the guard with refused targets only (no pod is touched). Every line must print REFUSED.
R=~/Code/inkling-abl
for t in "" all -a 1 3 prod-pod-a prod-pod-b "ink-research-1,prod-pod-a" eager-fox-53 ink-does-not-exist; do
  if bash "$R/scripts/ink_lium.sh" exec "$t" "true" 2>/dev/null; then echo "ALLOWED  exec '$t'  <-- BUG"; else echo "REFUSED  exec '$t'"; fi
done
for t in prod-pod-a 2 all; do
  if bash "$R/scripts/pod_rm.sh" "$t" 2>/dev/null; then echo "ALLOWED  rm '$t'  <-- BUG"; else echo "REFUSED  rm '$t'"; fi
done
if bash "$R/scripts/ink_lium.sh" rm ink-eval-2 --all 2>/dev/null; then echo "ALLOWED rm --all <-- BUG"; else echo "REFUSED  rm ink-eval-2 --all"; fi
if bash "$R/scripts/ink_lium.sh" up 1 --name nokeep-x 2>/dev/null; then echo "ALLOWED up non-ink <-- BUG"; else echo "REFUSED  up --name nokeep-x"; fi
echo "positive check (read-only): exec ink-eval-2 hostname"
bash "$R/scripts/ink_lium.sh" exec ink-eval-2 "hostname" 2>&1 | tail -1
