#!/bin/bash
# Guarded lium wrapper for every pod-targeting command (exec, scp, rsync, rm, logs, ssh, describe, reboot, cp).
# Usage: ink_lium.sh <subcommand> <ink-pod-name> [args...]
# Read-only account commands (ps, ls, spend, templates, whoami, balance, schedules list) pass through.
set -u
export LIUM_NONINTERACTIVE=1
export LIUM_NO_POD_INDEX=1
R=~/Code/inkling-abl
source "$R/scripts/ink_guard.sh"
SUB="${1:-}"; shift || true
case "$SUB" in
  ps|ls|spend|templates|whoami|balance) exec lium "$SUB" "$@" ;;
  schedules) [ "${1:-}" = "list" ] && exec lium schedules list || { echo "ink_lium: only 'schedules list' allowed" >&2; exit 2; } ;;
  cp) # pod-to-pod: SRC_POD:PATH DST_POD:PATH; guard both pod names; --delete not allowed
    SRC="${1:-}"; DST="${2:-}"
    for a in "$@"; do case "$a" in -a|--all|--delete) echo "ink_lium: refused $a" >&2; exit 2 ;; esac; done
    [[ "$SRC" == *:* && "$DST" == *:* ]] || { echo "ink_lium: cp needs POD:PATH POD:PATH" >&2; exit 2; }
    ink_guard "${SRC%%:*}" || exit 2
    ink_guard "${DST%%:*}" || exit 2
    exec lium cp "$@" ;;
  exec|scp|rsync|rm|logs|ssh|describe|reboot|record|port-forward)
    TARGET="${1:-}"
    for a in "$@"; do case "$a" in -a|--all) echo "ink_lium: refused --all" >&2; exit 2 ;; esac; done
    ink_guard "$TARGET" || exit 2
    if [ "$SUB" = "rm" ]; then exec "$R/scripts/pod_rm.sh" "$TARGET"; fi
    exec lium "$SUB" "$@" ;;
  up) # new pods must be named ink-*
    if ! printf '%s\n' "$@" | grep -qE '^ink-'; then echo "ink_lium: 'up' needs --name ink-*" >&2; exit 2; fi
    exec lium up "$@" ;;
  *) echo "ink_lium: subcommand '$SUB' not allowed" >&2; exit 2 ;;
esac
