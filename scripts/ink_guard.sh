#!/bin/bash
# Pod-name guard shared by every pod-targeting script.
# ink_guard <name>: succeeds only for an exact live pod NAME starting with "ink-".
# Rejects: empty, "all", -a/--all, digits-only (ps row numbers), anything not "ink-*",
# comma lists, and names not currently listed under that exact name.
ink_guard() {
  local name="$1"
  case "$name" in
    ""|all|ALL|-a|--all|*,*) echo "ink_guard: refused target '$name'" >&2; return 1 ;;
  esac
  if [[ "$name" =~ ^[0-9]+$ ]]; then echo "ink_guard: refused row number '$name'" >&2; return 1; fi
  if [[ "$name" != ink-* ]]; then echo "ink_guard: refused non-ink pod '$name'" >&2; return 1; fi
  if ! LIUM_NONINTERACTIVE=1 lium ps --format json | python3 -c "import sys,json; sys.exit(0 if any(p.get('name')=='$name' for p in json.load(sys.stdin)) else 1)"; then
    echo "ink_guard: no live pod named exactly '$name'" >&2; return 1
  fi
  return 0
}
