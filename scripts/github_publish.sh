#!/bin/bash
# Owner-requested PRIVATE GitHub repo. Publishes a clean single-commit snapshot of the current tree
# (respecting .gitignore) from a separate export dir, so local history (which contains raw outputs and
# vendored code from before the exclusions) is never pushed. gitleaks must be clean or it aborts.
set -euo pipefail
R=/Users/nikshepsvn/Code/inkling-abl
X=/Users/nikshepsvn/Code/inkling-abl-publish
rm -rf "$X"; mkdir -p "$X"
git -C "$R" add -A
git -C "$R" commit -q -m "chore: snapshot for private repo" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" || true
git -C "$R" archive HEAD | tar -x -C "$X"
git -C "$X" init -q -b main
git -C "$X" add -A
git -C "$X" commit -q -m "Inkling-2-Mini-Unbound: abliteration harness, methods, results" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
if ! gitleaks detect --source "$X" --redact --no-banner > /tmp/gl.out 2>&1; then echo "GITLEAKS FINDINGS — aborting"; tail -5 /tmp/gl.out; exit 1; fi
echo "gitleaks clean; size: $(du -sh "$X/.git" | cut -f1); files: $(git -C "$X" ls-files | wc -l)"
if ! gh repo view nokeepai/inkling-2-mini-unbound > /dev/null 2>&1; then
  gh repo create nokeepai/inkling-2-mini-unbound --private --description "Abliteration of Inkling-2-Mini (Inkling-2-Mini-Unbound): harness, methods, results" > /dev/null
fi
echo "visibility: $(gh repo view nokeepai/inkling-2-mini-unbound --json visibility -q .visibility)"
git -C "$X" remote add origin https://github.com/nokeepai/inkling-2-mini-unbound.git
git -C "$X" push -q -f -u origin main
echo "PUSHED https://github.com/nokeepai/inkling-2-mini-unbound"
