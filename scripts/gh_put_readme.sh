#!/bin/bash
# Update only README.md in the public GitHub repo (no other in-progress files).
set -euo pipefail
REPO=nokeepai/inkling-2-mini-unbound
SHA=$(gh api repos/$REPO/contents/README.md --jq .sha)
CONTENT=$(base64 -i /Users/nikshepsvn/Code/inkling-abl/README.md | tr -d '\n')
gh api -X PUT repos/$REPO/contents/README.md -f message="docs: nokeep link with ?ref=github" -f content="$CONTENT" -f sha="$SHA" --jq '.commit.html_url'
