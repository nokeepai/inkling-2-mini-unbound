#!/bin/bash
# Orchestrator checkpoint commit. Usage: commit.sh "<message>"
R=/Users/nikshepsvn/Code/inkling-abl
git -C "$R" add -A
git -C "$R" commit -q -m "$1" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && git -C "$R" log --oneline -1
