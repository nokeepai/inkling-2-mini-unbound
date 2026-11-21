#!/bin/bash
set -e
bash /Users/nikshepsvn/Code/inkling-abl/scripts/data/run_build.sh 2>&1 | grep -v -E "Loading weights|Redirects"
/Users/nikshepsvn/Code/inkling-abl/scripts/data/.venv/bin/python /Users/nikshepsvn/Code/inkling-abl/scripts/data/card.py
