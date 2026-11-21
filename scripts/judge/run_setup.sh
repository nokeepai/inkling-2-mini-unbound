#!/usr/bin/env bash
# Local launcher: copy setup to pod and run it with HF_TOKEN in env (never echoed).
set -euo pipefail
export LIUM_NONINTERACTIVE=1
POD=ink-eval-2
lium exec "$POD" "mkdir -p /srv/judge /root/logs /root/judge"
lium scp "$POD" /Users/nikshepsvn/Code/inkling-abl/scripts/judge/pod_setup.sh /root/judge/pod_setup.sh
HF_TOKEN_VAL="$(cat "$HOME/.cache/huggingface/token")"
lium exec "$POD" -e "HF_TOKEN=$HF_TOKEN_VAL" "bash /root/judge/pod_setup.sh > /root/logs/judge_setup.log 2>&1; tail -n 20 /root/logs/judge_setup.log"
