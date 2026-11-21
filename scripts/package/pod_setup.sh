#!/bin/bash
# Install peft into an isolated target dir (no deps) so the pod's shared environment is untouched.
bash ~/Code/inkling-abl/scripts/ink_lium.sh exec "${1:?pod}" "pip install -q --no-deps --target /root/pkg/pylib peft==0.17.1 2>&1 | tail -2; PYTHONPATH=/root/pkg/pylib python3 -c 'import peft;print(peft.__version__)'"
