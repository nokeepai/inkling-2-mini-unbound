#!/bin/bash
P=${1:-ink-research-2b}
bash ~/Code/inkling-abl/scripts/ink_lium.sh exec "$P" "df -h /root /srv 2>/dev/null; free -g; nproc; ls -la /root/w/orig | head -40; ls /srv/w 2>/dev/null; python3 -c 'import torch,safetensors;print(torch.__version__,safetensors.__version__)'; python3 -c 'import peft;print(peft.__version__)' 2>&1 | tail -1; uptime"
