#!/bin/bash
# bash stop_servers.sh <gpu list>
for g in ${1//,/ }; do [ -f /root/logs/srv_$g.pid ] && kill $(cat /root/logs/srv_$g.pid) 2>/dev/null; done; sleep 3; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
