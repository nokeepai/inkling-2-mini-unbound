#!/bin/bash
# ink-research-2b: install, wait for weights, rebuild ou/c0 + name map, rebuild the W4A16 cache from the uploaded
# continuations, noise floor (bs1), ou/c0, then start kl_server. Log: /root/logs/fid_setup.log
set -u
cd /root/fid
echo "== install $(date -u +%T)"
bash /root/fid/pod_install.sh 2>&1 | tail -2
echo "== wait weights $(date -u +%T)"
while ! grep -q FETCH_DONE /root/fetch.log 2>/dev/null; do sleep 10; done
nvidia-smi --query-gpu=name --format=csv,noheader
echo "== baselines + name map $(date -u +%T)"
cp hf_name_map.json hf_name_map.from_research2.json
python3 build_map_and_baselines.py 2>&1 | grep -E "baselines ok|BUILD_DONE|Error|Trace"
cmp -s hf_name_map.json hf_name_map.from_research2.json && echo "name map identical" || echo "NAME MAP DIFFERS"
sha256sum cands/ou/tensors.safetensors cands/c0/tensors.safetensors
echo "== cache $(date -u +%T)"
mkdir -p /srv/fid
AQ=0 python3 kl_eval.py cache 2>&1 | grep -E "CACHE_DONE|Error|Trace"
sha256sum /srv/fid/orig_lp.f16 orig_top1.npy orig_cont_val.pt
echo "== noise + baselines $(date -u +%T)"
AQ=0 python3 kl_eval.py eval orig --bs 1 --order sorted --tag noise_bs1_2b 2>&1 | grep -E "RESULT|Error|Trace"
AQ=0 python3 kl_eval.py eval orig --bs 8 --order shuffle1 --tag noise_shuffle_bs8_2b 2>&1 | grep -E "RESULT|Error|Trace"
AQ=0 python3 kl_eval.py eval /root/fid/cands/ou /root/fid/cands/c0 --bs 8 --order sorted --tag baseline_2b 2>&1 | grep -E "RESULT|Error|Trace"
echo "== start kl_server $(date -u +%T)"
mkdir -p /srv/cands
setsid nohup python3 /root/fid/kl_server.py > /root/logs/kl_server.log 2>&1 < /dev/null &
echo "SETUP_DONE $(date -u +%T)"
