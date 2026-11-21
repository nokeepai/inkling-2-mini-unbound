#!/bin/bash
# Pod: KL pipeline (gen, cache, noise floor, ou, c0), then HF MMLU-300 + GSM8K-100 on orig.
cd /root/fid
python3 kl_eval.py all /root/fid/cands/ou /root/fid/cands/c0 > /root/logs/fid_kl.log 2>&1
echo "KL_EXIT $?" >> /root/logs/fid_kl.log
python3 hf_cap_eval.py both orig > /root/logs/fid_cap.log 2>&1
echo "CAP_EXIT $?" >> /root/logs/fid_cap.log
