#!/bin/bash
# Wait for the W4A16 KL python to finish, test the fast W4A16 experts path, then run MMLU+GSM8K (fast only if exact).
KLPID=$1
while kill -0 $KLPID 2>/dev/null; do sleep 5; done
echo "KL_EXIT (python ended)" >> /root/logs/fid_kl16.log
cd /root/fid
python3 test_fast16.py > /root/logs/fid_test16.log 2>&1
if grep -q "exact True" /root/logs/fid_test16.log && grep -q "same tokens True" /root/logs/fid_test16.log; then FAST=1; else FAST=0; fi
echo "FAST16=$FAST" >> /root/logs/fid_test16.log
AQ=0 FAST16=$FAST python3 hf_cap_eval.py both orig > /root/logs/fid_cap16.log 2>&1
echo "CAP_EXIT $? FAST16=$FAST" >> /root/logs/fid_cap16.log
