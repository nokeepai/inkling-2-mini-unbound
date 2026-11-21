#!/bin/bash
# Noise-floor diagnosis: (a) determinism with identical batching (AQ on); (b) AQ off: own reference cache, then reshuffled.
cd /root/fid
python3 kl_eval.py eval orig --bs 8 --order sorted --tag det_same_batching > /root/logs/fid_diag.log 2>&1
AQ=0 CACHE_TAG=_aq0 python3 kl_eval.py cache >> /root/logs/fid_diag.log 2>&1
AQ=0 CACHE_TAG=_aq0 python3 kl_eval.py eval orig --bs 8 --order shuffle1 --tag aq0_noise_shuffle_bs8 >> /root/logs/fid_diag.log 2>&1
AQ=0 CACHE_TAG=_aq0 python3 kl_eval.py eval /root/fid/cands/c0 --bs 8 --order sorted --tag aq0_baseline >> /root/logs/fid_diag.log 2>&1
echo DIAG_EXIT >> /root/logs/fid_diag.log
