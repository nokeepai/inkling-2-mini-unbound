#!/bin/bash
# Re-run everything with act-quant OFF (W4A16). Old act-quant-ON reference files are kept with an _aq1 suffix.
cd /root/fid
for f in orig_cont_val.pt kl_val_ids.json orig_cache_meta.json orig_top1.npy; do [ -f $f ] && mv $f ${f%.*}_aq1.${f##*.}; done
[ -f /srv/fid/orig_lp.f16 ] && mv /srv/fid/orig_lp.f16 /srv/fid/orig_lp_aq1.f16
rm -f /srv/fid/orig_lp_aq0.f16
AQ=0 python3 kl_eval.py all /root/fid/cands/ou /root/fid/cands/c0 > /root/logs/fid_kl16.log 2>&1
echo "KL_EXIT $?" >> /root/logs/fid_kl16.log
AQ=0 python3 hf_cap_eval.py both orig > /root/logs/fid_cap16.log 2>&1
echo "CAP_EXIT $?" >> /root/logs/fid_cap16.log
