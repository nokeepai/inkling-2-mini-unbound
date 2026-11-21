#!/bin/bash
# Pod-side. fit_job.sh <cand_id>: ara_fit on /root/ara/jobs/<cid>/params.json, then build_cand + verify; log /root/logs/ara_<cid>.log
cid=$1
d=/root/ara/jobs/$cid
cd /root/ara
python ara_fit.py $d/params.json $d > /root/logs/ara_$cid.log 2>&1 || { echo FIT_FAIL >> /root/logs/ara_$cid.log; exit 1; }
mkdir -p $d/build
python build_cand.py $d /root/w/orig $d/build >> /root/logs/ara_$cid.log 2>&1 && echo BUILD_OK >> /root/logs/ara_$cid.log
python verify_build.py $d >> /root/logs/ara_$cid.log 2>&1
echo JOB_DONE >> /root/logs/ara_$cid.log
