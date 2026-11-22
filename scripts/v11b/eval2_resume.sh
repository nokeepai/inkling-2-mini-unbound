#!/bin/bash
# ON eval2: finish setup after the failed scratch download, then base KLD, v10 KLD recheck, A0/B screens.
set -u
cp /data/hfs/orig_attn/* /data/orig/
(apt-get install -y -q cmake > /dev/null 2>&1 || pip install -q cmake)
bash /root/v11b/eval_build.sh /data/hfs/llama_19bc.tar.gz
B=/data/llama.cpp/build/bin; W=/data/compare/wiki.test.raw; K=/data/compare/kld
rm -f $K/base.done
cd /root/v11b; nohup bash fetch_build_screen.sh A0 0 > /root/logs/fbs_A0.log 2>&1 &
nohup bash fetch_build_screen.sh B 1 > /root/logs/fbs_B.log 2>&1 &
cd /data; CUDA_VISIBLE_DEVICES=2 $B/llama-perplexity -m /data/base/Inkling-2-Mini-Q8_0.gguf -f $W -c 4096 --chunks 40 -ngl 99 -b 4096 -ub 512 --seed 1234 --kl-divergence-base $K/base.kld > /root/compare/logs/kldbase.log 2>&1 && touch $K/base.done
echo BASEKLD rc=$?
CUDA_VISIBLE_DEVICES=2 $B/llama-perplexity -m /data/v10/Inkling-2-Mini-Unbound-Q8_0.gguf -f $W -c 4096 --chunks 40 -ngl 99 -b 4096 -ub 512 --seed 1234 --kl-divergence-base $K/base.kld --kl-divergence > /root/compare/logs/v10_recheck.kld.log 2>&1
grep -E 'Mean +KLD|Same top' /root/compare/logs/v10_recheck.kld.log
echo RESUME_DONE
