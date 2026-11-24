#!/bin/bash
# ON eval pod: SFT(0.6)+DPO merge at DPO scale 1.0 -> KLD; if > 0.040 retry at 0.5; then full eval of the passing one.
#   bash dpo_auto.sh <name>      (expects /root/v3/loras/sft1.pt and /root/v3/loras/<name>.pt)
set -u
N=$1; cd /root/v11b; rm -f /root/v3/ev/sweep_$N.txt
kld_of() { grep -a 'Mean    KLD' /root/v3/ev/$1/kld.txt | awk '{print $3}'; }
pick=""
for s in 1.0 0.5; do
  bash kld_sweep.sh $N 0 "/root/v3/loras/sft1.pt:0.6" /root/v3/loras/$N.pt $s
  k=$(kld_of ${N}_s$s); echo "scale $s KLD $k" >> /root/v3/ev/auto_$N.txt
  if python3 -c "import sys; sys.exit(0 if float('$k') <= 0.040 else 1)"; then pick=${N}_s$s; break; fi
done
[ -z "$pick" ] && pick=${N}_s0.5 && echo "no scale passed KLD; evaluating s0.5 anyway" >> /root/v3/ev/auto_$N.txt
echo "EVAL $pick" >> /root/v3/ev/auto_$N.txt
bash eval_cand.sh $pick /data/v11/$pick-Q8_0.gguf 0,1,2,3,4,5 all > /root/logs/ev_$pick.log 2>&1
echo "AUTO_DONE $pick" >> /root/v3/ev/auto_$N.txt
