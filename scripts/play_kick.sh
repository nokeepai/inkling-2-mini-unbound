#!/bin/bash
# Kick the playground pod: weights (orig only) + pinned image pull, in parallel.
R=/Users/nikshepsvn/Code/inkling-abl
L="bash $R/scripts/ink_lium.sh"
cat > /tmp/play_fetch.sh <<'EOF'
#!/bin/bash
mkdir -p /root/w /root/logs
pip install -q -U "huggingface_hub[hf_transfer]" > /dev/null 2>&1
export HF_HUB_ENABLE_HF_TRANSFER=1
hf download thinkingmachines/Inkling-2-Mini --revision ed40e7da625f594fc9c4990f5f66be24873c22d7 --local-dir /root/w/orig > /root/logs/fetch.log 2>&1
echo FETCH_DONE >> /root/logs/fetch.log
EOF
$L scp ink-play-1 /tmp/play_fetch.sh /root/play_fetch.sh > /dev/null 2>&1
$L exec ink-play-1 "nohup bash /root/play_fetch.sh > /dev/null 2>&1 & nohup docker pull lmsysorg/sglang@sha256:421aed3ac11741381eac350bf7b1db28a0cafb6ae49b57ae78fba86a16a32306 > /root/logs/pull.log 2>&1 & sleep 60; du -sh /root/w; tail -1 /root/logs/pull.log"
