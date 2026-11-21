#!/usr/bin/env bash
# Research pod: stop SGLang, teacher-force both SGLang outputs through HF.
docker rm -f sgl >/dev/null 2>&1
cd /root/parity
python3 hf_run.py tf /root/parity/sg_research.json 2>&1 | grep -E 'TF top1|peak|Error|Trace'
python3 hf_run.py tf /root/parity/sg_eval.json 2>&1 | grep -E 'TF top1|Error|Trace'
