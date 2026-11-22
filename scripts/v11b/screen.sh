#!/bin/bash
# ON eval pod: val screen of one GGUF on one GPU with the compare pipeline (run_model.sh, identical flags),
# but on the v11b VAL splits (CMP_DATA=/root/v11b: harmful.jsonl = val 281, borderline.jsonl = val 773).
#   bash screen.sh <name> <gpu> <model.gguf> [--base-kld-first]
# Outputs /root/compare/out/<name>/metrics.json, gens_*.jsonl; KLD log /root/compare/logs/<name>.kld.log
set -u
export CMP_DATA=/root/v11b CMP_TOK=/data/compare/models/tok
cd /root/compare/bin
bash run_model.sh "$1" "$2" $((8200+$2)) "$3" - ${4:-}
