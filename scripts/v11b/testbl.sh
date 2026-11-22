#!/bin/bash
# ON eval pod: test-split over-refusal (borderline, thinking off, compare flags/caps) for one model.  bash testbl.sh <name> <gpu> <gguf>
set -u
export CMP_DATA=/root/v11b/test CMP_TOK=/data/compare/models/tok CMP_METRICS=borderline CMP_SKIP_KLD=1
cd /root/compare/bin && bash run_model.sh $1 $2 $((8200+$2)) "$3" -
