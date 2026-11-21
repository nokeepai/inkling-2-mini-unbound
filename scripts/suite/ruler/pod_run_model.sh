#!/bin/bash
# Pod-side: serve <model_dir> with the ENGINE_LC.md flags, run RULER gen + score, record wall time.
#   bash /srv/suite/ruler/pod_run_model.sh <name> </w/model_dir>
# Env RULER_N32 / RULER_N128 pass through to bench_ruler.py.
set -uo pipefail
NAME=$1; MODEL=$2
FLAGS="--context-length 135168 --mem-fraction-static 0.85 --chunked-prefill-size 8192 --max-running-requests 16 --random-seed 1234"
T0=$(date +%s)
echo "START $NAME $MODEL $(date -u +%FT%TZ)"
bash /srv/suite/suite_serve.sh 0 3100 $MODEL $FLAGS || { echo "SERVE_FAILED $NAME"; exit 1; }
T1=$(date +%s)
rm -f /srv/suite/out/$NAME/ruler/gens_*.jsonl
python3 /srv/suite/ruler/bench_ruler.py gen --port 3100 --model $NAME || { echo "GEN_FAILED $NAME"; exit 1; }
T2=$(date +%s)
python3 /srv/suite/ruler/bench_ruler.py score --model $NAME || { echo "SCORE_FAILED $NAME"; exit 1; }
echo "{\"model\": \"$NAME\", \"model_dir\": \"$MODEL\", \"serve_s\": $((T1-T0)), \"gen_s\": $((T2-T1)), \"total_s\": $(( $(date +%s)-T0 )), \"n32\": ${RULER_N32:-25}, \"n128\": ${RULER_N128:-10}}" > /srv/suite/out/$NAME/ruler/timing.json
cat /srv/suite/out/$NAME/ruler/timing.json
echo "RUN_DONE $NAME $(date -u +%FT%TZ)"
