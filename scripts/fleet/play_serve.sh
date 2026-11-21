#!/usr/bin/env bash
# Pod side (agent `fleet`): serve ONE candidate for interactive chat on host port 30000 with the frozen ENGINE.md flags.
#   bash /root/fleet/play_serve.sh <cand_id>      (needs /srv/cands/<cand_id>/{spec.json,lowrank.safetensors,sha256.txt},
#                                                 /srv/w/orig from the prep-only bring-up)  Log: /root/logs/play.log
set -uo pipefail
exec >> /root/logs/play.log 2>&1
CID=${1:?cand_id}
fail() { echo "PLAY_FAIL $*"; exit 1; }
echo "=== play $CID $(date -u +%FT%TZ)"
cd /srv/harness || fail "no harness"
D=/srv/cands/$CID
if [ ! -f $D/tensors.safetensors ] || [ "$(sha256sum $D/tensors.safetensors | cut -d' ' -f1)" != "$(cut -d' ' -f1 $D/sha256.txt)" ]; then
  python3 build_cand.py $D /srv/w/orig $D || fail "build_cand (exit $?; 3 = sha mismatch vs canonical)"
fi
[ "$(sha256sum $D/tensors.safetensors | cut -d' ' -f1)" = "$(cut -d' ' -f1 $D/sha256.txt)" ] || fail "sha mismatch"
echo "built $CID sha $(cut -c1-12 $D/sha256.txt)"
M=/srv/w/merged_$CID
[ -f $M/.merged_ok ] || { python3 sgl_merge.py $D $M || fail "sgl_merge"; touch $M/.merged_ok; }
echo "merged into $M"
bash /root/fleet/sglang_up.sh /w/merged_$CID 30000 \
  --mem-fraction-static 0.73815 --chunked-prefill-size 4096 --max-running-requests 31 --random-seed 1234 || fail "sglang_up"
R=$(curl -s localhost:30000/v1/chat/completions -H 'content-type: application/json' -d '{"model":"default","messages":[{"role":"user","content":"In one sentence, what are you?"}],"max_tokens":200,"temperature":0,"chat_template_kwargs":{"reasoning_effort":0}}')
echo "chat test: $R" | cut -c1-600
echo "$R" | grep -q '"content"' || fail "chat completion"
echo "PLAY_DONE $(date -u +%FT%TZ)"
