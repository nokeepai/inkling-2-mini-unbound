#!/bin/bash
# Mac: independent judge (Bison) on v2's sealed-test outputs (8192 cap) for v2(E), v1.0, original, AtomicChat.
set -u
S=~/.inkling-abl/v11b_private/val/out; O=~/.inkling-abl/v3_private/oldtest
for m in E v10 orig atomic; do for md in off e09; do
  python3 ~/Code/inkling-abl/scripts/v3/judge_v3.py $S/test8k_$m/gens_harmful_$md.jsonl $O/bison_${m}_$md.jsonl --backend bison --conc 8 --max-usd 0.6
done; done
echo BISON_OLD_DONE
