#!/usr/bin/env bash
# Re-run the whole package from the results on disk: intervals -> values.json -> final cards -> final charts ->
# per-item export -> preview. Run from anywhere; re-run every time new results land. Numbers only, no model text.
#
#   bash package_v3/refresh.sh            # defaults below; override any with env, e.g. R=... T=... bash refresh.sh
#
# Stops at the first failing step. render.py --final refuses (and writes nothing) while a used value is TBD and not
# listed in values.json meta.in_progress; its output names the missing keys.
set -euo pipefail
cd "$(dirname "$0")/.."

R="${R:-$HOME/.inkling-abl/stack_evals/final}"                 # main GPU results: stock prime v21 v3
T="${T:-$HOME/.inkling-abl/tinker_evals}"                       # Tinker service runs (adapter unmerged)
TCI="${TCI:-$T/ci_tinker_e03.json}"                             # its interval file (ref tinker_stock)
E_DEFAULT="$HOME/.inkling-abl/stack_evals/final_e03"
E="${E:-$( [ -d "$E_DEFAULT" ] && echo "$E_DEFAULT" )}"        # GPU low-effort root (merged weights at effort 0.3)
G="${G:-$HOME/.inkling-abl/gguf_evals/gguf_table_prime.tsv}"   # GGUF survival table, Prime quants + stock Q4_K_M
G2="${G2:-$HOME/.inkling-abl/gguf_evals/gguf_table_unbound.tsv}"   # GGUF survival table, Unbound v3 quants
POOLS="${POOLS:-$HOME/.inkling-abl/seeds}"                     # pool_prime.json / pool_v3.json are copied from here
# Hand-maintained inputs read every run: package_v3/drift.json, package_v3/seeds.json, package_v3/pool_*.json
MARGIN="${MARGIN:-2}"
PRIME_NAME="${PRIME_NAME:-Prime}"   # Prime or Honed: the one switch for every repo id, folder, link, file name and chart
SU=pub_humaneval,pub_mbpp,pub_humanevalfix,pub_quixbugs,pub_gsm8k_full,pub_gsm8k,pub_math500,pub_mmlu_2k,pub_mmlu,code,hardmath

echo "== 1. intervals"
python3 -I scripts/evalsuite/compare.py stock prime --ci --ref stock --margin "$MARGIN" --suites $SU --out-root "$R" \
    --json package_v3/ci_prime.json > package_v3/ci_prime.md
python3 -I scripts/evalsuite/compare.py v21 v3 --ci --ref v21 --margin "$MARGIN" --suites $SU --out-root "$R" \
    --json package_v3/ci_v3.json > package_v3/ci_v3.md

echo "== 2. collect -> package_v3/values.json"
for f in pool_prime.json pool_v3.json; do [ -f "$POOLS/$f" ] && cp "$POOLS/$f" "package_v3/$f"; done
EXTRA=()
[ -n "$E" ] && EXTRA+=(--gpu-low-root "$E" --e03-prime prime --e03-stock-gpu stock --e03-v21-gpu v21 --e03-v3-gpu v3)
[ -f "$G" ] && EXTRA+=(--gguf-table "$G")
[ -f "$G2" ] && EXTRA+=(--gguf-table-unbound "$G2")
[ -f "$TCI" ] && EXTRA+=(--tinker-root "$T" --tinker-ci "$TCI")
python3 -I package_v3/collect_v3.py --root "$R" --stock stock --prime prime --v21 v21 --v3 v3 --prime-name "$PRIME_NAME" \
    --indep-prime package_v3/ci_prime.json --indep-v3 package_v3/ci_v3.json --margin "$MARGIN" ${EXTRA[@]+"${EXTRA[@]}"} \
    | grep -E "^  (TBD|note)|^updated" || true

echo "== 3. final cards -> package_v3/dist"
# the other name's folders would otherwise linger beside this render's
for n in Prime Honed; do [ "$n" = "$PRIME_NAME" ] || rm -rf "package_v3/dist/Inkling-2-Mini-$n" \
    "package_v3/dist/Inkling-2-Mini-$n-LoRA" "package_v3/dist/Inkling-2-Mini-$n-GGUF"; done
python3 package_v3/render.py --final

echo "== 4. final charts -> package_v3/charts"
python3 scripts/charts/make_v3.py --final

echo "== 5. per-item export (no text) -> package_v3/results/data"
RES=(--root "$R" --tags stock,prime,v21,v3 --ci package_v3/ci_prime.json --ci package_v3/ci_v3.json)
[ -d "$T" ] && RES+=(--extra-root "tinker=$T@tinker_stock,tinker_stock_e03,tinker_prime")
[ -n "$E" ] && RES+=(--root-e03 "$E" --e03-tags prime,stock,v21,v3)
python3 -I package_v3/results/build_results.py "${RES[@]}" --out package_v3/results/data | tail -1
[ -f "$TCI" ] && cp "$TCI" package_v3/results/data/intervals/

echo "== 6. preview -> package_v3/preview.html"
python3 package_v3/preview.py
