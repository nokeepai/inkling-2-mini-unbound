# package_v3: release package for Unbound v3, Prime and Prime-LoRA (maintainer notes)

Everything a card, chart or launch draft says comes from `values.json`. Nothing here publishes anything; uploads
are a separate, manual step with the owner's OK.

## Morning commands

**One command, re-run whenever results land:** `bash package_v3/refresh.sh` (compare -> collect -> render --final ->
charts --final -> results export -> preview; roots overridable with `R=`, `T=`, `TCI=`, `E=`, `G=`). The steps
below are what it runs. Values for runs still in progress are listed in `values.json` `meta.in_progress` and
render as "run in progress, updated when it lands"; remove a key from that list once its run has landed.
Seed reruns go in `package_v3/seeds.json` (schema in the file).

Run from the repo root. Set the tags of the final runs first; nothing below writes outside `package_v3/` and the
two interval files.

```bash
R=~/.inkling-abl/stack_evals/judged          # graded results, <tag>/<suite>/
E=~/.inkling-abl/stack_evals/judged_e03      # low-effort arm (stock and v21 at effort 0.3), same layout
PRIME=<prime_tag>; V3=<v3_tag>               # e.g. stock_w18s60, v21_w18s60 in the dry run
G=<path>/gguf_table.tsv                      # from scripts/v3stack/gguf_eval_all.sh ($OUT/gguf_table.tsv on the pod)
SU=pub_humaneval,pub_mbpp,pub_humanevalfix,pub_quixbugs,pub_gsm8k_full,pub_gsm8k,pub_math500,pub_mmlu_2k,pub_mmlu,code,hardmath

# 1. intervals: one file per pair (the flag is --out-root, not --root; missing suites are skipped)
python3 -I scripts/evalsuite/compare.py stock $PRIME --ci --ref stock --margin 2 --suites $SU --out-root $R \
    --json package_v3/ci_prime.json > package_v3/ci_prime.md
python3 -I scripts/evalsuite/compare.py v21 $V3 --ci --ref v21 --margin 2 --suites $SU --out-root $R \
    --json package_v3/ci_v3.json > package_v3/ci_v3.md

# 2. collect into values.json (numbers only; add --dry-run to preview, or --out FILE for a scratch copy)
python3 -I package_v3/collect_v3.py --root $R --stock stock --prime $PRIME --v21 v21 --v3 $V3 \
    --indep-prime package_v3/ci_prime.json --indep-v3 package_v3/ci_v3.json --margin 2 \
    --root-e03 $E --e03-stock stock --e03-v21 v21 --gguf-table $G
#    --root-e03 / --gguf-table may be left out: those values then stay TBD and --final refuses until they exist.

# 3. fill by hand what is not on disk (see "Filled by hand"), and clear "provisional" on values you checked

# 4. render: draft first (TBD shown, provisional marked †), then final (refuses while anything used is TBD/provisional)
python3 package_v3/render.py --draft
python3 package_v3/render.py --final

# 5. charts (PNG + SVG into package_v3/charts/); --final refuses on TBD/provisional and drops the watermark
python3 scripts/charts/make_v3.py --final

# 6. per-item results for publication (no model text) and a one-page preview to read everything
python3 -I package_v3/results/build_results.py --root $R --tags stock,$PRIME,v21,$V3 --root-e03 $E --e03-tags stock,v21 \
    --ci package_v3/ci_prime.json --ci package_v3/ci_v3.json --out package_v3/results/data
#    add --extra-root LABEL=ROOT for any independent-suite runs kept under another root
python3 package_v3/preview.py          # -> package_v3/preview.html (cards, drafts, charts inlined)

# 7. check
python3 scripts/charts/data_v3.py                       # status of every chart value
grep -rn "TBD\|MISSING\|DRAFT" package_v3/dist package_v3/CHANGELOG.md package_v3/RELEASE_NOTES.md
```

**Dry run against other tags without touching values.json or dist/:** add `--out SCRATCH/values.json` to step 2,
then `render.py --draft --values SCRATCH/values.json --out SCRATCH/out` and
`make_v3.py --values SCRATCH/values.json --out SCRATCH/out/charts` (`data_v3.py --values` likewise).

**What the collector reads.** Public-benchmark scores, intervals, paired differences and verdicts come from the
compare.py `--json` files (`rows` per suite, `families` pooled); the larger suite wins when both exist
(`pub_gsm8k_full` over `pub_gsm8k`, `pub_mmlu_2k` over `pub_mmlu`), and `indep.*_source` and every n come from the
data. Maths always has two readings, `strict` and `number`; MATH-500 is `is_equiv` plus the row's
`correct_number`, which the collector pairs itself with stats.py. Lab code and hard maths intervals are computed
from the per-item files with the same stats.py code. Verdicts are recomputed at `--margin` with stats.py's rule.
`ubench.n` is whatever the run has (220 now, 1,100 for the full run); `auto.ubench_scope` words it.
`settings.*` (what the cards state and recommend) come from the runs' `meta_<suite>.json`; a mismatch between runs
is printed as a note.

**Name switch.** `names.prime_name` is one value. If any check ends below stock, the owner rule is to rename
Prime to Honed: change that value, re-render, re-chart. `render.py` prints a warning while `auto.worse_prime` is
non-empty.

**Optional explicit top-up.** `flags.explicit_topup` (default false) shows an Unbound-only section filled from
`topup.*`.

**Basalt.** v3 keeps the 0.5.5 `-q8` recipe and is checked by `scripts/v3rel/verify_basalt_v3.py` (v2.1's
`verify_basalt_v2.py` fails by design because v3 changes the experts). Put its one-line result into
`files.basalt_verify`; the Basalt card describes what the checker covers.

## Filled by hand (not on disk for collect_v3.py)

- `names`: adapter_rank, adapter_steps, train_prompts, anchor_answers (release_date is "November 2026" so the card
  is true on go-live day; adapter_checkpoint is set; Prime GGUF repo/file names are derived from prime_name);
  `training.*`; `reproduce.*`; `sysprompt.*` and `checkpoint.choice` are set (owner-supplied lab facts)
- `files.basalt_verify` (one-line summary of basalt_verify.log; hashes and sizes are not on the cards: they point at
  the hub file list / SHA256SUMS / MANIFEST.tsv, which exist only at go-live) (`basalt_verify`: what was actually checked, see the Basalt
  note below), `drift.kld_*`, `speed.hardware/quant/serving/decode_tps_*`
- `regressions.*_note` (set to "" when the QuixBugs note no longer applies; the dry run shows it no longer
  matches Prime's data)
- `ubench.willing_v2`, `ubench.delivery_v2` (no v2 run exists; the chart shows "not measured")
- `flags.per_item_published` and `reproduce.per_item_url`

Leaf format: `"TBD"` or `{"value": ..., "unit": "%|pts|x|tokens|s|n|text|bool", "ci": [lo, hi]?, "display": "..."?,
"provisional": true?}`. For the margin, use e.g. `{"value": 2.0, "unit": "pts", "display": "2 points"}`.

## Which file goes where

| Destination | Files |
|---|---|
| HF `nokeepai/Inkling-2-Mini-Unbound` | `dist/Inkling-2-Mini-Unbound/README.md` → `README.md`; all of `charts/` → `charts/` (the GGUF, Basalt and GitHub cards link their images here) |
| HF `nokeepai/Inkling-2-Mini-Unbound-GGUF` | `dist/Inkling-2-Mini-Unbound-GGUF/README.md` |
| HF `nokeepai/Inkling-2-Mini-Unbound-Basalt` | `dist/Inkling-2-Mini-Unbound-Basalt/README.md` |
| HF `nokeepai/Inkling-2-Mini-<prime_name>` | `dist/Inkling-2-Mini-<prime_name>/README.md`; `charts/hero_v3.*`, `speed_v3.*`, `nothing_lost_prime_v3.*`, `indep_prime_v3.*` → `charts/` |
| HF `nokeepai/Inkling-2-Mini-<prime_name>-GGUF` | `dist/Inkling-2-Mini-<prime_name>-GGUF/README.md` (stage_v3.py prime-gguf-card-only) |
| HF `nokeepai/Inkling-2-Mini-<prime_name>-LoRA` | `dist/Inkling-2-Mini-<prime_name>-LoRA/README.md`, plus the adapter and merge script |
| GitHub `nokeepai/inkling-2-mini-unbound` | `dist/github/README.md` → `README.md`; `package_v3/` as is (CHANGELOG.md, RELEASE_NOTES.md) |
| Owner, to post himself | `dist/launch/*.md` (Reddit posts, X thread, demo plan, objections) |

Images point at `https://huggingface.co/<repo>/resolve/<meta.chart_rev>/charts/`; set `meta.chart_rev` if the
charts go to a branch first.

## Files

- `LAUNCH_CHECKLIST.md` every go-live artefact, its source and the golive_v3.py step (or manual)
- `results/build_results.py` per-item export without text; `preview.py` single-page HTML preview

- `templates/` cards (`main`, `gguf`, `basalt`, `prime`, `prime_lora`, `github`), `CHANGELOG.md`, `RELEASE_NOTES.md`
- `launch/` launch drafts (rendered into `dist/launch/`)
- `values.json` every number; `render.py` cards; `collect_v3.py` results → values; `charts/` chart output
- `scripts/charts/data_v3.py`, `scripts/charts/make_v3.py` charts from values.json only
- `render.py` also reads `package_v21/values.json` (read-only) as `v21card.*` for v2.1 history and comparison rows
