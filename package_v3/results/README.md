# results: per-item scores for publication (no model text)

`build_results.py` exports one JSONL file per model and suite, plus the interval files and the run settings, so
anyone can recompute every number on the cards. It never copies a prompt, an answer or a thinking trace.

```bash
python3 -I package_v3/results/build_results.py --root ~/.inkling-abl/stack_evals/judged \
    --tags stock,$PRIME,v21,$V3 \
    --extra-root indep=<root of the independent-suite runs, if separate> \
    --root-e03 ~/.inkling-abl/stack_evals/judged_e03 --e03-tags stock,v21 \
    --ci package_v3/ci_prime.json --ci package_v3/ci_v3.json \
    --out package_v3/results/data
```

## What is in a row

| Suite | Fields |
|---|---|
| public benchmarks, lab code | `id`, `passed` or `correct`, `no_answer` / `no_box` where defined, `think_tokens`, `answer_tokens`, `think_cap_hit`, `answer_runaway`, `loop_any` / `loop_think` / `loop_answer`, `forced`, `seconds`, `finish`, `error` (true/false only) |
| maths (`pub_gsm8k*`, `hardmath`, `main`) | the above plus `strict` and `number` (number-correct, computed in memory by `scripts/evalsuite/stats.py`; only the boolean is written) |
| `main`, `rp` | the above plus the graders' probabilities `judge_p`, `alt_p` |
| `agent_hard_noforce` | `name`, `solved_at`, `stop`, `total_tokens`, `n_turns` |
| `ubench` | only `id`, `key`, `category`, `level`, `condition`, the grades (`ub_willing`, `ub_held`, `ub_complete`, `ub_delivery`, `ub_credit`, `ub_refused`, `ub_lectured`, `ub_strict`, `ub_probs`), token counts and cap/runaway flags |

Rows are built from a whitelist; any other field (including `prompt`, `answer`, `thinking`, `gold`, `pred`, error
messages) is dropped. String fields are short identifiers, capped at 80 characters.

Also written: `<root label>/<tag>/settings_<suite>.json` (decoding settings and run arguments, no paths or URLs),
`intervals/*.json` (compare.py `--ci --json` output, numbers only) and `MANIFEST.json` (row counts per file).

Before publishing, read `MANIFEST.json` and spot-check one file per suite: field names only.
