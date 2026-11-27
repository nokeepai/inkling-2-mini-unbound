# evalsuite

Portable eval for Inkling-2-Mini variants behind an OpenAI-compatible endpoint (vLLM or llama.cpp). Stdlib only.

| File | What it does |
|---|---|
| `run_suite.py` | Runs suites against an endpoint; `--judge-only` re-judges saved answers |
| `suites.py` | The seven lab suites (main, hardmath, code, agent_hard, agent_hard_noforce, rp, sysrules) |
| `ubench.py` | The six uncensoredbench suites, marked by importing the benchmark's own tools |
| `pub.py`, `build_pub.py` | Public benchmark suites (HumanEval, QuixBugs, GSM8K, MMLU, and the release suites: full GSM8K, MMLU 2k, MBPP, HumanEvalFix, MATH-500) and their data converter |
| `mathcheck.py` | Lenient number reading (`accuracy_number`) and the MATH benchmark's `is_equiv` |
| `stats.py` | Wilson intervals, paired bootstrap, exact McNemar, non-inferiority (`compare.py --ci`) |
| `compare.py` | Summary tables, mean ± sd over repeats |
| `select_ckpt.py` | Picks a checkpoint: per-item paired tests against a reference on every row |
| `selftest.py` | Offline end-to-end test against `mock_server.py` (no GPU, network or judge spend) |

## uncensoredbench suites

The question files come from `--ubench-dir` (default `~/Code/uncensoredbench`). The marking code is imported from
`EVALSUITE_UBENCH_TOOLS` (default `~/Code/uncensoredbench/tools`).

| Suite | Benchmark file | Scored by (benchmark tool) | Judge calls |
|---|---|---|---|
| `ubench` | 11 category files, 4 levels | `judge_v2.mark` and `delivery`, `analyze_v2.enrich` (refused, held back, credit, lectured), `strict_judge.classify` (Full / Softened / Incomplete / Refused) | 1 per answer per condition |
| `ubench_fa` | `false_alarms.jsonl` | same as `ubench`; `analyze_v2`'s false-alarm refused and delivery | 1 per answer per condition |
| `ubench_premises` | `premises.jsonl` | `judge_v2.PREMISE`, accepted or corrected (the benchmark dropped premises from v2 scoring) | 1 per answer |
| `ubench_rp` | `roleplay.jsonl` | `roleplay.TURN_Q`, per-turn held back, `roleplay.report` measures | 1 per turn |
| `ubench_smarts` | `smarts.jsonl` | `smarts_mark.right`, by code | 0 |
| `ubench_long` | `long_output.jsonl` | stability rates only (`stability.py`): cut, looping, empty. `tools/` defines no completeness score for these probes | 0 |

The `ubench` summary gives the following per condition, overall, for the headline (8 categories, without medical,
legal and financial), per category, per level, and per category x level:

- willing (not an explicit refusal) and complete (strict Full), with Wilson 95% intervals
- held back, lectured and delivery
- counts of refused, hedged (held back) and complete

Rows tagged with an exclusion are skipped unread, and `screen_canaries.jsonl` is never opened.

**Judges.** `bison` is the benchmark's main judge, the one behind its published `marks_v2_bison.jsonl` numbers. It runs
through `jev_judge`'s `bison` backend with the OpenRouter key. Jev, the benchmark's second opinion, needs a TypeSafe
key that this machine does not have. A `jev` request is dropped with a printed note. `glm` (judge3's GLM text judge on
Venice, a fixed sample of 150 answers) stands in as the second opinion on toning down.

**Comparability with published numbers.** The rubric and the scoring code are the same, and the judge matches the
benchmark's main (Bison) numbers. Sampling differs: this suite applies the lab's caps (4,096 thinking and 8,192 answer
tokens, with a forced answer after a cap hit), its temperature, and `--effort`. The benchmark used provider defaults
with a 4,000 to 16,000 token limit. Treat cross-tool comparisons as indicative, and compare checkpoints only with each
other, under one set of settings.

## Public benchmark suites (`pub.py`, alias `pub`)

These are public tests the training recipe was never tuned against. Every one is scored deterministically (no judge,
no cost). Generation uses the same client call, caps, effort and per-item seeds as `main` and `code`. Model code runs
in the `code` suite's sandbox (`metrics.run_program`: fresh `python -I` in a temp dir, 10 s), with at most half the
CPU cores running programs at once so the wall-clock timeout measures the program, not CPU contention.

| Suite | Source (MIT) | Items | Sampling | Scoring |
|---|---|---|---|---|
| `pub_humaneval` | openai/human-eval `HumanEval.jsonl.gz` | 164 | all | dataset prompt + last code block + `test` + `check(entry_point)` exits 0 |
| `pub_quixbugs` | jkoppel/QuixBugs `python_programs`, `json_testcases` | 31 | every program with JSON cases (the 9 graph/tree programs have none) | the model's full program passes every JSON case (generators list()-ed, `sqrt` within epsilon, tester.py rules) |
| `pub_gsm8k` | openai/grade-school-math `test.jsonl` | 300 | seed 1234, from the 1,269 test items not already in the project's prompt files (50 are in `prompts_hardmath.jsonl`) | `hardmath` scoring: `\boxed{}` (else last number) equals the number after `####` |
| `pub_mmlu` | HF `cais/mmlu` `all/test` parquet | 400 | seed 1234, 7 per subject x 57 plus 1, shuffled | last standalone A-D after "answer" on the last line that says "answer"; none = wrong |

QuixBugs leaves out three slow cases: the two its own pytest suite skips, and levenshtein's `rosettacode` case. The
exponential reference fix takes about 9.5 s of the 10 s limit on that case. `build_pub.py` checks that every included
reference program passes all its cases and every buggy one fails at least one (31/31 both ways). It also checks that
all 164 HumanEval canonical solutions pass.

```
# raw downloads: one folder per dataset under ~/.inkling-abl/tinker_lab/pub_raw/ (see build_pub.py's docstring)
python3 -I build_pub.py      # writes ~/.inkling-abl/tinker_lab/pub/<suite>.jsonl; MMLU needs a Python with pyarrow
python3 run_suite.py ... --suites pub       # reads <data-dir>/pub/<suite>.jsonl
python3 compare.py base_tag tuned_tag --suites pub_humaneval,pub_quixbugs,pub_gsm8k,pub_mmlu
```

### Release suites (alias `pub_new`; `pub_all` = all nine)

Full or large public sets, so a release claim can carry honest error bars. Same client settings, caps and seeds as
the four above; scored by code, never a judge. Each data row carries `provenance` (dataset, repo, config, split,
revision, licence, sampling, seed, raw file sha256) and `seen_in_lab_files` (project prompt files whose text
contains the item). The four original files are untouched, so earlier results stay comparable.

| Suite | Source @ revision | Split | n | Licence | Sampling | Scoring |
|---|---|---|---|---|---|---|
| `pub_gsm8k_full` | github openai/grade-school-math @3101c7d | test | 1,319 | MIT | all (50 are also in `prompts_hardmath.jsonl`, flagged) | as `pub_gsm8k` (same ids and prompts): `accuracy_strict` (= `maths_acc`) and `accuracy_number` |
| `pub_mmlu_2k` | HF cais/mmlu @c30699e, all | test | 2,000 | MIT | seed 1234, proportional to subject size across all 57 (largest remainder; 14 to 218 per subject, counts in the build report and `by_subject_n`) | as `pub_mmlu` |
| `pub_mbpp` | HF google-research-datasets/mbpp @4bb6404, sanitized | test | 257 | CC-BY-4.0 | all (35 are also in the lab's `code` suite, flagged) | MBPP paper prompt with all asserts shown; `test_imports` + last code block + every assert exits 0 (257/257 references pass) |
| `pub_humanevalfix` | HF bigcode/humanevalpack @9a41762, python | test | 164 | MIT | all | bigcode-evaluation-harness `humanevalfixtests-python`, `instruct` form: declaration + buggy body + the unit tests, then "Fix bugs in f."; the harness's imports + dataset prompt + the returned function + hidden tests exit 0 (164/164 fixes pass, 164/164 buggy fail) |
| `pub_math500` | HF HuggingFaceH4/MATH-500 @6e4ed1a (the PRM800K 500-problem test subset) | test | 500 | MIT (MATH, PRM800K; the HF card has no licence tag) | all | last `\boxed{}` vs gold with the MATH repo's `is_equiv` + string normalisation (`mathcheck.py`); 500/500 golds match their own solutions' boxes. `accuracy_number` is reported beside it |

**Maths, strict vs number.** The strict matcher (`metrics.maths_quality`, unchanged) marks `230\text{ miles}` wrong
against 230. `accuracy_number` (`maths_acc_number`; `mathcheck.number_value`) accepts the one number the final
answer states, ignoring units, currency, `\text{}`, thousands separators, percent signs and trailing words, and
rejects any answer with two different numbers. `main`, `hardmath`, `pub_gsm8k` and `pub_gsm8k_full` report both;
quote `accuracy_number` for accuracy claims and strict for format following.

**Statistics.** `compare.py TAGS --ci --ref TAG [--margin 2] [--json PATH]` (or `stats.py`) prints per suite and tag:
n, score, Wilson 95% interval; against the reference, the paired difference on items both answered, its 95% paired
bootstrap interval (10,000 resamples, seed 1234), the exact McNemar p, and a MISMATCH flag when item sets differ;
then pooled lines for coding (HumanEval + MBPP), bug-fixing (HumanEvalFix + QuixBugs), maths (GSM8K + MATH-500) and
knowledge (MMLU), each item once. The last column reads non-inferior when the bootstrap lower bound is at least
−margin points, and WORSE when the whole interval is below zero. Numbers only; answer text is never printed.

```
# data (raw downloads pinned in build_pub.py's docstring; MMLU, MBPP and HumanEvalPack need pyarrow)
uv run --no-project --with pyarrow python -I build_pub.py --suites new
# run the release suites against a server
python3 run_suite.py --base-url http://POD:8000/v1 --model inkling --tag TAG --suites pub_new --seed 1234
# intervals and non-inferiority against the base
python3 compare.py base_tag TAG --ci --ref base_tag --margin 2 --json ci.json
```

## Commands

```
cd scripts/evalsuite
# full run: lab suites + all benchmark suites, both benchmark conditions, paid judges on
python3 run_suite.py --base-url http://POD:8000/v1 --model inkling --tag c_step200 \
    --suites all --ubench-conditions command,permissive --judges bison,haiku,glm --seed 1234

# generate now, judge later (re-runs reuse the answers; ubench marks are cached in <suite>/judge_cache.jsonl)
python3 run_suite.py --base-url http://POD:8000/v1 --model inkling --tag c_step200 --suites ubench_all
python3 run_suite.py --base-url http://unused/v1 --model inkling --tag c_step200 --suites ubench_all \
    --judge-only --judges bison,glm --ubench-conditions command,permissive

# smoke run: one item in each category x level cell
python3 run_suite.py ... --suites ubench --limit 44

# pick a checkpoint against the released v2.1 (tags under ~/.inkling-abl/evalsuite, or full paths; DIR:STEP sets a step)
python3 select_ckpt.py --ref v21 c_step100 c_step200 c_step300
python3 compare.py v21 c_step200 --suites ubench,ubench_rp,ubench_smarts   # summary view
python3 selftest.py                                                         # offline check
```

A killed run resumes. Re-running the same tag skips answers already in `<suite>/r<i>.gen.partial.jsonl` and marks
already in `judge_cache.jsonl`. Delete the tag folder to start fresh.

`select_ckpt.py` builds about 80 rows per condition. Each row is tested per item, paired on matched items, with Holm
correction across rows at a one-sided 2.5%. Rows with too few matched items fall back to interval overlap. A candidate
is eligible only if no row is significantly worse than the reference and no row is missing (`--allow-missing` relaxes
the second rule). The earliest eligible step wins. The file is not called `select.py` on purpose: a file with that
name would shadow Python's `select` module and break `subprocess` and `http.server` for every script in this folder.

## Cost of a full run (one repeat)

| Suite | Items | Bison calls, command only | with permissive |
|---|---|---|---|
| ubench | 1,100 | 1,100 | 2,200 |
| ubench_fa | 100 | 100 | 200 |
| ubench_premises | 75 | 75 | 75 |
| ubench_rp | 30 chats x 15 turns | 450 | 450 |
| ubench_smarts | 1,000 | 0 | 0 |
| ubench_long | 50 | 0 | 0 |
| **total** | 2,775 generations (3,975 with permissive) | **1,725** | **2,925** |

Bison costs $0.10 per million input tokens (`jev_judge.BACKENDS`). The benchmark's notes put a full v2 run at about
$16 of Bison judging for about 54,000 calls (19 models x 1,199 answers x 2 conditions, plus 19 x 450 roleplay turns).
That is about $0.0003 a call, so one repeat costs about **$0.50** (command only) or **$0.90** (with permissive).
Answers cut at the length limit with under 150 words are not judged (the benchmark's Cut rule).

Optional `glm` adds up to 150 Venice calls per repeat. The benchmark records no price for these, so the cap counts
calls (`EVALSUITE_GLM_MAX_CALLS`, default 400). Bison spend is capped at `EVALSUITE_UBENCH_MAX_USD`, default $5,
tracked in `<out-root>/judge_spend_ubison.json`. The lab suites' judges (Bison $1, Haiku $2 caps) are unchanged.
