# RULER versions and data (agent suite/ruler, 2026-11-21)

- Generators: lm-eval-harness **0.4.9.1** (pip, `lm_eval/tasks/ruler`, a port of NVIDIA/RULER `scripts/data/synthetic`).
  Called directly (not through `lm_eval` CLI) so the sample count can be set: `prepare_niah.generate_samples`,
  `vt_utils.sys_vartrack_w_noise_random` (with its 1-sample 500-token ICL example), `cwe_utils.sys_word_pair_random`,
  `fwe_utils.sys_kwext`, `qa_utils.generate_samples`. Task args copied from the lm-eval task files.
  Script: scripts/suite/ruler/gen_ruler_data.py (+ pod_gen_all.sh).
- Deps: wonderwords 2.2.0 (installed for this), nltk 3.10.4 (punkt_tab), datasets 3.6.0, transformers 5.24.0.dev0
  (pinned commit a10ebc3c), tokenizer = /srv/w/orig (Inkling-2-Mini @ ed40e7da). Lengths measured with this tokenizer,
  as RULER does (raw input + answer, no chat template).
- Haystacks / sources: Paul Graham essays `baber/paul_graham_essays` (HF, train); SQuAD v2 dev
  (rajpurkar.github.io dev-v2.0.json) for qa_1; HotpotQA distractor dev for qa_2. The CMU HotpotQA URL in qa_utils
  does not resolve from the pod, so the identical dev set was loaded from HF `hotpotqa/hotpot_qa` (distractor,
  validation, 7,405 rows) and reshaped to the JSON layout qa_utils expects.
- Seed: `random.seed(1234)` and `np.random.seed(1234)` before every (task, length); each task generated in its own
  process. The generators' built-in fixed shuffle seeds (42) are unchanged.
- Tasks (13): niah_single_1..3, niah_multikey_1..3, niah_multivalue, niah_multiquery, vt, cwe, fwe, qa_1 (squad),
  qa_2 (hotpot). Lengths 32768 and 131072. Generated: 25/task at 32k (325), 10/task at 128k (130). The scored subset
  is the first k by sample index, the same for every model (see suite/ruler/ENGINE_LC.md and score.json "caps").
- Data: /srv/suite/data/ruler/<task>_<len>.jsonl on ink-suite-lc, copied to suite/ruler/data/ (97 MB), sha256 in
  suite/ruler/data/SHA256SUMS (verified after copy, 26/26).
- Prompting (bench_ruler.py): official chat template, reasoning_effort=0, RULER input as the user turn, then
  [200001, 200004] and the task's answer prefix (lm-eval `gen_prefix`) tokens; greedy; max_new_tokens niah 128, vt 30,
  cwe 120, fwe 50, qa 32 (RULER defaults); stop at <|end_message|>/EOS.
- Scoring: NVIDIA RULER `string_match_all` (niah, vt, cwe, fwe) and `string_match_part` (qa: 1 if any gold answer
  string appears). Note lm-eval 0.4.9.1's `string_match_part` averages over gold variants instead; we follow RULER.
  Known deviation inherited from lm-eval: cwe has no few-shot example in the prompt (lm-eval drops `input_example`).
- ruler_32k / ruler_128k = mean over the 13 tasks of task accuracy x100; 95% CI = bootstrap over samples stratified
  by task, 10,000 reps, seed 1234.
