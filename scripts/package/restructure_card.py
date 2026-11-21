"""One-off restructure of the card (owner feedback 2026-11-21): Highlights + Use it up top, results before method,
OU kept only as a table column plus one sentence. Edits R/package/MODEL_CARD_DRAFT.md in place."""
import os
import re

P = os.path.expanduser("~/Code/inkling-abl/package/MODEL_CARD_DRAFT.md")
s = open(P).read()

HIGHLIGHTS = """## Highlights

Measured on a sealed test split against the original Inkling-2-Mini.

- **Answers instead of refusing.** 0 of 277 harmful test prompts refused, with thinking off and at effort 0.9
  (the original refuses 97.5%).
- **Answers usefully.** 54% (thinking off) and 63% (effort 0.9) of answers to harmful requests are rated useful
  by an independent judge (Jev); the original: 0.4% and 1.1%.
- **No over-refusal.** 0% of 784 benign borderline prompts refused (the original refuses 38%).
- **Capability kept.** All 6 capability benchmarks (MMLU-Pro, GSM8K, MATH-500, GPQA Diamond, IFEval, HumanEval+)
  are inside the original's 95% confidence interval.
- **A clean, minimal edit.** 20 BF16 attention-output tensors changed with a convolution-aware, norm-preserving
  projection; no expert weights touched; a drop-in checkpoint that `abliteration/edit.pt` rebuilds bit for bit
  from the original.

Compared with OpenUncensored/Inkling-2-Mini-Uncensored, a prior release: more useful answers (54% / 63% against
34% / 43%) and lower drift from the original (KL 0.055 against 0.063).

Not everything improves: see [Known limits](#known-limits), in particular thinking loops at high effort and drift
from the original on harmless prompts.

## Use it

It loads and runs exactly like
[thinkingmachines/Inkling-2-Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini): same architecture,
files, chat template and settings. Point your existing setup at this repo.

- **SGLang** (the setup used for the evaluations below, image `lmsysorg/sglang:inkling-2-mini`):

  ```bash
  sglang serve --model-path flashvenom/Inkling-2-Mini-Unbound --tp 1 --attention-backend fa4 --dtype bfloat16 \\
    --page-size 128 --enable-multimodal --reasoning-parser inkling --tool-call-parser inkling2
  ```
- **Transformers:** Inkling-2-Mini support is in Transformers PR #51050 (tested at commit `a10ebc3c`); load it as
  the base model card describes. Experts are dequantised to BF16 on load (about 53 GB).
- **Thinking modes:** the chat template takes a thinking effort, passed as `reasoning_effort` (0 to 0.99, or
  `none`, `minimal`, `low`, `medium`, `high`, `xhigh`; default 0.9). It is rendered as a system line
  `Thinking effort level: <x>`. Effort 0 still thinks briefly. The reasoning parser splits the thinking from the
  answer.

  ```python
  from transformers import AutoTokenizer
  tok = AutoTokenizer.from_pretrained("flashvenom/Inkling-2-Mini-Unbound")
  prompt = tok.apply_chat_template([{"role": "user", "content": "..."}], tokenize=False, reasoning_effort=0.9)
  ```
- **Set a thinking cap.** At high effort the model can loop on some prompts (see Known limits); the evaluations
  used 4,096 thinking tokens.

"""

# 1. replace the "Why ..." section (up to "## What was changed") with Highlights + Use it
i = s.index("## Why Unbound is better")
j = s.index("## What was changed")
s = s[:i] + HIGHLIGHTS + s[j:]

# 2. move "What was changed" after the Evaluation section (just before Known limits), renamed "Method"
i = s.index("## What was changed")
j = s.index("## Results")
method = s[i:j].replace("## What was changed", "## Method")
s = s[:i] + s[j:]
k = s.index("## Known limits")
s = s[:k] + method + s[k:]

# 3. OU as a peer only: drop OU from prose outside tables
s = s.replace("*Harmless but borderline prompts refused. The original refuses 38%; the uncensored models refuse almost none.*",
              "*Harmless but borderline prompts refused. The original refuses 38%; Unbound refuses none.*")
s = s.replace("""short of the project's targets (≤ 0.02 and ≥ 96%); the original measured against itself gives {{KL_FLOOR}} and
  {{T1_FLOOR}}. It is closer to the original than OpenUncensored ({{KL_OU}}, {{T1_OU}}), but expect some wording
  and token-level differences on ordinary requests; the capability benchmarks above show no measurable loss.""",
              """short of the project's targets (≤ 0.02 and ≥ 96%); the original measured against itself gives {{KL_FLOOR}} and
  {{T1_FLOOR}}. Expect some wording and token-level differences on ordinary requests; the capability benchmarks
  above show no measurable loss.""")
s = s.replace(" against 0.0%, 1.4% and 0.0% for the original (OpenUncensored: 11.2%, 27.4%, 13.4%).",
              " against 0.0%, 1.4% and 0.0% for the original.")
s = s.replace("""OpenUncensored is [OpenUncensored/Inkling-2-Mini-Uncensored](https://huggingface.co/OpenUncensored/Inkling-2-Mini-Uncensored),
scored with the same harness.""",
              """For reference the tables include a prior uncensored release,
[OpenUncensored/Inkling-2-Mini-Uncensored](https://huggingface.co/OpenUncensored/Inkling-2-Mini-Uncensored),
scored with the same harness.""")
s = s.replace("- **Robustness.** Refusal and useful-answer rates when the harmful request is wrapped differently (see the\n  robustness table under Evaluation). Sets marked \"not completed\" were not measured.",
              "- **Robustness.** Refusal and useful-answer rates when the harmful request is wrapped differently are in the\n  robustness table under Evaluation. Sets marked \"not completed\" were not measured. Non-English rows use the\n  English refusal phrase list; refusals written in other languages may be undercounted.")
open(P, "w").write(s)
print([l for l in s.splitlines() if l.startswith("## ")])
print("OU mentions outside tables:", [l[:90] for l in s.splitlines() if "OpenUncensored" in l and not l.startswith("|")])
