## Inkling-2-Mini-Unbound v2.1

An uncensored Inkling 2 Mini that doesn't just stop refusing: it actually does what you ask. v2.1 is the current
version. Only 120 BF16 attention tensors differ from the original; experts, router, vision tower and tokenizer are
byte-identical, so it is a drop-in Inkling 2 Mini.

**Recommended settings: thinking on, temperature 0.6, top_p 0.95, thinking cap 4,096 tokens.** Do not use DRY sampling.

**Useful answers to 263 held-out harmful requests** (v2's sealed test without its CBRN category;
same prompts, llama.cpp engine, settings and judge for every model):

| Model | Thinking off | Thinking on |
|---|---|---|
| **Unbound v2.1** | **83.7%** | **85.2%** |
| IndrasMirror XQ | 58.9% | 68.1% |
| AtomicChat LoRA | 52.1% | 55.5% |
| OpenUncensored | 33.1% | 41.8% |
| JoaoZaokk Heretic | 17.5% | 26.2% |
| Original Inkling 2 Mini | 0.8% | 0.8% |

- Gain over the next best release (XQ), same prompts: +24.7 pts [+18.6, +30.8] off, +17.1 [+11.4, +22.8] on.
- Refuses 0.4% of harmful requests; 1 of 784 benign borderline prompts refused. MMLU-300 74.3 [69.3, 79.3],
  GSM8K-100 98 (original 76.3 / 97).
- Trade-offs: the most drift from the original of any release we measured (KLD 0.0439). With greedy decoding and
  thinking on, 22% of harmful prompts hit the thinking cap; temperature 0.6 brings that down (16.1% vs 29.1% greedy
  on v2.1's own test).
- Method: weight edit, then self-distillation, then an SFT + DPO round on the model's own answers.

**Downloads:** [weights](https://huggingface.co/nokeepai/Inkling-2-Mini-Unbound) ·
[GGUF](https://huggingface.co/nokeepai/Inkling-2-Mini-Unbound-GGUF) ·
[Basalt](https://huggingface.co/nokeepai/Inkling-2-Mini-Unbound-Basalt). Earlier versions: v2 on the `v2` branch of
each repo, v1.0 as legacy files. Run report: `v3/REPORT.md` section 0; release build: `scripts/v21/`.
