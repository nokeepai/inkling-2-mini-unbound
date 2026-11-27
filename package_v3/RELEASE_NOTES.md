## Inkling-2-Mini-Unbound v3 and Inkling-2-Mini-Prime

[Inkling 2 Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini) is Thinking Machines Lab's open-weights reasoning
model: it reads text and images and "thinks" in private before it answers. **Inkling-2-Mini-Unbound** is nokeep's
uncensored version of it, the same model with its refusals removed. **Unbound follows harmful instructions; do not
put it in front of end users without your own safeguards.**

This release fixes the biggest flaw of the previous version, v2.1: it thought too much, often hit the thinking limit
(the cap on how long it may think before it must answer), and stalled on multi-step work. One small extra training
step, the **Prime add-on**, teaches the model to stop deliberating and act: a small set of weight changes (a LoRA
adapter) trained only on harmless tasks. Merged into Unbound v2.1 it gives Unbound v3; merged into the original
model it gives Inkling-2-Mini-Prime; on its own it is the repo Inkling-2-Mini-Prime-LoRA. Because it was trained
only on harmless tasks, it changes how long the model thinks, not what it is willing to say.

What was released:

- **Unbound v3**: Unbound v2.1 plus the Prime add-on. Still uncensored, and no more so than v2.1.
- **Inkling-2-Mini-Prime** ("Prime"): the **original, still-censored** Inkling 2 Mini plus the same
  Prime add-on. Not uncensored. It is for agents and multi-turn work (it finishes multi-step debugging instead of
  stalling), for long system prompts, and for anyone who wants answers with about half the thinking.
- **Inkling-2-Mini-Prime-LoRA**: the Prime add-on on its own, as a small add-on file (a LoRA adapter) with its merge script.

**Results.** Each model was run once, before and after the Prime add-on, with the settings below.

| What we measured | Original → Prime | Unbound v2.1 → v3 |
|---|---|---|
| Hard debugging tasks solved (15 tasks) | 80.0% [54.8, 93.0] → 100.0% [79.6, 100.0] | 73.3% [48.0, 89.1] → 100.0% [79.6, 100.0] |
| Debugging turns with no code written | 86.2% → 0.0% | 88.7% → 0.0% |
| Thinking per everyday prompt (median tokens) | 318 → 174 | 344 → 184 |
| Everyday answers rated good by an independent grader | 96.8% [89.1, 99.1] → 96.8% [89.0, 99.1] | 88.9% [78.8, 94.5] → 96.8% [89.1, 99.1] |
| Sensitive requests: willing to answer (1,100 original → Prime; 1,099 / 1,100 graded v2.1 → v3) | 64.5% [61.6, 67.2] → 65.3% [62.4, 68.0] | 99.4% [98.7, 99.7] → 99.9% [99.5, 100.0] |

How to read it: each debugging task is a small program with three interacting bugs; the model sees the failing
tests, proposes a fix and repeats for up to 12 turns. Sensitive requests come from uncensoredbench, a
benchmark of requests models usually refuse. Numbers in brackets are 95% intervals, the range the true value
plausibly sits in.

**Useful answers to harmful requests.** Unbound v3 was also run on the comparison test from the v2.1 release (263 requests, same engine, settings and judge; v2.1's saved answers re-judged in the same pass): v3 81.7% [76.8, 86.3] with thinking off and 87.8% [83.7, 91.6] with thinking on, against v2.1 84.0% [79.5, 88.2] / 85.6% [81.0, 89.7]. Difference on the same requests: -2.3 pts [-6.8, +2.3] / +2.3 pts [-1.5, +6.1].

Both models think about half as much and finish the debugging work they used to stall on. **No
accuracy gain is claimed:** on public coding, maths and knowledge benchmarks both score level with their starting
points, within about 2 points. (Strict maths scoring rose +13 to +18 points, mostly answer
format; counting correct numbers the change is +0 to +3 points.)

**What got worse, or is unconfirmed.**

- On sensitive requests Prime answers as often as the original and slightly less fully (delivery
  score 35.9% [34.0, 37.7] → 34.6% [32.8, 36.4]).
- A possible small dip for Prime on MBPP (short Python problems) is unconfirmed:
  -1.2 pts [-2.5, +0.1], pooled over 3 runs with different random seeds.
- Unbound v3 is level with v2.1 on MBPP (-0.1 pts [-1.3, +1.2]), and a small loss on HumanEvalFix (bug-fixing
  problems, -1.4 pts [-3.5, +0.6]) is neither confirmed nor ruled out.
- Each model card lists every check that scored lower, however slightly.

**Settings used for every number:** `reasoning_effort` 0.9 (how much the model thinks), temperature
0.6, top_p 1.0, thinking limit 4096 tokens.

**Downloads:** [Unbound](https://huggingface.co/nokeepai/Inkling-2-Mini-Unbound) ·
[Unbound GGUF](https://huggingface.co/nokeepai/Inkling-2-Mini-Unbound-GGUF) (for llama.cpp, LM Studio, Ollama) ·
[Unbound Basalt](https://huggingface.co/nokeepai/Inkling-2-Mini-Unbound-Basalt) (one file for NVIDIA Blackwell GPUs) ·
[Prime](https://huggingface.co/nokeepai/Inkling-2-Mini-Prime) ·
[Prime GGUF](https://huggingface.co/nokeepai/Inkling-2-Mini-Prime-GGUF) ·
[Inkling-2-Mini-Prime-LoRA](https://huggingface.co/nokeepai/Inkling-2-Mini-Prime-LoRA)

**Earlier versions** stay available: v2.1 and v2 are branches of each Unbound repo on Hugging Face, and v1.0 is a
tagged revision. What each version changed and why is in the
[changelog](https://github.com/nokeepai/inkling-2-mini-unbound/blob/main/package_v3/CHANGELOG.md).
