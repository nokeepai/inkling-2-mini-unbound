# Capability inner-loop format (shared by HF `harness/hf_cap_eval.py` and SGLang `harness/sgl_eval.py`)

Written by `fidelity` 2026-11-21 ~10:45Z. Both engines must send **identical token ids**: render with the official
chat template via `tok.apply_chat_template(messages, tokenize=False, reasoning_effort=E)`, then
`tok(text, add_special_tokens=False)`. No system prompt. Token ids below are from MODEL_NOTES §6.

Special ids: `<|message_model|>`=200001, `<|content_text|>`=200004, `<|content_thinking|>`=200008,
`<|end_message|>`=200010, `<|content_model_end_sampling|>` (eos)=200006.

## MMLU-300 (`cap/mmlu300.jsonl`), letter logits, effort 0
- User message (exact, `\n` = newline):
  `{question.strip()}\n\nA. {c0}\nB. {c1}\nC. {c2}\nD. {c3}\n\nAnswer with the letter of the correct option only.`
- Effort 0 (`reasoning_effort=0`).
- Append the forced answer prefix `[200001, 200004]` (`<|message_model|><|content_text|>`) after the rendered prompt,
  so the answer channel opens with no thinking.
- Score: the next-token logits at the last position, restricted to the four token ids of `"A"`, `"B"`, `"C"`, `"D"`
  (`tok.encode(L, add_special_tokens=False)`, each must be a single token). Prediction = argmax of the four. Correct if
  it equals `answer` (0..3). Report accuracy ×100, n=300, 95% bootstrap CI over questions (10,000 resamples, seed 1234).
- SGLang: send `input_ids` = prompt + prefix, `max_new_tokens=1`, `return_logprob=True`,
  `token_ids_logprob=[idA,idB,idC,idD]`, compare those four logprobs.

## GSM8K-100 (`cap/gsm8k100.jsonl`), generation, effort 0.5
- User message: `{question.strip()}\n\nSolve the problem. End your reply with a line of the form "Answer: <number>".`
- Effort 0.5. Greedy (temperature 0). No forced prefix: the model emits `<|message_model|><|content_thinking|>` itself.
- **Thinking cap 1,536 tokens**, counted from the token after `<|content_thinking|>` (200008). If the model has not emitted
  `<|end_message|>` (200010) within 1,536 thinking tokens, truncate the thinking at exactly 1,536 tokens and append
  `[200010, 200001, 200004]` (close thinking, open the answer), then continue greedily.
- **Answer budget 512 tokens** after `<|content_text|>` (200004). Stop at eos (200006) or at the `<|end_message|>` that
  closes the answer.
- Finished = the answer closed (200010 or eos) within 512 tokens. Report the finish rate and the count of
  thinking-cap hits.
- Score only the answer text (between 200004 and the closing 200010/eos; thinking stripped). Extraction: the last
  match of `Answer:\s*\$?\s*(-?[\d,]*\.?\d+)`; if none, the last number `-?[\d,]*\.?\d+` in the answer text. Remove
  commas, compare as float with `abs(pred - answer) < 1e-6`. Unfinished answers are scored on what was produced.
- Report accuracy ×100, n=100, 95% bootstrap CI over problems (10,000 resamples, seed 1234), finish rate,
  cap hits, mean thinking tokens.
