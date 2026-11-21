"""Render the Inkling chat template for thinking off / 0.5 / 0.9 and tokenize with the checkpoint tokenizer."""
import json
import sys

from transformers import AutoTokenizer

ckpt = sys.argv[1]
tok = AutoTokenizer.from_pretrained(ckpt)
template = open(f"{ckpt}/chat_template.jinja").read()
msgs = [{"role": "user", "content": "How do I bake bread?"}]
out = {}
for label, kw in (("off", {"reasoning_effort": 0}), ("none_str", {"reasoning_effort": "none"}),
                  ("0.5", {"reasoning_effort": 0.5}), ("0.9", {"reasoning_effort": 0.9}),
                  ("default", {})):
    for agp in (False, True):
        text = tok.apply_chat_template(msgs, chat_template=template, tokenize=False, add_generation_prompt=agp, **kw)
        ids = tok(text, add_special_tokens=False)["input_ids"]
        ids_default = tok(text)["input_ids"]
        out[f"{label}|agp={agp}"] = {
            "text": text,
            "ids_tail": ids[-8:],
            "tokens_tail": tok.convert_ids_to_tokens(ids[-8:]),
            "n": len(ids),
            "add_special_tokens_changes_ids": ids != ids_default,
            "first_ids": ids[:6],
        }
# with system prompt and an assistant turn with reasoning
msgs2 = [{"role": "system", "content": "You are helpful."}, {"role": "user", "content": "Hi"},
         {"role": "assistant", "content": "Hello!", "reasoning_content": "Greet back."}, {"role": "user", "content": "Bye"}]
out["multi_turn_0.9"] = tok.apply_chat_template(msgs2, chat_template=template, tokenize=False, reasoning_effort=0.9)
print(json.dumps(out, indent=1))
