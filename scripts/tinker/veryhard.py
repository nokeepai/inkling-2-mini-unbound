"""Very hard maths set: AIME problems (integer answers 0-999), where long thinking should pay.

Why: on the lab's other sets more thinking does not help (stock e0.9 scores below stock e0.3),
so they cannot show whether a model trained to stop thinking sooner loses depth. This set is run
with a 16,000-token thinking cap so stock e0.9 gets the room it wants.

  $PY veryhard.py                      # -> prompts_veryhard.jsonl
  $PY eval.py --tag base_veryhard_e0.9 --prompts prompts_veryhard.jsonl --effort 0.9 \
      --think-cap 16000 --answer-cap 2048
"""
from __future__ import annotations

import lab_common as lc

SUFFIX = "\n\nGive the final answer in \\boxed{}."


def main() -> None:
    from datasets import load_dataset

    ds = load_dataset("AI-MO/aimo-validation-aime", split="train")
    rows = []
    for i, r in enumerate(ds):
        gold = str(r["answer"]).strip()
        if not gold.lstrip("-").isdigit():
            continue
        rows.append({"id": f"vh{i:03d}", "category": "maths", "source": "aime_validation",
                     "prompt": r["problem"].strip() + SUFFIX, "target_words": None,
                     "gold": str(int(gold))})
    lc.write_jsonl(lc.LAB_DIR / "prompts_veryhard.jsonl", rows)
    print(f"wrote {len(rows)} rows; first gold {rows[0]['gold']}")


if __name__ == "__main__":
    main()
