"""Length anchors: give Q&A and roleplay training prompts a target length taken from the stock model.

Why: the W4 models fix the thinking stall but their Q&A answers got ~45% shorter (311 -> 166
tokens) and the independent grader scores them slightly lower (paired -0.047 +- 0.021). Nothing
in the reward asks for that; it rides along with shorter thinking. Setting target_words to the
stock model's own median answer length lets the existing length term hold answers where they were.

  $PY anchors.py     # prompts_train_code.jsonl -> prompts_train_code_len.jsonl (free sampling)
"""
from __future__ import annotations

import asyncio
import statistics

import lab_common as lc
import rewards as rw
import rollout as ro

CATS = ("qa", "roleplay")
N = 3


async def main() -> None:
    rows = lc.read_jsonl(lc.LAB_DIR / "prompts_train_code.jsonl")
    renderer, tok = lc.renderer_and_tokenizer()
    sc = await lc.sampling_client(lc.service_client())
    cfg = ro.SampleCfg(temperature=0.6, effort=0.6, force_answer_after_cap=True)
    sem = asyncio.Semaphore(64)

    async def one(row: dict) -> None:
        if row["category"] not in CATS or row.get("target_words"):
            return
        rolls = await ro.rollout_group(sc, ro.build_prompt(renderer, row["prompt"], cfg), N, cfg, sem)
        words = [len(rw.WORD_RE.findall(ro.to_sample(r, tok, row, cfg).answer)) for r in rolls
                 if not r.answer_runaway]
        if words:
            row["target_words"] = int(statistics.median(words))

    await asyncio.gather(*(one(r) for r in rows))
    lc.write_jsonl(lc.LAB_DIR / "prompts_train_code_len.jsonl", rows)
    for c in CATS:
        t = [r["target_words"] for r in rows if r["category"] == c and r.get("target_words")]
        print(c, len(t), "anchored; median target words", statistics.median(t), "min", min(t), "max", max(t))


if __name__ == "__main__":
    asyncio.run(main())
