"""Generate RULER synthetic data ONCE (agent suite/ruler). Pod-side: python3 gen_ruler_data.py <task> <n32k> <n128k>
Generators: lm-eval 0.4.9.1 lm_eval/tasks/ruler (NVIDIA RULER port), tokenizer /srv/w/orig.
Seeding: random.seed(1234) and np.random.seed(1234) before every (task, length); the generators' internal fixed
shuffle seeds (42) are left as shipped. Each task runs in its own process so results do not depend on order.
Output: /srv/suite/data/ruler/<task>_<len>.jsonl rows {index, input, outputs, length, max_length, gen_prefix, task}.
"""
import json
import os
import random
import sys

import numpy as np

OUT = "/srv/suite/data/ruler"
TOK_PATH = "/srv/w/orig"
SEED = 1234
LENGTHS = [32768, 131072]

NIAH = {  # args copied from lm_eval/tasks/ruler/niah_utils.py
    "niah_single_1": dict(type_haystack="repeat", type_needle_k="words", type_needle_v="numbers"),
    "niah_single_2": dict(type_haystack="essay", type_needle_k="words", type_needle_v="numbers"),
    "niah_single_3": dict(type_haystack="essay", type_needle_k="words", type_needle_v="uuids"),
    "niah_multikey_1": dict(type_haystack="essay", type_needle_k="words", type_needle_v="numbers", num_needle_k=4),
    "niah_multikey_2": dict(type_haystack="needle", type_needle_k="words", type_needle_v="numbers"),
    "niah_multikey_3": dict(type_haystack="needle", type_needle_k="uuids", type_needle_v="uuids"),
    "niah_multivalue": dict(type_haystack="essay", type_needle_k="words", type_needle_v="numbers", num_needle_v=4),
    "niah_multiquery": dict(type_haystack="essay", type_needle_k="words", type_needle_v="numbers", num_needle_q=4),
}


def hotpot_from_hf():
    """The CMU hotpot URL in qa_utils does not resolve from the pod; rebuild the same JSON structure from the
    HF mirror of the distractor dev set (hotpotqa/hotpot_qa, config distractor, split validation)."""
    import datasets
    ds = datasets.load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
    return [{"question": d["question"], "answer": d["answer"],
             "context": [[t, s] for t, s in zip(d["context"]["title"], d["context"]["sentences"])]} for d in ds]


def gen(task, seq, n, tok):
    random.seed(SEED)
    np.random.seed(SEED)
    if task in NIAH:
        from lm_eval.tasks.ruler import niah_utils, prepare_niah
        cfg = NIAH[task]
        return prepare_niah.generate_samples(prepare_niah.get_haystack(type_haystack=cfg["type_haystack"]),
                                             max_seq_length=seq, template=niah_utils.TEMPLATE, num_samples=n,
                                             TOKENIZER=tok, **cfg)
    if task == "vt":
        from lm_eval.tasks.ruler import vt_utils
        icl = vt_utils.sys_vartrack_w_noise_random(tokenizer=tok, num_samples=1, max_seq_length=500, incremental=5)[0]
        return vt_utils.sys_vartrack_w_noise_random(tokenizer=tok, num_samples=n, max_seq_length=seq, icl_example=icl)
    if task == "cwe":
        from lm_eval.tasks.ruler import cwe_utils
        return cwe_utils.sys_word_pair_random(num_samples=n, max_seq_length=seq, tokenizer=tok)
    if task == "fwe":
        from lm_eval.tasks.ruler import fwe_utils
        return fwe_utils.sys_kwext(tokenizer=tok, max_seq_length=seq, num_samples=n)
    if task in ("qa_1", "qa_2"):
        from lm_eval.tasks.ruler import qa_utils
        if task == "qa_1":
            qas, docs = qa_utils.read_squad()
        else:
            orig = qa_utils.download_json
            qa_utils.download_json = lambda url: hotpot_from_hf() if "hotpot" in url else orig(url)
            qas, docs = qa_utils.read_hotpotqa()
        return qa_utils.generate_samples(tokenizer=tok, docs=docs, qas=qas, max_seq_length=seq, num_samples=n,
                                         tokens_to_generate=32)
    raise ValueError(task)


def main():
    task, n32, n128 = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOK_PATH)
    os.makedirs(OUT, exist_ok=True)
    for seq, n in zip(LENGTHS, [n32, n128]):
        p = f"{OUT}/{task}_{seq}.jsonl"
        if os.path.exists(p):
            print("exists", p)
            continue
        rows = gen(task, seq, n, tok)
        with open(p + ".tmp", "w") as f:
            for r in rows:
                r = {k: r[k] for k in ("index", "input", "outputs", "length", "max_length", "gen_prefix")}
                r["task"] = task
                f.write(json.dumps(r) + "\n")
        os.replace(p + ".tmp", p)
        print(f"DONE {task} {seq} n={len(rows)} max_len={max(r['length'] for r in rows)}", flush=True)


if __name__ == "__main__":
    main()
