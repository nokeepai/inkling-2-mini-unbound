"""Core capability suite (agent `suite`): mmlu_pro, gsm8k, math500, gpqa_diamond, ifeval, humaneval_plus.

  python3 bench_core.py <bench> gen --port P --model M [--shard k --nshards K]
  python3 bench_core.py <bench> score --model M

Settings (also in R/suite/PLAN.md):
- mmlu_pro: all 12,032 test questions, effort 0, 10-way letter logprobs after the forced answer prefix [200001, 200004]
  (the inner-loop MMLU format, extended to A-J). Zero-shot, no CoT.
- gsm8k (1,319), math500 (500), gpqa_diamond (198): effort 0.9, thinking cap 4,096 (force-close), answer cap 512/1024/1024;
  only the final answer (thinking stripped) is scored.
- ifeval (541): effort 0, answer cap 2,048, lm-eval 0.4.9.1 ifeval checkers; main = prompt-level strict.
- humaneval_plus (164): effort 0, answer cap 1,024, evalplus 0.3.1; main = plus (base + extra tests) pass@1, greedy.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import suite_common as S  # noqa: E402

LETTERS = "ABCDEFGHIJ"
CFG = {  # bench: (effort, think_cap, ans_cap)
    "gsm8k": (0.9, 4096, 512), "math500": (0.9, 4096, 1024), "gpqa_diamond": (0.9, 4096, 1024),
    "ifeval": (0, 2048, 2048), "humaneval_plus": (0, 1024, 1024)}


# Owner 15:15Z: fixed seeded subsets, identical for every model (idx keeps the full-set position).
SUBSET = {"mmlu_pro": 1000, "gsm8k": 500, "math500": 200}


def load(bench: str) -> list[dict]:
    rows = load_full(bench)
    if bench in SUBSET and os.environ.get("SUITE_FULL") != "1":
        import random
        keep = set(random.Random(1234).sample(range(len(rows)), SUBSET[bench]))
        rows = [r for r in rows if r["idx"] in keep]
    return rows


def load_full(bench: str) -> list[dict]:
    if bench == "humaneval_plus":
        from evalplus.data import get_human_eval_plus
        probs = get_human_eval_plus()
        return [{"idx": i, "task_id": t, "prompt": p["prompt"]} for i, (t, p) in enumerate(sorted(probs.items(),
                key=lambda kv: int(kv[0].split("/")[1])))]
    return [json.loads(l) for l in open(f"{S.DATA}/{bench}.jsonl")]


def message(bench: str, r: dict) -> str:
    if bench == "gsm8k":
        return f'{r["question"].strip()}\n\nSolve the problem. End your reply with a line of the form "Answer: <number>".'
    if bench == "math500":
        return f"{r['problem'].strip()}\n\nSolve the problem. Put your final answer within \\boxed{{}}."
    if bench == "gpqa_diamond":
        return (f"{r['question'].strip()}\n\nChoose the correct option. End your reply with a line of the form "
                f'"Answer: <letter>".')
    if bench == "ifeval":
        return r["prompt"]
    if bench == "humaneval_plus":
        return ("Complete the following Python function. Reply with the complete function (including the signature "
                "and any needed imports) in a single ```python code block.\n\n```python\n" + r["prompt"] + "```")
    raise ValueError(bench)


def gen_mmlu_pro(port: int, model: str, k: int, nsh: int) -> list[dict]:
    lid = []
    for L in LETTERS:
        t = S.tok().encode(L, add_special_tokens=False)
        assert len(t) == 1, L
        lid.append(t[0])

    def one(r):
        opts = r["options"]
        body = "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(opts))
        msg = f"{r['question'].strip()}\n\n{body}\n\nAnswer with the letter of the correct option only."
        ids = S.render(msg, 0) + [S.MSG_MODEL, S.TEXT]
        want = lid[:len(opts)]
        o = S.post(port, "/generate", {"input_ids": ids, "sampling_params": {"temperature": 0, "max_new_tokens": 1},
                                       "return_logprob": True, "token_ids_logprob": want,
                                       "logprob_start_len": len(ids) - 1})
        d = {t[1]: t[0] for t in o["meta_info"]["output_token_ids_logprobs"][0]}
        pred = max(range(len(opts)), key=lambda i: d[want[i]])
        return {"idx": r["idx"], "pred": pred, "answer": r["answer_index"], "category": r["category"]}
    return S.run_pool(one, S.shard(load("mmlu_pro"), k, nsh))


def gen_generative(bench: str):
    eff, tcap, acap = CFG[bench]

    def run(port: int, model: str, k: int, nsh: int) -> list[dict]:
        S.tok()  # load once in the main thread: transformers' lazy import is not thread-safe

        def one(r):
            o = S.capped(port, S.render(message(bench, r), eff), tcap, acap)
            o["idx"] = r["idx"]
            return o
        return S.run_pool(one, S.shard(load(bench), k, nsh))
    return run


NUM = re.compile(r"-?[\d,]*\.?\d+")
ANS = re.compile(r"Answer:\s*\$?\s*(-?[\d,]*\.?\d+)")
LET = re.compile(r"Answer\s*[:：]\s*\**\s*\(?\s*([A-D])\b")


def gsm8k_pred(text: str) -> float | None:
    m = ANS.findall(text) or NUM.findall(text)
    if not m:
        return None
    try:
        return float(m[-1].replace(",", ""))
    except ValueError:
        return None


def code_block(text: str, prompt: str) -> str:
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, re.S)
    if not blocks:
        m = re.search(r"```(?:python|py)?\s*\n(.*)", text, re.S)  # unterminated block
        blocks = [m.group(1)] if m else [text]
    code = max(blocks, key=len)
    if "def " not in code:  # body only: prepend the prompt
        code = prompt + code
    return code


def score(bench: str, model: str) -> None:
    data = {r["idx"]: r for r in S.shard(load(bench), 0, 1)}
    rows = [r for r in S.read_gens(model, bench) if r["idx"] in data]  # full-set runs are scored on the subset
    assert len(rows) == len(data), f"{bench}/{model}: {len(rows)} gens vs {len(data)} items"
    met: dict = {}
    if bench == "mmlu_pro":
        met["mmlu_pro"] = S.stat([float(r["pred"] == r["answer"]) for r in rows], 100)
        for c in sorted({r["category"] for r in rows}):
            met[f"mmlu_pro_{c.replace(' ', '_')}"] = S.stat([float(r["pred"] == r["answer"]) for r in rows
                                                             if r["category"] == c], 100)
        S.write_score(model, bench, met, "mmlu_pro")
        return
    if bench == "gsm8k":
        ok = []
        for r in rows:
            p = gsm8k_pred(r["answer"])
            ok.append(float(p is not None and abs(p - data[r["idx"]]["answer"]) < 1e-6))
        met["gsm8k"] = S.stat(ok, 100)
    elif bench == "math500":
        from math_verify import parse, verify
        ok = []
        for r in rows:
            gold = parse("$" + data[r["idx"]]["answer"] + "$")
            try:
                ok.append(float(bool(verify(gold, parse(r["answer"])))))
            except Exception:  # noqa: BLE001  parse failures count as wrong (loud in the per-item file)
                ok.append(0.0)
        met["math500"] = S.stat(ok, 100)
    elif bench == "gpqa_diamond":
        ok = []
        for r in rows:
            m = LET.findall(r["answer"])
            ok.append(float(bool(m) and m[-1] == data[r["idx"]]["answer"]))
        met["gpqa_diamond"] = S.stat(ok, 100)
    elif bench == "ifeval":
        from lm_eval.tasks.ifeval.utils import process_results
        ps, pl, ins, inl = [], [], [], []
        for r in rows:
            d = data[r["idx"]]
            doc = {"key": d["key"], "prompt": d["prompt"], "instruction_id_list": d["instruction_id_list"],
                   "kwargs": d["kwargs"]}
            o = process_results(doc, [r["answer"]])
            ps.append(float(o["prompt_level_strict_acc"]))
            pl.append(float(o["prompt_level_loose_acc"]))
            ins += [float(x) for x in o["inst_level_strict_acc"]]
            inl += [float(x) for x in o["inst_level_loose_acc"]]
        met["ifeval"] = S.stat(ps, 100)
        met["ifeval_prompt_loose"] = S.stat(pl, 100)
        met["ifeval_inst_strict"] = S.stat(ins, 100)
        met["ifeval_inst_loose"] = S.stat(inl, 100)
    elif bench == "humaneval_plus":
        d = S.out_dir(model, bench)
        samples = os.path.join(d, "samples.jsonl")
        with open(samples, "w") as f:
            for r in rows:
                it = data[r["idx"]]
                f.write(json.dumps({"task_id": it["task_id"], "solution": code_block(r["answer"], it["prompt"])}) + "\n")
        res_path = samples.replace(".jsonl", "_eval_results.json")
        if os.path.exists(res_path):
            os.remove(res_path)
        subprocess.run([sys.executable, "-m", "evalplus.evaluate", "--dataset", "humaneval", "--samples", samples,
                        "--parallel", "32", "--i_just_wanna_run"], check=True)
        ev = json.load(open(res_path))["eval"]
        base, plus = [], []
        for r in rows:
            e = ev[data[r["idx"]]["task_id"]][0]
            base.append(float(e["base_status"] == "pass"))
            plus.append(float(e["base_status"] == "pass" and e["plus_status"] == "pass"))
        met["humaneval_plus"] = S.stat(plus, 100)
        met["humaneval_base"] = S.stat(base, 100)
    for k, v in S.hygiene(rows).items():
        met[f"{bench}_{k}"] = v
    S.write_score(model, bench, met, bench)


GEN = {"mmlu_pro": gen_mmlu_pro, **{b: gen_generative(b) for b in CFG}}

if __name__ == "__main__":
    bench = sys.argv.pop(1)
    assert bench in GEN, bench
    S.cli(bench, GEN[bench], lambda m: score(bench, m))
