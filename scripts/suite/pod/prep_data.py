"""Pod: download + normalise the core suite datasets into /srv/suite/data/<bench>.jsonl (fixed order, idx field).
GPQA Diamond comes from a public mirror pushed from the Mac (/srv/suite/data/gpqa_raw/*.parquet)."""
import json
import os
import random
import subprocess

import pandas as pd

D = "/srv/suite/data"
RAW = "/srv/suite/data/raw"
os.makedirs(RAW, exist_ok=True)


def hf(repo: str, *files: str) -> str:
    out = os.path.join(RAW, repo.replace("/", "__"))
    subprocess.run(["hf", "download", repo, *files, "--repo-type", "dataset", "--local-dir", out], check=True,
                   capture_output=True)
    return out


def dump(name: str, rows: list[dict]) -> None:
    with open(f"{D}/{name}.jsonl", "w") as f:
        for i, r in enumerate(rows):
            r["idx"] = i
            f.write(json.dumps(r) + "\n")
    print(name, len(rows))


def main() -> None:
    p = hf("openai/gsm8k", "main/test-00000-of-00001.parquet")
    df = pd.read_parquet(f"{p}/main/test-00000-of-00001.parquet")
    dump("gsm8k", [{"question": q, "answer": float(a.split("####")[-1].strip().replace(",", ""))}
                   for q, a in zip(df.question, df.answer)])

    p = hf("HuggingFaceH4/MATH-500", "test.jsonl")
    dump("math500", [{"problem": r["problem"], "answer": r["answer"], "subject": r["subject"], "level": r["level"],
                      "unique_id": r["unique_id"]} for r in map(json.loads, open(f"{p}/test.jsonl"))])

    p = hf("TIGER-Lab/MMLU-Pro")
    files = sorted(os.path.join(dp, f) for dp, _, fs in os.walk(p) for f in fs if "test" in f and f.endswith(".parquet"))
    df = pd.concat([pd.read_parquet(f) for f in files]).sort_values("question_id")
    dump("mmlu_pro", [{"question_id": int(r.question_id), "question": r.question, "options": list(r.options),
                       "answer_index": int(r.answer_index), "category": r.category} for r in df.itertuples()])

    p = hf("google/IFEval")
    files = [os.path.join(dp, f) for dp, _, fs in os.walk(p) for f in fs if f.endswith((".parquet", ".jsonl"))]
    f = files[0]
    df = pd.read_parquet(f) if f.endswith(".parquet") else pd.read_json(f, lines=True)
    rows = []
    for r in df.itertuples():
        kw = [{k: v for k, v in (d or {}).items() if v is not None} for d in r.kwargs]
        rows.append({"key": int(r.key), "prompt": r.prompt, "instruction_id_list": list(r.instruction_id_list),
                     "kwargs": kw})
    dump("ifeval", rows)

    # fingertap/GPQA-Diamond: 198 questions with options already shuffled into the text (A-D) and the gold letter.
    g = pd.read_parquet("/srv/suite/data/gpqa_raw/gpqa_diamond.parquet")
    dump("gpqa_diamond", [{"question": q, "answer": a.strip()} for q, a in zip(g.question, g.answer)])


if __name__ == "__main__":
    main()
