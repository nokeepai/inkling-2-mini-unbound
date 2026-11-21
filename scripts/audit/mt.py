"""Auditor: fixed open machine translation for the language robustness test (Haiku is not available).

  python3 mt.py fwd <harmful.jsonl> <out.jsonl> [--limit N]          # English -> es, fr, de, zh
  python3 mt.py back <gens_robust_lang_*.jsonl ...>                  # answers -> English, writes *.en.jsonl beside each

Models: Helsinki-NLP/opus-mt-en-{es,fr,de,zh} and opus-mt-{es,fr,de,zh}-en (MarianMT), revisions recorded in the
output (snapshot commit). Decoding: each model's own generation config (beam search, deterministic), max 512 new
tokens per segment. Text is split on newlines and translated line by line (Marian's 512-token input limit), blank
lines kept. GPU if available. Back-translated rows get "answer_en" (and keep "answer").
"""
from __future__ import annotations

import json
import os
import re
import sys
import time

LANGS = ("es", "fr", "de", "zh")
_M: dict = {}


def model(name: str):
    if name not in _M:
        import torch
        from huggingface_hub import snapshot_download
        from transformers import MarianMTModel, MarianTokenizer
        path = snapshot_download(f"Helsinki-NLP/{name}", allow_patterns=["*.json", "*.spm", "*.bin", "*.safetensors",
                                                                         "*.txt", "vocab*"])
        rev = os.path.basename(path.rstrip("/"))
        tok = MarianTokenizer.from_pretrained(path)
        m = MarianMTModel.from_pretrained(path).to("cuda" if torch.cuda.is_available() else "cpu").eval()
        _M[name] = (tok, m, rev)
    return _M[name]


def segments(text: str) -> list[str]:
    out = []
    for line in text.split("\n"):
        if len(line) > 1500:  # very long lines: split on sentence ends
            out.extend(s for s in re.split(r"(?<=[.!?。！？])\s+", line) if s)
        else:
            out.append(line)
    return out


def translate(name: str, texts: list[str], bs: int = int(os.environ.get("MT_BS", "16"))) -> tuple[list[str], str]:
    import torch
    tok, m, rev = model(name)
    segs_per = [segments(t) for t in texts]
    flat = [s for segs in segs_per for s in segs if s.strip()]
    done: dict[str, str] = {}
    uniq = sorted(set(flat), key=len)
    with torch.no_grad():
        for i in range(0, len(uniq), bs):
            chunk = uniq[i:i + bs]
            enc = tok(chunk, return_tensors="pt", padding=True, truncation=True, max_length=512).to(m.device)
            gen = m.generate(**enc, max_new_tokens=512)
            for s, g in zip(chunk, tok.batch_decode(gen, skip_special_tokens=True)):
                done[s] = g
    outs = []
    for segs in segs_per:
        outs.append("\n".join(done[s] if s.strip() else s for s in segs))
    return outs, rev


def fwd(src: str, dst: str, limit: int) -> None:
    rows = [json.loads(l) for l in open(src) if l.strip()]
    if limit:
        rows = rows[:limit]
    t = time.time()
    with open(dst, "w") as f:
        for lang in LANGS:
            outs, rev = translate(f"opus-mt-en-{lang}", [r["text"] for r in rows])
            for r, o in zip(rows, outs):
                f.write(json.dumps({"id": r["id"], "lang": lang, "text": o, "mt": f"Helsinki-NLP/opus-mt-en-{lang}@{rev}"},
                                   ensure_ascii=False) + "\n")
    print(f"MT_FWD_DONE {len(rows)} prompts x {len(LANGS)} langs in {time.time() - t:.0f}s", flush=True)


def back(paths: list[str]) -> None:
    for p in paths:
        rows = [json.loads(l) for l in open(p) if l.strip()]
        by_lang: dict[str, list[int]] = {}
        for i, r in enumerate(rows):
            by_lang.setdefault(r.get("lang", "en"), []).append(i)
        for lang, idx in by_lang.items():
            if lang == "en":
                for i in idx:
                    rows[i]["answer_en"] = rows[i]["answer"]
                continue
            outs, rev = translate(f"opus-mt-{lang}-en", [rows[i]["answer"] for i in idx])
            for i, o in zip(idx, outs):
                rows[i]["answer_en"], rows[i]["mt_back"] = o, f"Helsinki-NLP/opus-mt-{lang}-en@{rev}"
        out = p[:-6] + ".en.jsonl"
        with open(out, "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"MT_BACK {p} -> {out} ({len(rows)} rows)", flush=True)


def main() -> None:
    cmd = sys.argv[1]
    lim = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else 0
    if cmd == "fwd":
        fwd(sys.argv[2], sys.argv[3], lim)
    elif cmd == "back":
        back([p for p in sys.argv[2:] if p.endswith(".jsonl")])
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
