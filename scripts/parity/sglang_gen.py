"""Greedy 64-token generation through a local SGLang server, from pre-tokenised prompts.

  python sglang_gen.py <port> <out.json> [n_prompts]
Sends input_ids directly (same token ids as the HF side), temperature 0, and asks for
per-position top-1 logprobs so SGLang's own top-1 is recorded next to the sampled token.
"""
import json
import sys
import time
import urllib.request

port, out_path = sys.argv[1], sys.argv[2]
n = int(sys.argv[3]) if len(sys.argv) > 3 else 50
P = json.load(open("/root/parity/prompts.json"))[:n]


def post(payload: dict) -> dict:
    req = urllib.request.Request(f"http://localhost:{port}/generate", data=json.dumps(payload).encode(),
                                 headers={"content-type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=600))


res = []
t0 = time.time()
import os
FLUSH = os.environ.get("FLUSH", "1") == "1"  # flush radix cache so every prompt is prefilled in full, like HF


def flush() -> None:
    req = urllib.request.Request(f"http://localhost:{port}/flush_cache", data=b"", method="POST")
    urllib.request.urlopen(req, timeout=60).read()


for r in P:
    if FLUSH:
        flush()
    o = post({"input_ids": r["input_ids"],
              "sampling_params": {"temperature": 0, "max_new_tokens": 64, "skip_special_tokens": False},
              "return_logprob": True, "top_logprobs_num": 2, "logprob_start_len": len(r["input_ids"]) - 1})
    meta = o["meta_info"]
    toks = o.get("output_ids") or [t[1] for t in meta["output_token_logprobs"]]
    tops = meta.get("output_top_logprobs", [])
    top1 = [row[0][1] for row in tops]
    margin = [round(row[0][0] - row[1][0], 4) for row in tops]
    res.append({"i": r["i"], "tokens": toks, "sg_top1": top1, "margin": margin, "text": o.get("text", "")[:400],
                "finish": meta.get("finish_reason")})
json.dump(res, open(out_path, "w"))
self_agree = sum(sum(int(a == b) for a, b in zip(x["tokens"], x["sg_top1"])) for x in res)
total = sum(len(x["tokens"]) for x in res)
print(f"{len(res)} prompts in {time.time() - t0:.1f}s; sampled==own top1 {self_agree}/{total}")
