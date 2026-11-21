"""Teacher-force another run's continuations through a local SGLang server; report top-1 agreement.
  python sglang_tf.py <port> <other.json> [n]
"""
import json
import sys
import urllib.request

port, path = sys.argv[1], sys.argv[2]
n = int(sys.argv[3]) if len(sys.argv) > 3 else 50
P = json.load(open("/root/parity/prompts.json"))
other = json.load(open(path))[:n]


def post(url: str, payload: dict | None) -> dict:
    data = json.dumps(payload).encode() if payload is not None else b""
    headers = {"content-type": "application/json"}
    try:  # admin endpoints (flush_cache, weight updates) need the per-pod key when the server has one
        headers["authorization"] = "Bearer " + open("/srv/admin_key").read().strip()
    except OSError:
        pass
    req = urllib.request.Request(f"http://localhost:{port}{url}", data=data, headers=headers)
    body = urllib.request.urlopen(req, timeout=600).read()
    return json.loads(body) if body.strip().startswith(b"{") else {}


agree = total = 0
for o in other:
    prompt = P[o["i"]]["input_ids"]
    post("/flush_cache", None)
    r = post("/generate", {"input_ids": prompt + o["tokens"],
                           "sampling_params": {"temperature": 0, "max_new_tokens": 1},
                           "return_logprob": True, "top_logprobs_num": 1,
                           "logprob_start_len": len(prompt) - 1})
    rows = r["meta_info"]["input_top_logprobs"]
    # rows[j] is the top-1 for predicting input position (len(prompt)-1+j)+1; first row may be None
    preds = [row[0][1] if row else None for row in rows]
    preds = preds[1:] if preds and preds[0] is None else preds
    preds = preds[: len(o["tokens"])]
    agree += sum(int(a == b) for a, b in zip(preds, o["tokens"]))
    total += len(o["tokens"])
print(f"SGLang TF top1 vs {path}: {agree}/{total} = {agree / max(total, 1):.4f}")
