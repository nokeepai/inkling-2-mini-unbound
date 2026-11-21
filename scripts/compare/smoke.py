"""ON pod: smoke-test a running llama-server against the HF tokenizer and the stop handling.  python3 smoke.py <port>"""
import json
import sys
import urllib.request

sys.argv = [sys.argv[0], "--port", sys.argv[1], "--out", "/tmp/smoke"]
sys.path.insert(0, "/root/compare/bin")
import llama_eval as E  # noqa: E402

E.PORT = int(sys.argv[2])
s = E.TOK.apply_chat_template([{"role": "user", "content": "Name three primes. Be brief."}], tokenize=False,
                              reasoning_effort=0)
hf_ids = E.TOK(s, add_special_tokens=False).input_ids
req = urllib.request.Request(f"http://127.0.0.1:{E.PORT}/tokenize", data=json.dumps(
    {"content": s, "add_special": False, "parse_special": True}).encode(), headers={"content-type": "application/json"})
ll_ids = json.loads(urllib.request.urlopen(req).read())["tokens"]
print("token ids identical:", hf_ids == ll_ids, len(hf_ids), len(ll_ids))
raw = E.post("/completion", {"prompt": hf_ids, "n_predict": 600, "temperature": 0.0, "top_k": 1, "return_tokens": True,
                             "stop": [E.END_S, E.EOS_S]})
print("stop_type", raw.get("stop_type"), "stopping_word", repr(raw.get("stopping_word")), "ntok", len(raw["tokens"]),
      "first", raw["tokens"][:3], "last", raw["tokens"][-3:])
for eff in (0, 0.9):
    r = E.capped(E.render("Name three primes. Be brief.", eff), 4096, 512)
    print(eff, {k: r[k] for k in ("think_len", "cap_hit", "finished")}, repr(r["answer"][:200]))
lid = [E.TOK.encode(L, add_special_tokens=False)[0] for L in "ABCD"]
ids = E.render("Which is a prime?\n\nA. 4\nB. 6\nC. 7\nD. 9\n\nAnswer with the letter of the correct option only.", 0)
o = E.post("/completion", {"prompt": ids + [E.MSG_MODEL, E.TEXT], "n_predict": 1, "temperature": 0.0, "top_k": 1,
                           "n_probs": 200, "post_sampling_probs": False})
top = o["completion_probabilities"][0]["top_logprobs"]
d = {t["id"]: t["logprob"] for t in top}
print("mmlu letters lp", [d.get(x) for x in lid], "ntop", len(top))
