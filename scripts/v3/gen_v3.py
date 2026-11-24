"""ON eval/gen pod: sample (or greedy-eval) completions from llama-servers. Extends scripts/v11b/gen_samples.py.

  python3 gen_v3.py --ports 8100,8101 --inp prompts.jsonl --out out.jsonl --effort 0 --n 4 --temp 0.8 \
      [--system sys.txt] [--prefill-file pre.txt] [--think-cap 1536] [--ans-cap 768] [--conc 12] [--greedy]

Input rows: {"id", "text", "category"?, "system"? (overrides --system), "prefill"? (overrides --prefill-file)}.
--greedy: temperature 0, n forced to 1, think cap 4096 / answer cap 8192 unless given.
Prefill: after the model's thinking closes (or at once when it does not open thinking), the answer channel is forced
open as [MSG_MODEL, TEXT] + tokens(prefill) and the model continues from there. The prefill tokens are part of
"completion" (they become training targets) and "answer" starts with the prefill text.
Output rows (gen_samples schema + new): id, prompt, category, effort, sys (0/1), sys_tag, k, think_len, cap_hit (thinking
unfinished at cap, force-closed), finished (answer closed before cap), ans_len, answer, think, completion, prefill.
Resumable: rows already in --out with the same (id, k) are skipped.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import threading
import urllib.request
import zlib
from concurrent.futures import ThreadPoolExecutor

from transformers import AutoTokenizer

MSG_MODEL, TEXT, THINK, END, EOS = 200001, 200004, 200008, 200010, 200006
TOK = AutoTokenizer.from_pretrained(os.environ.get("CMP_TOK", "/data/compare/models/tok"))
END_S, EOS_S = TOK.decode([END]), TOK.decode([EOS])
LOCK = threading.Lock()
Samp = dict


def post(port: int, payload: dict) -> dict:
    req = urllib.request.Request(f"http://127.0.0.1:{port}/completion", data=json.dumps(payload).encode(),
                                 headers={"content-type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=3600).read())


def render_msgs(msgs: list[dict], effort: float) -> list[int]:
    """Official template; it adds no generation prefix, the model emits <|message_model|> itself."""
    s = TOK.apply_chat_template(msgs, tokenize=False, reasoning_effort=effort)
    return TOK(s, add_special_tokens=False).input_ids


def render(text: str, effort: float, system: str | None) -> list[int]:
    msgs = [{"role": "user", "content": text}]
    if system:
        msgs.insert(0, {"role": "system", "content": system})
    return render_msgs(msgs, effort)


def gen(port: int, ids: list[int], n: int, samp: Samp) -> list[int]:
    o = post(port, {"prompt": ids, "n_predict": n, "cache_prompt": True, "return_tokens": True,
                    "stop": [END_S, EOS_S], **samp})
    toks = list(o.get("tokens") or [])
    st = o.get("stop_type")
    if st == "eos" and (not toks or toks[-1] not in (END, EOS)):
        toks.append(EOS)
    elif st == "word" and (not toks or toks[-1] not in (END, EOS)):
        toks.append(END if o.get("stopping_word") == END_S else EOS)
    return toks


def prefill_ids(prefill: str | None) -> list[int]:
    return TOK(prefill, add_special_tokens=False).input_ids if prefill else []


def sample(port: int, prompt: list[int], think_cap: int, ans_cap: int, samp: Samp, prefill: str | None = None) -> dict:
    """Thinking (capped, force-closed on cap) then the answer (capped). Same token handling as gen_samples.sample;
    with a prefill the answer header + prefill tokens are forced instead of letting the model open the answer."""
    pre = prefill_ids(prefill)
    # With a prefill we must not let the model write an answer before we can inject it: probe 2 tokens first.
    out = gen(port, prompt, 2 if pre else think_cap + 3, samp)
    if pre and out[:2] == [MSG_MODEL, THINK]:
        out = out + gen(port, prompt + out, think_cap + 1, samp)
    think, cap_hit, comp = [], False, []
    if out[:2] == [MSG_MODEL, THINK]:
        body = out[2:]
        if END in body[:think_cap + 1]:
            think = body[:body.index(END)]
            comp = [MSG_MODEL, THINK] + think + [END]
        else:
            think, cap_hit = body[:think_cap], True
            comp = [MSG_MODEL, THINK] + think + [END]
        if pre or cap_hit:
            head = [MSG_MODEL, TEXT] + pre
            rest = head + gen(port, prompt + comp + head, ans_cap - len(pre), samp)
        else:
            rest = gen(port, prompt + comp, ans_cap + 2, samp)
    elif pre:
        head = [MSG_MODEL, TEXT] + pre
        rest = head + gen(port, prompt + head, ans_cap - len(pre), samp)
    else:
        rest = out
    ans = rest[2:] if rest[:2] == [MSG_MODEL, TEXT] else rest
    stop = min([ans.index(t) for t in (END, EOS) if t in ans] or [len(ans)])
    finished = stop < len(ans) and stop <= ans_cap
    ans_body = ans[:min(stop, ans_cap)]
    comp = comp + rest[:2] + ans_body + ([ans[stop]] if finished else [])
    return {"think_len": len(think), "cap_hit": cap_hit, "finished": finished, "ans_len": len(ans_body),
            "answer": TOK.decode(ans_body), "think": TOK.decode(think) if think else "", "completion": comp}


EXTRA: dict = {}


def sampling(greedy: bool, temp: float, top_p: float, seed: int) -> Samp:
    if greedy:
        return {"temperature": 0.0, "top_k": 1, "seed": 1234}
    return {"temperature": temp, "top_p": top_p, "top_k": 0, "seed": seed}


def row_seed(seed0: int, rid: str, k: int, extra: int = 0) -> int:
    return seed0 + 7919 * k + zlib.crc32(rid.encode()) % 100000 + 31 * extra


def read_opt(path: str) -> str | None:
    return open(path).read().strip() if path else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ports", required=True)
    ap.add_argument("--inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--effort", type=float, default=0.0)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--system", default="")
    ap.add_argument("--sys-tag", default="", help="label stored in sys_tag (default: --system file basename)")
    ap.add_argument("--prefill-file", default="")
    ap.add_argument("--think-cap", type=int, default=None, help="default 1536 (4096 with --greedy)")
    ap.add_argument("--ans-cap", type=int, default=None, help="default 768 (8192 with --greedy)")
    ap.add_argument("--conc", type=int, default=12, help="concurrent requests per port")
    ap.add_argument("--seed0", type=int, default=1000)
    ap.add_argument("--extra", default="", help='extra llama-server sampler params as JSON, e.g. {"dry_multiplier":0.8}')
    a = ap.parse_args()
    global EXTRA
    EXTRA = json.loads(a.extra) if a.extra else {}
    think_cap = a.think_cap or (4096 if a.greedy else 1536)
    ans_cap = a.ans_cap or (8192 if a.greedy else 768)
    n = 1 if a.greedy else a.n
    ports = [int(p) for p in a.ports.split(",")]
    system, prefill = read_opt(a.system), read_opt(a.prefill_file)
    sys_tag = a.sys_tag or (os.path.basename(a.system) if a.system else "")
    rows = [json.loads(l) for l in open(a.inp) if l.strip()]
    done = set()
    if os.path.exists(a.out):
        done = {(r["id"], r["k"]) for r in map(json.loads, open(a.out))}
    jobs = [(r, k) for r in rows for k in range(n) if (r["id"], k) not in done]
    print(f"{len(jobs)} jobs over {len(ports)} ports (think_cap {think_cap}, ans_cap {ans_cap})", flush=True)
    port_cycle = itertools.cycle(ports)
    f = open(a.out, "a")
    n_done = [0]

    def one(job: tuple[dict, int]) -> None:
        r, k = job
        with LOCK:
            port = next(port_cycle)
        rsys = r.get("system", system)
        rpre = r.get("prefill", prefill)
        samp = {**sampling(a.greedy, a.temp, a.top_p, row_seed(a.seed0, r["id"], k)), **EXTRA}
        try:
            s = sample(port, render(r["text"], a.effort, rsys), think_cap, ans_cap, samp, rpre)
        except Exception as e:  # keep going; the row is retried on resume
            print("ERR", r["id"], k, repr(e)[:200], flush=True)
            return
        s.update({"id": r["id"], "prompt": r["text"], "category": r.get("category"), "effort": a.effort,
                  "sys": int(bool(rsys)), "sys_tag": r.get("sys_tag", "row" if "system" in r else sys_tag), "k": k,
                  "prefill": rpre or "", "greedy": a.greedy})
        with LOCK:
            f.write(json.dumps(s) + "\n")
            f.flush()
            n_done[0] += 1
            if n_done[0] % 200 == 0:
                print("done", n_done[0], flush=True)

    with ThreadPoolExecutor(a.conc * len(ports)) as ex:
        list(ex.map(one, jobs))
    print("GEN_DONE", a.out, n_done[0], flush=True)


if __name__ == "__main__":
    main()
