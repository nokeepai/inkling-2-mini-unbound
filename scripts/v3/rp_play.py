"""ON eval/gen pod: play scripted multi-turn roleplay chats against llama-servers (fixed user lines).

  python3 rp_play.py --ports 8100,8101 --inp chats.jsonl --out out.jsonl --effort 0 [--greedy | --n 4 --temp 0.8] \
      [--system sys.txt] [--ub] [--think-cap 1536] [--ans-cap 2048] [--conc 8] [--max-turns 0]

Input rows: {id, system?, setup, turns: [user strings], user_char}. --ub reads uncensoredbench questions/roleplay.jsonl
(id/theme/setup/turns/turn_levels; system = ~/Code/uncensoredbench/prompts/system_roleplay.txt unless --system;
user_char parsed from "I play <Name>" in the setup).
System message = system + "\\n\\nSetup: " + setup. Each turn: render the full history with the official template, generate
(thinking capped and force-closed as gen_v3.sample), then append the model's final ANSWER only as the assistant turn
(no reasoning_content, so earlier thinking is not in later prompts; that is also what the template does by default).
Output: one row per (chat, k, turn): id, k, turn (1-based), level, theme, user_char, setup, effort, sys, sys_tag, prompt
(this turn's user line), messages (JSON list rendered for this turn), think, answer, completion (token ids for SFT),
think_len, cap_hit, finished, ans_len. Rows of a chat are written together when the chat ends; resumable per (id, k).
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gen_v3 import render_msgs, row_seed, sample, sampling  # noqa: E402

LOCK = threading.Lock()
UB_SYSTEM = os.path.expanduser(os.environ.get("UB_SYSTEM", "~/Code/uncensoredbench/prompts/system_roleplay.txt"))
USER_CHAR_RX = re.compile(r"\b(?:I|I'll|I will) play ([A-Z][\w'\-]+)")


def user_char_of(chat: dict) -> str:
    if chat.get("user_char"):
        return chat["user_char"]
    m = USER_CHAR_RX.search(chat.get("setup", ""))
    return m.group(1) if m else ""


def play_chat(port: int, chat: dict, k: int, system: str, a: argparse.Namespace, think_cap: int, ans_cap: int) -> list[dict]:
    sysmsg = (chat.get("system") or system) + "\n\nSetup: " + chat["setup"]
    msgs: list[dict] = [{"role": "system", "content": sysmsg}]
    rows = []
    turns = chat["turns"][:a.max_turns] if a.max_turns else chat["turns"]
    levels = chat.get("turn_levels") or [None] * len(chat["turns"])
    for i, line in enumerate(turns):
        msgs.append({"role": "user", "content": line})
        samp = sampling(a.greedy, a.temp, a.top_p, row_seed(a.seed0, chat["id"], k, i + 1))
        s = sample(port, render_msgs(msgs, a.effort), think_cap, ans_cap, samp)
        rows.append({"id": chat["id"], "k": k, "turn": i + 1, "level": levels[i], "theme": chat.get("theme"),
                     "user_char": user_char_of(chat), "setup": chat["setup"], "effort": a.effort, "sys": 1,
                     "sys_tag": a.sys_tag, "prompt": line, "messages": json.loads(json.dumps(msgs)), "greedy": a.greedy, **s})
        msgs.append({"role": "assistant", "content": s["answer"]})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ports", required=True)
    ap.add_argument("--inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--effort", type=float, default=0.0)
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--system", default="")
    ap.add_argument("--sys-tag", default="")
    ap.add_argument("--ub", action="store_true", help="input is uncensoredbench questions/roleplay.jsonl")
    ap.add_argument("--think-cap", type=int, default=None, help="default 1536 (4096 with --greedy)")
    ap.add_argument("--ans-cap", type=int, default=2048)
    ap.add_argument("--max-turns", type=int, default=0)
    ap.add_argument("--conc", type=int, default=8, help="concurrent chats per port")
    ap.add_argument("--seed0", type=int, default=1000)
    a = ap.parse_args()
    think_cap = a.think_cap or (4096 if a.greedy else 1536)
    n = 1 if a.greedy else a.n
    ports = [int(p) for p in a.ports.split(",")]
    sys_path = a.system or (UB_SYSTEM if a.ub else "")
    if not sys_path:
        raise SystemExit("need --system (or --ub) for the roleplay system prompt")
    system = open(sys_path).read().strip()
    a.sys_tag = a.sys_tag or os.path.basename(sys_path)
    chats = [json.loads(l) for l in open(a.inp) if l.strip()]
    done = set()
    if os.path.exists(a.out):
        done = {(r["id"], r["k"]) for r in map(json.loads, open(a.out))}
    jobs = [(c, k) for c in chats for k in range(n) if (c["id"], k) not in done]
    print(f"{len(jobs)} chats over {len(ports)} ports", flush=True)
    port_cycle = itertools.cycle(ports)
    f = open(a.out, "a")
    n_done = [0]

    def one(job: tuple[dict, int]) -> None:
        c, k = job
        with LOCK:
            port = next(port_cycle)
        try:
            rows = play_chat(port, c, k, system, a, think_cap, a.ans_cap)
        except Exception as e:  # keep going; the whole chat is replayed on resume
            print("ERR", c["id"], k, repr(e)[:200], flush=True)
            return
        with LOCK:
            for r in rows:
                f.write(json.dumps(r) + "\n")
            f.flush()
            n_done[0] += 1
            if n_done[0] % 20 == 0:
                print("chats done", n_done[0], flush=True)

    with ThreadPoolExecutor(a.conc * len(ports)) as ex:
        list(ex.map(one, jobs))
    print("RP_DONE", a.out, n_done[0], flush=True)


if __name__ == "__main__":
    main()
