"""Local mock of an OpenAI-compatible Inkling server for the self-test (no GPU, no network).

One "token" = one whitespace-delimited chunk. Canned behaviour keyed on the request text:
  creative prompt mentioning "dragon"  -> endless looping thinking (thinking-cap hit; forced answer follows)
  creative prompt mentioning "letter"  -> endless looping answer (answer runaway)
  maths (asks for \\boxed{})            -> correct gold answer for 2 of 3 prompts
  code (MBPP)                           -> correct for mbpp163 only, a stub otherwise
  agent_hard                            -> turn 1 repeats the buggy file, turn 2 sends fixed.py;
                                           t03 never sends code; t05 loops in thinking on turn 1
  roleplay                              -> scene text; rpe000 repeats its previous reply from turn 3
  sysrules                              -> greeting, headings, bullets, sign-off parsed from the rules
  pub_* (if <data-dir>/pub exists)      -> reference code / gold letter for even crc32(prompt), else the buggy
                                           program, a stub or the next letter; pub_gsm8k uses the maths branch
  release pub suites (same rule)          -> pub_mbpp reference / stub, pub_humanevalfix fixed / buggy function,
                                           pub_mmlu_2k letter; pub_gsm8k_full and pub_math500 use the maths branch
--style field: thinking in delta.reasoning_content, vLLM-like /tokenize (messages + chat_template_kwargs).
--style markers: Inkling channel tokens in content when skip_special_tokens=false (no reasoning parser),
                 llama.cpp-like /apply-template and /tokenize {"content"}.

  python3 mock_server.py --port 18080 --style field --data-dir ~/.inkling-abl/tinker_lab --agent-dir DIR
"""
from __future__ import annotations

import argparse
import itertools
import json
import re
import threading
import zlib
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterator

import template as tpl

TOKEN_RE = re.compile(r"\S+\s*|\s+")
STATS: dict[str, Any] = {"chat": 0, "completions": 0, "effort": Counter(), "missing_kwargs": 0}
LOCK = threading.Lock()
WORLD: dict[str, Any] = {}


def toks(text: str) -> list[str]:
    return TOKEN_RE.findall(text)


def endless(sentence: str) -> Iterator[str]:
    return itertools.cycle(toks(sentence))


def load_world(data_dir: Path, agent_dir: Path) -> None:
    gold = {}
    for f in ("prompts_eval.jsonl", "prompts_hardmath.jsonl"):
        for line in (data_dir / f).read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                if r.get("gold"):
                    gold[r["prompt"]] = r["gold"]
    tasks = {d.name: ((d / "buggy.py").read_text(), (d / "fixed.py").read_text())
             for d in sorted(agent_dir.glob("t[0-9][0-9]_*"))}
    WORLD.update(gold=gold, tasks=tasks, pub_code={}, pub_mmlu={})
    load_pub_world(data_dir / "pub")


def pub_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()] if path.exists() else []


def load_pub_world(pub: Path) -> None:
    """Public suites (pub.py): reference answers keyed by prompt; pub_right() picks which ones to get right."""
    for r in pub_rows(pub / "pub_humaneval.jsonl"):
        stub = f"def {r['entry_point']}(*args, **kwargs):\n    return None\n"
        WORLD["pub_code"][r["prompt"]] = (r["reference"], stub)
    for r in pub_rows(pub / "pub_quixbugs.jsonl"):
        WORLD["pub_code"][r["prompt"]] = (r["reference"], r["buggy"])
    for r in pub_rows(pub / "pub_gsm8k.jsonl"):
        WORLD["gold"][r["prompt"]] = r["gold"]   # answered by the maths branch, like hardmath
    for r in pub_rows(pub / "pub_mmlu.jsonl"):
        WORLD["pub_mmlu"][r["prompt"]] = r["gold"]
    load_release_world(pub)


def load_release_world(pub: Path) -> None:
    """Release suites: MBPP gets a stub that defines nothing (its asserts then fail), HumanEvalFix the buggy
    function; the maths suites' golds join the maths branch, whose wrong answer for a non-integer gold is 'x'."""
    for r in pub_rows(pub / "pub_mbpp.jsonl"):
        WORLD["pub_code"][r["prompt"]] = (r["reference"], "def solution(*args):\n    return None\n")
    for r in pub_rows(pub / "pub_humanevalfix.jsonl"):
        WORLD["pub_code"][r["prompt"]] = (r["reference"], r["buggy"])
    for name in ("pub_gsm8k_full", "pub_math500"):
        for r in pub_rows(pub / f"{name}.jsonl"):
            WORLD["gold"][r["prompt"]] = r["gold"]
    for r in pub_rows(pub / "pub_mmlu_2k.jsonl"):
        WORLD["pub_mmlu"][r["prompt"]] = r["gold"]


def pub_right(prompt: str) -> bool:
    """Code and MMLU items: right for even crc32 of the prompt (selftest recomputes the expected count)."""
    return zlib.crc32(prompt.encode()) % 2 == 0


def plan_pub(u: str) -> Iterator[str] | None:
    if u in WORLD["pub_code"]:
        right, wrong = WORLD["pub_code"][u]
        return iter(toks(f"Here is the program.\n\n```python\n{right if pub_right(u) else wrong}```\n"))
    if u in WORLD["pub_mmlu"]:
        gold = WORLD["pub_mmlu"][u]
        say = gold if pub_right(u) else "ABCD"[("ABCD".index(gold) + 1) % 4]
        return iter(toks(f"Option A and option B look close; checking each one.\n\nAnswer: **{say}**"))
    return None


def prose(n: int, salt: int) -> str:
    names = ["Elara", "Juno", "Kofi", "Mara", "Tomas"]
    out = []
    i = 0
    while sum(len(s.split()) for s in out) < n:
        out.append(f"Then {names[(i + salt) % 5]} walked past lantern {i}-{salt}, and counted {i * 7 + salt} stars.")
        i += 1
    return " ".join(out)


def agent_task(text: str) -> tuple[str, str, str] | None:
    for name, (buggy, fixed) in WORLD["tasks"].items():
        if buggy.strip()[:400] in text:
            return name, buggy, fixed
    return None


def code_reply(code: str) -> str:
    return f"Here is the corrected module.\n\n```python\n{code}```\n"


def plan_agent(name: str, buggy: str, fixed: str, turn: int) -> tuple[Iterator[str], Iterator[str]]:
    think = iter(toks("I should trace each function against the tests and fix the bugs one by one."))
    if name.startswith("t03"):
        return think, iter(toks("I believe the problem is in the date arithmetic; let me think more about it."))
    if name.startswith("t05") and turn == 1:
        return endless("Let me re-read the merge loop once more and check the boundaries again. "), iter([])
    return think, iter(toks(code_reply(buggy if turn == 1 else fixed)))


def plan_sysrules(system: str) -> tuple[Iterator[str], Iterator[str]]:
    p = re.search(r"support assistant for ([A-Za-z]+)", system)
    sign = re.search(r'End every reply with this exact final line: "(.+)"', system)
    product = p.group(1) if p else "Product"
    body = (f"Hello from {product}!\n## Answer\nOpen Settings, choose the panel and follow the steps. "
            "Your reference is REF-1234.\n## Next steps\n- Restart the app\n- Update it\n- Check the panel\n"
            "Would you like more detail on any step?\n" + (sign.group(1) if sign else ""))
    return iter(toks("The rules list a greeting, headings and a sign-off; keep it short.")), iter(toks(body))


MOCK_MARK = re.compile(r"\[\[mock:(\w+)(?:=(\w+))?\]\]")


def marked_reply(kind: str, value: str) -> str:
    """Synthetic ubench fixtures carry [[mock:...]] tags so the self-test knows each item's expected mark."""
    if kind == "refuse":
        return "I can't help with that request."
    if kind == "hedge":
        return "Please be careful and consult a professional first. " + prose(60, 1)
    if kind == "correct":
        return "Actually, that claim is false; the record shows otherwise. " + prose(40, 2)
    return f"Working through it step by step, the result follows.\nAnswer: {value}"


def plan_chat(messages: list[dict[str, Any]]) -> tuple[Iterator[str], Iterator[str]]:
    users = [m["content"] for m in messages if m["role"] == "user"]
    replies = [m["content"] for m in messages if m["role"] == "assistant"]
    system = " ".join(m["content"] for m in messages if m["role"] == "system")
    u0, u = users[0], users[-1]
    short_think = iter(toks("Let me plan the reply briefly and then write it out clearly for the user."))
    if (t := agent_task(u0)) is not None:
        return plan_agent(*t, turn=len(users))
    if (pub_answer := plan_pub(u)) is not None:
        return short_think, pub_answer
    if "RULES" in system:
        return plan_sysrules(system)
    if "Your function must pass this test" in u:
        ok = "area_polygon" in u
        code = ("import math\ndef area_polygon(s, l):\n    return s * l ** 2 / (4 * math.tan(math.pi / s))\n"
                if ok else "def solution(*args):\n    return None\n")
        return short_think, iter(toks(f"```python\n{code}```\n"))
    if "\\boxed{}" in u:
        gold = WORLD["gold"].get(u, "0")
        right = zlib.crc32(u.encode()) % 3 != 0
        val = gold if right else str(int(float(gold)) + 1) if re.fullmatch(r"-?\d+", gold) else "x"
        return short_think, iter(toks(f"Working through it carefully, the result is \\boxed{{{val}}}."))
    if (mk := MOCK_MARK.search(u)) is not None:
        return short_think, iter(toks(marked_reply(mk.group(1), mk.group(2) or "")))
    if "dragon" in u.lower():
        return endless("Wait, let me reconsider the opening line of the story once more before writing. "), iter([])
    if any(k in u0.lower() for k in ("roleplay", "please play")):
        if "stargazing" in u0 and len(users) >= 3 and replies:
            return short_think, iter(toks(replies[-1]))
        return short_think, iter(toks(prose(60, len(users)) + " What would you like to do next?"))
    if "letter" in u.lower():
        return short_think, endless("The letter went on and on, repeating the same line again and again. ")
    m = re.search(r"(\d{2,4})[- ]word|(\d{2,4}) words", u)
    n = int(m.group(1) or m.group(2)) if m else 150
    return short_think, iter(toks(prose(n, zlib.crc32(u.encode()) % 5) + " The end."))


def plan_completion(prompt: str) -> Iterator[str]:
    """Forced answer after a cap hit: prompt = rendered chat + capped thinking + answer header."""
    users = re.findall(r"<\|message_user\|><\|content_text\|>(.*?)<\|end_message\|>", prompt, re.S)
    u0 = users[0] if users else ""
    if (t := agent_task(u0)) is not None:
        return iter(toks(code_reply(t[2])))
    if users and "letter" in users[-1].lower():
        return endless("The letter went on and on, repeating the same line again and again. ")
    return iter(toks("After a long think, here is a short and complete answer to the request."))


class Handler(BaseHTTPRequestHandler):
    style = "field"
    protocol_version = "HTTP/1.0"

    def log_message(self, *args: Any) -> None:  # quiet
        return

    def _json(self, obj: Any, code: int = 200) -> None:
        raw = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self) -> None:
        if self.path.endswith("/v1/models"):
            self._json({"data": [{"id": "mock-inkling"}]})
        elif self.path.endswith("/mock/stats"):
            with LOCK:
                self._json({**STATS, "effort": dict(STATS["effort"])})
        else:
            self._json({"status": "ok"})

    def do_POST(self) -> None:
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))) or b"{}")
        if self.path.endswith("/v1/chat/completions"):
            self._chat(body)
        elif self.path.endswith("/v1/completions"):
            self._completion(body)
        elif self.path.endswith("/tokenize"):
            self._tokenize(body)
        elif self.path.endswith("/apply-template") and self.style == "markers":
            eff = (body.get("chat_template_kwargs") or {}).get("reasoning_effort")
            self._json({"prompt": tpl.render(body["messages"], eff)})
        else:
            self._json({"error": "not found"}, 404)

    def _tokenize(self, body: dict[str, Any]) -> None:
        if self.style == "field" and "messages" in body:
            eff = (body.get("chat_template_kwargs") or {}).get("reasoning_effort")
            t = toks(tpl.render(body["messages"], eff))
        elif self.style == "field":
            t = toks(body.get("prompt", ""))
        elif "content" in body:
            t = toks(body["content"])
        else:
            self._json({"error": "bad request"}, 400)
            return
        ids = [zlib.crc32(x.encode()) % 200000 for x in t]
        self._json({"tokens": ids, "count": len(ids)} if self.style == "field" else {"tokens": ids})

    def _chat(self, body: dict[str, Any]) -> None:
        kw = body.get("chat_template_kwargs") or {}
        with LOCK:
            STATS["chat"] += 1
            if "reasoning_effort" in kw:
                STATS["effort"][str(kw["reasoning_effort"])] += 1
            else:
                STATS["missing_kwargs"] += 1
        think, answer = plan_chat(body["messages"])
        markers = self.style == "markers" and body.get("skip_special_tokens") is False
        events: list[tuple[str, Iterator[str]]] = []
        if self.style == "field":
            events = [("reasoning_content", think), ("content", answer)]
        elif markers:
            events = [("content", iter(["<|message_model|>", "<|content_", "thinking|>"])), ("content", think),
                      ("content", iter(["<|end_message|>", "<|message_model|>", "<|content_text|>"])),
                      ("content", answer), ("content", iter(["<|end_message|>"]))]
        else:  # no reasoning parser and specials skipped: thinking leaks into content unmarked
            events = [("content", think), ("content", answer)]
        self._sse(body, events, chat=True)

    def _completion(self, body: dict[str, Any]) -> None:
        with LOCK:
            STATS["completions"] += 1
        self._sse(body, [("text", plan_completion(body["prompt"]))], chat=False)

    def _sse(self, body: dict[str, Any], events: list[tuple[str, Iterator[str]]], chat: bool) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        max_tokens = int(body.get("max_tokens") or 1 << 30)
        cont = bool((body.get("stream_options") or {}).get("continuous_usage_stats"))
        n, finish = 0, "stop"
        try:
            for field, it in events:
                for tok in it:
                    if n >= max_tokens:
                        finish = "length"
                        break
                    n += 1
                    choice = ({"index": 0, "delta": {field: tok}} if chat else {"index": 0, "text": tok})
                    ev: dict[str, Any] = {"choices": [choice]}
                    if cont:
                        ev["usage"] = {"completion_tokens": n}
                    self._send(ev)
                if finish == "length":
                    break
            last = {"index": 0, "delta": {}, "finish_reason": finish} if chat else \
                {"index": 0, "text": "", "finish_reason": finish}
            self._send({"choices": [last]})
            self._send({"choices": [], "usage": {"completion_tokens": n}})
            self.wfile.write(b"data: [DONE]\n\n")
        except (BrokenPipeError, ConnectionResetError):
            return  # client cut the stream at a cap

    def _send(self, ev: dict[str, Any]) -> None:
        self.wfile.write(b"data: " + json.dumps(ev).encode() + b"\n\n")
        self.wfile.flush()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=18080)
    ap.add_argument("--style", choices=["field", "markers"], default="field")
    ap.add_argument("--data-dir", default=str(Path("~/.inkling-abl/tinker_lab").expanduser()))
    ap.add_argument("--agent-dir", default=str(Path(__file__).resolve().parent.parent / "tinker" / "agent_hard_tasks"))
    a = ap.parse_args()
    load_world(Path(a.data_dir).expanduser(), Path(a.agent_dir).expanduser())
    Handler.style = a.style
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    srv.daemon_threads = True
    print(f"mock listening on 127.0.0.1:{a.port} style={a.style}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
