"""OpenAI-compatible client with the Tinker lab's sampling semantics, enforced client-side.

Lab semantics (scripts/tinker/rollout.py), reproduced over /v1/chat/completions:
  - thinking is capped at think_cap tokens; if it has not closed by then, that is a thinking-cap hit;
  - on a cap hit (eval default, force_answer_after_cap) the thinking is closed and an answer is still
    sampled; here that continuation goes to /v1/completions with a raw prompt (the official template,
    rendered locally) + the capped thinking + <|end_message|><|message_model|><|content_text|>;
  - the answer is capped at answer_cap tokens; reaching it without the model ending = answer runaway.
The stream is read token by token and cut (connection closed, so the server aborts the request) the
moment a cap is reached. Token counts come from per-chunk usage when the server sends it
(vLLM continuous_usage_stats), otherwise one token per text chunk (llama.cpp streams one token per chunk).

Thinking/answer separation, in order: the server's reasoning field (`reasoning_content`, or `reasoning`
on newer vLLM); with --raw-markers, Inkling's channel tokens parsed from content
(<|content_thinking|> ... <|end_message|> ... <|content_text|>); content that starts with <think>.
"""
from __future__ import annotations

import http.client
import json
import random
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Iterator
from urllib.parse import urlsplit

import template as tpl

SPECIAL_RE = re.compile(r"<\|[a-z_0-9]+\|>")
PARTIAL_RE = re.compile(r"<(\|[a-z_0-9]*\|?)?$|</?t?h?i?n?k?$")


@dataclass(frozen=True)
class ClientCfg:
    base_url: str
    model: str
    effort: float
    temperature: float
    top_p: float = 1.0
    think_cap: int = 4096
    answer_cap: int = 8192
    effort_transport: str = "kwargs"   # kwargs: chat_template_kwargs per request; none: set on the server
    raw_markers: bool = False          # request special tokens and split channels client-side
    force_mode: str = "completions"    # completions | none: how the answer is sampled after a cap hit
    concurrency: int = 32
    api_key: str = ""
    timeout: float = 900.0
    extra_body: dict[str, Any] = field(default_factory=dict)


@dataclass
class Gen:
    thinking: str = ""
    answer: str = ""
    think_tokens: int = 0
    answer_tokens: int = 0
    think_cap_hit: bool = False
    answer_runaway: bool = False
    forced: bool = False          # answer came from the post-cap continuation
    finish: str | None = None
    split: str = "none"           # field | markers | think_tags | none
    count_source: str = "chunks"  # usage | chunks
    seconds: float = 0.0
    error: str | None = None


class ChannelParser:
    """Splits streamed content into (channel, text) using Inkling channel tokens or <think> tags.
    Markers split across chunks are held back until they can be recognised."""

    def __init__(self, raw_markers: bool, start: str = "answer") -> None:
        self.raw = raw_markers
        self.channel = start
        self.pending = ""
        self.think_closed = False
        self.used = False
        self._first = True

    def _apply(self, marker: str) -> None:
        self.used = True
        if marker in ("<|content_thinking|>", "<think>"):
            self.channel = "think"
        elif marker == "<|content_text|>":
            self.channel = "answer"
        elif marker == "</think>" or (marker == "<|end_message|>" and self.channel == "think"):
            self.think_closed, self.channel = True, "answer"
        elif marker == "<|end_message|>" and self.channel == "answer":
            self.channel = "done"

    def _marker_at(self, buf: str, j: int) -> str | None:
        if buf.startswith("<think>", j) or buf.startswith("</think>", j):
            return "<think>" if buf.startswith("<think>", j) else "</think>"
        if self.raw:
            m = SPECIAL_RE.match(buf, j)
            return m.group(0) if m else None
        return None

    def feed(self, text: str) -> list[tuple[str, str]]:
        buf, self.pending = self.pending + text, ""
        if self._first and buf.strip():
            self._first = False
            # Without raw markers only a leading <think> block is treated as thinking.
            if not self.raw and not buf.lstrip().startswith("<think>"):
                self.feed = self._plain  # type: ignore[method-assign]
                return self._plain(buf)
        out: list[tuple[str, str]] = []
        i = 0
        while i < len(buf):
            j = buf.find("<", i)
            if j < 0:
                out.append((self.channel, buf[i:]))
                break
            out.append((self.channel, buf[i:j]))
            mk = self._marker_at(buf, j)
            if mk:
                self._apply(mk)
                i = j + len(mk)
            elif PARTIAL_RE.fullmatch(buf[j:]) and len(buf) - j < 48:
                self.pending = buf[j:]
                break
            else:
                out.append((self.channel, "<"))
                i = j + 1
        return [(c, t) for c, t in out if t and c != "done"]

    def _plain(self, text: str) -> list[tuple[str, str]]:
        return [(self.channel, text)] if text else []

    def flush(self) -> list[tuple[str, str]]:
        rest, self.pending = self.pending, ""
        return [(self.channel, rest)] if rest and self.channel != "done" else []


class Client:
    def __init__(self, cfg: ClientCfg) -> None:
        self.cfg = cfg
        u = urlsplit(cfg.base_url.rstrip("/"))
        self._https = u.scheme == "https"
        self._host, self._port = u.hostname or "127.0.0.1", u.port
        root = u.path.rstrip("/")
        self._root = root[:-3] if root.endswith("/v1") else root
        self.sem = threading.BoundedSemaphore(cfg.concurrency)
        self.tok_mode: str | None = None  # vllm | llama | None, set by probe()

    # ---------- HTTP ----------

    def _conn(self) -> http.client.HTTPConnection:
        cls = http.client.HTTPSConnection if self._https else http.client.HTTPConnection
        return cls(self._host, self._port, timeout=self.cfg.timeout)

    def _headers(self) -> dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self.cfg.api_key:
            h["Authorization"] = "Bearer " + self.cfg.api_key
        return h

    def post_json(self, path: str, body: dict[str, Any], timeout: float | None = None) -> Any:
        conn = self._conn()
        if timeout:
            conn.timeout = timeout
        try:
            conn.request("POST", self._root + path, json.dumps(body), self._headers())
            resp = conn.getresponse()
            raw = resp.read()
            if resp.status >= 400:
                raise RuntimeError(f"HTTP {resp.status} {path}: {raw[:300]!r}")
            return json.loads(raw)
        finally:
            conn.close()

    def get_json(self, path: str) -> Any:
        conn = self._conn()
        try:
            conn.request("GET", self._root + path, headers=self._headers())
            resp = conn.getresponse()
            raw = resp.read()
            if resp.status >= 400:
                raise RuntimeError(f"HTTP {resp.status} {path}")
            return json.loads(raw)
        finally:
            conn.close()

    def _stream(self, path: str, body: dict[str, Any]) -> tuple[http.client.HTTPConnection, Iterator[dict[str, Any]]]:
        conn = self._conn()
        conn.request("POST", self._root + path, json.dumps(body), self._headers())
        resp = conn.getresponse()
        if resp.status >= 400:
            raw = resp.read()
            conn.close()
            raise RuntimeError(f"HTTP {resp.status} {path}: {raw[:300]!r}")

        def events() -> Iterator[dict[str, Any]]:
            while True:
                line = resp.readline()
                if not line:
                    return
                line = line.strip()
                if not line.startswith(b"data:"):
                    continue
                data = line[5:].strip()
                if data == b"[DONE]":
                    return
                yield json.loads(data)
        return conn, events()

    # ---------- request bodies ----------

    def _sampling(self, max_tokens: int, seed: int | None) -> dict[str, Any]:
        body: dict[str, Any] = {"model": self.cfg.model, "stream": True, "max_tokens": max_tokens,
                                "temperature": self.cfg.temperature, "top_p": self.cfg.top_p,
                                "stream_options": {"include_usage": True, "continuous_usage_stats": True}}
        if seed is not None:
            body["seed"] = seed
        body.update(self.cfg.extra_body)
        return body

    def chat_body(self, messages: list[dict[str, Any]], seed: int | None) -> dict[str, Any]:
        # Backstop only: the caps are enforced per channel while streaming.
        body = self._sampling(self.cfg.think_cap + self.cfg.answer_cap + 16, seed)
        body["messages"] = messages
        if self.cfg.effort_transport == "kwargs":
            body["chat_template_kwargs"] = {"reasoning_effort": self.cfg.effort}
        if self.cfg.raw_markers:
            body["skip_special_tokens"] = False
        return body

    # ---------- generation ----------

    def generate(self, messages: list[dict[str, Any]], seed: int | None = None,
                 force: bool = True) -> Gen:
        """One sample with the lab's caps. force=False: a cap hit ends with no answer (agent no-force)."""
        t0 = time.time()
        g = Gen()
        for attempt in range(4):
            try:
                g = Gen()
                with self.sem:
                    self._chat_stage(messages, seed, g)
                if g.think_cap_hit and force and self.cfg.force_mode == "completions":
                    with self.sem:
                        self._forced_answer(messages, seed, g)
                break
            except (OSError, http.client.HTTPException, RuntimeError, ValueError) as e:
                g.error = f"{type(e).__name__}: {str(e)[:300]}"
                if "HTTP 4" in g.error and "HTTP 429" not in g.error:
                    break
                time.sleep(min(30.0, 2.0 * (attempt + 1) + random.random()))
        g.seconds = round(time.time() - t0, 2)
        return g

    def _chat_stage(self, messages: list[dict[str, Any]], seed: int | None, g: Gen) -> None:
        parser = ChannelParser(self.cfg.raw_markers)
        conn, events = self._stream("/v1/chat/completions", self.chat_body(messages, seed))
        think: list[str] = []
        answer: list[str] = []
        usage_prev, cut, saw_field = 0, False, False
        try:
            for ev in events:
                pieces, usage_now = self._pieces_chat(ev, parser)
                if pieces and pieces[0][0] == "field":
                    saw_field = True
                    pieces = pieces[1:]
                n_new, usage_prev = self._count(pieces, usage_now, usage_prev, g)
                if pieces:
                    self._absorb(pieces, n_new, think, answer, g)
                fr = (ev.get("choices") or [{}])[0].get("finish_reason")
                if fr:
                    g.finish = fr
                answer_started = bool(answer) or parser.think_closed
                if not answer_started and g.think_tokens >= self.cfg.think_cap:
                    g.think_cap_hit, cut = True, True
                    break
                if g.answer_tokens >= self.cfg.answer_cap:
                    g.answer_runaway, cut = True, True
                    break
            if not cut:
                rest = parser.flush()
                if rest:
                    self._absorb(rest, 0, think, answer, g)
        finally:
            conn.close()
        if not cut and g.finish == "length":
            if answer or parser.think_closed:
                g.answer_runaway = True
            else:
                g.think_cap_hit = True
        g.thinking, g.answer = "".join(think), "".join(answer)
        g.split = "field" if saw_field else ("markers" if parser.used and self.cfg.raw_markers
                                             else "think_tags" if parser.used else "none")
        if g.think_cap_hit:
            g.think_tokens = max(g.think_tokens, self.cfg.think_cap)
            g.answer, g.answer_tokens = "", 0

    @staticmethod
    def _pieces_chat(ev: dict[str, Any], parser: ChannelParser) -> tuple[list[tuple[str, str]], int | None]:
        usage = ev.get("usage") or {}
        usage_now = usage.get("completion_tokens")
        choices = ev.get("choices") or []
        if not choices:
            return [], usage_now
        delta = choices[0].get("delta") or {}
        r = delta.get("reasoning_content") or delta.get("reasoning") or ""
        c = delta.get("content") or ""
        pieces: list[tuple[str, str]] = []
        if r:
            pieces += [("field", ""), ("think", r)]
        if c:
            got = parser.feed(c)
            if r:
                pieces += got
            else:
                pieces = got
        return pieces, usage_now

    def _count(self, pieces: list[tuple[str, str]], usage_now: int | None, usage_prev: int,
               g: Gen) -> tuple[int, int]:
        if usage_now is not None and usage_now >= usage_prev:
            g.count_source = "usage"
            n = usage_now - usage_prev
            return (n if pieces else 0), usage_now
        return (1 if pieces else 0), usage_prev

    @staticmethod
    def _absorb(pieces: list[tuple[str, str]], n_new: int, think: list[str], answer: list[str],
                g: Gen) -> None:
        for ch, text in pieces:
            (think if ch == "think" else answer).append(text)
        # Attribute this chunk's tokens to the channel holding most of its text.
        t = sum(len(x) for c, x in pieces if c == "think")
        a = sum(len(x) for c, x in pieces if c != "think")
        if t >= a and t > 0:
            g.think_tokens += n_new
        else:
            g.answer_tokens += n_new

    def _forced_answer(self, messages: list[dict[str, Any]], seed: int | None, g: Gen) -> None:
        prompt = tpl.forced_answer_prompt(messages, self.cfg.effort, g.thinking)
        body = self._sampling(self.cfg.answer_cap + 8, seed)
        body["prompt"] = prompt
        if self.cfg.raw_markers:
            body["skip_special_tokens"] = False
        parser = ChannelParser(True, start="answer")
        conn, events = self._stream("/v1/completions", body)
        answer: list[str] = []
        usage_prev, n_ans, cut, finish = 0, 0, False, None
        try:
            for ev in events:
                usage_now = (ev.get("usage") or {}).get("completion_tokens")
                choices = ev.get("choices") or []
                text = choices[0].get("text") or "" if choices else ""
                pieces = [p for p in parser.feed(text) if p[0] == "answer"] if text else []
                if usage_now is not None and usage_now >= usage_prev:
                    n_ans += (usage_now - usage_prev) if pieces else 0
                    usage_prev = usage_now
                elif pieces:
                    n_ans += 1
                answer += [t for _, t in pieces]
                if choices and choices[0].get("finish_reason"):
                    finish = choices[0]["finish_reason"]
                if n_ans >= self.cfg.answer_cap:
                    cut = True
                    break
        finally:
            conn.close()
        g.forced, g.answer, g.answer_tokens = True, "".join(answer), n_ans
        g.answer_runaway = cut or finish == "length"
        g.finish = finish or ("length" if cut else g.finish)

    # ---------- token counting and probes ----------

    def count_prompt_tokens(self, messages: list[dict[str, Any]]) -> tuple[int, str]:
        kw = {"reasoning_effort": self.cfg.effort}
        if self.tok_mode == "vllm":
            try:
                r = self.post_json("/tokenize", {"model": self.cfg.model, "messages": messages,
                                                 "chat_template_kwargs": kw, "add_generation_prompt": True})
                return int(r.get("count", len(r.get("tokens", [])))), "server"
            except (OSError, RuntimeError, ValueError, http.client.HTTPException):
                pass
        if self.tok_mode == "llama":
            try:
                p = self.post_json("/apply-template", {"messages": messages, "chat_template_kwargs": kw})["prompt"]
                return len(self.post_json("/tokenize", {"content": p})["tokens"]), "server"
            except (OSError, RuntimeError, ValueError, KeyError, http.client.HTTPException):
                pass
        return int(len(tpl.render(messages, self.cfg.effort)) / 3.5), "estimate"

    def probe(self) -> dict[str, Any]:
        """Preflight: model list, does reasoning_effort reach the template, does thinking come back split."""
        info: dict[str, Any] = {}
        try:
            info["models"] = [m.get("id") for m in self.get_json("/v1/models").get("data", [])]
        except (OSError, RuntimeError, ValueError, http.client.HTTPException) as e:
            info["models_error"] = str(e)[:200]
        info["effort_check"] = self._effort_check()
        g = self.generate([{"role": "user", "content": "What is 17 * 23? Answer briefly."}], seed=1)
        info["split_check"] = {"split": g.split, "think_tokens": g.think_tokens, "answer_tokens": g.answer_tokens,
                               "answer_head": g.answer[:120], "count_source": g.count_source, "error": g.error}
        if not g.thinking and self.cfg.effort > 0:
            info["split_warning"] = ("no thinking came back separately; start vLLM with a reasoning parser or "
                                     "llama-server with --jinja --reasoning-format auto, or pass --raw-markers")
        return info

    def _effort_check(self) -> dict[str, Any]:
        msgs = [{"role": "user", "content": "hi"}]
        try:
            toks = [self.post_json("/tokenize", {"model": self.cfg.model, "messages": msgs,
                                                 "chat_template_kwargs": {"reasoning_effort": e},
                                                 "add_generation_prompt": True})["tokens"] for e in (0, 0.9)]
            self.tok_mode = "vllm"
            return {"method": "vllm /tokenize", "differs": toks[0] != toks[1]}
        except (OSError, RuntimeError, ValueError, KeyError, http.client.HTTPException):
            pass
        try:
            ps = [self.post_json("/apply-template", {"messages": msgs,
                                                     "chat_template_kwargs": {"reasoning_effort": e}})["prompt"]
                  for e in (0, 0.9)]
            self.tok_mode = "llama"
            return {"method": "llama.cpp /apply-template", "differs": ps[0] != ps[1],
                    "level_0.9_rendered": "Thinking effort level: 0.9" in ps[1]}
        except (OSError, RuntimeError, ValueError, KeyError, http.client.HTTPException):
            return {"method": "unavailable", "differs": None}
