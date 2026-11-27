"""OpenAI-style bridge from scripts/evalsuite to one Tinker sampling client (stock or stock + live LoRA).

evalsuite/client.py speaks streaming /v1/chat/completions (+ /v1/completions for the forced answer after a
thinking-cap hit, + /tokenize for its effort probe). Tinker only offers token-level sampling, so this server
renders the chat with the lab's TMLv0 renderer (rollout.build_chat_prompt) and runs the lab's two-stage
sampling (rollout.py): thinking until <|end_message|> or the thinking cap, then the answer until
<|content_model_end_sampling|> or the answer cap. It answers with what client.py reads:
  delta.reasoning_content (thinking), delta.content (answer), cumulative usage.completion_tokens per chunk
  (continuous_usage_stats), finish_reason "stop" | "length".
Thinking-cap hit, POD semantics (stack_test.sh runs run_suite.py with the default --force-mode completions):
the chat stream returns the capped thinking with usage == think_cap, so client.py cuts and marks the cap hit,
then asks /v1/completions with the thinking closed (<|end_message|><|message_model|><|content_text|>) for a
forced answer capped at answer_cap. Here that continuation reuses the exact sampled thinking tokens (cached
per chat request) instead of re-tokenising the text. With --force-mode none (agent_hard_noforce) the reply
ends at the cap with no answer, as on the pods.
Counts are text tokens per channel (specials excluded), which is what client.py counts on the pods.

Usage (Python with the SDK; the key is read by lab_common into the SDK and never printed):
  PY=~/.inkling-abl/tinker_lab/.venv/bin/python
  # stock (free variant, exactly as the lab: lab_common.FREE_MODEL)
  $PY scripts/tinker/oai_proxy.py --base --mode free --port 8411 --model-name stock
  # stock + adapter, live (un-merged); the checkpoint's base model is asserted to be the free model
  $PY scripts/tinker/oai_proxy.py --sampler tinker://<run>/sampler_weights/<name> --port 8412 --model-name v3
  # the suite, unchanged, pod decoding settings (stack_test.sh: effort 0.9, temp 0.6, seed 1234; caps 4096/8192)
  cd scripts/evalsuite && python3 run_suite.py --base-url http://127.0.0.1:8411/v1 --model stock \
      --tag tinker_stock --suites pub --effort 0.9 --temperature 0.6 --seed 1234 --concurrency 16
Optional thinking off (Tinker only; the pods have no equivalent): add --extra-body '{"thinking": false}'
(effort 0 + prefilled <|message_model|><|content_text|>, as rollout.build_prompt does for thinking=False).
Throughput: lab evals on the free tier logged roughly 500-1,600 generated tokens/s at 32 in flight;
each request is 2 SDK calls (3 after a cap hit), so a 300-item suite at ~3k tokens/item takes ~10-30 min.
The free tier may queue or rate-limit; transient errors are retried here and again by client.py.

  $PY scripts/tinker/oai_proxy.py --selftest      # offline: fake sampler, no network, no key
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
import sys
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable

HERE = Path(__file__).resolve().parent
EVALSUITE = HERE.parent / "evalsuite"
sys.path.insert(0, str(HERE))
sys.path.insert(1, str(EVALSUITE))

import lab_common as lc  # noqa: E402
import rollout as ro  # noqa: E402
import template as tpl  # noqa: E402  (evalsuite/template.py: same forced-answer string the client builds)

# rollout.SPECIAL_MIN is 199990, but ids 199990-199997 are ordinary words (" VIP", " GON", ...); the
# tokenizer's specials start at <|unused|> 199998. Counting/decoding here uses the true boundary.
SPECIAL_FIRST = 199998
FORCED_TAIL = [ro.END_MESSAGE, ro.MESSAGE_MODEL, ro.CONTENT_TEXT]
SPECIAL_RE = re.compile(r"(<\|[a-z_0-9]+\|>)")
PING_SECONDS = 10.0
LOG_EVERY = 50
FORCED_CACHE_MAX = 4096
CHAT_ROLES = frozenset({"system", "user", "assistant"})


@dataclass(frozen=True)
class ProxyCfg:
    model_name: str = "inkling-tinker"
    think_cap: int = 4096
    answer_cap: int = 8192
    effort: float = 0.9          # used only when a request carries no reasoning_effort
    concurrency: int = 16
    retries: int = 5
    pass_seed: bool = True


@dataclass
class Proxy:
    cfg: ProxyCfg
    sampler: Any
    renderer: Any
    tok: Any
    sem: asyncio.Semaphore
    forced: "OrderedDict[str, list[int]]" = field(default_factory=OrderedDict)
    stats: dict[str, float] = field(default_factory=lambda: dict.fromkeys(
        ("n", "chat", "completions", "cap_hit", "runaway", "errors", "retries", "think_tok", "answer_tok",
         "seconds", "forced_cache_hit", "forced_reparsed"), 0))


@dataclass
class Reply:
    think: list[int] = field(default_factory=list)    # thinking text tokens (specials stripped)
    answer: list[int] = field(default_factory=list)   # answer text tokens
    cap_hit: bool = False
    runaway: bool = False
    forced_prefix: list[int] | None = None            # prompt + capped thinking + forced tail


class BadRequest(ValueError):
    """Client error: answered with HTTP 400 so client.py records it without retrying."""


# ---------------------------------------------------------------- tokens


def is_special(t: int) -> bool:
    return t >= SPECIAL_FIRST


def text_tokens(tokens: list[int], channel: int, start: int | None) -> list[int]:
    """Text tokens of one channel (CONTENT_THINKING or CONTENT_TEXT), tracking the content_* marker."""
    out, cur = [], start
    for t in tokens:
        if t in (ro.CONTENT_THINKING, ro.CONTENT_TEXT):
            cur = t
        elif not is_special(t) and cur == channel:
            out.append(t)
    return out


def cut_at_cap(tokens: list[int], channel: int, start: int | None, cap: int) -> tuple[list[int], bool]:
    """Keep tokens up to the cap-th text token of `channel`. True when a further text token was cut,
    i.e. the pods' client would have stopped the stream at exactly `cap` counted tokens."""
    n, cur = 0, start
    for i, t in enumerate(tokens):
        if t in (ro.CONTENT_THINKING, ro.CONTENT_TEXT):
            cur = t
        elif not is_special(t) and cur == channel:
            if n == cap:
                return tokens[:i], True
            n += 1
    return tokens, False


def encode_with_specials(tok: Any, text: str) -> list[int]:
    """Raw prompt text -> ids, mapping <|name|> to the special id (the adapter's encode() treats them as text)."""
    tml = tok.tml_tokenizer
    out: list[int] = []
    for part in SPECIAL_RE.split(text):
        if not part:
            continue
        if SPECIAL_RE.fullmatch(part):
            out.append(tml.encode_special(part[2:-2]))
        else:
            out += list(tml.encode_ordinary(part))
    return out


def decode(tok: Any, tokens: list[int]) -> str:
    return tok.decode(tokens) if tokens else ""


# ---------------------------------------------------------------- request parsing


def message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):  # OpenAI content parts; only text exists in these suites
        return "".join(p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text")
    raise BadRequest("message content must be a string or a list of text parts")


def parse_chat(messages: Any) -> ro.Chat:
    """role/content only: a prior assistant turn carries its answer and never any thinking (as rollout.row_chat)."""
    if not isinstance(messages, list) or not messages:
        raise BadRequest("messages must be a non-empty list")
    chat = []
    for m in messages:
        if not isinstance(m, dict) or m.get("role") not in CHAT_ROLES:
            raise BadRequest("only system/user/assistant messages are supported")
        chat.append({"role": m["role"], "content": message_text(m.get("content"))})
    if chat[-1]["role"] != "user":
        raise BadRequest("messages must end with a user turn")
    return chat


def parse_effort(body: dict[str, Any], default: float) -> float:
    raw = (body.get("chat_template_kwargs") or {}).get("reasoning_effort", default)
    try:
        eff = tpl.EFFORT_MAP[raw.strip()] if isinstance(raw, str) else float(raw)
    except (KeyError, TypeError, ValueError) as e:
        raise BadRequest(f"bad reasoning_effort {raw!r}") from e
    if not 0.0 <= eff < 1.0:
        raise BadRequest("reasoning_effort must be in [0, 1)")
    return eff


def thinking_on(body: dict[str, Any]) -> bool:
    kw = body.get("chat_template_kwargs") or {}
    return not (body.get("thinking") is False or kw.get("enable_thinking") is False)


def caps_for(px: Proxy, body: dict[str, Any]) -> tuple[int, int]:
    """Caps come from the proxy flags; a body may override (--extra-body '{"think_cap":..,"answer_cap":..}')."""
    return int(body.get("think_cap", px.cfg.think_cap)), int(body.get("answer_cap", px.cfg.answer_cap))


def sample_cfg(px: Proxy, body: dict[str, Any]) -> ro.SampleCfg:
    think_cap, answer_cap = caps_for(px, body)
    on = thinking_on(body)
    return ro.SampleCfg(think_cap=think_cap, answer_cap=answer_cap,
                        temperature=float(body.get("temperature", 0.6)),
                        effort=parse_effort(body, px.cfg.effort) if on else 0.0,
                        thinking=on, force_answer_after_cap=True)


def check_context(prompt: list[int], cfg: ro.SampleCfg) -> None:
    need = len(prompt) + cfg.think_cap + cfg.answer_cap + 8
    if need > lc.CONTEXT_LIMIT:
        raise BadRequest(f"prompt+caps={need} exceeds the {lc.CONTEXT_LIMIT} context")


def warn_backstop(body: dict[str, Any], cfg: ro.SampleCfg) -> None:
    # client.py sends max_tokens = think_cap + answer_cap + 16; a mismatch means run_suite and this
    # proxy disagree on the caps, which would silently change the cap-hit rate.
    want = cfg.think_cap + cfg.answer_cap + 16
    if body.get("max_tokens") not in (None, want) and not getattr(warn_backstop, "done", False):
        warn_backstop.done = True  # type: ignore[attr-defined]
        print(f"WARNING: request max_tokens={body.get('max_tokens')} but proxy caps imply {want}; "
              "pass the same --think-cap/--answer-cap to both", flush=True)


# ---------------------------------------------------------------- sampling


def transient(e: BaseException) -> bool:
    import tinker

    if isinstance(e, (tinker.APIConnectionError, tinker.RateLimitError, tinker.InternalServerError,
                      asyncio.TimeoutError, ConnectionError)):
        return True
    if isinstance(e, tinker.APIStatusError):
        return e.status_code in (408, 409, 429) or e.status_code >= 500
    return isinstance(e, tinker.RequestFailedError) and str(getattr(e, "category", "")).lower() != "user"


async def with_retry(px: Proxy, call: Callable[[], Awaitable[Any]]) -> Any:
    for attempt in range(px.cfg.retries):
        try:
            return await call()
        except Exception as e:  # noqa: BLE001 - classified below; non-transient errors propagate
            if not transient(e) or attempt == px.cfg.retries - 1:
                raise
            px.stats["retries"] += 1
            await asyncio.sleep(min(60.0, 2.0 ** (attempt + 1)) + random.random())
    raise RuntimeError("unreachable")


async def sample_once(px: Proxy, tokens: list[int], max_tokens: int, stop: list[int], temperature: float,
                      top_p: float, seed: int | None) -> tuple[list[int], str]:
    """One n=1 sample; mirrors rollout._sample (paid pre-check/record) and adds top_p and the request seed."""
    import tinker

    params = tinker.SamplingParams(max_tokens=max_tokens, temperature=temperature, stop=stop, top_p=top_p,
                                   seed=seed if px.cfg.pass_seed else None)
    lc.paid_precheck(lc.paid_cost(prefill=len(tokens), sample=max_tokens))

    async def call() -> Any:
        return await px.sampler.sample(prompt=tinker.ModelInput.from_ints(tokens), num_samples=1,
                                       sampling_params=params)
    seq = (await with_retry(px, call)).sequences[0]
    out = list(seq.tokens)
    lc.paid_record(lc.paid_cost(prefill=len(tokens), sample=len(out)), "sample")
    return out, seq.stop_reason


async def think_stage(px: Proxy, prompt: list[int], cfg: ro.SampleCfg, knobs: dict[str, Any]) -> tuple[list[int], bool]:
    # +2: the model samples its own <|message_model|><|content_thinking|> header; the pods' client counts
    # only thinking text against the cap, so the header must not eat into it (the lab's think_cap does).
    toks, stop = await sample_once(px, prompt, cfg.think_cap + 2, [ro.END_MESSAGE], cfg.temperature,
                                   knobs["top_p"], knobs["seed"])
    toks, cut = cut_at_cap(toks, ro.CONTENT_THINKING, None, cfg.think_cap)
    cap_hit = cut or (stop == "length" and ro.END_MESSAGE not in toks)
    if not cap_hit and (not toks or toks[-1] != ro.END_MESSAGE):
        toks = toks + [ro.END_MESSAGE]  # whether the SDK returns the stop token is not documented
    return toks, cap_hit


async def answer_stage(px: Proxy, context: list[int], header_done: bool, cfg: ro.SampleCfg,
                       knobs: dict[str, Any], seed_offset: int) -> tuple[list[int], bool]:
    """Answer until <|content_model_end_sampling|> (rollout._finish_answer), capped at answer_cap text tokens."""
    allowance = 0 if header_done else 2  # same reasoning as think_stage, for <|message_model|><|content_text|>
    seed = None if knobs["seed"] is None else knobs["seed"] + seed_offset
    toks, stop = await sample_once(px, context, cfg.answer_cap + allowance, [ro.END_SAMPLING],
                                   cfg.temperature, knobs["top_p"], seed)
    start = ro.CONTENT_TEXT if header_done else None
    toks, cut = cut_at_cap(toks, ro.CONTENT_TEXT, start, cfg.answer_cap)
    ended = ro.END_SAMPLING in toks or ro.END_MESSAGE in toks
    return toks, cut or (stop == "length" and not ended)


async def generate_chat(px: Proxy, prompt: list[int], cfg: ro.SampleCfg, knobs: dict[str, Any]) -> Reply:
    if not cfg.thinking:
        toks, runaway = await answer_stage(px, prompt, True, cfg, knobs, 0)
        return Reply(answer=text_tokens(toks, ro.CONTENT_TEXT, ro.CONTENT_TEXT), runaway=runaway)
    t1, cap_hit = await think_stage(px, prompt, cfg, knobs)
    if cap_hit:
        return Reply(think=text_tokens(t1, ro.CONTENT_THINKING, None), cap_hit=True,
                     forced_prefix=prompt + t1 + FORCED_TAIL)
    t2, runaway = await answer_stage(px, prompt + t1, False, cfg, knobs, 1)
    both = t1 + t2
    return Reply(think=text_tokens(both, ro.CONTENT_THINKING, None),
                 answer=text_tokens(both, ro.CONTENT_TEXT, None), runaway=runaway)


# ---------------------------------------------------------------- stats


def record(px: Proxy, kind: str, reply: Reply | None, seconds: float) -> None:
    s = px.stats
    s["n"] += 1
    s[kind] += 1
    s["seconds"] += seconds
    if reply is None:
        s["errors"] += 1
    else:
        s["cap_hit"] += reply.cap_hit
        s["runaway"] += reply.runaway
        s["think_tok"] += len(reply.think)
        s["answer_tok"] += len(reply.answer)
    if s["n"] % LOG_EVERY == 0:
        n = s["n"]
        print(json.dumps({"t": round(time.time()), "requests": int(n), "chat": int(s["chat"]),
                          "completions": int(s["completions"]), "cap_hit": int(s["cap_hit"]),
                          "runaway": int(s["runaway"]), "errors": int(s["errors"]), "retries": int(s["retries"]),
                          "mean_think_tok": round(s["think_tok"] / n, 1), "mean_answer_tok": round(s["answer_tok"] / n, 1),
                          "mean_s": round(s["seconds"] / n, 1), "forced_cache_hit": int(s["forced_cache_hit"]),
                          "forced_reparsed": int(s["forced_reparsed"])}), flush=True)


def remember_forced(px: Proxy, key: str, tokens: list[int]) -> None:
    px.forced[key] = tokens
    while len(px.forced) > FORCED_CACHE_MAX:
        px.forced.popitem(last=False)


# ---------------------------------------------------------------- HTTP / SSE


def sse(obj: Any) -> bytes:
    return b"data: " + json.dumps(obj, ensure_ascii=False).encode() + b"\n\n"


def chat_chunks(px: Proxy, rid: str, think: str, answer: str, n_think: int, n_answer: int,
                n_prompt: int, finish: str) -> list[bytes]:
    """Thinking and answer as one chunk each, usage cumulative, so client.py attributes the tokens per channel."""
    base = {"id": rid, "object": "chat.completion.chunk", "created": int(time.time()), "model": px.cfg.model_name}

    def usage(n: int) -> dict[str, int]:
        return {"prompt_tokens": n_prompt, "completion_tokens": n, "total_tokens": n_prompt + n}
    out = []
    if think:
        out.append(sse({**base, "choices": [{"index": 0, "delta": {"role": "assistant", "reasoning_content": think}}],
                        "usage": usage(n_think)}))
    if answer:
        out.append(sse({**base, "choices": [{"index": 0, "delta": {"content": answer}}],
                        "usage": usage(n_think + n_answer)}))
    out.append(sse({**base, "choices": [{"index": 0, "delta": {}, "finish_reason": finish}],
                    "usage": usage(n_think + n_answer)}))
    out.append(sse({**base, "choices": [], "usage": usage(n_think + n_answer)}))
    return out + [b"data: [DONE]\n\n"]


def completion_chunks(px: Proxy, rid: str, text: str, n: int, n_prompt: int, finish: str) -> list[bytes]:
    base = {"id": rid, "object": "text_completion", "created": int(time.time()), "model": px.cfg.model_name}
    usage = {"prompt_tokens": n_prompt, "completion_tokens": n, "total_tokens": n_prompt + n}
    out = [sse({**base, "choices": [{"index": 0, "text": text}], "usage": usage})] if text else []
    out.append(sse({**base, "choices": [{"index": 0, "text": "", "finish_reason": finish}], "usage": usage}))
    return out + [sse({**base, "choices": [], "usage": usage}), b"data: [DONE]\n\n"]


async def stream_work(request: Any, work: Awaitable[list[bytes]]) -> Any:
    """Headers go out at once and a comment ping every 10 s, so the client's 900 s socket timeout never fires
    while a request waits for the semaphore or a long sample. A failure after the headers is sent as a
    non-JSON data line: client.py's json.loads raises ValueError, which it records as an error and retries,
    rather than silently scoring an empty answer."""
    from aiohttp import web

    resp = web.StreamResponse(headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache"})
    await resp.prepare(request)
    task = asyncio.ensure_future(work)
    try:
        while not task.done():
            done, _ = await asyncio.wait({task}, timeout=PING_SECONDS)
            if not done:
                await resp.write(b": ping\n\n")
        try:
            chunks = task.result()
        except Exception as e:  # noqa: BLE001 - reported to the client, scrubbed of any key
            chunks = [f"data: proxy-error {lc.scrub(type(e).__name__ + ': ' + str(e))[:300]}\n\n".encode()]
        for c in chunks:
            await resp.write(c)
    except ConnectionError:
        task.cancel()  # client cut the stream at a cap or gave up; free the slot
        return resp
    return resp


async def run_chat(px: Proxy, body: dict[str, Any], prompt: list[int], cfg: ro.SampleCfg,
                   chat: ro.Chat) -> tuple[Reply, str, str]:
    knobs = {"top_p": float(body.get("top_p", 1.0)), "seed": body.get("seed")}
    t0 = time.time()
    try:
        async with px.sem:
            reply = await generate_chat(px, prompt, cfg, knobs)
    except Exception:
        record(px, "chat", None, time.time() - t0)
        raise
    record(px, "chat", reply, time.time() - t0)
    think, answer = decode(px.tok, reply.think), decode(px.tok, reply.answer)
    if reply.forced_prefix is not None:
        remember_forced(px, tpl.forced_answer_prompt(body["messages"], cfg.effort, think), reply.forced_prefix)
    return reply, think, answer


def finish_of(reply: Reply) -> str:
    return "length" if reply.cap_hit or reply.runaway else "stop"


async def handle_chat(px: Proxy, request: Any) -> Any:
    from aiohttp import web

    try:
        body = await request.json()
        chat = parse_chat(body.get("messages"))
        cfg = sample_cfg(px, body)
        prompt = ro.build_chat_prompt(px.renderer, chat, cfg)
        check_context(prompt, cfg)
    except (BadRequest, ValueError) as e:
        return web.json_response({"error": {"message": str(e), "type": "invalid_request_error"}}, status=400)
    warn_backstop(body, cfg)
    rid = f"chatcmpl-{int(time.time() * 1000)}-{random.randrange(1 << 30)}"

    async def work() -> list[bytes]:
        reply, think, answer = await run_chat(px, body, prompt, cfg, chat)
        return chat_chunks(px, rid, think, answer, len(reply.think), len(reply.answer), len(prompt),
                           finish_of(reply))
    if body.get("stream"):
        return await stream_work(request, work())
    reply, think, answer = await run_chat(px, body, prompt, cfg, chat)
    n = len(reply.think) + len(reply.answer)
    return web.json_response({
        "id": rid, "object": "chat.completion", "created": int(time.time()), "model": px.cfg.model_name,
        "choices": [{"index": 0, "finish_reason": finish_of(reply),
                     "message": {"role": "assistant", "content": answer, "reasoning_content": think}}],
        "usage": {"prompt_tokens": len(prompt), "completion_tokens": n, "total_tokens": len(prompt) + n}})


def forced_tokens(px: Proxy, prompt_text: str) -> list[int]:
    """Exact sampled tokens when this continuation follows one of our cap hits; else tokenise the text."""
    hit = px.forced.pop(prompt_text, None)
    if hit is not None:
        px.stats["forced_cache_hit"] += 1
        return hit
    px.stats["forced_reparsed"] += 1
    return encode_with_specials(px.tok, prompt_text)


async def run_completion(px: Proxy, body: dict[str, Any], tokens: list[int], cap: int) -> tuple[str, int, str]:
    knobs = {"top_p": float(body.get("top_p", 1.0)), "seed": body.get("seed")}
    cfg = ro.SampleCfg(answer_cap=cap, temperature=float(body.get("temperature", 0.6)))
    header_done = bool(tokens) and tokens[-1] == ro.CONTENT_TEXT
    t0 = time.time()
    try:
        async with px.sem:
            toks, runaway = await answer_stage(px, tokens, header_done, cfg, knobs, 2)
    except Exception:
        record(px, "completions", None, time.time() - t0)
        raise
    ans = text_tokens(toks, ro.CONTENT_TEXT, ro.CONTENT_TEXT if header_done else None)
    record(px, "completions", Reply(answer=ans, runaway=runaway), time.time() - t0)
    return decode(px.tok, ans), len(ans), "length" if runaway else "stop"


async def handle_completions(px: Proxy, request: Any) -> Any:
    """The forced answer after a thinking-cap hit (client.py _forced_answer); raw prompt in, answer text out."""
    from aiohttp import web

    try:
        body = await request.json()
        if not isinstance(body.get("prompt"), str):
            raise BadRequest("prompt must be a string")
        tokens = forced_tokens(px, body["prompt"])
        cap = caps_for(px, body)[1]
        if len(tokens) + cap + 8 > lc.CONTEXT_LIMIT:
            raise BadRequest("prompt+answer cap exceeds the context")
    except (BadRequest, ValueError) as e:
        return web.json_response({"error": {"message": str(e), "type": "invalid_request_error"}}, status=400)
    rid = f"cmpl-{int(time.time() * 1000)}-{random.randrange(1 << 30)}"

    async def work() -> list[bytes]:
        text, n, finish = await run_completion(px, body, tokens, cap)
        return completion_chunks(px, rid, text, n, len(tokens), finish)
    if body.get("stream"):
        return await stream_work(request, work())
    text, n, finish = await run_completion(px, body, tokens, cap)
    return web.json_response({"id": rid, "object": "text_completion", "model": px.cfg.model_name,
                              "choices": [{"index": 0, "text": text, "finish_reason": finish}],
                              "usage": {"prompt_tokens": len(tokens), "completion_tokens": n}})


async def handle_tokenize(px: Proxy, request: Any) -> Any:
    """vLLM-style /tokenize: lets client.py verify the effort reaches the prompt and count prompt tokens exactly."""
    from aiohttp import web

    try:
        body = await request.json()
        if "messages" in body:
            cfg = ro.SampleCfg(effort=parse_effort(body, px.cfg.effort))
            ids = ro.build_chat_prompt(px.renderer, parse_chat(body["messages"]), cfg)
        else:
            ids = encode_with_specials(px.tok, str(body.get("prompt", "")))
    except (BadRequest, ValueError) as e:
        return web.json_response({"error": {"message": str(e)}}, status=400)
    return web.json_response({"tokens": ids, "count": len(ids), "max_model_len": lc.CONTEXT_LIMIT})


def build_app(px: Proxy) -> Any:
    from aiohttp import web

    async def models(_: Any) -> Any:
        return web.json_response({"object": "list", "data": [{"id": px.cfg.model_name, "object": "model",
                                                              "owned_by": "tinker"}]})

    async def health(_: Any) -> Any:
        return web.json_response({"status": "ok"})
    app = web.Application(client_max_size=64 * 1024 * 1024)
    app.router.add_get("/v1/models", models)
    app.router.add_get("/health", health)
    app.router.add_post("/v1/chat/completions", lambda r: handle_chat(px, r))
    app.router.add_post("/v1/completions", lambda r: handle_completions(px, r))
    app.router.add_post("/tokenize", lambda r: handle_tokenize(px, r))
    return app


async def start_site(px: Proxy, port: int) -> Any:
    from aiohttp import web

    runner = web.AppRunner(build_app(px), access_log=None)  # no access log: nothing about requests is printed
    await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", port).start()
    return runner


# ---------------------------------------------------------------- live entry point


async def make_sampler(a: argparse.Namespace) -> Any:
    if a.mode == "paid":
        lc.enable_paid(a.paid_run)  # refuses unless LAB_PAID_MAX_USD > 0
    svc = lc.service_client()
    if a.sampler:
        sc = await svc.create_sampling_client(model_path=a.sampler)
        lc.assert_free(await sc.get_base_model_async())  # as eval.py: the adapter must sit on the allowed base
        return sc
    return await lc.sampling_client(svc)


async def serve(a: argparse.Namespace) -> None:
    renderer, tok = lc.renderer_and_tokenizer()
    cfg = ProxyCfg(model_name=a.model_name, think_cap=a.think_cap, answer_cap=a.answer_cap, effort=a.effort,
                   concurrency=a.concurrency, pass_seed=not a.no_seed)
    px = Proxy(cfg=cfg, sampler=await make_sampler(a), renderer=renderer, tok=tok,
               sem=asyncio.Semaphore(cfg.concurrency))
    await start_site(px, a.port)
    src = a.sampler or f"base {lc.default_model()}"
    print(f"oai_proxy on http://127.0.0.1:{a.port}/v1 model={cfg.model_name} source={src} "
          f"caps={cfg.think_cap}/{cfg.answer_cap} concurrency={cfg.concurrency}", flush=True)
    await asyncio.Event().wait()


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--base", action="store_true", help="stock model")
    src.add_argument("--sampler", help="tinker://.../sampler_weights/<name> (stock + adapter, live)")
    ap.add_argument("--mode", choices=["free", "paid"], default="free")
    ap.add_argument("--paid-run", default="oai_proxy", help="ledger run name in paid mode")
    ap.add_argument("--port", type=int, default=8411)
    ap.add_argument("--model-name", default="inkling-tinker", help="id served at /v1/models (pass as --model)")
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--think-cap", type=int, default=4096)
    ap.add_argument("--answer-cap", type=int, default=8192)
    ap.add_argument("--effort", type=float, default=0.9, help="only for requests without reasoning_effort")
    ap.add_argument("--no-seed", action="store_true", help="do not forward the request seed to Tinker")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if not a.selftest and not (a.base or a.sampler):
        ap.error("choose --base or --sampler (or --selftest)")
    return a


def main() -> None:
    a = parse_args()
    if a.selftest:
        selftest()
        return
    asyncio.run(serve(a))


# ---------------------------------------------------------------- offline self-test


def fake_messages(prompt_text: str) -> list[dict[str, str]]:
    """Recover the chat from a rendered prompt so the fake can reuse mock_server's canned replies."""
    roles = {"user": "user", "model": "assistant", "system": "system"}
    msgs = [{"role": roles[r], "content": c} for r, c in re.findall(
        r"<\|message_(user|model|system)\|><\|content_text\|>(.*?)<\|end_message\|>", prompt_text, re.S)]
    return [m for m in msgs if not (m["role"] == "system" and m["content"].startswith("Thinking effort level:"))]


def fake_tokens(tok: Any, words: Any, max_tokens: int) -> list[int]:
    import itertools

    return list(tok.tml_tokenizer.encode_ordinary("".join(itertools.islice(words, max_tokens + 8))))[:max_tokens + 8]


def make_fake_sampler(tok: Any, log: list[dict[str, Any]]) -> Any:
    """A stand-in for tinker's SamplingClient: mock_server's canned behaviour, real token ids, honours
    max_tokens/stop, fails once with a transient error to exercise the retry path."""
    from types import SimpleNamespace

    import mock_server as ms

    state = {"failed": False}

    async def sample(prompt: Any, num_samples: int, sampling_params: Any) -> Any:
        ids = prompt.to_ints()
        if not state["failed"]:
            state["failed"] = True
            raise ConnectionError("fake transient failure")
        mx, stop = sampling_params.max_tokens, list(sampling_params.stop)
        text = tok.decode(ids)
        log.append({"ids": ids, "stop": stop, "max_tokens": mx, "seed": sampling_params.seed})
        prefilled = ids[-1] == ro.CONTENT_TEXT
        # forced continuation = answer header right after a thinking message (thinking-off has none)
        if prefilled and "<|content_thinking|>" in text.rsplit("<|message_user|>", 1)[-1]:
            out = fake_tokens(tok, ms.plan_completion(text), mx) + [ro.END_MESSAGE, ro.END_SAMPLING]
        else:
            think, answer = ms.plan_chat(fake_messages(text))
            if stop == [ro.END_MESSAGE]:
                out = [ro.MESSAGE_MODEL, ro.CONTENT_THINKING] + fake_tokens(tok, think, mx) + [ro.END_MESSAGE]
            else:
                head = [] if prefilled else [ro.MESSAGE_MODEL, ro.CONTENT_TEXT]
                out = head + fake_tokens(tok, answer, mx) + [ro.END_MESSAGE, ro.END_SAMPLING]
        cut = [i for i, t in enumerate(out) if t in stop]
        if cut and cut[0] < mx:
            out, reason = out[:cut[0] + 1], "stop"
        else:
            out, reason = out[:mx], "length"
        seq = SimpleNamespace(tokens=out, logprobs=None, stop_reason=reason)
        return SimpleNamespace(sequences=[seq])
    return SimpleNamespace(sample=sample)


def free_port() -> int:
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def start_mock(port: int) -> None:
    import threading
    from http.server import ThreadingHTTPServer

    import mock_server as ms

    ms.WORLD.update(gold={}, tasks={}, pub_code={}, pub_mmlu={})  # canned prose/dragon/letter branches only
    ms.Handler.style = "field"
    srv = ThreadingHTTPServer(("127.0.0.1", port), ms.Handler)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()


def start_proxy(px_factory: Callable[[], Proxy], port: int) -> Proxy:
    import threading

    ready: dict[str, Any] = {}
    ev = threading.Event()

    def run() -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        ready["px"] = px_factory()  # Semaphore must be created inside this loop
        loop.run_until_complete(start_site(ready["px"], port))
        ev.set()
        loop.run_forever()
    threading.Thread(target=run, daemon=True).start()
    ev.wait(30)
    return ready["px"]


SAME = ("think_cap_hit", "answer_runaway", "forced", "finish", "split", "count_source", "error")


def compare(name: str, a: Any, b: Any, texts: str) -> list[str]:
    """Same record fields from the mock (pod) path and the proxy path; texts: both | think | answer | none."""
    bad = [f"{name}: {k} mock={getattr(a, k)!r} proxy={getattr(b, k)!r}" for k in SAME if getattr(a, k) != getattr(b, k)]
    if texts in ("both", "think") and a.thinking != b.thinking:
        bad.append(f"{name}: thinking differs")
    if texts in ("both", "answer") and a.answer != b.answer:
        bad.append(f"{name}: answer differs")
    return bad


def selftest() -> None:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")  # tokenizer from the local cache only; no network
    from client import Client, ClientCfg

    renderer, tok = lc.renderer_and_tokenizer()
    log: list[dict[str, Any]] = []
    caps = {"think_cap": 64, "answer_cap": 512}
    mport, pport = free_port(), free_port()
    start_mock(mport)
    px = start_proxy(lambda: Proxy(cfg=ProxyCfg(model_name="selftest", **caps, concurrency=4), sampler=make_fake_sampler(tok, log),
                                   renderer=renderer, tok=tok, sem=asyncio.Semaphore(4)), pport)

    def client(port: int, **kw: Any) -> Any:
        return Client(ClientCfg(base_url=f"http://127.0.0.1:{port}/v1", model="m", effort=0.9, temperature=0.6,
                                concurrency=4, **caps, **kw))
    mock, prox = client(mport), client(pport)
    errs: list[str] = []
    checks: list[str] = []

    def both(msgs: list[dict[str, str]], **kw: Any) -> tuple[Any, Any]:
        return mock.generate(msgs, seed=7, **kw), prox.generate(msgs, seed=7, **kw)

    # 1. plain prompt
    a, b = both([{"role": "user", "content": "Tell me about rivers."}])
    errs += compare("plain", a, b, "both")
    errs += [] if (b.finish, b.split, b.think_tokens > 0, b.answer_tokens > 0) == ("stop", "field", True, True) \
        else [f"plain: unexpected {b}"]
    errs += [] if px.stats["retries"] == 1 else ["retry path not exercised"]
    checks.append(f"plain: think {b.think_tokens} tok, answer {b.answer_tokens} tok, finish {b.finish}, 1 retry")

    # 2. system + multi-turn: the proxy's prompt must be the renderer's tokens and the official template text
    chat = [{"role": "system", "content": "You are terse."}, {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello."}, {"role": "user", "content": "Write a 40-word story about a fox."}]
    n0 = len(log)
    a, b = both(chat)
    errs += compare("system+multi", a, b, "both")
    sent = log[n0]["ids"]
    if sent != ro.build_chat_prompt(renderer, chat, ro.SampleCfg(effort=0.9)) or tok.decode(sent) != tpl.render(chat, 0.9):
        errs.append("system+multi: prompt tokens differ from build_chat_prompt / template.render")
    if log[n0]["seed"] != 7 or log[n0]["stop"] != [ro.END_MESSAGE] or log[n0]["max_tokens"] != caps["think_cap"] + 2:
        errs.append(f"system+multi: stage-1 params {log[n0]}")
    checks.append("system+multi: prompt == build_chat_prompt == template.render; seed/stop/max_tokens forwarded")

    # 3. thinking off (Tinker-only switch): effort 0 + prefilled answer header, no thinking returned
    off = client(pport, extra_body={"thinking": False})
    n0 = len(log)
    b = off.generate([{"role": "user", "content": "Tell me about rivers."}], seed=7)
    a = mock.generate([{"role": "user", "content": "Tell me about rivers."}], seed=7)
    ids = log[n0]["ids"]
    if b.thinking or b.think_tokens or b.answer != a.answer or b.finish != "stop" or b.error:
        errs.append(f"thinking off: {b}")
    if ids[-2:] != [ro.MESSAGE_MODEL, ro.CONTENT_TEXT] or "Thinking effort level: 0<" not in tok.decode(ids):
        errs.append("thinking off: prompt is not effort 0 + prefilled answer header")
    checks.append("thinking off: no thinking, same answer text, prompt ends <|message_model|><|content_text|>")

    # 4. thinking-cap hit: capped thought, forced answer via /v1/completions (pod semantics)
    a, b = both([{"role": "user", "content": "Write a story about a dragon."}])
    errs += compare("think cap", a, b, "answer")
    hit = px.stats["forced_cache_hit"]
    if not (b.think_cap_hit and b.forced and b.think_tokens == caps["think_cap"] and b.answer and hit == 1):
        errs.append(f"think cap: {b} cache_hit={hit}")
    if not a.thinking[:40] == b.thinking[:40]:
        errs.append("think cap: capped thinking is not the same canned loop")
    checks.append(f"thinking-cap hit: cap_hit/forced both paths, think_tokens={b.think_tokens}, "
                  f"forced answer identical, exact-token continuation used")

    # 4b. no-force (agent_hard_noforce): the reply ends at the cap with no answer
    a, b = both([{"role": "user", "content": "Write a story about a dragon."}], force=False)
    errs += compare("think cap no-force", a, b, "answer")
    checks.append("thinking-cap hit, force=False: no answer on both paths")

    # 5. answer length stop
    a, b = both([{"role": "user", "content": "Write a letter to my aunt."}])
    errs += compare("answer cap", a, b, "think")
    if not (b.answer_runaway and b.answer_tokens == caps["answer_cap"] and a.answer[:60] == b.answer[:60]):
        errs.append(f"answer cap: runaway={b.answer_runaway} answer_tokens={b.answer_tokens}")
    checks.append(f"answer length stop: runaway both paths, answer_tokens={b.answer_tokens}")

    # 6. probe (models, effort reaches the prompt via /tokenize) and the text fallback for /v1/completions
    info = prox.probe()
    if info.get("models") != ["selftest"] or info["effort_check"].get("differs") is not True:
        errs.append(f"probe: {info}")
    text = tpl.forced_answer_prompt(chat, 0.9, "some thinking")
    want = ro.build_chat_prompt(renderer, chat, ro.SampleCfg(effort=0.9))
    if encode_with_specials(tok, text)[:len(want)] != want:
        errs.append("fallback tokenisation of a raw forced prompt differs from the renderer")
    checks.append("probe: /v1/models + /tokenize effort check pass; raw-prompt fallback tokenises like the renderer")

    # 7. bad request -> HTTP 400, recorded by client.py without retry
    g = prox.generate([{"role": "assistant", "content": "x"}], seed=1)
    if not (g.error and "HTTP 400" in g.error):
        errs.append(f"bad request: {g.error}")
    checks.append("bad request -> HTTP 400 error recorded")

    print("\n".join("  checked " + c for c in checks))
    if errs:
        print("SELFTEST FAIL\n  " + "\n  ".join(errs))
        raise SystemExit(1)
    print(f"SELFTEST PASS ({len(log)} fake sampling calls, stats {json.dumps({k: int(v) for k, v in px.stats.items()})})")


if __name__ == "__main__":
    main()
