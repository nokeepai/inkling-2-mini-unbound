"""Two-stage sampling with separate thinking and answer caps on Inkling-2-Mini.

Inkling's TMLv0 format puts thinking and answer in separate messages:
  <|message_model|><|content_thinking|>...<|end_message|><|message_model|><|content_text|>...
  <|end_message|><|content_model_end_sampling|>
Stage 1 samples up to think_cap and stops at the first <|end_message|>; if it ran out first,
that is a thinking-cap hit. Stage 2 continues from exactly those tokens up to answer_cap, so
in the normal case the whole completion is sampled (no forced tokens) and trainable.
Thinking "off" = effort 0 plus a prefilled <|message_model|><|content_text|> header,
because effort 0 alone still emits a short thinking message.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

import tinker

import lab_common as lc
from rewards import Sample

MESSAGE_MODEL, CONTENT_TEXT, CONTENT_THINKING = 200001, 200004, 200008
END_MESSAGE, END_SAMPLING = 200010, 200006
SPECIAL_MIN = 199990


@dataclass(frozen=True)
class SampleCfg:
    think_cap: int = 4096
    answer_cap: int = 8192
    temperature: float = 0.6
    effort: float = 0.6
    thinking: bool = True
    # Eval only: after a thinking-cap hit, force-close thinking and still sample an answer.
    force_answer_after_cap: bool = False


@dataclass
class Rollout:
    prompt_tokens: list[int]
    completion: list[int] = field(default_factory=list)
    logprobs: list[float] = field(default_factory=list)
    # 1 for sampled tokens, 0 for forced ones (never trained on).
    trainable: list[int] = field(default_factory=list)
    think_cap_hit: bool = False
    answer_runaway: bool = False
    seconds: float = 0.0

    def append(self, tokens: list[int], logprobs: list[float] | None, sampled: bool) -> None:
        self.completion += tokens
        self.logprobs += list(logprobs) if logprobs is not None else [0.0] * len(tokens)
        self.trainable += [1 if sampled else 0] * len(tokens)


def build_prompt(renderer: Any, text: str, cfg: SampleCfg) -> list[int]:
    from tinker_cookbook.renderers import Message

    effort = cfg.effort if cfg.thinking else 0.0
    toks = renderer.build_generation_prompt([Message(role="user", content=text)], effort=effort).to_ints()
    if not cfg.thinking:
        toks = toks + [MESSAGE_MODEL, CONTENT_TEXT]
    return toks


Chat = list[dict[str, str]]
CHAT_ROLES = frozenset({"user", "assistant"})


def has_context(row: dict[str, Any]) -> bool:
    """True for rows that need build_chat_prompt: a system prompt or an explicit message list.
    Plain `prompt` rows and system-less `turns` rows keep their old render paths."""
    return bool(row.get("system")) or "messages" in row


def row_chat(row: dict[str, Any]) -> Chat:
    """[system?] + messages (or the plain prompt as one user turn), validated.

    Only role and content are kept, so a prior assistant turn carries its answer text and never
    any thinking field a source file might hold. Raises ValueError on a malformed row."""
    system = row.get("system")
    if system is not None and not (isinstance(system, str) and system.strip()):
        raise ValueError(f"{row.get('id')}: system must be a non-empty string")
    msgs = row.get("messages")
    if msgs is None:
        msgs = [{"role": "user", "content": row["prompt"]}]
    if not msgs or msgs[-1].get("role") != "user":
        raise ValueError(f"{row.get('id')}: messages must end with a user turn")
    for m in msgs:
        if m.get("role") not in CHAT_ROLES or not isinstance(m.get("content"), str):
            raise ValueError(f"{row.get('id')}: messages hold only user/assistant turns with string content")
    head = [{"role": "system", "content": system}] if system else []
    return head + [{"role": m["role"], "content": m["content"]} for m in msgs]


def build_chat_prompt(renderer: Any, chat: Chat, cfg: SampleCfg) -> list[int]:
    """As build_prompt, for a full conversation (optional leading system turn, earlier turns).

    The TMLv0 renderer emits the effort system message after any leading system turns and before
    the first user turn, frames each prior assistant turn as
    <|message_model|><|content_text|>...<|end_message|><|content_model_end_sampling|>, and adds no
    generation header; that matches the official chat_template.jinja token for token (checked
    offline against out/final/configs/chat_template.jinja, see build_wide_train.py --check)."""
    from tinker_cookbook.renderers import Message

    effort = cfg.effort if cfg.thinking else 0.0
    conv = [Message(role=m["role"], content=m["content"]) for m in chat]
    toks = renderer.build_generation_prompt(conv, effort=effort).to_ints()
    if not cfg.thinking:
        toks = toks + [MESSAGE_MODEL, CONTENT_TEXT]
    return toks


def split_channels(tokens: list[int], start_channel: int | None) -> tuple[list[int], list[int]]:
    """Return (thinking_tokens, answer_tokens) by tracking the content_* marker in force."""
    think, answer = [], []
    channel = start_channel
    for t in tokens:
        if t in (CONTENT_THINKING, CONTENT_TEXT):
            channel = t
            continue
        if t >= SPECIAL_MIN:
            continue
        (think if channel == CONTENT_THINKING else answer).append(t)
    return think, answer


def to_sample(r: Rollout, tok: Any, row: dict[str, Any], cfg: SampleCfg) -> Sample:
    start = None if cfg.thinking else CONTENT_TEXT
    think, answer = split_channels(r.completion, start)
    return Sample(category=row["category"], thinking=tok.decode(think), answer=tok.decode(answer),
                  think_tokens=len(think), answer_tokens=len(answer),
                  think_cap_hit=r.think_cap_hit, answer_runaway=r.answer_runaway,
                  target_words=row.get("target_words"), gold=row.get("gold"))


async def _sample(sc: Any, tokens: list[int], n: int, max_tokens: int, stop: list[int],
                  temperature: float) -> list[Any]:
    params = tinker.SamplingParams(max_tokens=max_tokens, temperature=temperature, stop=stop)
    # Paid mode only: refuse if the worst case would cross the cap, then record actual usage.
    lc.paid_precheck(lc.paid_cost(prefill=len(tokens), sample=n * max_tokens))
    resp = await sc.sample(prompt=tinker.ModelInput.from_ints(tokens), num_samples=n,
                           sampling_params=params)
    seqs = list(resp.sequences)
    lc.paid_record(lc.paid_cost(prefill=len(tokens), sample=sum(len(s.tokens) for s in seqs)), "sample")
    return seqs


async def _finish_answer(sc: Any, r: Rollout, cfg: SampleCfg, sem: asyncio.Semaphore) -> None:
    async with sem:
        seq = (await _sample(sc, r.prompt_tokens + r.completion, 1, cfg.answer_cap,
                             [END_SAMPLING], cfg.temperature))[0]
    r.append(list(seq.tokens), seq.logprobs, sampled=True)
    r.answer_runaway = seq.stop_reason == "length"


async def rollout_group(sc: Any, prompt: list[int], n: int, cfg: SampleCfg,
                        sem: asyncio.Semaphore) -> list[Rollout]:
    """n rollouts of one prompt. Stage 1 is one request with num_samples=n."""
    t0 = time.time()
    rolls = [Rollout(prompt_tokens=prompt) for _ in range(n)]
    if cfg.thinking:
        async with sem:
            seqs = await _sample(sc, prompt, n, cfg.think_cap, [END_MESSAGE], cfg.temperature)
        for r, s in zip(rolls, seqs):
            r.append(list(s.tokens), s.logprobs, sampled=True)
            r.think_cap_hit = s.stop_reason == "length"
            if r.think_cap_hit and cfg.force_answer_after_cap:
                r.append([END_MESSAGE, MESSAGE_MODEL, CONTENT_TEXT], None, sampled=False)
        todo = [r for r in rolls if not r.think_cap_hit or cfg.force_answer_after_cap]
        await asyncio.gather(*(_finish_answer(sc, r, cfg, sem) for r in todo))
    else:
        async with sem:
            seqs = await _sample(sc, prompt, n, cfg.answer_cap, [END_SAMPLING], cfg.temperature)
        for r, s in zip(rolls, seqs):
            r.append(list(s.tokens), s.logprobs, sampled=True)
            r.answer_runaway = s.stop_reason == "length"
    for r in rolls:
        r.seconds = time.time() - t0
    return rolls


def assert_context(prompt: list[int], cfg: SampleCfg) -> None:
    need = len(prompt) + cfg.think_cap + cfg.answer_cap + 8
    if need > lc.CONTEXT_LIMIT:
        raise SystemExit(f"prompt+caps={need} exceeds the free tier's {lc.CONTEXT_LIMIT} context")
