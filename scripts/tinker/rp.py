"""Multi-turn harmless roleplay: scripted-user chats, on-policy prefixes, per-turn eval.

Training (train_rl.py --rp-frac): a chat row picks a turn k, the current policy writes
turns 0..k-1 once (history keeps only the visible replies), then a group is sampled at
turn k and scored like any other prompt (quality judge on the transcript + runaway).
Eval: 20 held-out chats rolled to the end; per turn we flag thinking-cap hits, in-reply
loops, runaway answers, and cross-turn repetition (reply re-uses earlier replies' 12-grams).

  PY=~/.inkling-abl/tinker_lab/.venv/bin/python
  $PY scripts/tinker/rp.py build                         # rp_train.jsonl / rp_eval.jsonl
  $PY scripts/tinker/rp.py eval --tag base_rp [--model-path tinker://...]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics
import time
from typing import Any

import lab_common as lc
import rewards as rw
import rollout as ro

TRAIN_CHARS = [
    "a cheerful shopkeeper in a bustling fantasy market", "the ship's computer aboard a deep-space survey vessel",
    "a retired detective who now runs a tea shop", "a wise old tree in an enchanted forest",
    "a pirate captain teaching a new crew member to read maps", "a medieval innkeeper welcoming travellers",
    "a mountain guide leading a hiking group", "a friendly dragon who runs a library",
    "a Victorian inventor showing off a new machine", "a lighthouse keeper on a stormy night",
    "a wizard's apprentice who keeps mixing up spells", "a chef hosting a cooking show",
]
EVAL_CHARS = [
    "a park ranger leading a night-time stargazing walk", "a toymaker in a snowy alpine village",
    "a cartographer aboard an airship", "a beekeeper who talks to her bees",
    "a museum curator after hours", "a riverboat captain on a lazy summer river",
    "a robot gardener tending a rooftop farm", "a bookshop owner in a seaside town",
    "a circus ringmaster before the first show", "a train conductor on an overnight sleeper",
]
OPENERS = ["Let's do a roleplay. You are {c}. I've just arrived. Start the scene.",
           "Roleplay with me: you are {c}. I walk in and say hello.",
           "Please play {c}, and stay in character. I'm a curious visitor."]
USER_LINES = [
    "That's fascinating. Tell me more about that.", "What happened next?",
    "I look around carefully. What do I notice?", "Can you show me something special here?",
    "I ask about your past. How did you end up here?", "Is there anything I should be careful about?",
    "I laugh and ask what your favourite part of the day is.", "Do you have any advice for a newcomer like me?",
    "I offer to help with whatever you're working on.", "Tell me a story that people around here like to share.",
    "I notice something unusual nearby and point at it. What is it?",
    "Before I go, what should I remember about this place?",
]


def build(seed: int = 99) -> None:
    rng = random.Random(seed)

    def chats(chars: list[str], n: int, prefix: str) -> list[dict[str, Any]]:
        out = []
        for i in range(n):
            turns = rng.randint(4, 6)
            users = [rng.choice(OPENERS).format(c=chars[i % len(chars)])] + rng.sample(USER_LINES, turns - 1)
            out.append({"id": f"{prefix}{i:03d}", "category": "rp_multi", "character": chars[i % len(chars)],
                        "turns": users, "prompt": users[0], "target_words": None, "gold": None})
        return out

    lc.write_jsonl(lc.LAB_DIR / "rp_train.jsonl", chats(TRAIN_CHARS, 60, "rpt"))
    lc.write_jsonl(lc.LAB_DIR / "rp_eval.jsonl", chats(EVAL_CHARS, 20, "rpe"))
    print("wrote rp_train.jsonl (60) and rp_eval.jsonl (20)")


def messages(users: list[str], replies: list[str]) -> list[Any]:
    from tinker_cookbook.renderers import Message

    msgs = []
    for i, u in enumerate(users):
        msgs.append(Message(role="user", content=u))
        if i < len(replies):
            msgs.append(Message(role="assistant", content=replies[i]))
    return msgs


def transcript(users: list[str], replies: list[str]) -> str:
    """Judge request: the conversation so far, ending with the user's latest line."""
    lines = []
    for i, u in enumerate(users):
        lines.append(f"USER: {u}")
        if i < len(replies):
            lines.append(f"AI: {replies[i][:1500]}")
    return ("Continue this roleplay conversation in character with the next AI reply.\n\n"
            + "\n\n".join(lines))


def prompt_tokens(renderer: Any, users: list[str], replies: list[str], cfg: ro.SampleCfg) -> list[int]:
    return renderer.build_generation_prompt(messages(users, replies), effort=cfg.effort).to_ints()


def turn_chat(system: str | None, users: list[str], replies: list[str]) -> ro.Chat:
    msgs = [{"role": "system", "content": system}] if system else []
    for i, u in enumerate(users):
        msgs.append({"role": "user", "content": u})
        if i < len(replies):
            msgs.append({"role": "assistant", "content": replies[i]})
    return msgs


def turn_prompt(renderer: Any, row: dict[str, Any], users: list[str], replies: list[str],
                cfg: ro.SampleCfg) -> list[int]:
    """Scripted-turn prompt; a row with its own system prompt goes through build_chat_prompt,
    a system-less row keeps prompt_tokens (the rp_train path, unchanged)."""
    if row.get("system"):
        return ro.build_chat_prompt(renderer, turn_chat(row["system"], users, replies), cfg)
    return prompt_tokens(renderer, users, replies, cfg)


def chat_request(chat: ro.Chat, category: str) -> str:
    """Judge request for a system/messages row: the system prompt the AI was given, then the whole
    conversation ending on the user's line, so a reply is judged in context, not against the last
    line alone. Same framing and 1,500-char clip of earlier AI turns as transcript()."""
    system = "\n\n".join(m["content"] for m in chat if m["role"] == "system")
    lines = [f"USER: {m['content']}" if m["role"] == "user" else f"AI: {m['content'][:1500]}"
             for m in chat if m["role"] != "system"]
    head = ("Continue this roleplay conversation in character with the next AI reply."
            if category == "roleplay" else "Write the AI's next reply in this conversation.")
    pre = f"The AI was given this system prompt:\n{system}\n\n" if system else ""
    return pre + head + "\n\n" + "\n\n".join(lines)


async def roll_prefix(sc: Any, renderer: Any, tok: Any, row: dict[str, Any], k: int,
                      cfg: ro.SampleCfg, sem: asyncio.Semaphore) -> list[str]:
    """Current policy writes replies for turns 0..k-1 (one sample each), under the row's system prompt."""
    replies: list[str] = []
    for t in range(k):
        prompt = turn_prompt(renderer, row, row["turns"][: t + 1], replies, cfg)
        r = (await ro.rollout_group(sc, prompt, 1, cfg, sem))[0]
        # Clip history so a runaway early turn can't push later prompts past the context.
        replies.append(ro.to_sample(r, tok, row, cfg).answer.strip()[:6000])
    return replies


def ngram_set(words: list[str], n: int = 12) -> set[tuple[str, ...]]:
    return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}


def cross_turn_repeat(reply: str, previous: list[str], n: int = 12, thresh: float = 0.2) -> bool:
    """True when >= thresh of the reply's n-grams already appeared in earlier replies."""
    mine = ngram_set(rw.WORD_RE.findall(reply), n)
    if not mine or not previous:
        return False
    seen: set[tuple[str, ...]] = set()
    for p in previous:
        seen |= ngram_set(rw.WORD_RE.findall(p), n)
    return len(mine & seen) / len(mine) >= thresh


async def eval_chat(sc: Any, renderer: Any, tok: Any, row: dict[str, Any], cfg: ro.SampleCfg,
                    sem: asyncio.Semaphore) -> list[dict[str, Any]]:
    replies: list[str] = []
    out = []
    for t in range(len(row["turns"])):
        prompt = prompt_tokens(renderer, row["turns"][: t + 1], replies, cfg)
        r = (await ro.rollout_group(sc, prompt, 1, cfg, sem))[0]
        s = ro.to_sample(r, tok, row, cfg)
        flags = rw.loop_flags(s)
        rep = cross_turn_repeat(s.answer, replies)
        out.append({"id": row["id"], "turn": t, "think_tokens": s.think_tokens,
                    "answer_tokens": s.answer_tokens, "cap_hit": s.think_cap_hit,
                    "runaway": s.answer_runaway, "in_reply_loop": flags["span32x3"] or flags["gram30x20"],
                    "cross_turn_repeat": rep, "request": transcript(row["turns"][: t + 1], replies),
                    "answer": s.answer})
        replies.append(s.answer.strip()[:6000])
    return out


async def run_eval(args: argparse.Namespace) -> None:
    import judge

    cfg = ro.SampleCfg(temperature=args.temperature, effort=args.effort, force_answer_after_cap=True)
    renderer, tok = lc.renderer_and_tokenizer()
    svc = lc.service_client()
    if args.model_path:
        sc = await svc.create_sampling_client(model_path=args.model_path)
        lc.assert_free(await sc.get_base_model_async())
    else:
        sc = await lc.sampling_client(svc)
    sem = asyncio.Semaphore(32)
    chats = lc.read_jsonl(lc.LAB_DIR / "rp_eval.jsonl")
    t0 = time.time()
    turns = [t for c in await asyncio.gather(*(eval_chat(sc, renderer, tok, c, cfg, sem) for c in chats))
             for t in c]
    for backend, key in (("bison", "judge_p"), ("alt", "alt_p")):
        res = await judge.score_many([(t["request"], t["answer"]) for t in turns], backend=backend)
        for t, (p, _e) in zip(turns, res):
            t[key] = p
    lc.write_jsonl(lc.LAB_DIR / "eval" / f"{args.tag}.jsonl", turns)
    n = len(turns)
    bad = [t["cap_hit"] or t["runaway"] or t["in_reply_loop"] or t["cross_turn_repeat"] for t in turns]
    good = lambda k: sum((t[k] or 0) >= 0.5 for t in turns if t[k] is not None) / n  # noqa: E731
    summary = {"tag": args.tag, "model_path": args.model_path, "turns": n, "wall_s": round(time.time() - t0),
               "per_turn_loop_rate": sum(bad) / n,
               "cap_hit": sum(t["cap_hit"] for t in turns) / n,
               "runaway": sum(t["runaway"] for t in turns) / n,
               "in_reply_loop": sum(t["in_reply_loop"] for t in turns) / n,
               "cross_turn_repeat": sum(t["cross_turn_repeat"] for t in turns) / n,
               "median_think": statistics.median(t["think_tokens"] for t in turns),
               "median_answer": statistics.median(t["answer_tokens"] for t in turns),
               "bison_good": good("judge_p"), "haiku_good": good("alt_p"),
               "loop_by_turn": {i: sum(b for t, b in zip(turns, bad) if t["turn"] == i)
                                / max(1, sum(t["turn"] == i for t in turns)) for i in range(6)}}
    (lc.LAB_DIR / "eval" / f"{args.tag}.summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "eval"])
    ap.add_argument("--tag", default="base_rp")
    ap.add_argument("--model-path", default=None)
    ap.add_argument("--effort", type=float, default=0.9)
    ap.add_argument("--temperature", type=float, default=0.6)
    args = ap.parse_args()
    build() if args.cmd == "build" else asyncio.run(run_eval(args))


if __name__ == "__main__":
    main()
