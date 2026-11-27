"""Build the harmless free-lab prompt pool (PLAN.md Phase 0.5, hard rule 2).

Mix: ~40% creative with a stated word count, 25% maths with gold answers (GSM8K train +
MATH train, public HF), 20% general/technical Q&A, 15% harmless roleplay openers.
Writes prompts_train.jsonl and prompts_eval.jsonl (eval stratified, fixed seed).

  ~/.inkling-abl/tinker_lab/.venv/bin/python scripts/tinker/prompts.py
"""
from __future__ import annotations

import random
import re
from typing import Any

import lab_common as lc

SEED = 20261125
COUNTS = {"creative": 160, "maths": 100, "qa": 80, "roleplay": 60}
EVAL_FRACTION = 0.25

CREATIVE_FORMS = ["short story", "fable", "diary entry", "letter", "travelogue", "myth",
                  "bedtime story", "science-fiction vignette", "mystery scene", "essay"]
CREATIVE_TOPICS = [
    "a lighthouse keeper who befriends a migrating whale", "a bakery that only opens at dawn",
    "two rival gardeners preparing for a village fair", "a robot learning to paint sunsets",
    "a lost umbrella finding its way home", "the first snowfall in a desert town",
    "a librarian who discovers a book that rewrites itself", "a chess match between grandparents",
    "a tiny dragon afraid of heights", "a train journey across the mountains at night",
    "an astronaut growing tomatoes on Mars", "a street musician and a curious pigeon",
    "a map that leads to a forgotten orchard", "a kite festival on a windy coast",
    "a clockmaker who repairs a stranger's watch", "a fox and an owl planning a picnic",
    "a family reunion during a thunderstorm", "a new teacher's first day at a mountain school",
    "a scientist cataloguing glowing mushrooms", "a ferry captain's last voyage before retiring",
    "a city where everyone forgets one word each day", "a beekeeper and a stubborn queen bee",
    "a child building a boat from driftwood", "an old radio that picks up songs from the future",
    "a cartographer mapping an underwater cave", "a quiet café on the edge of the world",
]
LENGTH_PHRASES = ["Write a {form} of about {n} words about {topic}.",
                  "In roughly {n} words, write a {form} about {topic}.",
                  "Please write a ~{n}-word {form} about {topic}.",
                  "Compose a {form} about {topic}. Aim for approximately {n} words."]
WORD_TARGETS = [150, 200, 250, 300, 400, 500, 600, 800, 1000, 1200]

QA_TOPICS = [
    "how a hash table handles collisions", "why the sky appears blue", "how vaccines train the immune system",
    "the difference between TCP and UDP", "how photosynthesis converts light into chemical energy",
    "what a database index is and when it helps", "how compound interest works",
    "why we have seasons on Earth", "how a refrigerator moves heat", "what Big-O notation measures",
    "how public-key cryptography lets strangers communicate securely", "how tides are caused by the Moon",
    "the difference between a process and a thread", "how a bill becomes law in a parliamentary system",
    "how GPS determines your position", "what inflation is and how central banks respond to it",
    "how git branches and merges work", "why bread rises", "how noise-cancelling headphones work",
    "what a neural network learns during training", "how plate tectonics shapes continents",
    "the causes of the French Revolution", "how a CPU cache improves performance",
    "how recycling of aluminium works", "what makes a good unit test", "how rainbows form",
    "the difference between weather and climate", "how a transistor acts as a switch",
    "what REST APIs are", "how bees communicate the location of flowers",
    "how garbage collection works in managed languages", "why ice floats on water",
    "how solar panels generate electricity", "what the Doppler effect is",
    "how DNS resolves a domain name", "how a binary search works", "why leaves change colour in autumn",
    "how electric motors turn electricity into motion", "the main ideas of supply and demand",
    "how containers differ from virtual machines",
]
QA_FRAMES = ["Explain {t}.", "Can you explain {t} to a curious beginner?",
             "Give a clear, concise explanation of {t}.", "I'd like to understand {t}. Walk me through it."]

RP_CHARACTERS = [
    "a cheerful shopkeeper in a bustling fantasy market", "the ship's computer aboard a deep-space survey vessel",
    "a retired detective who now runs a tea shop", "a museum tour guide in ancient Rome, in character",
    "a wise old tree in an enchanted forest", "a pirate captain teaching a new crew member to read maps",
    "a medieval innkeeper welcoming travellers", "a mountain guide leading a hiking group",
    "a friendly dragon who runs a library", "a Victorian inventor showing off a new machine",
    "a starship engineer explaining a strange noise in the engine", "a lighthouse keeper on a stormy night",
    "a wizard's apprentice who keeps mixing up spells", "a chef hosting a cooking show",
    "a park ranger leading a night-time stargazing walk",
]
RP_OPENERS = [
    "Let's do a roleplay. You are {c}. I walk in and say: \"Hello! What's good today?\"",
    "Roleplay with me: you are {c}. I'm a newcomer who just arrived. Start the scene.",
    "Please play {c}. I approach and ask what makes this place special. Stay in character.",
    "You are {c}. I'm a curious visitor. Greet me and tell me a little story about your day.",
]
BLOCKLIST = re.compile(r"\b(weapon|gun|bomb|drug|kill|murder|suicide|self-harm|sex|nude|"
                       r"steal|crime|poison|explosive|blood)\w*", re.I)


def creative(rng: random.Random, n: int) -> list[dict[str, Any]]:
    rows = []
    for i in range(n):
        target = WORD_TARGETS[i % len(WORD_TARGETS)]
        text = rng.choice(LENGTH_PHRASES).format(
            form=rng.choice(CREATIVE_FORMS), topic=rng.choice(CREATIVE_TOPICS), n=target)
        rows.append({"category": "creative", "prompt": text, "target_words": target, "gold": None})
    return rows


def _gsm8k(n: int, rng: random.Random) -> list[dict[str, Any]]:
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split="train")
    idx = rng.sample(range(len(ds)), n)
    rows = []
    for i in idx:
        gold = ds[i]["answer"].split("####")[-1].strip().replace(",", "")
        rows.append({"category": "maths", "source": "gsm8k_train",
                     "prompt": ds[i]["question"] + "\n\nGive the final answer in \\boxed{}.",
                     "target_words": None, "gold": gold})
    return rows


def _math(n: int, rng: random.Random) -> list[dict[str, Any]]:
    """MATH train, levels 1-3, integer answers only so exact match is unambiguous."""
    from datasets import load_dataset

    subjects = ["algebra", "prealgebra", "number_theory", "counting_and_probability"]
    pool = []
    for s in subjects:
        for r in load_dataset("EleutherAI/hendrycks_math", s, split="train"):
            if r["level"] not in ("Level 1", "Level 2", "Level 3"):
                continue
            gold = _last_boxed(r["solution"])
            if gold is not None and re.fullmatch(r"-?\d+", gold):
                pool.append({"category": "maths", "source": f"math_train_{s}",
                             "prompt": r["problem"] + "\n\nGive the final answer in \\boxed{}.",
                             "target_words": None, "gold": gold})
    return rng.sample(pool, n)


def _last_boxed(text: str) -> str | None:
    start = text.rfind("\\boxed{")
    if start < 0:
        return None
    depth, i = 1, start + len("\\boxed{")
    for j in range(i, len(text)):
        depth += {"{": 1, "}": -1}.get(text[j], 0)
        if depth == 0:
            return text[i:j].strip()
    return None


def templated(rng: random.Random, n: int, category: str, frames: list[str],
              fills: list[str], key: str) -> list[dict[str, Any]]:
    combos = [(f, x) for f in frames for x in fills]
    rng.shuffle(combos)
    return [{"category": category, "prompt": f.format(**{key: x}), "target_words": None,
             "gold": None} for f, x in combos[:n]]


def split(rows: list[dict[str, Any]], rng: random.Random) -> tuple[list[dict], list[dict]]:
    train, eval_ = [], []
    for cat in COUNTS:
        group = [r for r in rows if r["category"] == cat]
        rng.shuffle(group)
        k = round(len(group) * EVAL_FRACTION)
        eval_ += group[:k]
        train += group[k:]
    return train, eval_


def main() -> None:
    rng = random.Random(SEED)
    n_gsm = COUNTS["maths"] * 7 // 10
    rows = (creative(rng, COUNTS["creative"])
            + _gsm8k(n_gsm, rng) + _math(COUNTS["maths"] - n_gsm, rng)
            + templated(rng, COUNTS["qa"], "qa", QA_FRAMES, QA_TOPICS, "t")
            + templated(rng, COUNTS["roleplay"], "roleplay", RP_OPENERS, RP_CHARACTERS, "c"))
    flagged = [r["prompt"][:80] for r in rows if BLOCKLIST.search(r["prompt"])]
    # Maths word problems sometimes mention "stole a cookie" etc.; drop rather than judge.
    rows = [r for r in rows if not BLOCKLIST.search(r["prompt"])]
    for i, r in enumerate(rows):
        r["id"] = f"{r['category'][:2]}{i:04d}"
    train, eval_ = split(rows, rng)
    lc.write_jsonl(lc.LAB_DIR / "prompts_train.jsonl", train)
    lc.write_jsonl(lc.LAB_DIR / "prompts_eval.jsonl", eval_)
    by = {c: (sum(r["category"] == c for r in train), sum(r["category"] == c for r in eval_))
          for c in COUNTS}
    print(f"train={len(train)} eval={len(eval_)} by_category(train,eval)={by} "
          f"screened_out={len(flagged)}")
    for f in flagged:
        print("  screened:", f)


if __name__ == "__main__":
    main()
