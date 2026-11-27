"""Rule-based reward terms from PLAN.md §4, minus the judge-based quality.

Pure functions over text and token counts so they can be ablated term by term
(RewardConfig toggles) and checked without the network.

  ~/.inkling-abl/tinker_lab/.venv/bin/python scripts/tinker/rewards.py   # runs the inline checks
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable

WORD_RE = re.compile(r"\S+")


@dataclass(frozen=True)
class RewardConfig:
    think_cap: int = 4096
    answer_cap: int = 8192
    # DAPO soft zone starts at 75% of the cap: 3072/4096 in PLAN.md, scaled with the cap.
    overlong_start_frac: float = 0.75
    use_quality: bool = True
    use_loop: bool = True
    use_overlong: bool = True
    use_runaway: bool = True
    use_length: bool = True
    use_brevity: bool = True
    brevity_gate: float = 0.7
    use_diversity: bool = False


@dataclass(frozen=True)
class Sample:
    """One rollout, already split into thinking and answer by the sampler."""
    category: str
    thinking: str
    answer: str
    think_tokens: int
    answer_tokens: int
    think_cap_hit: bool      # thinking never closed within think_cap
    answer_runaway: bool     # answer hit answer_cap without end-of-sampling
    target_words: int | None = None
    gold: str | None = None


@dataclass
class Terms:
    quality: float = 0.0
    loop: float = 0.0
    overlong: float = 0.0
    runaway: float = 0.0
    length: float = 0.0
    brevity: float = 0.0
    diversity: float = 0.0
    flags: dict[str, bool] = field(default_factory=dict)

    @property
    def total(self) -> float:
        return (self.quality + self.loop + self.overlong + self.runaway + self.length
                + self.brevity + self.diversity)


# ---------- loop detection ----------

def has_repeated_span(words: list[str], span: int = 32, times: int = 3) -> bool:
    """True if some `span`-word window occurs at least `times` times (any positions)."""
    if len(words) < span * times:
        return False
    counts = Counter(tuple(words[i:i + span]) for i in range(len(words) - span + 1))
    return max(counts.values()) >= times


def has_frequent_ngram(words: list[str], n: int = 30, times: int = 20) -> bool:
    """30-gram repeated 20x. Word-level, so it is tokenizer-agnostic and unit-checkable."""
    if len(words) < n + times - 1:
        return False
    counts = Counter(tuple(words[i:i + n]) for i in range(len(words) - n + 1))
    return max(counts.values()) >= times


def loop_flags(s: Sample) -> dict[str, bool]:
    # Thinking and answer are checked separately: drafting a story in the thinking and then
    # writing it out in the answer repeats spans across the two, and that is fine.
    parts = [WORD_RE.findall(s.thinking), WORD_RE.findall(s.answer)]
    return {"cap_hit": s.think_cap_hit,
            "span32x3": any(has_repeated_span(w) for w in parts),
            "gram30x20": any(has_frequent_ngram(w) for w in parts)}


def loop_term(s: Sample) -> tuple[float, dict[str, bool]]:
    flags = loop_flags(s)
    return (-1.0 if any(flags.values()) else 0.0), flags


# ---------- length-shaped terms ----------

def overlong_term(think_tokens: int, cap: int, start_frac: float = 0.75) -> float:
    """Linear 0 -> -0.5 between start_frac*cap and cap (DAPO soft overlong punishment)."""
    start = start_frac * cap
    if think_tokens <= start:
        return 0.0
    return -0.5 * min(1.0, (think_tokens - start) / (cap - start))


def runaway_term(answer_runaway: bool) -> float:
    return -1.0 if answer_runaway else 0.0


def length_term(answer: str, target_words: int | None) -> float:
    if not target_words:
        return 0.0
    words = len(WORD_RE.findall(answer))
    return -0.3 * min(1.0, abs(words - target_words) / target_words)


def brevity_terms(think_tokens: list[int], qualities: list[float], gate: float) -> list[float]:
    """+0.1*(1 - rank/(n-1)) by thinking length within the group; shortest gets +0.1.

    Gated on quality so the model can't buy reward by thinking less and failing.
    Ties share the average rank so identical lengths get identical bonuses.
    """
    n = len(think_tokens)
    if n < 2:
        return [0.0] * n
    order = sorted(range(n), key=lambda i: think_tokens[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and think_tokens[order[j + 1]] == think_tokens[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2
        i = j + 1
    return [0.1 * (1 - ranks[k] / (n - 1)) if qualities[k] >= gate else 0.0 for k in range(n)]


# ---------- name diversity ----------

NAME_RE = re.compile(r"(?<=[a-z,;] )([A-Z][a-z]{2,})\b")
NOT_NAMES = {"The", "And", "But", "She", "His", "Her", "They", "This", "That", "When", "Then",
             "Its", "Our", "You", "Your", "There", "What", "Mars", "Earth", "Moon", "Sun",
             "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
             "God", "Mr", "Mrs", "Dear", "English", "French", "Captain", "Doctor"}
# The stock model's favourites in base creative evals (Elias, Mara, Wren dominate).
OVERUSED_NAMES = {"Elias", "Mara", "Wren", "Maren", "Elara", "Tomas", "Pip", "Lumen"}


def main_name(text: str) -> str | None:
    """Most frequent mid-sentence capitalised word: a cheap proxy for the protagonist."""
    c = Counter(m for m in NAME_RE.findall(text) if m not in NOT_NAMES)
    return c.most_common(1)[0][0] if c else None


def diversity_terms(samples: list[Sample]) -> list[float]:
    """-0.1 for a creative answer whose main name repeats another sample's in the group
    or is one of the model's overused names. Group-level, so it rewards spread, not a word."""
    names = [main_name(s.answer) if s.category == "creative" else None for s in samples]
    counts = Counter(n for n in names if n)
    return [-0.1 if n and (counts[n] >= 2 or n in OVERUSED_NAMES) else 0.0 for n in names]


# ---------- quality ----------

def extract_final_answer(text: str) -> str | None:
    """Last \\boxed{...} if present, else the last number in the text."""
    start = text.rfind("\\boxed{")
    if start >= 0:
        depth, i = 1, start + len("\\boxed{")
        for j in range(i, len(text)):
            depth += {"{": 1, "}": -1}.get(text[j], 0)
            if depth == 0:
                return text[i:j].strip()
    nums = re.findall(r"-?\d[\d,]*\.?\d*", text)
    return nums[-1] if nums else None


def _norm_num(x: str) -> str:
    x = x.replace(",", "").replace("$", "").replace("\\%", "").replace("%", "").strip()
    x = re.sub(r"^\\text\{(.*)\}$", r"\1", x)
    try:
        f = float(x)
        return str(int(f)) if f == int(f) else str(f)
    except ValueError:
        return x


def maths_quality(answer: str, gold: str) -> float:
    pred = extract_final_answer(answer)
    return 1.0 if pred is not None and _norm_num(pred) == _norm_num(gold) else 0.0


def heuristic_quality(s: Sample) -> float:
    """PLACEHOLDER for creative/Q&A/roleplay until a judge is wired in.

    It only checks that an answer exists and finishes cleanly; it cannot tell good
    writing from bad, so it must not be used to pick checkpoints.
    """
    words = len(WORD_RE.findall(s.answer))
    if words < 20:
        return 0.2
    ends_clean = s.answer.rstrip()[-1:] in ".!?\"')*”’"
    return 0.8 if ends_clean else 0.5


QualityFn = Callable[[Sample], float]


def default_quality(s: Sample) -> float:
    if s.category == "maths" and s.gold is not None:
        return maths_quality(s.answer, s.gold)
    return heuristic_quality(s)


# ---------- group scoring ----------

def score_group(samples: list[Sample], cfg: RewardConfig,
                quality_fn: QualityFn = default_quality) -> list[Terms]:
    out: list[Terms] = []
    for s in samples:
        t = Terms()
        # PLAN.md: runaway answers are not quality-scored; a capped thinking has no answer.
        scoreable = not s.answer_runaway and not s.think_cap_hit
        if cfg.use_quality and scoreable:
            t.quality = quality_fn(s)
        loop, t.flags = loop_term(s)
        t.flags["runaway"] = s.answer_runaway
        if cfg.use_loop:
            t.loop = loop
        if cfg.use_overlong:
            t.overlong = overlong_term(s.think_tokens, cfg.think_cap, cfg.overlong_start_frac)
        if cfg.use_runaway:
            t.runaway = runaway_term(s.answer_runaway)
        if cfg.use_length and scoreable:
            t.length = length_term(s.answer, s.target_words)
        out.append(t)
    if cfg.use_diversity:
        for t, d in zip(out, diversity_terms(samples)):
            t.diversity = d
    if cfg.use_brevity:
        bonus = brevity_terms([s.think_tokens for s in samples],
                              [t.quality for t in out], cfg.brevity_gate)
        for t, b in zip(out, bonus):
            t.brevity = b
    return out


# ---------- inline checks ----------

def _checks() -> None:
    para = " ".join(f"w{i}" for i in range(32))
    assert has_repeated_span((para + " x " + para + " y " + para).split())
    assert not has_repeated_span((para + " " + para).split())
    assert has_frequent_ngram(("a b c " * 300).split())
    assert not has_frequent_ngram(" ".join(f"t{i}" for i in range(2000)).split())
    assert overlong_term(3000, 4096) == 0.0
    assert abs(overlong_term(3584, 4096) + 0.25) < 1e-9
    assert overlong_term(5000, 4096) == -0.5
    assert abs(overlong_term(1792, 2048) + 0.25) < 1e-9  # scales with the cap
    assert length_term(" ".join(["w"] * 400), 400) == 0.0
    assert abs(length_term(" ".join(["w"] * 200), 400) + 0.15) < 1e-9
    assert length_term(" ".join(["w"] * 2000), 400) == -0.3
    assert maths_quality("so the answer is \\boxed{1,234}", "1234") == 1.0
    assert maths_quality("The total is 42 dollars.", "42") == 1.0
    assert maths_quality("\\boxed{41}", "42") == 0.0
    b = brevity_terms([100, 300, 200, 300], [1.0, 1.0, 1.0, 0.0], 0.7)
    assert abs(b[0] - 0.1) < 1e-9 and b[3] == 0.0 and abs(b[2] - 0.1 * (1 - 1 / 3)) < 1e-9
    cfg = RewardConfig()
    good = Sample("maths", "short", "\\boxed{7}", 500, 20, False, False, gold="7")
    capped = Sample("maths", ("loop " * 50), "", 4096, 0, True, False, gold="7")
    runaway = Sample("creative", "t", "word " * 9000, 900, 8192, False, True, target_words=400)
    g, c, r = score_group([good, capped, runaway], cfg)
    assert g.quality == 1.0 and g.loop == 0.0 and abs(g.brevity - 0.1) < 1e-9
    assert c.loop == -1.0 and c.overlong == -0.5 and c.quality == 0.0
    assert r.runaway == -1.0 and r.quality == 0.0 and r.length == 0.0
    draft = Sample("creative", para + " " + para, para, 200, 40, False, False)
    assert not loop_flags(draft)["span32x3"]  # draft x2 in thinking + once in answer is fine
    looped = Sample("creative", para + " a " + para + " b " + para, "", 200, 0, False, False)
    assert loop_flags(looped)["span32x3"]
    dv = diversity_terms([Sample("creative", "", a, 10, 10, False, False) for a in (
        "then Elias walked, and Elias sang", "then Juno ran, so Juno hid",
        "then Kofi ran, and Kofi hid", "then Kofi sat, and Kofi ate")])
    assert dv == [-0.1, 0.0, -0.1, -0.1], dv
    off = score_group([capped], RewardConfig(use_loop=False, use_overlong=False))[0]
    assert off.total == 0.0
    print("rewards.py checks passed")


if __name__ == "__main__":
    _checks()
