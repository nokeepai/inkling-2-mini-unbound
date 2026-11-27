"""Metric functions ported verbatim from the Tinker lab (scripts/tinker/rewards.py, eval.py, rp.py,
code_eval.py) so numbers here match the lab's definitions. Pure functions, stdlib only.
"""
from __future__ import annotations

import math
import re
import statistics
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import mathcheck

WORD_RE = re.compile(r"\S+")
CODE_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.S)

# ---------- loops (rewards.py) ----------


def has_repeated_span(words: list[str], span: int = 32, times: int = 3) -> bool:
    """True if some `span`-word window occurs at least `times` times (any positions)."""
    if len(words) < span * times:
        return False
    counts = Counter(tuple(words[i:i + span]) for i in range(len(words) - span + 1))
    return max(counts.values()) >= times


def has_frequent_ngram(words: list[str], n: int = 30, times: int = 20) -> bool:
    if len(words) < n + times - 1:
        return False
    counts = Counter(tuple(words[i:i + n]) for i in range(len(words) - n + 1))
    return max(counts.values()) >= times


def text_loops(text: str) -> bool:
    words = WORD_RE.findall(text)
    return has_repeated_span(words) or has_frequent_ngram(words)


def loop_flags(thinking: str, answer: str, cap_hit: bool) -> dict[str, bool]:
    """The lab's flags (cap_hit, span32x3, gram30x20; thinking and answer checked separately so a
    drafted story repeated in the answer is fine), plus per-channel loop flags."""
    parts = [WORD_RE.findall(thinking), WORD_RE.findall(answer)]
    return {"cap_hit": cap_hit,
            "span32x3": any(has_repeated_span(w) for w in parts),
            "gram30x20": any(has_frequent_ngram(w) for w in parts),
            "loop_think": has_repeated_span(parts[0]) or has_frequent_ngram(parts[0]),
            "loop_answer": has_repeated_span(parts[1]) or has_frequent_ngram(parts[1])}


LAB_LOOP_KEYS = ("cap_hit", "span32x3", "gram30x20")

# ---------- maths (rewards.py) ----------


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


def maths_number_quality(answer: str, gold: str) -> float:
    """accuracy_number: strict, or else the one number the final answer states equals a numeric gold
    (units, currency, \\text{}, separators and trailing words ignored; two different numbers never pass).
    Non-numeric golds keep the strict result. Strict stays the lab's metric; this one is reported beside it."""
    if maths_quality(answer, gold):
        return 1.0
    g = mathcheck.gold_number(gold)
    v = mathcheck.number_value(extract_final_answer(answer))
    return 1.0 if g is not None and v is not None and mathcheck.same_number(v, g) else 0.0

# ---------- creative names (eval.py's version, used for its reports) ----------


NAME_RE = re.compile(r"(?<=[a-z,;] )([A-Z][a-z]{2,})\b")
NOT_NAMES = {"The", "And", "But", "She", "His", "Her", "They", "This", "That", "When", "Then",
             "Its", "Our", "You", "Your", "There", "What", "Mars", "Earth", "Moon", "Sun",
             "Monday", "Sunday", "God", "Mr", "Mrs", "Dear", "I'm", "English", "French"}


def creative_names(answers: list[str], top: int = 10) -> list[tuple[str, int]]:
    """Capitalised mid-sentence words, counted once per story (name-collapse proxy)."""
    c: Counter[str] = Counter()
    for a in answers:
        c.update({m for m in NAME_RE.findall(a) if m not in NOT_NAMES})
    return c.most_common(top)

# ---------- roleplay (rp.py) ----------


def ngram_set(words: list[str], n: int = 12) -> set[tuple[str, ...]]:
    return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}


def cross_turn_repeat(reply: str, previous: list[str], n: int = 12, thresh: float = 0.2) -> bool:
    """True when >= thresh of the reply's n-grams already appeared in earlier replies."""
    mine = ngram_set(WORD_RE.findall(reply), n)
    if not mine or not previous:
        return False
    seen: set[tuple[str, ...]] = set()
    for p in previous:
        seen |= ngram_set(WORD_RE.findall(p), n)
    return len(mine & seen) / len(mine) >= thresh

# ---------- code (code_eval.py) ----------


def run_tests(answer: str, tests: list[str], timeout: int = 10) -> tuple[bool, str]:
    """Last ```python block + the problem's asserts, fresh `python -I` in a temp dir."""
    blocks = CODE_RE.findall(answer)
    if not blocks:
        return False, "no code block"
    return run_program(blocks[-1] + "\n\n" + "\n".join(tests) + "\n", timeout)


def last_code_block(answer: str) -> str | None:
    """The block run_tests would run: the last ```python (or bare ```) block."""
    blocks = CODE_RE.findall(answer)
    return blocks[-1] if blocks else None


def run_program(program: str, timeout: int = 10) -> tuple[bool, str]:
    """The code suites' sandbox: fresh `python -I` in a temp dir, pass = exit code 0."""
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "solution.py"
        f.write_text(program)
        try:
            res = subprocess.run([sys.executable, "-I", str(f)], cwd=d, capture_output=True,
                                 text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return False, "timeout"
    return res.returncode == 0, res.stderr[-300:]

# ---------- aggregation helpers ----------


def median(xs: list[float]) -> float:
    return float(statistics.median(xs)) if xs else 0.0


def rate(rows: list[dict[str, Any]], key: str) -> float:
    return sum(bool(r[key]) for r in rows) / len(rows) if rows else 0.0


def judge_block(rows: list[dict[str, Any]], prefixes: tuple[str, ...] = ("judge", "alt")) -> dict[str, Any]:
    """eval.py's judge aggregation: mean p and share >= 0.5 over judged rows."""
    m: dict[str, Any] = {}
    for prefix in prefixes:
        jq = [r[f"{prefix}_p"] for r in rows if r.get(f"{prefix}_p") is not None]
        if jq:
            m[f"{prefix}_mean_p"] = sum(jq) / len(jq)
            m[f"{prefix}_good_rate"] = sum(p >= 0.5 for p in jq) / len(jq)
            m[f"{prefix}_n"] = len(jq)
    return m


def category_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """eval.py category_metrics + separate thinking/answer loop rates."""
    m: dict[str, Any] = {
        "n": len(rows),
        "think_cap_hit": rate(rows, "think_cap_hit"),
        "answer_runaway": rate(rows, "answer_runaway"),
        "loop_any": rate(rows, "loop_any"),
        "loop_think": rate(rows, "loop_think"),
        "loop_answer": rate(rows, "loop_answer"),
        "median_think_tokens": median([r["think_tokens"] for r in rows]),
        "median_answer_tokens": median([r["answer_tokens"] for r in rows]),
    }
    cr = [r for r in rows if r.get("target_words")]
    if cr:
        m["length_within_30pct"] = sum(
            abs(r["answer_words"] - r["target_words"]) <= 0.3 * r["target_words"]
            and not r["answer_runaway"] for r in cr) / len(cr)
        m["median_words_over_target"] = median([r["answer_words"] / r["target_words"] for r in cr])
        m["top_names"] = creative_names([r["answer"] for r in cr])
    mx = [r for r in rows if r.get("gold") is not None]
    if mx:
        m["maths_acc"] = sum(r["maths_correct"] for r in mx) / len(mx)
        # Additive: maths_acc keeps its meaning; these name the strict/number pair explicitly. Computed from the
        # saved answer, so --judge-only re-summaries of old runs gain the number reading too.
        m["maths_acc_strict"] = m["maths_acc"]
        m["maths_acc_number"] = sum(maths_number_quality(r.get("answer") or "", r["gold"]) for r in mx) / len(mx)
    m.update(judge_block(rows))
    return m


# ---------- intervals and paired tests (ubench summaries, select_ckpt.py) ----------

Z95 = 1.959964


def wilson(k: int, n: int, z: float = Z95) -> tuple[float, float]:
    """Wilson score interval: honest at small n and at rates near 0 or 1, where the normal interval is not."""
    if n == 0:
        return 0.0, 1.0
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, centre - half), min(1.0, centre + half)


def rate_ci(flags: list[bool]) -> dict[str, float]:
    k, n = sum(bool(f) for f in flags), len(flags)
    lo, hi = wilson(k, n)
    return {"k": k, "n": n, "rate": k / n if n else 0.0, "lo": lo, "hi": hi}


def mean_ci(xs: list[float]) -> dict[str, float]:
    """Normal-approximation 95% interval of a mean (graded scores such as delivery)."""
    n = len(xs)
    if n == 0:
        return {"n": 0, "mean": 0.0, "lo": 0.0, "hi": 0.0}
    m = sum(xs) / n
    se = statistics.stdev(xs) / math.sqrt(n) if n > 1 else 0.0
    return {"n": n, "mean": m, "lo": m - Z95 * se, "hi": m + Z95 * se}


def paired_diff(ref: dict[Any, float], cand: dict[Any, float]) -> dict[str, float] | None:
    """Mean of per-item differences (cand - ref) over shared items, with a 95% interval and z score."""
    keys = [k for k in ref if k in cand]
    if not keys:
        return None
    d = [cand[k] - ref[k] for k in keys]
    n, m = len(d), sum(d) / len(d)
    se = statistics.stdev(d) / math.sqrt(n) if n > 1 else 0.0
    z = m / se if se > 0 else (0.0 if m == 0 else math.copysign(math.inf, m))
    return {"n": n, "diff": m, "lo": m - Z95 * se, "hi": m + Z95 * se, "z": z}


def upper_tail_p(z: float) -> float:
    """P(Z >= z) for a standard normal."""
    return 0.5 * math.erfc(z / math.sqrt(2))


def holm(pvals: dict[str, float], alpha: float) -> set[str]:
    """Holm step-down: the rows whose null is rejected at family-wise level alpha."""
    order = sorted(pvals, key=lambda k: pvals[k])
    rejected: set[str] = set()
    for i, k in enumerate(order):
        if pvals[k] > alpha / (len(order) - i):
            break
        rejected.add(k)
    return rejected


def checks() -> None:
    """The lab's inline checks for the ported functions."""
    para = " ".join(f"w{i}" for i in range(32))
    assert has_repeated_span((para + " x " + para + " y " + para).split())
    assert not has_repeated_span((para + " " + para).split())
    assert has_frequent_ngram(("a b c " * 300).split())
    assert not has_frequent_ngram(" ".join(f"t{i}" for i in range(2000)).split())
    assert maths_quality("so the answer is \\boxed{1,234}", "1234") == 1.0
    assert maths_quality("The total is 42 dollars.", "42") == 1.0
    assert maths_quality("\\boxed{41}", "42") == 0.0
    assert not loop_flags(para + " " + para, para, False)["span32x3"]
    f = loop_flags(para + " a " + para + " b " + para, "", False)
    assert f["span32x3"] and f["loop_think"] and not f["loop_answer"]
    assert cross_turn_repeat(" ".join(f"x{i}" for i in range(40)), [" ".join(f"x{i}" for i in range(40))])
    ok, _ = run_tests("```python\ndef add(a, b):\n    return a + b\n```", ["assert add(1, 2) == 3"])
    bad, _ = run_tests("```python\ndef add(a, b):\n    return a - b\n```", ["assert add(1, 2) == 3"])
    assert ok and not bad
    lo, hi = wilson(0, 10)
    assert lo == 0.0 and 0.27 < hi < 0.29
    lo, hi = wilson(50, 100)
    assert abs(lo - 0.4038) < 1e-3 and abs(hi - 0.5962) < 1e-3
    pd = paired_diff({1: 1.0, 2: 1.0, 3: 0.0}, {1: 0.0, 2: 0.0, 3: 0.0})
    assert pd is not None and pd["n"] == 3 and abs(pd["diff"] + 2 / 3) < 1e-9 and pd["hi"] < 0.5
    assert holm({"a": 0.001, "b": 0.02, "c": 0.5}, 0.05) == {"a", "b"}
    assert abs(upper_tail_p(1.959964) - 0.025) < 1e-4
