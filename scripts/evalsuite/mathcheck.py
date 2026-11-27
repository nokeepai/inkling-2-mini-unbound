"""Maths answer checks that do not depend on a judge model. Stdlib only, no imports from this folder.

Two independent checks:
  number_value()  the lenient "right number" reading used for accuracy_number (GSM8K, hardmath, main maths rows).
                  The strict matcher (metrics.maths_quality) marks `230\\text{ miles}` wrong against gold 230; this
                  one reads one number regardless of units, currency, \\text{}, thousands separators, percent signs
                  and trailing words, and refuses (None) whenever the answer holds two different numbers, so it
                  can never rescue an answer by picking the right one out of several.
  is_equiv()      the MATH benchmark's own answer equivalence (string normalisation then exact match),
                  re-implemented from hendrycks/math, modeling/math_equivalence.py (MIT), for pub_math500.
                  Boxed-answer extraction follows the same repo's last_boxed_only_string / remove_boxed.
"""
from __future__ import annotations

import re

# ---------- lenient number reading (accuracy_number) ----------

# Thousands separators first, before `\,` is dropped as spacing: 1,234 / 1{,}234 / 1\,234 all mean 1234.
# A comma is a separator only when exactly three digits follow, so "1, 2" or "1,23" stay two numbers.
_THOUSANDS_RE = re.compile(r"(?<=\d)(?:,|\{,\}|\\,)(?=\d{3}(?!\d))")
# Text-like wrappers keep their content as plain words: \text{ miles} -> " miles".
_WRAP_RE = re.compile(r"\\(?:text|textbf|textit|textrm|mathrm|mathbf|mbox|operatorname)\s*\{([^{}]*)\}")
# Exponents on a unit (m^2, \text{cm}^3) are part of the unit, not a second number. Only after a letter or a
# closing brace, so 5^2 still counts as two numbers and is refused.
_UNIT_POWER_RE = re.compile(r"([A-Za-z}])\s*\^\{?\d\}?")
_SPACE_RE = re.compile(r"\\[,;:! ]|\\q?quad|\\left|\\right|\\displaystyle|~")
# Currency and percent signs vanish without a gap so "-\$5" stays one negative number.
_SIGN_RE = re.compile(r"\\\$|\$|\\%|%")
_MIXED_RE = re.compile(r"(?<=\d)\s*(\\[dt]?frac)")
_FRAC_RE = re.compile(r"\\[dt]?frac\s*\{\s*(-?\d+(?:\.\d+)?)\s*\}\s*\{\s*(-?\d+(?:\.\d+)?)\s*\}")
_SLASH_RE = re.compile(r"(?<![\d.])(-?\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)(?![\d.])")
_NUM_RE = re.compile(r"-?(?:\d+(?:\.\d+)?|\.\d+)")
_REL_TOL = 1e-6


def _ratio(m: re.Match[str]) -> str:
    """A fraction becomes its value in plain decimals (never e-notation, which would read as two numbers)."""
    den = float(m.group(2))
    return f"{float(m.group(1)) / den:.15f}".rstrip("0") + " " if den else " "


def _clean(text: str) -> str:
    t = text.replace("\u2212", "-").replace("\u2013", "-")
    t = _THOUSANDS_RE.sub("", t)
    t = _UNIT_POWER_RE.sub(r"\1", t)
    for _ in range(3):  # nested wrappers such as \text{\textbf{5}}
        t = _WRAP_RE.sub(lambda m: f" {m.group(1)} ", t)
    t = _UNIT_POWER_RE.sub(r"\1", t)
    t = _SIGN_RE.sub("", _SPACE_RE.sub(" ", t))
    t = _MIXED_RE.sub(r" \1", t)  # 2\frac{1}{2} reads as two numbers and is refused, not fused into 20.5
    t = _FRAC_RE.sub(_ratio, t)
    return _SLASH_RE.sub(_ratio, t)


def number_value(text: str | None) -> float | None:
    """The single number an answer states, or None when it states none or more than one distinct value.
    With an '=' only the part after the last '=' counts (x = 7, 5 + 2 = 7): that is the stated result."""
    if text is None:
        return None
    t = _clean(text)
    if "=" in t:
        t = t.rsplit("=", 1)[1]
    values = {float(x) for x in _NUM_RE.findall(t)}
    return values.pop() if len(values) == 1 else None


def same_number(a: float, b: float) -> bool:
    return abs(a - b) <= _REL_TOL * max(1.0, abs(b))


def gold_number(gold: str | None) -> float | None:
    """GSM8K-style golds are plain numbers; anything else returns None (the caller then keeps strict)."""
    try:
        return float(str(gold).replace(",", "").strip())
    except ValueError:
        return None

# ---------- MATH equivalence (hendrycks/math, modeling/math_equivalence.py) ----------
# Kept close to the original on purpose, including its quirks (e.g. "k = 5" loses "k =" only when the left side
# has at most two characters, and an exception anywhere falls back to raw string equality), so pub_math500
# scores match the published MATH protocol rather than an improved one.


def _fix_frac_tail(substr: str) -> str | None:
    """\\frac12 -> \\frac{1}{2}, \\frac1{72} -> \\frac{1}{72}; None when the original gives up."""
    if substr[:1] == "{":
        return substr
    if len(substr) < 2:
        return None
    a, b, post = substr[0], substr[1], substr[2:]
    return "{" + a + "}{" + b + "}" + post if b != "{" else "{" + a + "}" + b + post


def _fix_fracs(string: str) -> str:
    parts = string.split("\\frac")
    out = parts[0]
    for substr in parts[1:]:
        fixed = _fix_frac_tail(substr)
        if fixed is None:
            return string
        out += "\\frac" + fixed
    return out


def _fix_a_slash_b(string: str) -> str:
    parts = string.split("/")
    if len(parts) != 2:
        return string
    try:
        a, b = int(parts[0]), int(parts[1])
    except ValueError:
        return string
    return "\\frac{" + str(a) + "}{" + str(b) + "}" if string == f"{a}/{b}" else string


def _remove_right_units(string: str) -> str:
    # The original: "\text{ " only ever occurs (at least in the val set) when describing units.
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        assert len(splits) == 2  # the original asserts; is_equiv turns the failure into raw equality
        return splits[0]
    return string


def _fix_sqrt(string: str) -> str:
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    out = splits[0]
    for split in splits[1:]:
        out += "\\sqrt" + split if split[0] == "{" else "\\sqrt{" + split[0] + "}" + split[1:]
    return out


_STRIP_PAIRS = (("\n", ""), ("\\!", ""), ("\\\\", "\\"), ("tfrac", "frac"), ("dfrac", "frac"),
                ("\\left", ""), ("\\right", ""), ("^{\\circ}", ""), ("^\\circ", ""), ("\\$", ""))


def strip_string(string: str) -> str:
    for old, new in _STRIP_PAIRS:
        string = string.replace(old, new)
    string = _remove_right_units(string)
    string = string.replace("\\%", "").replace(" .", " 0.").replace("{.", "{0.")
    if not string:
        return string
    if string[0] == ".":
        string = "0" + string
    halves = string.split("=")
    if len(halves) == 2 and len(halves[0]) <= 2:
        string = halves[1]
    string = _fix_sqrt(string).replace(" ", "")
    string = _fix_fracs(string)
    if string == "0.5":
        string = "\\frac{1}{2}"
    return _fix_a_slash_b(string)


def is_equiv(pred: str | None, gold: str | None) -> bool:
    if pred is None or gold is None:
        return pred is None and gold is None
    try:
        return strip_string(pred) == strip_string(gold)
    except Exception:  # the original's bare except: any normalisation failure means raw comparison
        return pred == gold


def last_boxed(text: str) -> str | None:
    """Content of the last \\boxed{...} (or \\fbox{...}, or \\boxed X), as MATH's last_boxed_only_string +
    remove_boxed read it; None when there is no complete box."""
    idx = text.rfind("\\boxed")
    if idx < 0:
        idx = text.rfind("\\fbox")
        if idx < 0:
            return None
    head = text[idx:]
    if head.startswith("\\boxed "):  # "\boxed 5": the Minerva / lm-eval form, up to the next "$" or space
        return head[len("\\boxed "):].split("$")[0].strip() or None
    depth = 0
    for i, ch in enumerate(head):
        depth += {"{": 1, "}": -1}.get(ch, 0)
        if ch == "}" and depth == 0:
            box = head[:i + 1]
            start = box.find("{")
            return box[start + 1:-1] if start >= 0 else None
    return None


def math_correct(answer: str, gold: str) -> bool:
    return is_equiv(last_boxed(answer), gold)

# ---------- unit tests (selftest.py runs these; no data, no network) ----------


def _num_ok(boxed: str, gold: str) -> bool:
    v, g = number_value(boxed), gold_number(gold)
    return v is not None and g is not None and same_number(v, g)


def checks() -> None:
    assert _num_ok("230\\text{ miles}", "230") and _num_ok("230 miles per hour", "230")      # units, words
    assert _num_ok("25\\text{ m}^2", "25") and _num_ok("\\text{\\textbf{5}} apples", "5")
    assert _num_ok("\\$1,250", "1250") and _num_ok("\\$18.00 per hour", "18")                 # currency
    assert _num_ok("-\\$5", "-5") and _num_ok("\\$-5", "-5")
    assert _num_ok("1{,}000{,}000", "1000000") and _num_ok("12\\,500", "12500")              # thousands
    assert _num_ok("1,234,567", "1234567")
    assert _num_ok("25\\%", "25") and _num_ok("25%", "25")                                    # percent
    assert _num_ok("\\frac{3}{4}", "0.75") and _num_ok("\\dfrac{3}{4}", "0.75")             # fractions
    assert _num_ok("3/4", "0.75") and _num_ok("-\\frac{1}{2}", "-0.5") and _num_ok("\\frac{-1}{2}", "-0.5")
    assert _num_ok("-12\\text{ degrees}", "-12") and _num_ok("\u221212", "-12")              # negatives
    assert not _num_ok("12", "-12") and not _num_ok("-12", "12")
    assert not _num_ok("231\\text{ miles}", "230") and not _num_ok("\\$1,251", "1250")       # wrong stays wrong
    assert not _num_ok("12 \\text{ or } 15", "12") and not _num_ok("12 \\text{ or } 15", "15")  # two numbers
    assert not _num_ok("3, 4", "3") and not _num_ok("2\\frac{1}{2}", "2.5") and not _num_ok("5^2", "25")
    assert not _num_ok("18 dollars and 50 cents", "18") and number_value("10:30") is None
    assert _num_ok("x = 7", "7") and _num_ok("5 + 2 = 7", "7") and not _num_ok("5 + 2 = 7", "5")
    assert _num_ok("18 \\text{ (18 dollars)}", "18") and number_value("no number here") is None
    assert number_value(None) is None and gold_number("x") is None
    _math_checks()


def _math_checks() -> None:
    assert is_equiv("\\frac{1}{2}", "0.5") and is_equiv("\\dfrac12", "\\frac{1}{2}") and is_equiv("1/2", "\\frac12")
    assert is_equiv("\\left( 3, \\frac{\\pi}{2} \\right)", "(3,\\frac{\\pi}{2})")
    assert is_equiv("10\\%", "10") and is_equiv("x = 5", "5") and is_equiv("90^\\circ", "90")
    assert is_equiv("5\\text{ cm}", "5") and is_equiv("\\sqrt3", "\\sqrt{3}") and is_equiv(".5", "\\frac{1}{2}")
    assert not is_equiv("\\frac{1}{3}", "0.33") and not is_equiv("6", "5") and not is_equiv(None, "5")
    assert last_boxed("so \\boxed{\\frac{1}{2}} and finally \\boxed{\\left( 3, \\frac{\\pi}{2} \\right)}.") \
        == "\\left( 3, \\frac{\\pi}{2} \\right)"
    assert last_boxed("\\fbox{7}") == "7" and last_boxed("\\boxed 7$") == "7" and last_boxed("\\boxed{7") is None
    assert math_correct("The answer is $\\boxed{\\dfrac{3}{4}}$.", "\\frac{3}{4}")
    assert not math_correct("The answer is 3/4.", "\\frac{3}{4}")  # MATH requires a box; none = wrong
