"""Long-system-prompt rule compliance, ported from scripts/tinker/sysrules_eval.py (snapshot 2026-11-25 21:05).

Tasks come from the lab's sysrules_tasks.jsonl (built by the lab script); this file only rebuilds the
checkers from each task's stored parameters. Code-checked rules are scored here; the four tone-style
rules (question_end, no_dates, tone, bullets) need the Haiku judge and are left unscored without it.
If the lab changes rule_pool, re-sync it here (self-test checks every task's rule names resolve).
"""
from __future__ import annotations

import re
from typing import Any, Callable

import judges

COMPETITORS = ["Zentrix", "Plumio", "Corvane", "Daylix"]
JUDGE_SYSTEM = ("You check whether an assistant's reply follows ONE specific rule from its system prompt. "
                "Rule: {rule} Reply with a score from 0 (clearly broken) to 100 (clearly followed).")
Checker = Callable[[str], bool] | None


def _words(a: str) -> list[str]:
    return re.findall(r"[A-Za-z']+", a)


def rule_pool(p: str, prm: dict[str, Any]) -> list[tuple[str, str, Checker]]:
    banned, max_words, sign, ref = prm["banned"], prm["max_words"], prm["sign"], prm["ref"]
    return [
        ("start", f'Begin every reply with exactly "Hello from {p}!" on its own first line.',
         lambda a: a.strip().splitlines()[0].strip() == f"Hello from {p}!" if a.strip() else False),
        ("signoff", f'End every reply with this exact final line: "{sign}"',
         lambda a: a.strip().splitlines()[-1].strip() == sign if a.strip() else False),
        ("banned", f"Never use these words: {', '.join(banned)}.",
         lambda a: not any(re.search(rf"\b{w}\b", a, re.I) for w in banned)),
        ("maxwords", f"Keep every reply under {max_words} words.", lambda a: len(_words(a)) < max_words),
        ("no_excl", 'Never use exclamation marks, except in the required greeting line.',
         lambda a: "!" not in "\n".join(a.strip().splitlines()[1:])),
        ("sections", 'Structure every reply with the two headings "## Answer" and "## Next steps", in that order.',
         lambda a: "## Answer" in a and "## Next steps" in a and a.index("## Answer") < a.index("## Next steps")),
        ("no_urls", "Never include URLs or web addresses.", lambda a: not re.search(r"https?://|www\.|\.com\b", a)),
        ("no_prices", "Never state prices or amounts of money.",
         lambda a: not re.search(r"[$€£]\s?\d|\d+\s?(dollars|euros|pounds|USD|EUR)", a, re.I)),
        ("competitors", f"Never mention competitor products ({', '.join(COMPETITORS)}).",
         lambda a: not any(c.lower() in a.lower() for c in COMPETITORS)),
        ("product_case", f'Always write the product name exactly as "{p}".',
         lambda a: all(m == p for m in re.findall(re.escape(p), a, re.I))),
        ("ref", f'Include a reference code in the form "{ref}-" followed by four digits (e.g. {ref}-1234).',
         lambda a: bool(re.search(rf"\b{ref}-\d{{4}}\b", a))),
        ("no_emoji", "Never use emoji.", lambda a: not re.search("[\U0001F300-\U0001FAFF☀-➿]", a)),
        ("question_end", "Finish the body (before the sign-off) with exactly one clarifying question.", None),
        ("no_dates", "Never promise specific release dates or delivery dates.", None),
        ("tone", "Stay calm and professional; no slang and no jokes.", None),
        ("bullets", "When listing more than two items, use a bulleted list with '- '.", None),
    ]


def checkers(task: dict[str, Any]) -> dict[str, Checker]:
    pool = {r[0]: r[2] for r in rule_pool(task["product"], task["params"])}
    return {n: pool[n] for n in task["rule_names"]}


def score_task(task: dict[str, Any], answer: str, use_judge: bool) -> dict[str, Any]:
    res: dict[str, bool | None] = {n: (bool(fn(answer)) if fn is not None else None)
                                   for n, fn in checkers(task).items()}
    errors = 0
    if use_judge:
        for name in [n for n, v in res.items() if v is None]:
            p, err = judges.call("haiku", task["user"], answer,
                                 JUDGE_SYSTEM.format(rule=task["rules_text"][name]))
            res[name] = None if p is None else p >= 0.5
            errors += err is not None
    known = {k: v for k, v in res.items() if v is not None}
    return {"rules": res, "kept": sum(known.values()), "n": len(known), "all_kept": all(known.values()),
            "tempted_kept": [res.get(t) for t in task["tempted"]], "judge_errors": errors}
