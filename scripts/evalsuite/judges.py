"""Quality judges, ported from scripts/tinker/judge.py (eval-only here; nothing trains on them).

  bison: OpenRouter decisions API, baseten/bison-1, p = P(yes) for "fully and well fulfils the request".
  haiku: OpenRouter chat, anthropic/claude-haiku-5.5, 0-100 score via JSON schema, p = score / 100.

Spend caps as in judge.py ($1 Bison, $2 Haiku), shared across processes through locked spend files in
the suite's output root; override with EVALSUITE_JUDGE_MAX_USD / EVALSUITE_ALT_JUDGE_MAX_USD.
EVALSUITE_JUDGE_MOCK=1 returns deterministic scores with no network and no key lookup (self-test).
"""
from __future__ import annotations

import fcntl
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

BISON_URL, BISON_MODEL = "https://openrouter.ai/api/alpha/decisions", "baseten/bison-1"
ALT_URL, ALT_MODEL = "https://openrouter.ai/api/v1/chat/completions", "anthropic/claude-haiku-5.5"
PRICE_PER_M_INPUT = 0.10  # Bison fallback when OpenRouter omits usage.cost
MAX_CHARS = 24000
INSTRUCTIONS = (
    "Judge the AI's answer to the user's request. Answer yes only if the answer fully and well "
    "fulfils the request: it does what was asked, is accurate and coherent, is well written for its "
    "purpose (story, explanation or in-character reply), and comes to a proper finish rather than "
    "trailing off, padding or repeating itself. Does the answer fully and well fulfil the request?")
QUESTION = {"fulfils_well": {"type": "noul", "instructions": INSTRUCTIONS}}
ALT_SYSTEM = ("You grade an AI assistant's answer to a user's request. " + INSTRUCTIONS
              + " Reply with a score from 0 (clearly no) to 100 (clearly yes).")
ALT_SCHEMA = {"type": "json_schema", "json_schema": {"name": "grade", "strict": True, "schema": {
    "type": "object", "properties": {"score": {"type": "integer", "minimum": 0, "maximum": 100}},
    "required": ["score"], "additionalProperties": False}}}
RETRY_CODES = (408, 429, 500, 502, 503, 504)

SPEND_DIR = Path(os.environ.get("EVALSUITE_SPEND_DIR", "~/.inkling-abl/evalsuite")).expanduser()
CAPS = {"bison": float(os.environ.get("EVALSUITE_JUDGE_MAX_USD", "1.0")),
        "haiku": float(os.environ.get("EVALSUITE_ALT_JUDGE_MAX_USD", "2.0")),
        # ubench marks with the benchmark's Bison rubric (about $0.0003 a call); a full run is about $0.5-1
        "ubison": float(os.environ.get("EVALSUITE_UBENCH_MAX_USD", "5.0"))}
CALL_CAPS = {"glm": int(os.environ.get("EVALSUITE_GLM_MAX_CALLS", "400"))}  # Venice reports no cost: cap calls
_key: list[str] = []


def mock_mode() -> bool:
    return os.environ.get("EVALSUITE_JUDGE_MOCK") == "1"


def set_spend_dir(path: Path) -> None:
    global SPEND_DIR
    SPEND_DIR = path


def _spend_file(backend: str) -> Path:
    return SPEND_DIR / f"judge_spend_{backend}.json"


def spent(backend: str) -> float:
    try:
        return float(json.loads(_spend_file(backend).read_text())["usd"])
    except (FileNotFoundError, ValueError, KeyError):
        return 0.0


def _spend_add(backend: str, usd: float) -> None:
    path = _spend_file(backend)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.seek(0)
        raw = f.read().strip()
        data = json.loads(raw) if raw else {"usd": 0.0, "calls": 0}
        data["usd"] += usd
        data["calls"] += 1
        f.seek(0)
        f.truncate()
        f.write(json.dumps(data))


def calls(backend: str) -> int:
    try:
        return int(json.loads(_spend_file(backend).read_text())["calls"])
    except (FileNotFoundError, ValueError, KeyError):
        return 0


def add_spend(backend: str, usd: float) -> None:
    """Public entry for ubench's judge transport (shares the locked spend files)."""
    _spend_add(backend, usd)


def _get_key() -> str:
    """OPENROUTER_API_KEY, else the lab's source (uncensoredbench pilot.key)."""
    if not _key:
        k = os.environ.get("OPENROUTER_API_KEY", "")
        if not k:
            sys.path.insert(0, os.path.expanduser("~/Code/uncensoredbench/tools"))
            import pilot  # type: ignore[import-not-found]  # noqa: E402

            k = pilot.key("openrouter") or ""
        if not k:
            raise SystemExit("OpenRouter key not found (set OPENROUTER_API_KEY)")
        _key.append(k)
    return _key[0]


def _scrub(text: str, key: str) -> str:
    return re.sub(r"(sk-or-|tml-)[A-Za-z0-9_\-]+", "<key>", text.replace(key, "<key>"))


def _post(url: str, body: dict[str, object]) -> tuple[dict[str, object] | None, str | None]:
    key = _get_key()
    data = json.dumps(body).encode()
    err = "unknown"
    for attempt in range(6):
        req = urllib.request.Request(url, data, {"Authorization": "Bearer " + key,
                                                 "Content-Type": "application/json"})
        try:
            return json.load(urllib.request.urlopen(req, timeout=120)), None
        except urllib.error.HTTPError as e:
            err = _scrub(f"HTTP {e.code}: {e.read()[:200].decode('utf8', 'replace')}", key)
            if e.code not in RETRY_CODES:
                return None, err
        except Exception as e:  # network / parse errors are retried, then reported
            err = _scrub(f"{type(e).__name__}: {str(e)[:200]}", key)
        time.sleep(min(30, 3 * (attempt + 1)))
    return None, err


def _mock(request: str, answer: str, system: str | None) -> tuple[float | None, str | None]:
    """Deterministic stand-in: clean, finished answers of 20+ words score high."""
    words = len(answer.split())
    clean = answer.rstrip()[-1:] in ".!?\"')*”’`"
    base = 0.85 if words >= 20 and clean else 0.3
    jitter = (zlib.crc32((request[:200] + answer[:200] + (system or "")).encode()) % 100) / 1000
    return round(base + jitter, 3), None


def call(backend: str, request: str, answer: str, system: str | None = None) -> tuple[float | None, str | None]:
    """(p, error). `system` replaces the Haiku grading prompt (sysrules asks about one rule)."""
    if mock_mode():
        return _mock(request, answer, system)
    if spent(backend) >= CAPS[backend]:
        return None, f"{backend} judge budget ${CAPS[backend]} reached"
    if backend == "bison":
        resp, err = _post(BISON_URL, {"model": BISON_MODEL, "questions": QUESTION,
                                      "state": {"request": request, "answer": answer[:MAX_CHARS]}})
        if resp is None:
            return None, err
        usage = resp.get("usage") or {}
        cost = usage.get("cost")  # type: ignore[union-attr]
        _spend_add("bison", float(cost) if cost is not None
                   else int(usage.get("input_tokens") or 0) * PRICE_PER_M_INPUT / 1e6)  # type: ignore[union-attr]
        try:
            return float(resp["answers"]["fulfils_well"]["noul"]), None  # type: ignore[index]
        except (KeyError, TypeError, ValueError) as e:
            return None, f"parse: {e}"
    user = f"REQUEST:\n{request}\n\n=========== ANSWER ===========\n{answer[:MAX_CHARS]}"
    resp, err = _post(ALT_URL, {"model": ALT_MODEL, "max_tokens": 40, "response_format": ALT_SCHEMA,
                                "usage": {"include": True},
                                "messages": [{"role": "system", "content": system or ALT_SYSTEM},
                                             {"role": "user", "content": user}]})
    if resp is None:
        return None, err
    _spend_add("haiku", float((resp.get("usage") or {}).get("cost") or 0.0))  # type: ignore[union-attr]
    try:
        score = json.loads(resp["choices"][0]["message"]["content"])["score"]  # type: ignore[index]
        return max(0.0, min(1.0, float(score) / 100)), None
    except (KeyError, TypeError, ValueError) as e:
        return None, f"parse: {e}"


def score_many(backend: str, pairs: list[tuple[str, str]],
               concurrency: int = int(os.environ.get("EVALSUITE_JUDGE_CONC", "16"))) -> list[tuple[float | None, str | None]]:
    """Judge (request, answer) pairs; empty answers score 0 without a call (as in judge.py)."""
    def one(pair: tuple[str, str]) -> tuple[float | None, str | None]:
        req, ans = pair
        return (0.0, None) if not ans.strip() else call(backend, req, ans)

    with ThreadPoolExecutor(max(1, concurrency)) as ex:
        return list(ex.map(one, pairs))
