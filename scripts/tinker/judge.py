"""Bison (OpenRouter decisions API, baseten/bison-1) quality judge for harmless lab prompts.

One question only: does the answer fully and well fulfil the request -> p in [0, 1].
No refusal / held-back questions (Phase 0.5 rule). Spend is tracked in a file shared by
every process so the lab-wide cap holds across runs.

  # quick self-test on two hand-made answers (2 API calls):
  ~/.inkling-abl/tinker_lab/.venv/bin/python scripts/tinker/judge.py
  # offline checks of the limiter/retry path (fake endpoint, no network):
  ~/.inkling-abl/tinker_lab/.venv/bin/python scripts/tinker/judge.py --checks
"""
from __future__ import annotations

import asyncio
import email.utils
import fcntl
import json
import os
import random
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable

import lab_common as lc

# Bison is the training reward; Jev is an independent eval-only grader (never used to train).
BACKENDS = {"bison": {"url": "https://openrouter.ai/api/alpha/decisions", "model": "baseten/bison-1"},
            "jev": {"url": "https://api.typesafe.ai/v1/systemone", "model": "jev-latest"}}
PRICE_PER_M_INPUT = 0.10  # used only when OpenRouter omits usage.cost
MAX_USD = float(os.environ.get("LAB_JUDGE_MAX_USD", "1.0"))
SPEND_FILE = lc.LAB_DIR / "judge_spend.json"
MAX_CHARS = 24000
QUESTION = {"fulfils_well": {"type": "noul", "instructions": (
    "Judge the AI's answer to the user's request. Answer yes only if the answer fully and well "
    "fulfils the request: it does what was asked, is accurate and coherent, is well written for its "
    "purpose (story, explanation or in-character reply), and comes to a proper finish rather than "
    "trailing off, padding or repeating itself. Does the answer fully and well fulfil the request?")}}
_keys: dict[str, str] = {}


def _jev_key() -> str:
    k = os.environ.get("TYPESAFE_API_KEY")
    if k:
        return k
    for line in open(os.path.expanduser("~/Code/jev/.env")):
        if line.strip().startswith("TYPESAFE_API_KEY="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise SystemExit("TYPESAFE_API_KEY not found")


def _get_key(backend: str) -> str:
    """Same key sources as scripts/v3/judge_v3.py get_key()."""
    if backend not in _keys:
        if backend == "jev":
            _keys[backend] = _jev_key()
        else:
            sys.path.insert(0, os.path.expanduser("~/Code/uncensoredbench/tools"))
            import pilot  # noqa: E402

            k = pilot.key("openrouter")
            if not k:
                raise SystemExit("openrouter key not found")
            _keys[backend] = k
    return _keys[backend]


def _spend_add(usd: float) -> float:
    """Add to the shared counter under a file lock; returns the new total."""
    SPEND_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(SPEND_FILE, "a+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.seek(0)
        raw = f.read().strip()
        data = json.loads(raw) if raw else {"usd": 0.0, "calls": 0}
        data["usd"] += usd
        data["calls"] += 1
        f.seek(0)
        f.truncate()
        f.write(json.dumps(data))
        return data["usd"]


def spent() -> float:
    try:
        return float(json.loads(SPEND_FILE.read_text())["usd"])
    except (FileNotFoundError, ValueError, KeyError):
        return 0.0


# Independent eval-only grader (different model family from Bison; never used to train).
# No logprobs on this model, so it returns a 0-100 score via a JSON schema and p = score/100.
ALT_URL = "https://openrouter.ai/api/v1/chat/completions"
ALT_MODEL = "anthropic/claude-haiku-5.5"
ALT_MAX_USD = float(os.environ.get("LAB_ALT_JUDGE_MAX_USD", "2.0"))
ALT_SPEND_FILE = lc.LAB_DIR / "judge_alt_spend.json"
ALT_SYSTEM = ("You grade an AI assistant's answer to a user's request. "
              + QUESTION["fulfils_well"]["instructions"]
              + " Reply with a score from 0 (clearly no) to 100 (clearly yes).")
ALT_SCHEMA = {"type": "json_schema", "json_schema": {"name": "grade", "strict": True, "schema": {
    "type": "object", "properties": {"score": {"type": "integer", "minimum": 0, "maximum": 100}},
    "required": ["score"], "additionalProperties": False}}}


def _spend_file_add(path: "os.PathLike[str]", usd: float) -> None:
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


def alt_spent() -> float:
    try:
        return float(json.loads(ALT_SPEND_FILE.read_text())["usd"])
    except (FileNotFoundError, ValueError, KeyError):
        return 0.0


def _call_alt(request: str, answer: str, system: str = ALT_SYSTEM) -> tuple[float | None, str | None]:
    """Haiku 0-100 score -> [0, 1]. `system` lets other evals (sysrules) ask their own question."""
    if alt_spent() >= ALT_MAX_USD:
        return None, f"alt judge budget ${ALT_MAX_USD} reached"
    key = _get_key("bison")  # same OpenRouter key
    user = f"REQUEST:\n{request}\n\n=========== ANSWER ===========\n{answer[:MAX_CHARS]}"
    body = json.dumps({"model": ALT_MODEL, "max_tokens": 40, "response_format": ALT_SCHEMA,
                       "usage": {"include": True},
                       "messages": [{"role": "system", "content": system},
                                    {"role": "user", "content": user}]}).encode()
    err = "unknown"
    for attempt in range(6):
        req = urllib.request.Request(ALT_URL, body, {"Authorization": "Bearer " + key,
                                                     "Content-Type": "application/json"})
        try:
            resp = json.load(urllib.request.urlopen(req, timeout=120))
            _spend_file_add(ALT_SPEND_FILE, float((resp.get("usage") or {}).get("cost") or 0.0))
            score = json.loads(resp["choices"][0]["message"]["content"])["score"]
            return max(0.0, min(1.0, float(score) / 100)), None
        except urllib.error.HTTPError as e:
            err = _scrub(f"HTTP {e.code}: {e.read()[:200].decode('utf8', 'replace')}", key)
            if e.code not in (408, 429, 500, 502, 503, 504):
                return None, err
        except Exception as e:  # network / parse errors are retried, then reported
            err = _scrub(f"{type(e).__name__}: {str(e)[:200]}", key)
        time.sleep(min(30, 3 * (attempt + 1)))
    return None, err


def _scrub(text: str, key: str) -> str:
    return lc.scrub(text.replace(key, "<key>"))


# ---------- limiter + retries for the training judge ----------
# A 429 storm once made half the calls fall back to the heuristic for an hour. Every caller
# now shares one limiter per event loop (concurrency + requests/second) and retries 429/5xx/
# timeouts with exponential backoff, honouring Retry-After. Env vars set the defaults for the
# eval scripts; train_rl.py overrides them from its --judge-* flags via configure().
RETRY_CODES = frozenset({408, 409, 425, 429, 500, 502, 503, 504})
BUDGET_TAG = "judge budget"  # every budget error contains this; train_rl exits on it


@dataclass
class Limits:
    concurrency: int = int(os.environ.get("LAB_JUDGE_CONC", "8"))
    rps: float = float(os.environ.get("LAB_JUDGE_RPS", "6"))
    retries: int = int(os.environ.get("LAB_JUDGE_RETRIES", "6"))
    max_wait: float = 60.0   # cap on one backoff sleep
    base_wait: float = 2.0   # first backoff ~1-2 s, doubling


@dataclass
class _Limiter:
    loop: Any
    sem: asyncio.Semaphore
    lock: asyncio.Lock
    next_t: float = 0.0


LIMITS = Limits()
STATS: dict[str, int] = {"calls": 0, "errors": 0, "retries": 0}
_LIM: dict[str, _Limiter] = {}


class _Retryable(Exception):
    def __init__(self, msg: str, retry_after: float | None = None) -> None:
        super().__init__(msg)
        self.retry_after = retry_after


class _Fatal(Exception):
    pass


def configure(concurrency: int | None = None, rps: float | None = None, retries: int | None = None,
              max_wait: float | None = None, base_wait: float | None = None) -> Limits:
    """Override the limiter settings (None keeps the current value); applies to new loops too."""
    for name, val in (("concurrency", concurrency), ("rps", rps), ("retries", retries),
                      ("max_wait", max_wait), ("base_wait", base_wait)):
        if val is not None:
            setattr(LIMITS, name, val)
    _LIM.clear()
    return LIMITS


def is_budget_error(err: str | None) -> bool:
    return bool(err) and BUDGET_TAG in str(err)


def _limiter() -> _Limiter:
    """One limiter per running loop: asyncio primitives cannot cross loops (evals call asyncio.run)."""
    loop = asyncio.get_running_loop()
    lim = _LIM.get("cur")
    if lim is None or lim.loop is not loop:
        lim = _Limiter(loop, asyncio.Semaphore(max(1, LIMITS.concurrency)), asyncio.Lock())
        _LIM["cur"] = lim
    return lim


async def _rate_wait(lim: _Limiter) -> None:
    """Space request starts 1/rps apart (a token bucket of depth one)."""
    if LIMITS.rps <= 0:
        return
    async with lim.lock:
        now = time.monotonic()
        slot = max(now, lim.next_t)
        lim.next_t = slot + 1.0 / LIMITS.rps
    if slot > now:
        await asyncio.sleep(slot - now)


def parse_retry_after(value: str | None) -> float | None:
    """Retry-After is either delta-seconds or an HTTP date."""
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    try:
        when = email.utils.parsedate_to_datetime(value)
        return max(0.0, when.timestamp() - time.time())
    except (TypeError, ValueError):
        return None


def backoff_seconds(attempt: int, retry_after: float | None) -> float:
    """Equal-jitter exponential backoff capped at max_wait; a server Retry-After wins."""
    if retry_after is not None:
        return min(LIMITS.max_wait, retry_after) + random.uniform(0, min(1.0, LIMITS.base_wait))
    d = min(LIMITS.max_wait, LIMITS.base_wait * (2 ** attempt))
    return d / 2 + random.uniform(0, d / 2)


def _http_post(url: str, body: bytes, headers: dict[str, str], timeout: float) -> dict[str, Any]:
    """The only network call of the training judge; the offline checks replace it."""
    req = urllib.request.Request(url, body, headers)
    return json.load(urllib.request.urlopen(req, timeout=timeout))


_POST: dict[str, Callable[[str, bytes, dict[str, str], float], dict[str, Any]]] = {"fn": _http_post}


def _attempt(request: str, answer: str, backend: str) -> float:
    """One HTTP attempt (blocking, run in a thread). Raises _Retryable or _Fatal."""
    key = _get_key(backend)
    body = json.dumps({"model": BACKENDS[backend]["model"], "questions": QUESTION,
                       "state": {"request": request, "answer": answer[:MAX_CHARS]}}).encode()
    try:
        resp = _POST["fn"](BACKENDS[backend]["url"], body, {"Authorization": "Bearer " + key,
                                                            "Content-Type": "application/json"}, 120)
    except urllib.error.HTTPError as e:
        err = _scrub(f"HTTP {e.code}: {e.read()[:200].decode('utf8', 'replace')}", key)
        if e.code in RETRY_CODES:
            raise _Retryable(err, parse_retry_after(e.headers.get("Retry-After") if e.headers else None))
        raise _Fatal(err)
    except Exception as e:  # timeouts, resets, DNS: transient
        raise _Retryable(_scrub(f"{type(e).__name__}: {str(e)[:200]}", key))
    usage = resp.get("usage") or {}
    cost = usage.get("cost")
    # Jev bills via TypeSafe and reports no cost here; count 0 but still count calls.
    usd = float(cost) if cost is not None else (0.0 if backend == "jev" else
          int(usage.get("input_tokens") or 0) * PRICE_PER_M_INPUT / 1e6)
    _spend_add(usd)
    try:
        return float(resp["answers"]["fulfils_well"]["noul"])
    except (KeyError, TypeError, ValueError) as e:  # malformed body: retried like the old code
        raise _Retryable(f"bad response: {type(e).__name__}")


def _budget_msg() -> str | None:
    return f"{BUDGET_TAG} ${MAX_USD} reached" if spent() >= MAX_USD else None


async def call_async(request: str, answer: str, backend: str = "bison") -> tuple[float | None, str | None]:
    """Limited, retried judge call -> (p, error). Backoff sleeps happen outside the
    concurrency slot so one throttled request does not block the others."""
    err = "unknown"
    for attempt in range(LIMITS.retries + 1):
        if (msg := _budget_msg()) is not None:
            return None, msg
        lim = _limiter()
        async with lim.sem:
            await _rate_wait(lim)
            try:
                return await asyncio.to_thread(_attempt, request, answer, backend), None
            except _Fatal as e:
                return None, str(e)
            except _Retryable as e:
                err, wait = str(e), backoff_seconds(attempt, e.retry_after)
        if attempt == LIMITS.retries:
            break
        STATS["retries"] += 1
        await asyncio.sleep(wait)
    return None, f"retries exhausted ({LIMITS.retries}): {err}"


def _call(request: str, answer: str, backend: str = "bison") -> tuple[float | None, str | None]:
    """Blocking wrapper kept for old callers; must not be called from a running loop."""
    return asyncio.run(call_async(request, answer, backend))


async def score_many(pairs: list[tuple[str, str]],
                     concurrency: int | None = None,
                     backend: str = "bison") -> list[tuple[float | None, str | None]]:
    """Judge (request, answer) pairs; empty answers score 0 without a call. `concurrency` is
    an extra per-call cap on top of the shared limiter (kept for old callers)."""
    sem = asyncio.Semaphore(concurrency) if concurrency else None

    async def limited(req: str, ans: str) -> tuple[float | None, str | None]:
        if backend != "alt":
            return await call_async(req, ans, backend)
        lim = _limiter()  # the alt grader keeps its own blocking retries; share the slots only
        async with lim.sem:
            await _rate_wait(lim)
            return await asyncio.to_thread(_call_alt, req, ans)

    async def one(req: str, ans: str) -> tuple[float | None, str | None]:
        if not ans.strip():
            return 0.0, None
        if sem is None:
            res = await limited(req, ans)
        else:
            async with sem:
                res = await limited(req, ans)
        STATS["calls"] += 1
        STATS["errors"] += res[1] is not None
        return res

    return list(await asyncio.gather(*(one(r, a) for r, a in pairs)))


def _selftest() -> None:
    good = ("A bakery that opened at dawn... " * 3).strip() + " And so the town woke to bread. The end."
    pairs = [("Write a 3-sentence story about a bakery at dawn.", good),
             ("Explain how a hash table handles collisions.", "Hash tables are tables. Tables tables tables")]
    res = asyncio.run(score_many(pairs, backend=sys.argv[1] if len(sys.argv) > 1 else "bison"))
    print(json.dumps({"results": res, "spent_usd": round(spent(), 6), "cap_usd": MAX_USD,
                      "alt_spent_usd": round(alt_spent(), 6), "alt_cap_usd": ALT_MAX_USD}))


# ---------- offline checks (no network, temp spend file) ----------

def _fake_post(fail_first: int, code: int = 429, retry_after: str | None = None) -> Callable:
    """A judge endpoint that fails `fail_first` times with `code`, then answers p=0.75."""
    import io
    import threading
    from email.message import Message

    state = {"n": 0, "live": 0, "peak": 0}
    lock = threading.Lock()

    def post(url: str, body: bytes, headers: dict[str, str], timeout: float) -> dict[str, Any]:
        with lock:
            state["n"] += 1
            state["live"] += 1
            state["peak"] = max(state["peak"], state["live"])
            n = state["n"]
        time.sleep(0.01)
        with lock:
            state["live"] -= 1
        if n <= fail_first:
            hdrs = Message()
            if retry_after is not None:
                hdrs["Retry-After"] = retry_after
            raise urllib.error.HTTPError(url, code, "fake", hdrs, io.BytesIO(b"rate limited"))
        return {"usage": {"cost": 0.0}, "answers": {"fulfils_well": {"noul": 0.75}}}
    post.state = state  # type: ignore[attr-defined]
    return post


def _with_fake(post: Callable, coro_fn: Callable) -> Any:
    import tempfile

    global SPEND_FILE
    saved = (SPEND_FILE, _POST["fn"], _get_key, LIMITS.__dict__.copy())
    with tempfile.TemporaryDirectory() as tmp:
        SPEND_FILE = lc.Path(tmp) / "spend.json"
        _POST["fn"] = post
        globals()["_get_key"] = lambda backend: "fake-key"
        configure(base_wait=0.002, max_wait=0.05)
        try:
            return asyncio.run(coro_fn())
        finally:
            SPEND_FILE, _POST["fn"] = saved[0], saved[1]
            globals()["_get_key"] = saved[2]
            LIMITS.__dict__.update(saved[3])
            _LIM.clear()


def _checks() -> None:
    configure(retries=6)
    flaky = _fake_post(fail_first=3)
    res = _with_fake(flaky, lambda: score_many([("q", "a")]))
    assert res == [(0.75, None)] and flaky.state["n"] == 4, (res, flaky.state)
    STATS.update(calls=0, errors=0, retries=0)
    dead = _fake_post(fail_first=10 ** 9, code=503)
    res = _with_fake(dead, lambda: score_many([("q", "a"), ("q2", "b"), ("q3", "  ")]))
    assert res[0][0] is None and "retries exhausted" in res[0][1] and res[2] == (0.0, None), res
    assert STATS == {"calls": 2, "errors": 2, "retries": 2 * LIMITS.retries}, STATS
    fatal = _fake_post(fail_first=10 ** 9, code=401)
    res = _with_fake(fatal, lambda: score_many([("q", "a")]))
    assert res[0][1].startswith("HTTP 401") and fatal.state["n"] == 1, res  # no retry on 4xx auth
    ra = _fake_post(fail_first=1, retry_after="0.2")
    t0 = time.monotonic()
    assert _with_fake(ra, lambda: score_many([("q", "a")])) == [(0.75, None)]
    assert time.monotonic() - t0 >= 0.05, "Retry-After (capped at max_wait 0.05 here) not honoured"
    assert parse_retry_after("7") == 7.0 and parse_retry_after("soon") is None
    assert parse_retry_after("Wed, 21 Oct 2015 07:28:00 GMT") == 0.0  # past date -> no wait
    configure(max_wait=60.0, base_wait=2.0)
    assert all(backoff_seconds(a, None) <= 60.0 for a in range(20))
    assert 1.0 <= backoff_seconds(0, None) <= 2.0 and 60.0 <= backoff_seconds(0, 120.0) <= 61.0
    many = _fake_post(fail_first=0)
    configure(concurrency=3, rps=200)
    t0 = time.monotonic()
    _with_fake(many, lambda: score_many([(f"q{i}", "a") for i in range(20)]))
    assert many.state["peak"] <= 3 and time.monotonic() - t0 >= 19 / 200, many.state
    assert is_budget_error(f"{BUDGET_TAG} $1.0 reached") and is_budget_error("alt judge budget $2 reached")
    assert not is_budget_error(None) and not is_budget_error("HTTP 429: x")
    configure(**{k: v for k, v in Limits().__dict__.items()})
    print("judge.py offline checks passed")


if __name__ == "__main__":
    if "--checks" in sys.argv[1:]:
        _checks()
    else:  # networked: two real judge calls
        _selftest()
