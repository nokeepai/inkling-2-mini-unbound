"""Shared guard rails for the Tinker free lab (PLAN.md Phase 0.5).

Every script imports clients from here so the free-model assertion and the key handling
live in exactly one place. By default nothing may touch a paid model. Paid mode exists only
behind an explicit --paid flag (train_rl.py, eval.py) AND a positive LAB_PAID_MAX_USD; every
paid call is pre-checked against and recorded in a locked spend ledger.
"""
from __future__ import annotations

import fcntl
import json
import os
import re
import time
from pathlib import Path
from typing import Any

FREE_MODEL: str = "thinkingmachines/Inkling-2-Mini:peft:65536:free"
# Tokenizer/renderer lookups want the bare model name; the :peft:...:free suffix is a
# serving variant, not a different vocabulary.
TOKENIZER_MODEL: str = FREE_MODEL.split(":", 1)[0]
CONTEXT_LIMIT: int = 65536
# Paid 64K LoRA tier (docs: models_and_pricing, 2026-10-19). Prices are USD per 1M tokens.
PAID_MODEL: str = "thinkingmachines/Inkling-2-Mini"
PAID_PRICES: dict[str, float] = {"prefill": 0.15, "sample": 0.50, "train": 0.75}
PAID_LEDGER: Path = Path("~/.inkling-abl/tinker_lab/paid_ledger.json").expanduser()
_PAID: dict[str, Any] = {"on": False, "run": None}

LAB_DIR: Path = Path("~/.inkling-abl/tinker_lab").expanduser()
KEY_PATH: Path = Path("~/.inkling-abl/tinker_key").expanduser()
_KEY_RE = re.compile(r"tml-[A-Za-z0-9_\-]+")


def scrub(text: str) -> str:
    """Remove anything that looks like a Tinker key before it reaches a log."""
    return _KEY_RE.sub("tml-[REDACTED]", text)


def assert_free(model: str) -> None:
    """Refuse to proceed unless the model is the exact free ID, or paid mode was enabled."""
    if model == FREE_MODEL:
        return
    if _PAID["on"] and model == PAID_MODEL:
        return
    raise SystemExit(f"refusing: model {model!r} is not the free model {FREE_MODEL!r}")


def paid_max_usd() -> float:
    return float(os.environ.get("LAB_PAID_MAX_USD", "0"))


def enable_paid(run: str) -> str:
    """Switch this process to the paid model. Refuses unless a positive cap is set."""
    if paid_max_usd() <= 0:
        raise SystemExit("refusing --paid: LAB_PAID_MAX_USD is 0 (set it to authorise spend)")
    _PAID.update(on=True, run=run)
    return PAID_MODEL


def default_model() -> str:
    return PAID_MODEL if _PAID["on"] else FREE_MODEL


def _ledger_update(fn: Any) -> dict[str, Any]:
    PAID_LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with open(PAID_LEDGER, "a+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.seek(0)
        raw = f.read().strip()
        data = json.loads(raw) if raw else {"usd": 0.0, "by_run": {}, "calls": 0}
        fn(data)
        f.seek(0)
        f.truncate()
        f.write(json.dumps(data, indent=1))
        return data


def paid_spent() -> float:
    try:
        return float(json.loads(PAID_LEDGER.read_text())["usd"])
    except (FileNotFoundError, ValueError, KeyError):
        return 0.0


def paid_cost(prefill: int = 0, sample: int = 0, train: int = 0) -> float:
    p = PAID_PRICES
    return (prefill * p["prefill"] + sample * p["sample"] + train * p["train"]) / 1e6


def paid_precheck(est_usd: float) -> None:
    """Refuse a paid call whose worst-case estimate would cross the cap. No-op when free."""
    if _PAID["on"] and paid_spent() + est_usd > paid_max_usd():
        raise SystemExit(f"paid cap reached: spent ${paid_spent():.4f} + est ${est_usd:.4f} "
                         f"> LAB_PAID_MAX_USD ${paid_max_usd():.2f}")


def paid_record(usd: float, kind: str) -> None:
    if not _PAID["on"]:
        return
    run = _PAID["run"] or "unknown"

    def add(d: dict[str, Any]) -> None:
        d["usd"] += usd
        d["calls"] += 1
        r = d["by_run"].setdefault(run, {"usd": 0.0})
        r["usd"] += usd
        r[kind] = r.get(kind, 0.0) + usd
    _ledger_update(add)


def load_key() -> None:
    """Put the key in the environment only; it is never printed or written elsewhere."""
    os.environ["TINKER_API_KEY"] = KEY_PATH.read_text().strip()


def service_client() -> Any:
    load_key()
    # Evals of paid-trained checkpoints bill at paid rates but have no --paid flag; the arm
    # script exports LAB_PAID_RUN so their sampling is pre-checked and recorded too.
    if os.environ.get("LAB_PAID_RUN") and not _PAID["on"]:
        enable_paid(os.environ["LAB_PAID_RUN"])
    import tinker

    return tinker.ServiceClient()


def sampling_client(svc: Any, model: str | None = None) -> Any:
    model = model or default_model()
    assert_free(model)
    return svc.create_sampling_client(base_model=model)


def training_client(svc: Any, rank: int, train_attn: bool, train_mlp: bool,
                    train_unembed: bool, model: str | None = None) -> Any:
    model = model or default_model()
    assert_free(model)
    return svc.create_lora_training_client(
        base_model=model, rank=rank, train_attn=train_attn, train_mlp=train_mlp,
        train_unembed=train_unembed,
    )


def renderer_and_tokenizer() -> tuple[Any, Any]:
    from tinker_cookbook.renderers.tml_v0 import TmlV0Renderer
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    tok = get_tokenizer(TOKENIZER_MODEL)
    return TmlV0Renderer(tok), tok


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


class Log:
    """Append-only JSONL + stdout logger that scrubs keys from every line."""

    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, event: str, **fields: Any) -> None:
        row = {"t": round(time.time(), 2), "event": event, **fields}
        line = scrub(json.dumps(row, ensure_ascii=False, default=str))
        print(line, flush=True)
        with self.path.open("a") as f:
            f.write(line + "\n")
