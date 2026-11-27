"""Wide RL prompt file: prompts_train_code.jsonl + the harmless pool rows that train_rl.py's plain
(non --uncensor) path renders with the meaning the row intends.

The pool was built for a GPU-side environment that also renders `system`, `messages` and `prefill`;
the plain path renders only `prompt` (or `turns` without any system). Rows whose meaning depends on
a dropped field are excluded, never rewritten. Harmful rows are dropped right after json-loading,
before anything else looks at them, and are only counted.

  PY=~/.inkling-abl/tinker_lab/.venv/bin/python
  $PY build_wide_train.py            # write LAB/prompts_train_wide.jsonl (leak check runs first)
  $PY build_wide_train.py --check    # offline: leak, near-dup (new rows), render, debug, maths checks
  $PY build_wide_train.py --allow-context          # LAB/prompts_train_wide_ctx.jsonl (see below)
  $PY build_wide_train.py --allow-context --check  # same checks + token match vs the chat template

Optional third source: LAB/prompts_extra_harmless.jsonl (hand-written, same dedupe and checks).

--allow-context keeps harmless rows that need `system` and/or prior turns (train_rl.collect renders
them with rollout.build_chat_prompt): `messages` rows keep system + messages, scripted `turns` rows
keep system + turns (on-policy prefix at train time). As in the pool's own environment, `setup` and
`user_char` are judge-side notes the model never sees, so a row with a system prompt keeps its setup
out of the render. `prefill` rows stay out, the harmless-source allowlist and the leak check are
unchanged, and every system/message/turn text is leak-checked, not just `prompt`. It also reads
LAB/prompts_extra_context.jsonl and LAB/prompts_extra_long.jsonl (system/messages rows) when present.
"""
from __future__ import annotations

import argparse
import ast
import glob
import json
import os
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

import lab_common as lc

HERE = Path(__file__).resolve().parent
POOL = Path("~/.inkling-abl/v3_private/pool_v3/pool.jsonl").expanduser()
BASE = lc.LAB_DIR / "prompts_train_code.jsonl"
OUT = lc.LAB_DIR / "prompts_train_wide.jsonl"
# Hand-written harmless qa/chat/roleplay rows (third source, optional); ids xq/xc/xr####.
EXTRA = lc.LAB_DIR / "prompts_extra_harmless.jsonl"
EXTRA_SOURCE = "new:extra_harmless"
NEAR_DUP_CHARS = 60
EVAL_FILES = ("prompts_eval.jsonl", "prompts_hardmath.jsonl", "prompts_code.jsonl", "rp_eval.jsonl",
              "sysrules_tasks.jsonl", "prompts_veryhard.jsonl")
UBENCH_GLOB = str(Path("~/Code/uncensoredbench/questions/*.jsonl").expanduser())
HARD_TASKS = HERE / "agent_hard_tasks"
TRAIN_TASKS = HERE / "agent_train_tasks"
CATS = frozenset({"debug", "maths", "creative", "qa", "chat", "roleplay"})
# prompts_train_code.jsonl rows whose text equals a prompts_eval.jsonl prompt (cr0148, cr0111):
# template collisions already in every earlier run. Dropped by id so any NEW leak still fails.
KNOWN_BASE_LEAKS = frozenset({"cr0008", "cr0061"})
# Their `setup` only restates the character line already in the prompt (checked by eye).
SETUP_OK_SOURCES = frozenset({"lab_roleplay", "lab_rp_train"})
# Built to probe refusals for the uncensor environment; not lab-harmless even when category is not.
EDGY_SOURCES = frozenset({"pool_creative", "pool_rp"})
TEXT_KEYS = ("prompt", "turns", "user", "system", "wordings", "text")
OUT_KEYS = ("id", "category", "prompt", "target_words", "gold", "tests", "module", "buggy_pass",
            "source", "orig_source", "sub")
# --allow-context only: written to a NEW file so prompts_train_wide.jsonl (W16) is never replaced.
OUT_CTX = lc.LAB_DIR / "prompts_train_wide_ctx.jsonl"
CTX_KEYS = ("system", "messages", "turns")
# Hand-written system/messages rows from another agent; may not exist yet.
EXTRA_CTX = {lc.LAB_DIR / "prompts_extra_context.jsonl": "new:extra_context",
             lc.LAB_DIR / "prompts_extra_long.jsonl": "new:extra_long"}
NEW_SOURCES = frozenset({EXTRA_SOURCE, *EXTRA_CTX.values()})
EXTRA_CATS = CATS - {"debug", "maths"}
TOK_DIR = Path("~/.inkling-abl/v3_private/tok").expanduser()  # official tokenizer + chat_template.jinja


def norm_full(s: str) -> str:
    return " ".join(s.lower().split())


def norm200(s: str) -> str:
    return norm_full(s)[:200]


def load_pool() -> tuple[list[dict[str, Any]], int]:
    """Harmless pool rows and the number of harmful rows dropped unread."""
    rows, n_harm = [], 0
    for line in POOL.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("category") == "harmful":
            n_harm += 1
            continue
        rows.append(r)
    return rows, n_harm


def buggy_source(prompt: str) -> str | None:
    import code_eval as ce

    blocks = ce.CODE_RE.findall(prompt)
    return blocks[-1] if blocks else None


def context_reason(r: dict[str, Any]) -> str | None:
    """Plain-path reasons only: the row needs system/turns/messages the plain path cannot render."""
    msgs = r.get("messages")
    if "turns" in r:
        return "multi_turn_needs_system_setup"
    if r.get("system") or any(m.get("role") != "user" for m in msgs or []):
        return "system_dropped"
    if msgs is not None and msgs != [{"role": "user", "content": r["prompt"]}]:
        return "messages_not_prompt"
    return None


def split_system(r: dict[str, Any]) -> tuple[str | None, list[dict[str, Any]]] | None:
    """(system, messages without a system turn), or None if a system turn is not a single leading
    one or contradicts the row's `system` field. Compares strings only, never prints them."""
    msgs = list(r.get("messages") or [])
    system = r.get("system") or None
    sys_turns = [i for i, m in enumerate(msgs) if m.get("role") == "system"]
    if not sys_turns:
        return system, msgs
    if sys_turns != [0] or (system is not None and msgs[0].get("content") != system):
        return None
    return msgs[0].get("content") or None, msgs[1:]


def context_row(r: dict[str, Any]) -> dict[str, Any] | None:
    """The row with only the context fields train_rl renders (system + messages, or system +
    turns), `prompt` = the final user turn; None if malformed. A `turns` row's messages must be
    just its first scripted turn (the pool's shape), else the two would disagree."""
    import rollout as ro

    split = split_system(r)
    if split is None:
        return None
    system, msgs = split
    out = {k: v for k, v in r.items() if k not in CTX_KEYS}
    if system:
        out["system"] = system
    turns = r.get("turns")
    if turns is not None:
        ok = (isinstance(turns, list) and len(turns) >= 2 and all(isinstance(t, str) and t.strip() for t in turns)
              and msgs in ([], [{"role": "user", "content": turns[0]}]))
        return {**out, "turns": turns, "prompt": turns[0]} if ok else None
    if not msgs:
        msgs = [{"role": "user", "content": r.get("prompt")}]
    try:
        chat = ro.row_chat({"id": r.get("id"), "messages": msgs})
    except (ValueError, AttributeError):
        return None
    out["prompt"] = chat[-1]["content"]
    if len(chat) > 1:  # a lone user turn is just `prompt`; keep the row plain-shaped
        out["messages"] = chat
    return out if out["prompt"].strip() and (system or len(chat) > 1) else None


def exclusion_reason(r: dict[str, Any], allow_context: bool = False) -> str | None:
    """Why the training path cannot use this pool row as it stands, or None. Order matters only
    for which reason a row is counted under. allow_context keeps rows whose system/turns/messages
    build_chat_prompt renders; everything after that point is the same allowlist as before."""
    if "prefill" in r:
        return "prefill_dropped"
    if not allow_context and (why := context_reason(r)):
        return why
    ctx = allow_context and context_reason(r) is not None
    if ctx and context_row(r) is None:
        return "context_malformed"
    # In the pool's own environment setup/user_char only reach the judge; with the system prompt
    # rendered, dropping setup from the model's view keeps the row's meaning.
    if "setup" in r and r.get("source") not in SETUP_OK_SOURCES and not (ctx and r.get("system")):
        return "setup_dropped"
    if r.get("source") in EDGY_SOURCES:
        return "uncensor_pool_source"
    if r.get("category") not in CATS:
        return "unknown_category"
    if r.get("sub") == "chat/false_premise":
        # The judge scores "did it do what was asked", which would pay for going along with the
        # false premise. Leave these out of training until the judge sees the premise.
        return "false_premise_judge_blind"
    if r["category"] == "debug" and not (r.get("tests") and r.get("module") and buggy_source(r["prompt"])):
        return "debug_fields_missing"
    if r["category"] == "maths" and r.get("gold") is None:
        return "maths_no_gold"
    return None


def to_out(r: dict[str, Any], source: str) -> dict[str, Any]:
    """Keep only fields the plain path reads (plus provenance); `messages`/`setup` are dropped
    only after exclusion_reason verified they add nothing to `prompt`."""
    out = {k: r[k] for k in OUT_KEYS + CTX_KEYS if k in r and k not in ("source", "orig_source")}
    out.setdefault("target_words", None)
    out.setdefault("gold", None)
    if "source" in r:
        out["orig_source"] = r["source"]
    out["source"] = source
    return out


def dialog_text(r: dict[str, Any]) -> str:
    """The conversation's user/assistant text (scripted turns for a turns row); `prompt` otherwise."""
    if r.get("turns"):
        return "\n".join(r["turns"])
    if r.get("messages"):
        return "\n".join(m["content"] for m in r["messages"])
    return r["prompt"]


def full_text(r: dict[str, Any]) -> str:
    """Dedupe key: the whole conversation incl. system for a context row; for a plain row exactly
    the old key, norm_full(prompt)."""
    return norm_full("\n".join(filter(None, [r.get("system") or "", dialog_text(r)])))


def plain_row(r: dict[str, Any]) -> dict[str, Any]:
    """Pool row on the plain path: its messages were verified to equal `prompt`, so drop them."""
    return {k: v for k, v in r.items() if k not in CTX_KEYS}


def merge(base: list[dict[str, Any]], pool: list[dict[str, Any]],
          extra: list[tuple[dict[str, Any], str]] = ()) -> tuple[list[dict[str, Any]], Counter]:
    """Base rows first so a pool copy of a lab row loses; dedupe by id and full normalised text.
    Full text, not 200 chars: every mutant of one debug module shares its first 200 chars.
    `extra` is (row, source tag) pairs from the hand-written files."""
    out: list[dict[str, Any]] = []
    dropped: Counter = Counter()
    ids: set[str] = set()
    texts: set[str] = set()
    tagged = ([(r, "lab:prompts_train_code") for r in base] + [(r, "pool:" + r["source"]) for r in pool]
              + list(extra))
    for r, src in tagged:
        if r["id"] in ids:
            dropped[f"dup_id({src.split(':')[0]})"] += 1
        elif full_text(r) in texts:
            dropped[f"dup_text({src.split(':')[0]})"] += 1
        else:
            ids.add(r["id"])
            texts.add(full_text(r))
            out.append(to_out(r, src))
    return out, dropped


# ---------- leak check ----------

def strings(v: Any) -> Iterator[str]:
    if isinstance(v, str):
        yield v
    elif isinstance(v, list):
        for x in v:
            yield from strings(x)
    elif isinstance(v, dict):
        for x in v.values():
            yield from strings(x)


def row_texts(r: dict[str, Any]) -> Iterator[str]:
    for k in TEXT_KEYS:
        yield from strings(r.get(k))


def task_name(d: str) -> str:
    """t01_inventory_ledger / m01_shopping_cart -> inventory_ledger / shopping_cart."""
    return d.split("_", 1)[1] if d[:1] in "tm" and d[1:3].isdigit() else d


def heldout_index() -> dict[str, Any]:
    """ids and 200-char texts of every held-out set; ubench texts are kept hashed-in-set only."""
    ids, texts, counts = set(), set(), Counter()
    for f in EVAL_FILES:
        for r in lc.read_jsonl(lc.LAB_DIR / f):
            ids.add(str(r["id"]))
            texts |= {norm200(t) for t in row_texts(r) if t.strip()}
            counts[f] += 1
    for f in sorted(glob.glob(UBENCH_GLOB)):
        for line in Path(f).read_text().splitlines():
            if line.strip():
                texts |= {norm200(t) for t in row_texts(json.loads(line)) if t.strip()}
                counts["ubench"] += 1
    names, srcs = set(), set()
    for d in sorted(HARD_TASKS.glob("t[0-9][0-9]_*")):
        names |= {task_name(d.name), json.loads((d / "meta.json").read_text()).get("name", "")}
        srcs |= {norm200((d / f).read_text()) for f in ("buggy.py", "fixed.py") if (d / f).exists()}
        counts["agent_hard_tasks"] += 1
    return {"ids": ids, "texts": texts, "names": names - {""}, "srcs": srcs, "counts": dict(counts)}


def context_texts(r: dict[str, Any]) -> list[str]:
    out = [r["system"]] if r.get("system") else []
    out += [m["content"] for m in r.get("messages") or []] + list(r.get("turns") or [])
    return [t for t in out if t.strip()]


def leak_check(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Fails loudly on any shared id / 200-char text / hard-task module or source. Prints ids only."""
    idx = heldout_index()
    hits: Counter = Counter()
    bad: list[str] = []
    for r in rows:
        why = [k for k, hit in (
            ("id", r["id"] in idx["ids"]),
            ("text", norm200(r["prompt"]) in idx["texts"]),
            # Context rows: every system/message/turn text, so a held-out line cannot hide in history.
            ("context_text", any(norm200(t) in idx["texts"] for t in context_texts(r))),
            ("hard_module", r["category"] == "debug" and task_name(r.get("module", "")) in idx["names"]),
            ("hard_source", r["category"] == "debug" and norm200(buggy_source(r["prompt"]) or "") in idx["srcs"]),
        ) if hit]
        hits.update(why)
        if why:
            bad.append(f"{r['id']}:{'+'.join(why)}")
    report = {"rows_checked": len(rows), "heldout_rows": idx["counts"], "heldout_texts": len(idx["texts"]),
              "hits": dict(hits)}
    if bad:
        raise SystemExit(f"LEAK: {len(bad)} rows overlap held-out sets {report} first ids {bad[:20]}")
    return report


def near_dup_check(rows: list[dict[str, Any]]) -> dict[str, int]:
    """New hand-written rows only: fail if one shares its first NEAR_DUP_CHARS normalised chars
    with any other training row or any held-out text (template collisions an exact check misses)."""
    new = [r for r in rows if r["source"] in NEW_SOURCES]
    if not new:
        return {"new_rows_checked": 0}
    def head(r: dict[str, Any]) -> str:
        # dialog_text, not system: shared system boilerplate would flag every context row.
        return norm_full(dialog_text(r))[:NEAR_DUP_CHARS]
    train_heads = Counter(head(r) for r in rows)
    held = {t[:NEAR_DUP_CHARS] for t in heldout_index()["texts"]}
    bad = [f"{r['id']}:{'train' if train_heads[head(r)] > 1 else 'heldout'}" for r in new
           if train_heads[head(r)] > 1 or head(r) in held]
    if bad:
        raise SystemExit(f"NEAR-DUP: {len(bad)} new rows share their first {NEAR_DUP_CHARS} chars {bad[:30]}")
    return {"new_rows_checked": len(new)}


# ---------- offline checks ----------

def render_check(rows: list[dict[str, Any]]) -> dict[str, int]:
    """Exactly the prompt build collect() does before sampling, thinking on and off."""
    for k in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        os.environ[k] = "1"  # tokenizer from the local cache only; never the network
    import rollout as ro
    import rp

    renderer, _tok = lc.renderer_and_tokenizer()
    n = 0
    for r in rows:
        for thinking in (True, False):
            cfg = ro.SampleCfg(thinking=thinking)
            if "turns" in r:  # collect(): turns rows never switch thinking off; empty prefix at k=0
                toks = rp.turn_prompt(renderer, r, r["turns"][:1], [], ro.SampleCfg())
            elif ro.has_context(r):
                toks = ro.build_chat_prompt(renderer, ro.row_chat(r), cfg)
            else:
                toks = ro.build_prompt(renderer, r["prompt"], cfg)
            assert toks and all(isinstance(t, int) for t in toks), r["id"]
            ro.assert_context(toks, cfg)
            n += 1
    return {"renders": n, "rows": len(rows)}


def check_chats(rows: list[dict[str, Any]]) -> list[tuple[str, list[dict[str, str]]]]:
    """(id, chat) per context row; a turns row is checked at its LAST turn with placeholder
    replies, so the system turn, every user turn and the assistant framing all get compared."""
    import rollout as ro
    import rp

    out = []
    for r in rows:
        if "turns" in r:
            replies = [f"Placeholder reply {i}." for i in range(len(r["turns"]) - 1)]
            out.append((r["id"], rp.turn_chat(r.get("system"), r["turns"], replies)))
        elif ro.has_context(r):
            out.append((r["id"], ro.row_chat(r)))
    return out


def template_check(rows: list[dict[str, Any]], n_plain: int = 50) -> dict[str, Any]:
    """Token-for-token: the lab render == the official chat template (TOK_DIR/chat_template.jinja,
    the file the served model uses) tokenised by the official tokenizer.json, both offline.
    Thinking on at several efforts; thinking off = the effort-0 template + the 2-token answer header."""
    import rollout as ro

    if not (TOK_DIR / "chat_template.jinja").exists():
        return {"skipped": f"no {TOK_DIR}/chat_template.jinja"}
    from transformers import AutoTokenizer

    hf = AutoTokenizer.from_pretrained(str(TOK_DIR))
    renderer, tok = lc.renderer_and_tokenizer()
    plain = [r for r in rows if not ro.has_context(r) and "turns" not in r]
    chats = check_chats(rows) + [(r["id"], ro.row_chat(r)) for r in random.Random(0).sample(plain, min(n_plain, len(plain)))]
    n, bad, bpe_only = 0, [], []
    for rid, chat in chats:
        for effort in (0.0, 0.5, 0.6, 0.7, 0.8, 0.9):
            text = hf.apply_chat_template(chat, tokenize=False, reasoning_effort=effort)
            ref = hf.encode(text, add_special_tokens=False)
            ours = ro.build_chat_prompt(renderer, chat, ro.SampleCfg(effort=effort))
            agree = True
            if len(chat) == 1:  # plain row: the unchanged build_prompt path must agree too
                agree = ro.build_prompt(renderer, chat[0]["content"], ro.SampleCfg(effort=effort)) == ours
            if effort == 0.0:
                off = ro.build_chat_prompt(renderer, chat, ro.SampleCfg(thinking=False))
                agree = agree and off == ours + [ro.MESSAGE_MODEL, ro.CONTENT_TEXT]
            n += 1
            # Same string, different BPE split: the Tinker tokenizer and tokenizer.json disagree on a
            # rare word (a tokenizer difference, not a template one); counted, not fatal.
            if agree and ours != ref and tok.decode(ours) == text:
                bpe_only.append(f"{rid}@{effort}")
            elif not agree or ours != ref:
                bad.append(f"{rid}@{effort}")
    if bad:
        raise SystemExit(f"TEMPLATE MISMATCH: {len(bad)}/{n} renders, first {bad[:20]}")
    return {"chats": len(chats), "context_chats": len(chats) - min(n_plain, len(plain)), "renders": n,
            "token_identical": n - len(bpe_only), "string_identical_bpe_differs": bpe_only}


def debug_check(rows: list[dict[str, Any]], n: int = 20) -> dict[str, Any]:
    """On n pool debug rows: the buggy code scores the declared buggy_pass, and the clean
    agent_train_tasks module (formatted as debugpool does) passes every assert."""
    import debugpool

    pool_dbg = [r for r in rows if r["category"] == "debug" and r["source"].startswith("pool:")]
    sample = random.Random(0).sample(pool_dbg, min(n, len(pool_dbg)))
    same_tests = 0
    for r in sample:
        got = debugpool.pass_fraction(buggy_source(r["prompt"]), r["tests"])
        assert abs(got - r["buggy_pass"]) < 1.5e-3, (r["id"], got, r["buggy_pass"])
        assert got < 1.0, r["id"]
        mod = TRAIN_TASKS / r["module"]
        ref = ast.unparse(ast.parse((mod / "module.py").read_text())) + "\n"
        assert debugpool.pass_fraction(ref, r["tests"]) == 1.0, r["id"]
        same_tests += r["tests"] == (mod / "tests.py").read_text()
    return {"debug_rows_checked": len(sample), "tests_identical_to_module": same_tests}


def maths_check(rows: list[dict[str, Any]]) -> dict[str, int]:
    """Every gold round-trips through rewards.maths_quality; count golds that only a boxed
    answer can match (non-numeric), since the last-number fallback cannot."""
    import rewards as rw

    nonnum = 0
    for r in (r for r in rows if r["category"] == "maths"):
        g = r["gold"]
        assert isinstance(g, str) and g.strip(), r["id"]
        assert rw.maths_quality(f"so the answer is \\boxed{{{g}}}.", g) == 1.0, r["id"]
        assert rw.maths_quality("\\boxed{definitely wrong}", g) == 0.0, r["id"]
        nonnum += re.fullmatch(r"-?\d+(\.\d+)?", rw._norm_num(g)) is None
    return {"maths_rows": sum(r["category"] == "maths" for r in rows), "nonnumeric_golds": nonnum}


def structure_check(rows: list[dict[str, Any]], allow_context: bool = False) -> None:
    import rollout as ro

    assert len({r["id"] for r in rows}) == len(rows), "duplicate id"
    assert len({full_text(r) for r in rows}) == len(rows), "duplicate prompt"
    banned = {"prefill", "setup"} | (set() if allow_context else set(CTX_KEYS))
    for r in rows:
        assert r["category"] in CATS and isinstance(r["prompt"], str) and r["prompt"].strip(), r["id"]
        assert not banned & set(r), r["id"]
        assert r["category"] != "harmful" and not r.get("orig_source", "").startswith("pool_"), r["id"]
        assert r.get("source", "").split(":")[-1] not in EDGY_SOURCES, r["id"]
        if "turns" in r:
            assert "messages" not in r and r["prompt"] == r["turns"][0] and len(r["turns"]) >= 2, r["id"]
        elif ro.has_context(r):
            assert ro.row_chat(r)[-1]["content"] == r["prompt"], r["id"]  # raises on a malformed chat


# ---------- main ----------

def extra_ctx_row(r: dict[str, Any], tag: str) -> tuple[dict[str, Any] | None, str]:
    """(row ready for merge, "") or (None, why). Category is checked first, so a harmful or
    unknown row is counted and never looked at further."""
    if r.get("category") not in EXTRA_CATS:
        return None, "harmful_dropped_unread(extra)" if r.get("category") == "harmful" else "bad_category(extra)"
    if "prefill" in r:
        return None, "prefill_dropped(extra)"
    if not isinstance(r.get("id"), str) or not r["id"]:
        return None, "no_id(extra)"
    if r.get("system") or "messages" in r or "turns" in r:
        row = context_row(r)
    else:
        row = plain_row(r) if isinstance(r.get("prompt"), str) and r["prompt"].strip() else None
    return (None, "context_malformed(extra)") if row is None else ({**row, "source": tag}, "")


def load_extra_ctx() -> tuple[list[tuple[dict[str, Any], str]], Counter]:
    """(row, source tag) pairs from the hand-written context files that exist (missing = counted)."""
    out: list[tuple[dict[str, Any], str]] = []
    skipped: Counter = Counter()
    for path, tag in EXTRA_CTX.items():
        if not path.exists():
            skipped[f"missing_file({path.name})"] += 1
            continue
        for r in lc.read_jsonl(path):
            row, why = extra_ctx_row(r, tag)
            if row is None:
                skipped[why] += 1
            else:
                out.append((row, tag))
    return out, skipped


def pool_usable(pool: list[dict[str, Any]], allow_context: bool) -> tuple[list[dict[str, Any]], Counter]:
    excluded = Counter(why for r in pool if (why := exclusion_reason(r, allow_context)))
    usable = [context_row(r) if allow_context and context_reason(r) else plain_row(r)
              for r in pool if exclusion_reason(r, allow_context) is None]
    return usable, excluded


def drop_heldout_context(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Context rows sharing ANY system/message/turn text with a held-out set are left out and
    counted (ids only), not a build failure: short scripted lines ("What happened next?") recur by
    design, and one such line must not block the file. leak_check still runs on what is left and
    fails loudly on anything this misses."""
    texts = heldout_index()["texts"]
    keep, hit = [], []
    for r in rows:
        ctx = "turns" in r or ("messages" in r or bool(r.get("system")))
        if ctx and any(norm200(t) in texts for t in context_texts(r) + [r["prompt"]]):
            hit.append(r)
        else:
            keep.append(r)
    return keep, {"n": len(hit), "by_source": dict(Counter(r["source"] for r in hit)), "ids": [r["id"] for r in hit]}


def newly_kept(rows: list[dict[str, Any]], pool: list[dict[str, Any]]) -> dict[str, Any]:
    """Kept rows the plain build would have excluded, by old reason and by category|source."""
    old = {r["id"]: why for r in pool if (why := context_reason(r))}
    new = [r for r in rows if r["source"].startswith("pool:") and r["id"] in old]
    return {"n": len(new), "by_old_reason": dict(Counter(old[r["id"]] for r in new)),
            "by_category_source": dict(Counter(f"{r['category']}|{r['source']}" for r in new)),
            "turns_rows": sum("turns" in r for r in new), "messages_rows": sum("messages" in r for r in new)}


def build(allow_context: bool, out_path: Path) -> None:
    if allow_context and out_path == OUT:
        raise SystemExit(f"--allow-context must not overwrite {OUT}; use --out")
    pool, n_harm = load_pool()
    usable, excluded = pool_usable(pool, allow_context)
    base = lc.read_jsonl(BASE)
    excluded["known_heldout_leak(lab)"] = sum(r["id"] in KNOWN_BASE_LEAKS for r in base)
    extra = lc.read_jsonl(EXTRA) if EXTRA.exists() else []
    assert all(r.get("source") == EXTRA_SOURCE and r["category"] in EXTRA_CATS
               for r in extra), "extra rows must be qa/chat/roleplay/creative from " + EXTRA_SOURCE
    extra_ctx, skipped_ctx = load_extra_ctx() if allow_context else ([], Counter())
    rows, dups = merge([r for r in base if r["id"] not in KNOWN_BASE_LEAKS], usable,
                       [(r, EXTRA_SOURCE) for r in extra] + extra_ctx)
    rows, overlap = drop_heldout_context(rows) if allow_context else (rows, {"n": 0})
    structure_check(rows, allow_context)
    leaks = leak_check(rows)
    near = near_dup_check(rows)
    lc.write_jsonl(out_path, rows)
    os.chmod(out_path, 0o600)
    report = {"out": str(out_path), "rows": len(rows), "allow_context": allow_context,
              "by_category": dict(Counter(r["category"] for r in rows)),
              "by_source": dict(Counter(r["source"] for r in rows)),
              "by_category_source": dict(Counter(f"{r['category']}|{r['source']}" for r in rows)),
              "extra_rows_read": len(extra), "extra_ctx_rows_read": len(extra_ctx),
              "excluded": {"harmful_dropped_unread": n_harm, **excluded, **dups, **skipped_ctx},
              "leak_check": leaks, "near_dup_check": near, "heldout_context_overlap_dropped": overlap}
    if allow_context:
        report["newly_kept_context"] = newly_kept(rows, pool)
    print(json.dumps(report))


def check(path: Path) -> None:
    import rollout as ro

    rows = lc.read_jsonl(path)
    ctx = any(ro.has_context(r) or "turns" in r for r in rows)
    structure_check(rows, ctx)
    out = {"leak_check": leak_check(rows), "near_dup_check": near_dup_check(rows), "render": render_check(rows),
           "debug": debug_check(rows), "maths": maths_check(rows)}
    if ctx:
        out["template"] = template_check(rows)
    print(json.dumps({"check": "passed", "file": str(path), **out}))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true", help="offline checks of the written file")
    ap.add_argument("--allow-context", action="store_true",
                    help="keep harmless system/messages/turns rows; writes --out (default prompts_train_wide_ctx)")
    ap.add_argument("--out", default="", help="output path (lab-relative ok)")
    a = ap.parse_args()
    path = (lc.LAB_DIR / a.out) if a.out else (OUT_CTX if a.allow_context else OUT)
    check(path) if a.check else build(a.allow_context, path)


if __name__ == "__main__":
    main()
