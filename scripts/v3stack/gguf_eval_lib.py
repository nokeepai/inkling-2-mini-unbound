"""Helpers for gguf_eval.sh / gguf_eval_all.sh (stdlib only). Numbers only: no prompt or answer text is printed.

  plan       --gguf F --free-mib N [--np N --ctx N]   -> "NP CTX NEED_MIB" (exit 2: does not fit fully on the GPU)
  probe      --base http://127.0.0.1:PORT --out probe.json [--strict]
  brief      --summary <tag>/<suite>/summary.json      -> one line of headline numbers
  sanity     --summary <tag>/pub_gsm8k/summary.json --min 0.5   (exit 1 below the floor)
  table      --root /root/gguf_evals --tags a,b,c [--tsv out.tsv]
  types      --gguf F                                  -> tensor-type histogram
  keep-flags --gguf PRIME_SUBQ8.gguf                   -> the --tensor-type overrides that file was built with
  types-diff --a F --b G                               -> exit 1 if any tensor's type differs (names + types only)
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import sys
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any, BinaryIO, Callable

HERE = Path(__file__).resolve().parent
EVALSUITE = Path(os.environ.get("EVALSUITE_DIR", HERE.parent / "evalsuite"))

# ---------------------------------------------------------------- VRAM plan

# Per-token KV of the 4 global-attention layers, counted once per loop pass (layers 11 and 17 sit in the looped
# core, 3 passes: 1 + 3 + 3 + 1 = 8 layer instances x 2 (K,V) x 8 kv heads x 128 x 2 bytes = 32 KiB). Upper bound:
# if the loop fix shares KV across passes the real figure is lower, which only adds headroom.
KV_KIB_PER_TOKEN = 32
# Sliding-window layers (window 1024): 40 layer instances x 4 KiB per token, (1024 + ubatch 128) tokens per slot.
SWA_MIB_PER_SLOT = 180
OVERHEAD_MIB = 1500   # CUDA context, compute buffers at -ub 128, output buffer
MARGIN_MIB = 400      # the estimates are unmeasured for v3 files; an OOM mid-suite costs more than a smaller pool
# Largest first. Agent turns need prompt + 4,096 + 8,192 tokens of room; the unified KV pool lets one long turn
# borrow from idle slots, so a pool of ~8k tokens per slot is enough for the observed prompt sizes.
CANDIDATES = ((16, 196608), (16, 131072), (12, 98304), (8, 65536), (6, 65536), (4, 49152))


def need_mib(file_mib: int, np_: int, ctx: int) -> int:
    return file_mib + ctx * KV_KIB_PER_TOKEN // 1024 + np_ * SWA_MIB_PER_SLOT + OVERHEAD_MIB + MARGIN_MIB


def cmd_plan(a: argparse.Namespace) -> int:
    file_mib = os.path.getsize(a.gguf) // (1 << 20)
    if a.np and a.ctx:
        cands: tuple[tuple[int, int], ...] = ((a.np, a.ctx),)
    else:
        cands = CANDIDATES
    for np_, ctx in cands:
        n = need_mib(file_mib, np_, ctx)
        if n <= a.free_mib:
            print(np_, ctx, n)
            return 0
    smallest = need_mib(file_mib, *cands[-1])
    print(f"does not fit fully on the GPU: file {file_mib} MiB, need >= {smallest} MiB with np {cands[-1][0]} "
          f"ctx {cands[-1][1]}, free {a.free_mib} MiB (partial CPU offload would not be comparable)", file=sys.stderr)
    return 2

# ---------------------------------------------------------------- probe


def post(base: str, path: str, body: dict[str, Any], timeout: float = 600) -> dict[str, Any]:
    req = urllib.request.Request(base + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def stream(base: str, path: str, body: dict[str, Any], timeout: float = 600) -> list[dict[str, Any]]:
    """SSE events of a streamed request: the suite's client streams, so the probe reads the same shape."""
    req = urllib.request.Request(base + path, json.dumps({**body, "stream": True}).encode(),
                                 {"Content-Type": "application/json"})
    events = []
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for line in r:
            line = line.strip()
            if line.startswith(b"data:") and line[5:].strip() != b"[DONE]":
                events.append(json.loads(line[5:]))
    return events


def joined(events: list[dict[str, Any]], *fields: str) -> str:
    out = []
    for ev in events:
        for ch in ev.get("choices") or []:
            src = ch.get("delta") or ch
            out += [src.get(f) or "" for f in fields]
    return "".join(out)


def finish_of(events: list[dict[str, Any]]) -> str | None:
    return next((ch.get("finish_reason") for ev in events for ch in ev.get("choices") or []
                 if ch.get("finish_reason")), None)


def guarded(fn: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    try:
        return fn()
    except (OSError, urllib.error.URLError, ValueError, KeyError, IndexError) as e:
        return {"ok": False, "error": f"{type(e).__name__}: {str(e)[:160]}"}


def effort_check(base: str) -> dict[str, Any]:
    """The rendered prompt must carry the effort line; the suite runs with --skip-probe, so this is the only check."""
    msgs = [{"role": "user", "content": "hi"}]
    p = {e: post(base, "/apply-template", {"messages": msgs, "chat_template_kwargs": {"reasoning_effort": e}})["prompt"]
         for e in (0, 0.9)}
    rendered = "Thinking effort level: 0.9" in p[0.9]
    return {"ok": rendered and p[0] != p[0.9], "differs": p[0] != p[0.9], "level_0.9_rendered": rendered}


def split_check(base: str) -> dict[str, Any]:
    """Thinking must come back in reasoning_content (llama-server --reasoning-format auto, Inkling PEG parser)."""
    body = {"model": "x", "messages": [{"role": "user", "content": "What is 17 * 23? Answer briefly."}],
            "max_tokens": 3000, "temperature": 0, "seed": 1, "chat_template_kwargs": {"reasoning_effort": 0.9}}
    ev = stream(base, "/v1/chat/completions", body)
    think, ans = joined(ev, "reasoning_content", "reasoning"), joined(ev, "content")
    markers = "<|" in ans or "<|" in think
    usage = next((e["usage"] for e in reversed(ev) if e.get("usage")), {})
    return {"ok": bool(think) and bool(ans) and not markers, "think_chars": len(think), "answer_chars": len(ans),
            "markers_in_text": markers, "has_391": "391" in ans, "finish": finish_of(ev), "chunks": len(ev),
            "completion_tokens": usage.get("completion_tokens")}


def forced_check(base: str) -> dict[str, Any]:
    """The client's post-cap answer: raw prompt (template port) to /v1/completions; special tokens must parse."""
    sys.path.insert(0, str(EVALSUITE))
    import template as tpl  # noqa: E402  (evalsuite's port of the official chat template)
    msgs = [{"role": "user", "content": "What is 17 * 23? Answer briefly."}]
    prompt = tpl.forced_answer_prompt(msgs, 0.9, "17 * 23 = 391.")
    tok = post(base, "/tokenize", {"content": prompt})["tokens"]
    ev = stream(base, "/v1/completions", {"model": "x", "prompt": prompt, "max_tokens": 64, "temperature": 0,
                                          "seed": 1})
    text = joined(ev, "text")
    return {"ok": bool(text.strip()) and "<|" not in text, "prompt_tokens": len(tok), "prompt_chars": len(prompt),
            "answer_chars": len(text), "has_391": "391" in text, "finish": finish_of(ev)}


def cmd_probe(a: argparse.Namespace) -> int:
    res = {"effort": guarded(lambda: effort_check(a.base)), "split": guarded(lambda: split_check(a.base)),
           "forced": guarded(lambda: forced_check(a.base))}
    Path(a.out).write_text(json.dumps(res, indent=1))
    print("probe " + " ".join(f"{k}={'ok' if v.get('ok') else 'FAIL'}" for k, v in res.items()) + " "
          + json.dumps({k: {x: y for x, y in v.items() if x != "ok"} for k, v in res.items()}))
    bad = [k for k, v in res.items() if not v.get("ok")]
    return 1 if (a.strict and bad) else 0

# ---------------------------------------------------------------- summaries


def load_agg(path: str | Path) -> dict[str, float]:
    agg = json.loads(Path(path).read_text())["agg"]
    return {k: v["mean"] for k, v in agg.items() if isinstance(v, dict) and v.get("mean") is not None}


# (summary key, short label): agent stall numbers first, then the sanity suites' scores
BRIEF_KEYS = (("solve_rate", "solve"), ("no_code_rate", "no_code"), ("cap_hit_per_turn", "cap_hit"),
              ("mean_total_tokens", "mean_tokens"), ("mean_turns_to_solve", "turns_to_solve"),
              ("turns_total", "turns"), ("stops.error", "err_stops"), ("stops.context", "ctx_stops"),
              ("all.maths_acc_number", "acc_number"), ("all.maths_acc", "acc_strict"),
              ("all.think_cap_hit", "cap_hit"), ("pass_at_1", "pass@1"), ("cap_hit", "cap_hit"),
              ("n_ok", "n_ok"), ("n", "n"), ("n_failed", "n_failed"), ("wall_seconds", "wall_s"))


def brief_line(agg: dict[str, float]) -> str:
    seen: set[str] = set()
    parts = []
    for key, label in BRIEF_KEYS:
        if key in agg and label not in seen:
            seen.add(label)
            parts.append(f"{label}={agg[key]:.4g}")
    return " ".join(parts)


def cmd_brief(a: argparse.Namespace) -> int:
    try:
        print(brief_line(load_agg(a.summary)))
    except (OSError, ValueError, KeyError) as e:
        print(f"no summary ({type(e).__name__})")
        return 1
    return 0


def cmd_sanity(a: argparse.Namespace) -> int:
    agg = load_agg(a.summary)
    acc = agg.get("all.maths_acc_number", agg.get("all.maths_acc"))
    print(f"acc_number={acc}")
    return 0 if acc is not None and acc >= a.min else 1


TABLE_ROWS = (("agent_hard_noforce", "solve_rate"), ("agent_hard_noforce", "no_code_rate"),
              ("agent_hard_noforce", "cap_hit_per_turn"), ("agent_hard_noforce", "mean_total_tokens"),
              ("agent_hard_noforce", "turns_total"), ("agent_hard_noforce", "stops.error"),
              ("pub_gsm8k", "all.maths_acc_number"), ("pub_gsm8k", "all.think_cap_hit"),
              ("pub_humaneval", "pass_at_1"), ("pub_humaneval", "cap_hit"))


def cmd_table(a: argparse.Namespace) -> int:
    tags = [t for t in a.tags.split(",") if t]
    root = Path(a.root)

    def cell(tag: str, suite: str, key: str) -> str:
        f = root / tag / suite / "summary.json"
        if not f.exists():
            return "-"
        v = load_agg(f).get(key)
        return "-" if v is None else (f"{v:.0f}" if abs(v) >= 100 else f"{v:.3g}")
    lines = ["metric\t" + "\t".join(tags)]
    lines += [f"{s}.{k}\t" + "\t".join(cell(t, s, k) for t in tags) for s, k in TABLE_ROWS]
    out = "\n".join(lines) + "\n"
    if a.tsv:
        Path(a.tsv).write_text(out)
    sys.stdout.write(out)
    return 0

# ---------------------------------------------------------------- GGUF header (tensor types)

GGML_TYPES = {0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0", 9: "Q8_1", 10: "Q2_K",
              11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K", 15: "Q8_K", 16: "IQ2_XXS", 17: "IQ2_XS",
              18: "IQ3_XXS", 19: "IQ1_S", 20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS", 24: "I8",
              25: "I16", 26: "I32", 27: "I64", 28: "F64", 29: "IQ1_M", 30: "BF16", 34: "TQ1_0", 35: "TQ2_0",
              39: "MXFP4"}
SCALAR = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}


def rd(f: BinaryIO, fmt: str) -> Any:
    return struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]


def rd_str(f: BinaryIO) -> str:
    return f.read(rd(f, "<Q")).decode("utf-8", "replace")


def skip_value(f: BinaryIO, vtype: int) -> None:
    if vtype in SCALAR:
        f.seek(struct.calcsize(SCALAR[vtype]), 1)
    elif vtype == 8:
        f.seek(rd(f, "<Q"), 1)
    elif vtype == 9:
        etype, n = rd(f, "<I"), rd(f, "<Q")
        if etype in SCALAR:
            f.seek(n * struct.calcsize(SCALAR[etype]), 1)
        else:
            for _ in range(n):
                skip_value(f, etype)
    else:
        raise ValueError(f"unknown GGUF value type {vtype}")


def tensor_types(path: str) -> dict[str, str]:
    """name -> ggml type name, read from the header only (no tensor data)."""
    with open(path, "rb") as f:
        if f.read(4) != b"GGUF":
            raise ValueError(f"{path}: not a GGUF file")
        rd(f, "<I")
        n_tensors, n_kv = rd(f, "<Q"), rd(f, "<Q")
        for _ in range(n_kv):
            rd_str(f)
            skip_value(f, rd(f, "<I"))
        out = {}
        for _ in range(n_tensors):
            name = rd_str(f)
            f.seek(8 * rd(f, "<I"), 1)
            t = rd(f, "<I")
            rd(f, "<Q")
            out[name] = GGML_TYPES.get(t, str(t))
        return out


def cmd_types(a: argparse.Namespace) -> int:
    print(json.dumps(dict(Counter(tensor_types(a.gguf).values()).most_common())))
    return 0


# Attention tensors gguf_v3.sh keeps at q8_0 below Q8_0, plus attn_rel (v3/PLAN.md: "keep attn_rel at q8_0 too").
KEEP_CANDIDATES = ("attn_q", "attn_k", "attn_v", "attn_output", "attn_gate", "attn_rel")


def cmd_keep_flags(a: argparse.Namespace) -> int:
    """Read which attention tensors the Prime file holds at Q8_0, so the stock quant gets the same overrides."""
    types = tensor_types(a.gguf)
    flags = []
    for c in KEEP_CANDIDATES:
        ts = [t for n, t in types.items() if n.endswith(f".{c}.weight")]
        if ts and all(t == "Q8_0" for t in ts):
            flags += ["--tensor-type", f"{c}=q8_0"]
    print(" ".join(flags))
    return 0


def cmd_types_diff(a: argparse.Namespace) -> int:
    ta, tb = tensor_types(a.a), tensor_types(a.b)
    diff = sorted(n for n in set(ta) | set(tb) if ta.get(n) != tb.get(n))
    print(f"tensors a={len(ta)} b={len(tb)} differing={len(diff)}")
    for n in diff[:12]:
        print(f"  {n}: {ta.get(n)} vs {tb.get(n)}")
    return 1 if diff else 0

# ---------------------------------------------------------------- main


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--gguf", required=True)
    p.add_argument("--free-mib", type=int, required=True)
    p.add_argument("--np", type=int, default=0)
    p.add_argument("--ctx", type=int, default=0)
    p = sub.add_parser("probe")
    p.add_argument("--base", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--strict", action="store_true")
    for name in ("brief", "sanity"):
        p = sub.add_parser(name)
        p.add_argument("--summary", required=True)
        p.add_argument("--min", type=float, default=0.5)
    p = sub.add_parser("table")
    p.add_argument("--root", required=True)
    p.add_argument("--tags", required=True)
    p.add_argument("--tsv", default="")
    for name in ("types", "keep-flags"):
        sub.add_parser(name).add_argument("--gguf", required=True)
    p = sub.add_parser("types-diff")
    p.add_argument("--a", required=True)
    p.add_argument("--b", required=True)
    a = ap.parse_args()
    handlers = {"plan": cmd_plan, "probe": cmd_probe, "brief": cmd_brief, "sanity": cmd_sanity, "table": cmd_table,
                "types": cmd_types, "keep-flags": cmd_keep_flags, "types-diff": cmd_types_diff}
    return handlers[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
