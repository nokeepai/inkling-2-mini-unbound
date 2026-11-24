"""ON ink-v3-train: DPO (sigmoid) LoRA on the BF16 attention linears of v2 (+ an SFT LoRA folded in first).

  python3 train_dpo.py --data /root/v3/dpo.jsonl --base /root/w/v2 --out /root/v3/runs/<name> \
      [--sft-lora /root/v3/runs/sft/lora_final.pt --sft-scale 1.0] [--beta 0.1] [--sft-weight 0] \
      [--rank 32] [--alpha 32] [--lr 5e-6] [--epochs 1] [--accum 8] [--max-len 8192] [--targets q,k,v,o,g] \
      [--save-every 50] [--stop-acc 0.95] [--limit N] [--seed 1234] [--dry]

dpo.jsonl rows: {"id", "text" (user prompt) | "messages" ([{role, content}]), "system" (optional), "effort",
 "chosen", "rejected" (completion token ids from <|message_model|>), "kind": "full"|"loop"|"opener",
 "prefix" (optional token ids appended to the prompt, unmasked shared context), "w" (optional weight, default 1)}.

Masks (over the completion, i.e. the tokens after prompt+prefix):
 full / opener : answer tokens only = everything after the first [MSG_MODEL, TEXT] marker (thinking-on:
                 [MSG_MODEL, THINK] .. END [MSG_MODEL, TEXT] answer END; thinking-off: [MSG_MODEL, TEXT] answer END),
                 closing END included. If the completion has no marker but the prefix does (opener pairs that
                 start inside the answer), every completion token counts. No answer tokens -> pair dropped.
 loop          : every completion token (thinking included).
 SFT anchor (--sft-weight > 0): lambda * token-mean NLL over the chosen ANSWER tokens (RPO-style).

Exact low-memory DPO gradient. Per pair: reference logps rc, rr are cached (first pass, DPO LoRA disabled, SFT fold
applied); 2 no-grad policy forwards give pc, pr; z = beta*((pc-rc)-(pr-rr)); L = -w*logsigmoid(z);
dL/dpc = g = -w*beta*sigmoid(-z), dL/dpr = -g. Then two separate forward+backward passes on the surrogates
g*logp_c and -g*logp_r (g a constant), so only one sequence graph is alive at a time. Their gradient sum equals
autograd of the joint loss exactly (chain rule, g evaluated at the current weights). Cost per pair ~
2 no-grad forwards + 2 checkpointed forward/backward ~= 2.5x one SFT row of the same length per sequence pair.

LoRA: W x + (alpha/r) B A x, fp32 A/B on the bf16 base, B = 0 at init (policy == reference at step 0).
Saves lora_<step>.pt / lora_final.pt in train_lora.py's format so merge_lora.py / merge_stack.py read them.
Final model = v2 + sft_scale*SFT LoRA + DPO LoRA (use merge_stack.py with both).
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import random
import sys
import time
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F

MSG_MODEL, TEXT, THINK, END, EOS = 200001, 200004, 200008, 200010, 200006
TARGETS = {"q": "self_attn.q_proj.linear", "k": "self_attn.k_proj.linear", "v": "self_attn.v_proj.linear",
           "o": "self_attn.o_proj.linear", "g": "self_attn.o_proj.gate_proj"}
N_LAYERS = 24


# ---------------------------------------------------------------- LoRA ----------------------------------------------
class LoRA(nn.Module):
    """Same math and init as v11b/train_lora.py; `LoRA.enabled = False` turns every adapter off (reference model)."""
    enabled = True

    def __init__(self, base: nn.Linear, r: int, alpha: float):
        super().__init__()
        self.base = base
        self.scale = alpha / r
        self.A = nn.Parameter(torch.empty(r, base.in_features, dtype=torch.float32, device=base.weight.device))
        self.B = nn.Parameter(torch.zeros(base.out_features, r, dtype=torch.float32, device=base.weight.device))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        if not LoRA.enabled:
            return y
        d = (x.float() @ self.A.t()) @ self.B.t()
        return y + (d * self.scale).to(y.dtype)


@contextlib.contextmanager
def lora_disabled():
    prev = LoRA.enabled
    LoRA.enabled = False
    try:
        yield
    finally:
        LoRA.enabled = prev


def inject(model: nn.Module, keys: list[str], r: int, alpha: float, n_layers: int = N_LAYERS) -> dict[str, LoRA]:
    suffixes = [TARGETS[k] for k in keys]
    found = {}
    for name, mod in list(model.named_modules()):
        if "language_model.layers." in name and any(name.endswith(s) for s in suffixes) and isinstance(mod, nn.Linear):
            parent_name, child = name.rsplit(".", 1)
            lora = LoRA(mod, r, alpha)
            setattr(model.get_submodule(parent_name), child, lora)
            found[name] = lora
    assert len(found) == n_layers * len(keys), len(found)
    return found


@torch.no_grad()
def fold_lora(model: nn.Module, lora_path: str, merge_scale: float) -> dict:
    """W <- bf16(W.float() + merge_scale*(alpha/r)*B@A), in place, BEFORE inject (names are the plain nn.Linear ones)."""
    d = torch.load(lora_path, map_location="cpu", weights_only=False)
    a = d["meta"]["args"]
    scale = a["alpha"] / a["rank"] * merge_scale
    rel = []
    for name, (A, B) in d["lora"].items():
        mod = model.get_submodule(name)
        assert isinstance(mod, nn.Linear) and not isinstance(mod, LoRA), name
        W = mod.weight
        assert tuple(W.shape) == (B.shape[0], A.shape[1]), (name, W.shape, A.shape, B.shape)
        delta = scale * (B.to(W.device, torch.float32) @ A.to(W.device, torch.float32))
        rel.append(float(delta.norm() / W.float().norm()))
        W.copy_((W.float() + delta).to(W.dtype))
    return {"n": len(rel), "scale": scale, "mean_rel_delta": sum(rel) / max(1, len(rel))}


def save(loras: dict[str, LoRA], path: str, meta: dict) -> None:
    torch.save({"meta": meta, "lora": {n: (m.A.detach().cpu(), m.B.detach().cpu()) for n, m in loras.items()}}, path)


# ---------------------------------------------------------------- masking -------------------------------------------
def find_marker(ids: list[int]) -> int | None:
    """Index just past the [MSG_MODEL, TEXT] marker that opens the answer, or None.
    Thinking-on completions open with [MSG_MODEL, THINK]; the answer marker must follow the first END."""
    if len(ids) >= 2 and ids[0] == MSG_MODEL and ids[1] == THINK:
        try:
            e = ids.index(END, 2)
        except ValueError:
            return None  # thinking never closed (truncated)
        return e + 3 if ids[e + 1: e + 3] == [MSG_MODEL, TEXT] else None
    for i in range(len(ids) - 1):
        if ids[i] == MSG_MODEL and ids[i + 1] == TEXT:
            return i + 2
    return None


def answer_mask(comp: list[int], prefix: list[int]) -> list[float]:
    s = find_marker(comp)
    if s is None:
        s = 0 if (prefix and find_marker(prefix) is not None) else len(comp)
    return [0.0] * s + [1.0] * (len(comp) - s)


def build_masks(kind: str, comp: list[int], prefix: list[int]) -> tuple[list[float], list[float]]:
    """(dpo mask, anchor mask) over completion positions."""
    ans = answer_mask(comp, prefix)
    if kind == "loop":
        return [1.0] * len(comp), ans
    if kind == "loopthink":  # runaway-thinking pair whose chosen answer is not a target: DPO on thinking tokens only
        return [1.0 - x for x in ans], ans
    assert kind in ("full", "opener"), kind
    return ans, ans


# ---------------------------------------------------------------- data ----------------------------------------------
def prompt_ids(proc, row: dict, encode) -> list[int]:
    if "messages" in row:
        msgs = list(row["messages"])
        if row.get("system") and not (msgs and msgs[0]["role"] == "system"):
            msgs.insert(0, {"role": "system", "content": row["system"]})
        return encode(proc, msgs, effort=row.get("effort", 0.9), device="cpu")[0].tolist()
    return encode(proc, row["text"], effort=row.get("effort", 0.9), device="cpu", system=row.get("system"))[0].tolist()


def prepare(rows: list[dict], ctx_of, max_len: int) -> tuple[list[dict], dict]:
    """ctx_of(row) -> prompt ids. Returns usable pairs and drop/truncation counts."""
    pairs, stats = [], {"rows": len(rows), "drop_ctx": 0, "drop_nomask": 0, "drop_same": 0, "trunc": 0}  # trunc = dropped as too long
    for i, r in enumerate(rows):
        prefix = list(r.get("prefix") or [])
        ctx = ctx_of(r) + prefix
        if len(ctx) >= max_len - 8:
            stats["drop_ctx"] += 1
            continue
        room = max_len - len(ctx)
        c, j = list(r["chosen"]), list(r["rejected"])
        kind0 = r.get("kind", "full")
        # owner rule: never train TOWARD an answer cut mid-way. Chosen must fit whole, else drop the pair. A loop
        # pair's rejected side (a runaway think) may be cut at the window: it is only pushed down, never imitated.
        if len(c) > room or (len(j) > room and kind0 not in ("loop", "loopthink")):
            stats["trunc"] += 1
            continue
        if len(j) > room:
            stats["rej_cut_loop"] = stats.get("rej_cut_loop", 0) + 1
            j = j[:room]
        if c == j:
            stats["drop_same"] += 1
            continue
        kind = r.get("kind", "full")
        mc, ac = build_masks(kind, c, prefix)
        mr, _ = build_masks(kind, j, prefix)
        if sum(mc) == 0 or sum(mr) == 0:
            stats["drop_nomask"] += 1
            continue
        pairs.append({"key": f"{i}:{r.get('id', i)}", "id": r.get("id", i), "kind": kind, "ctx": ctx,
                      "c": c, "r": j, "mc": mc, "mr": mr, "ac": ac, "w": float(r.get("w", 1.0))})
    stats["pairs"] = len(pairs)
    return pairs, stats


# ---------------------------------------------------------------- log-probs -----------------------------------------
def seq_logp(model, ctx: list[int], comp: list[int], masks: list[list[float]], device) -> list[torch.Tensor]:
    """Summed token log-probs of `comp` given `ctx`, one scalar per mask. Grad flows iff grad mode is on.
    Only the logit rows spanning the masked positions are upcast (memory)."""
    nz = [k for m in masks for k, v in enumerate(m) if v]
    a, b = min(nz), max(nz) + 1
    ids = torch.tensor([ctx + comp], device=device)
    logits = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).logits
    s = len(ctx) - 1
    lp = logits[0, s + a: s + b].float()
    tok_lp = -F.cross_entropy(lp, torch.tensor(comp[a:b], device=device), reduction="none")
    return [(tok_lp * torch.tensor(m[a:b], device=device)).sum() for m in masks]


@torch.no_grad()
def pair_logps(model, p: dict, device) -> tuple[float, float]:
    lc = seq_logp(model, p["ctx"], p["c"], [p["mc"]], device)[0]
    lr = seq_logp(model, p["ctx"], p["r"], [p["mr"]], device)[0]
    return float(lc), float(lr)


def dpo_pair_backward(model, p: dict, ref: tuple[float, float], beta: float, sft_weight: float, accum: int,
                      device) -> dict:
    """Exact DPO (+ optional anchor) gradient for one pair, accumulated into .grad, divided by `accum`."""
    pc, pr = pair_logps(model, p, device)
    rc, rr = ref
    z = beta * ((pc - rc) - (pr - rr))
    w = p["w"]
    g = -w * beta * torch.sigmoid(torch.tensor(-z, dtype=torch.float64)).item()  # dL/dpc ; dL/dpr = -g
    loss = -w * F.logsigmoid(torch.tensor(z, dtype=torch.float64)).item()
    n_anchor = sum(p["ac"])
    use_anchor = sft_weight > 0 and n_anchor > 0
    masks = [p["mc"], p["ac"]] if use_anchor else [p["mc"]]
    out = seq_logp(model, p["ctx"], p["c"], masks, device)
    surr = g * out[0]
    nll = 0.0
    if use_anchor:
        anchor = -out[1] / n_anchor
        surr = surr + sft_weight * w * anchor
        nll = float(anchor.detach())
    pc_grad = float(out[0].detach())
    (surr / accum).backward()
    del out, surr
    lr_g = seq_logp(model, p["ctx"], p["r"], [p["mr"]], device)[0]
    pr_grad = float(lr_g.detach())
    ((-g * lr_g) / accum).backward()
    del lr_g
    return {"loss": loss + sft_weight * w * nll, "dpo_loss": loss, "z": z, "acc": float(z > 0),
            "rw_c": beta * (pc - rc), "rw_r": beta * (pr - rr), "drift_c": (pc - rc) / sum(p["mc"]),
            "nll": nll, "pc": pc, "pr": pr, "rc": rc, "rr": rr, "g": g,
            "recompute_err": max(abs(pc_grad - pc), abs(pr_grad - pr)), "tok": len(p["c"]) + len(p["r"])}


# ---------------------------------------------------------------- reference cache -----------------------------------
def fingerprint(a: argparse.Namespace) -> str:
    h = hashlib.sha256()
    with open(a.data, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    sft = ""
    if a.sft_lora:
        st = os.stat(a.sft_lora)
        sft = f"{os.path.abspath(a.sft_lora)}:{st.st_size}:{st.st_mtime_ns}"
    h.update(f"|{os.path.abspath(a.base)}|{sft}|{a.sft_scale}|{a.max_len}".encode())
    return h.hexdigest()[:16]


def ref_logps(model, pairs: list[dict], path: str | None, fp: str, device) -> dict[str, tuple[float, float]]:
    if path and os.path.exists(path):
        d = json.load(open(path))
        if d.get("fp") == fp and all(p["key"] in d["ref"] for p in pairs):
            print("ref cache hit", path, flush=True)
            return {k: tuple(v) for k, v in d["ref"].items()}
        print("ref cache stale/incomplete, recomputing", flush=True)
    ref, t0 = {}, time.time()
    with lora_disabled():
        for i, p in enumerate(pairs):
            ref[p["key"]] = pair_logps(model, p, device)
            if (i + 1) % 50 == 0:
                print(f"ref {i + 1}/{len(pairs)} {time.time() - t0:.0f}s", flush=True)
    if path:
        json.dump({"fp": fp, "ref": ref}, open(path, "w"))
    return ref


# ---------------------------------------------------------------- main ----------------------------------------------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--base", default="/root/w/v2")
    ap.add_argument("--out", required=True)
    ap.add_argument("--sft-lora", default="")
    ap.add_argument("--sft-scale", type=float, default=1.0)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--sft-weight", type=float, default=0.0)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--alpha", type=float, default=32)  # alpha/r = 1.0 at the defaults
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=8192)
    ap.add_argument("--targets", default="q,k,v,o,g")
    ap.add_argument("--save-every", type=int, default=50)
    ap.add_argument("--stop-acc", type=float, default=0.0, help="stop when running acc over last 32 pairs >= this")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--dry", action="store_true", help="2 pairs, no saving; checks shapes, z=0 at init, sign of update")
    ap.add_argument("--dry-lr", type=float, default=1e-5)
    return ap.parse_args()


def make_order(n: int, epochs: float, seed: int, limit: int) -> list[int]:
    rng, order = random.Random(seed), []
    for _ in range(math.ceil(epochs)):
        o = list(range(n))
        rng.shuffle(o)
        order += o
    order = order[: int(n * epochs)]
    return order[:limit] if limit else order


def mem_gb() -> float:
    return round(torch.cuda.max_memory_allocated() / 2**30, 1) if torch.cuda.is_available() else 0.0


def load_policy(a: argparse.Namespace):
    sys.path.insert(0, "/root/parity")
    sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "parity"))
    from hf_load import encode, load  # noqa: E402

    model, proc = load(a.base)
    model.requires_grad_(False)
    fold = fold_lora(model, a.sft_lora, a.sft_scale) if a.sft_lora else None
    print("sft fold", fold, flush=True)
    loras = inject(model, a.targets.split(","), a.rank, a.alpha)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train(True)  # checkpointing is active only in train mode; the model has no dropout
    return model, proc, encode, loras, fold


def run_dry(model, loras, pairs, a, device) -> None:
    params = [p for m in loras.values() for p in (m.A, m.B)]
    pairs = pairs[:2]
    with lora_disabled():
        ref = {p["key"]: pair_logps(model, p, device) for p in pairs}
    for p in pairs:
        print("pair", p["id"], p["kind"], "ctx", len(p["ctx"]), "c", len(p["c"]), "r", len(p["r"]),
              "masked c/r/anchor", sum(p["mc"]), sum(p["mr"]), sum(p["ac"]), flush=True)
        rec = dpo_pair_backward(model, p, ref[p["key"]], a.beta, a.sft_weight, 2, device)
        print({k: round(v, 5) for k, v in rec.items()}, "mem_gb", mem_gb(), flush=True)
    gn = float(torch.nn.utils.clip_grad_norm_(params, 1e9))
    print("grad norm", gn, "(z should be 0 at init: B=0 so policy == reference)", flush=True)
    opt = torch.optim.AdamW(params, lr=a.dry_lr, weight_decay=0.0, betas=(0.9, 0.99))
    opt.step()
    opt.zero_grad(set_to_none=True)
    for p in pairs:
        pc, pr = pair_logps(model, p, device)
        rc, rr = ref[p["key"]]
        print("after 1 step", p["id"], "dpc", round(pc - rc, 4), "dpr", round(pr - rr, 4),
              "z", round(a.beta * ((pc - rc) - (pr - rr)), 5), "(expect z > 0)", flush=True)
    print("DRY_DONE mem_gb", mem_gb(), flush=True)


def train(model, loras, pairs, ref, a, meta, device) -> None:
    params = [p for m in loras.values() for p in (m.A, m.B)]
    order = make_order(len(pairs), a.epochs, a.seed, a.limit)
    steps = math.ceil(len(order) / a.accum)
    meta["steps"] = steps
    json.dump(meta, open(f"{a.out}/meta.json", "w"), indent=1)
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0, betas=(0.9, 0.99))
    warm = max(1, int(0.05 * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm))))
    log = open(f"{a.out}/train_log.jsonl", "a")
    t0, step, buf, recent = time.time(), 0, [], deque(maxlen=32)
    for i, idx in enumerate(order):
        p = pairs[idx]
        rec = dpo_pair_backward(model, p, ref[p["key"]], a.beta, a.sft_weight, a.accum, device)
        buf.append(rec)
        recent.append(rec["acc"])
        if (i + 1) % a.accum and i + 1 != len(order):
            continue
        gn = float(torch.nn.utils.clip_grad_norm_(params, 1.0))
        lr_now = sched.get_last_lr()[0]
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        mean = lambda k: sum(r[k] for r in buf) / len(buf)  # noqa: E731
        out = {"step": step, "loss": mean("loss"), "dpo_loss": mean("dpo_loss"), "z": mean("z"), "acc": mean("acc"),
               "run_acc32": sum(recent) / len(recent), "rw_c": mean("rw_c"), "rw_r": mean("rw_r"),
               "drift_c": mean("drift_c"), "nll": mean("nll"), "recompute_err": max(r["recompute_err"] for r in buf),
               "gnorm": gn, "lr": lr_now, "tok": sum(r["tok"] for r in buf), "elapsed": round(time.time() - t0, 1),
               "mem_gb": mem_gb()}
        log.write(json.dumps(out) + "\n")
        log.flush()
        buf = []
        if step % 5 == 0 or step == 1:
            print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in out.items()}, flush=True)
        if step % a.save_every == 0:
            save(loras, f"{a.out}/lora_{step}.pt", meta)
        if a.stop_acc and len(recent) == recent.maxlen and out["run_acc32"] >= a.stop_acc:
            print(f"EARLY_STOP step {step} run_acc32 {out['run_acc32']:.3f}", flush=True)
            meta["early_stop"] = step
            break
    meta["final_step"] = step
    save(loras, f"{a.out}/lora_final.pt", meta)
    json.dump(meta, open(f"{a.out}/meta.json", "w"), indent=1)
    print("TRAIN_DONE", step, round(time.time() - t0), flush=True)


def main() -> None:
    a = parse_args()
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    torch.cuda.manual_seed_all(a.seed)
    device = "cuda:0"
    if not a.dry:
        os.makedirs(a.out, exist_ok=True)
    model, proc, encode, loras, fold = load_policy(a)
    params = [p for m in loras.values() for p in (m.A, m.B)]
    print("lora params", sum(p.numel() for p in params), "alpha/r", a.alpha / a.rank, flush=True)
    rows = [json.loads(line) for line in open(a.data) if line.strip()]
    if a.dry:
        rows = rows[:16]  # enough to find 2 usable pairs
    pairs, stats = prepare(rows, lambda r: prompt_ids(proc, r, encode), a.max_len)
    print("data", stats, flush=True)
    assert pairs, "no usable pairs"
    if a.dry:
        run_dry(model, loras, pairs, a, device)
        return
    meta = {"args": vars(a), "data_stats": stats, "sft_fold": fold, "n_rows": len(pairs)}
    ref = ref_logps(model, pairs, f"{a.out}/ref_logps.json", fingerprint(a), device)
    train(model, loras, pairs, ref, a, meta, device)


if __name__ == "__main__":
    main()
