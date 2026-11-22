"""ON ink-v11-res: LoRA SFT on the BF16 attention linears of v1.0 (experts and everything else frozen).

  python3 train_lora.py --data /root/v11b/sft.jsonl --base /root/w/v10 --out /root/v11b/runs/<name> \
      [--rank 16] [--alpha 32] [--lr 1e-4] [--epochs 1] [--accum 8] [--max-len 3072] [--targets q,k,v,o,g] [--save-every 100]

sft.jsonl rows: {"id", "text" (user prompt), "effort" (0 or 0.9), "completion" (token ids the model should produce,
from <|message_model|> through <|end_message|>), "w" (loss weight, default 1)}.
Prompt ids = official chat template, no system prompt (hf_load.encode). Loss = token-mean CE on completion tokens only.
LoRA: W x + (alpha/r) B A x, A ~ kaiming, B = 0, fp32 params. Saves lora_<step>.pt = {hf_module_name: (A, B)}, + meta.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time

import torch
import torch.nn as nn

sys.path.insert(0, "/root/parity")
from hf_load import encode, load  # noqa: E402

TARGETS = {"q": "self_attn.q_proj.linear", "k": "self_attn.k_proj.linear", "v": "self_attn.v_proj.linear",
           "o": "self_attn.o_proj.linear", "g": "self_attn.o_proj.gate_proj"}


class LoRA(nn.Module):
    def __init__(self, base: nn.Linear, r: int, alpha: float):
        super().__init__()
        self.base = base
        self.scale = alpha / r
        self.A = nn.Parameter(torch.empty(r, base.in_features, dtype=torch.float32, device=base.weight.device))
        self.B = nn.Parameter(torch.zeros(base.out_features, r, dtype=torch.float32, device=base.weight.device))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        d = (x.float() @ self.A.t()) @ self.B.t()
        return y + (d * self.scale).to(y.dtype)


def inject(model, keys: list[str], r: int, alpha: float) -> dict[str, LoRA]:
    suffixes = [TARGETS[k] for k in keys]
    found = {}
    for name, mod in list(model.named_modules()):
        if "language_model.layers." in name and any(name.endswith(s) for s in suffixes) and isinstance(mod, nn.Linear):
            parent_name, child = name.rsplit(".", 1)
            parent = model.get_submodule(parent_name)
            lora = LoRA(mod, r, alpha)
            setattr(parent, child, lora)
            found[name] = lora
    assert len(found) == 24 * len(keys), len(found)
    return found


def save(loras: dict[str, LoRA], path: str, meta: dict) -> None:
    torch.save({"meta": meta, "lora": {n: (m.A.detach().cpu(), m.B.detach().cpu()) for n, m in loras.items()}}, path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--base", default="/root/w/v10")
    ap.add_argument("--out", required=True)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=float, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=3072)
    ap.add_argument("--targets", default="q,k,v,o,g")
    ap.add_argument("--save-every", type=int, default=100)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=1234)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(a.seed)
    model, proc = load(a.base)
    model.requires_grad_(False)
    loras = inject(model, a.targets.split(","), a.rank, a.alpha)
    params = [p for m in loras.values() for p in (m.A, m.B)]
    print("lora params", sum(p.numel() for p in params), flush=True)

    rows = [json.loads(l) for l in open(a.data)]
    rng = random.Random(a.seed)
    seqs = []
    for r in rows:
        p = encode(proc, r["text"], effort=r["effort"])[0].tolist()
        c = r["completion"]
        if len(p) + len(c) > a.max_len:
            c = c[: a.max_len - len(p)]
        seqs.append((p, c, float(r.get("w", 1.0))))
    order = []
    for _ in range(math.ceil(a.epochs)):
        o = list(range(len(seqs)))
        rng.shuffle(o)
        order += o
    order = order[: int(len(seqs) * a.epochs)]
    if a.limit:
        order = order[: a.limit]
    steps = math.ceil(len(order) / a.accum)
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0, betas=(0.9, 0.99))
    warm = max(1, int(0.05 * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm))))
    meta = {"args": vars(a), "n_rows": len(seqs), "steps": steps}
    json.dump(meta, open(f"{a.out}/meta.json", "w"), indent=1)
    log = open(f"{a.out}/train_log.jsonl", "a")
    t0, step, acc_loss, acc_tok = time.time(), 0, 0.0, 0
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train(True)  # checkpointing is active only in train mode; the model has no dropout
    for i, idx in enumerate(order):
        p, c, w = seqs[idx]
        ids = torch.tensor([p + c], device="cuda:0")
        logits = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).logits
        lp = logits[0, len(p) - 1: len(p) - 1 + len(c)].float()
        tgt = torch.tensor(c, device=lp.device)
        loss = nn.functional.cross_entropy(lp, tgt, reduction="mean") * w
        (loss / a.accum).backward()
        acc_loss += loss.item()
        acc_tok += len(c)
        del logits, lp, loss
        if (i + 1) % a.accum == 0 or i + 1 == len(order):
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            rec = {"step": step, "loss": acc_loss / a.accum, "tok": acc_tok, "lr": sched.get_last_lr()[0],
                   "elapsed": round(time.time() - t0, 1), "mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 1)}
            log.write(json.dumps(rec) + "\n")
            log.flush()
            if step % 5 == 0 or step == 1:
                print(rec, flush=True)
            acc_loss, acc_tok = 0.0, 0
            if step % a.save_every == 0:
                save(loras, f"{a.out}/lora_{step}.pt", meta)
    save(loras, f"{a.out}/lora_final.pt", meta)
    print("TRAIN_DONE", step, round(time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
