"""Loop-aware ARA fit (agent ara, runs on ink-research-1). Train split captures from ara_capture.py only.

  python ara_fit.py <params.json> <out_dir>

Per layer l in [l_min, l_max] (wo_ud = self_attn.o_proj.linear, W [2304, 3072], unit-norm rows):
  W' = diag(r) (W + A Bᵀ), A [2304, k], B [3072, k]; r_i = ||W_i|| / ||(W + ABᵀ)_i|| when mp (row norms preserved).
  The objective is measured on the POST-CONV write (post_attention_convolution), so the conv leak is inside it:
    y_t = ((1 + w3) h_t + w2 h_{t-1} + w1 h_{t-2} + w0 h_{t-3}) / sqrt(5),  h = W' x,
  and r commutes with the depthwise conv, so y' = r * (y + conv(A Bᵀ x)).
  Every pass of a looped layer (6-17) is a separate set of points; the loss averages over all of them (sum over passes).
  Groups g = (effort, pass, position); bad = harmful prompts, good = harmless prompts (fixed original outputs).
    preserve = mean_good ||y'_good - y_good||² / mean ||y_good||²
    steer    = mean_g [ kNN(y'_bad, Y_good,g) - overcorrect * kNN(y'_bad, Y_bad,g) ] / scale_g   (heretic ara_loss)
    loss     = preserve_w * preserve + steer
  Position weights: thinking-token positions are weighted by think_w (template positions weight 1).
Writes <out_dir>/lowrank.safetensors (checkpoint-name keys .A/.B/.r, fp32) and <out_dir>/fit.json.
"""
import json
import math
import os
import sys
import time

import torch
from safetensors import safe_open
from safetensors.torch import save_file

CAP = "/root/ara/cap"
ORIG = "/root/w/orig"
P = json.load(open(sys.argv[1]))
OUT = sys.argv[2]
os.makedirs(OUT, exist_ok=True)
dev = "cuda:0"
torch.manual_seed(int(P.get("seed", 0)))
meta = json.load(open(f"{CAP}/meta.json"))
VALID = torch.load(f"{CAP}/valid.pt").to(dev)            # [N, NPOS]
CONV = torch.load(f"{CAP}/conv.pt")
PVALID = torch.load(f"{CAP}/pvalid.pt").to(dev) if os.path.exists(f"{CAP}/pvalid.pt") else None
NPOS = VALID.shape[1]
sets = meta["sets"]
efforts = meta["efforts"]
EFF = [e for e in sorted(set(efforts)) if e in P.get("efforts", [0.0, 0.9])]
idx_bad = {e: torch.tensor([i for i, (s, ee) in enumerate(zip(sets, efforts)) if s == "harmful" and ee == e], device=dev) for e in EFF}
idx_good = {e: torch.tensor([i for i, (s, ee) in enumerate(zip(sets, efforts)) if s == "harmless" and ee == e], device=dev) for e in EFF}
idx = json.load(open(f"{ORIG}/model.safetensors.index.json"))["weight_map"]


def wname(l):
    return f"model.llm.layers.{l}.attn.wo_ud.weight"


def orig_w(l):
    with safe_open(f"{ORIG}/{idx[wname(l)]}", "pt") as f:
        return f.get_tensor(wname(l))


def conv_post(h, w):
    """h [..., WIN, D] -> post-conv at the last NPOS window positions [..., NPOS, D] (fp32)."""
    c0 = 1.0 + w[:, 3]
    T = h.shape[-2]
    out = c0 * h[..., 3:, :] + w[:, 2] * h[..., 2:T - 1, :] + w[:, 1] * h[..., 1:T - 2, :] + w[:, 0] * h[..., 0:T - 3, :]
    return out / math.sqrt(5.0)


def knn_mean(a, b, k):
    """a [G, n, D], b [G, m, D] -> mean distance of each a-point to its k nearest b-points, [G, n]."""
    d = torch.cdist(a, b)
    return d.topk(min(k, b.shape[1]), dim=-1, largest=False).values.mean(-1)


def fit_layer(l):
    t0 = time.time()
    W = orig_w(l).float().to(dev)                      # [2304, 3072]
    w = CONV[l].to(dev)                                # [2304, 4]
    X = torch.load(f"{CAP}/L{l}.pt").to(dev)           # [N, Pp, WIN, 3072] fp16
    Np, Pp, WIN, _ = X.shape
    with torch.no_grad():
        Y = conv_post((X.float() @ W.T), w)            # [N, Pp, NPOS, 2304]  (fp32 reference)
    k = int(P["rank"])
    A = torch.zeros(W.shape[0], k, device=dev, requires_grad=True)
    B = (torch.randn(W.shape[1], k, device=dev) / math.sqrt(W.shape[1])).requires_grad_(True)
    opt = torch.optim.Adam([A, B], lr=float(P["lr"]))
    rown2 = (W * W).sum(1)
    posw = torch.ones(NPOS, device=dev)
    posw[4:] = float(P.get("think_w", 1.0))
    gi = torch.cat([idx_good[e] for e in EFF])
    gscale = Y[gi].norm(dim=-1).pow(2).mean()
    # group-wise fixed clouds: [G = E*Pp*NPOS, n, D]
    def clouds(ids):
        Z = Y[ids]                                      # [n, Pp, NPOS, D]
        return Z.permute(1, 2, 0, 3).reshape(Pp * NPOS, len(ids), -1)
    Yg = torch.cat([clouds(idx_good[e]) for e in EFF])
    Yb = torch.cat([clouds(idx_bad[e]) for e in EFF])
    Vb = torch.cat([VALID[idx_bad[e]].T.repeat(Pp, 1) for e in EFF]).float()       # [G, n]
    Vg = torch.cat([VALID[idx_good[e]].T.repeat(Pp, 1) for e in EFF]).float()
    pw_g = posw.repeat(Pp * len(EFF))                                                # [G]
    # per-group spread of the good cloud: normalises distances across positions/passes
    mu_g = (Yg * Vg[..., None]).sum(1, keepdim=True) / Vg.sum(1).clamp(min=1)[:, None, None]
    scale = ((Yg - mu_g).norm(dim=-1) * Vg).sum(1) / Vg.sum(1).clamp(min=1)          # [G]
    Xb = torch.cat([X[idx_bad[e]] for e in EFF])                                       # [nb*E, Pp, WIN, 3072]
    Xg = X[gi]
    kk = int(P["knn_k"])
    oc = float(P["overcorrect"])
    pres_w = float(P["preserve_w"])
    pres2_w = float(P.get("pres2_w", 0.0))
    if pres2_w > 0:
        Xp = torch.load(f"{CAP}/P{l}.pt").to(dev)                                      # [Np, Pp, 35, 3072]
        Vp = PVALID[:, None, :].float()
        with torch.no_grad():
            Yp = conv_post(Xp.float() @ W.T, w)
        pscale = (Yp.pow(2).sum(-1) * Vp).sum() / (Vp.sum() * Pp)
    mean_w = float(P.get("mean_w", 0.0))
    gamma = float(P.get("gamma", 0.0))
    mb0 = (Yb * Vb[..., None]).sum(1) / Vb.sum(1).clamp(min=1)[:, None]               # [G, D]
    mg0 = mu_g[:, 0]
    tgt = mg0 + gamma * (mg0 - mb0)
    gap2 = (mg0 - mb0).pow(2).sum(-1).clamp(min=1e-8)
    hist = []

    def rscale():
        if not P.get("mp", True):
            return torch.ones(W.shape[0], device=dev)
        WB = W @ B
        n2 = rown2 + 2 * (A * WB).sum(1) + ((A @ (B.T @ B)) * A).sum(1)
        return (rown2 / n2.clamp(min=1e-8)).sqrt()

    def forward(Xs, Ys):
        z = Xs.float() @ B                              # [n, Pp, WIN, k]
        dy = conv_post(z @ A.T, w)                      # [n, Pp, NPOS, D]
        return rscale() * (Ys + dy)

    def loss_fn():
        r = None
        yg = forward(Xg, Y[gi])
        pres = ((yg - Y[gi]).pow(2).sum(-1) * VALID[gi][:, None, :]).sum() / (VALID[gi].sum() * Pp) / gscale
        loss = pres_w * pres
        if pres2_w > 0:
            yp = forward(Xp, Yp)
            loss = loss + pres2_w * ((yp - Yp).pow(2).sum(-1) * Vp).sum() / (Vp.sum() * Pp) / pscale
        yb = torch.cat([forward(X[idx_bad[e]], Y[idx_bad[e]]).permute(1, 2, 0, 3).reshape(Pp * NPOS, len(idx_bad[e]), -1)
                        for e in EFF])
        to_good = knn_mean(yb, Yg, kk)
        to_bad = knn_mean(yb, Yb, kk + 1)              # +1: the point's own original is its nearest bad neighbour
        st = (to_good - oc * to_bad) / scale[:, None]
        steer = (st * Vb * pw_g[:, None]).sum() / (Vb * pw_g[:, None]).sum()
        if mean_w > 0:
            # group-mean shift: harmful mean -> harmless mean + gamma * (harmless - harmful) (overshoot lets the
            # attention write compensate for refusal that the MoE writes later)
            mb = (yb * Vb[..., None]).sum(1) / Vb.sum(1).clamp(min=1)[:, None]
            sh = (mb - tgt).pow(2).sum(-1) / gap2
            steer = steer + mean_w * (sh * pw_g).sum() / pw_g.sum()
        return loss + steer, pres, steer

    with torch.no_grad():
        _, p0, s0 = loss_fn()
    for step in range(int(P["steps"])):
        opt.zero_grad(set_to_none=True)
        loss, pres, steer = loss_fn()
        loss.backward()
        opt.step()
        if step % 50 == 0:
            hist.append([step, round(float(pres), 5), round(float(steer), 5)])
    with torch.no_grad():
        loss, pres, steer = loss_fn()
        r = rscale()
        dW = A @ B.T
        Wn = r[:, None] * (W + dW)
        rel = float((Wn - W).norm() / W.norm())
        # how far harmful post-conv writes moved toward the harmless group means (diagnostic)
        yb = torch.cat([forward(X[idx_bad[e]], Y[idx_bad[e]]) for e in EFF])
        Yb0 = torch.cat([Y[idx_bad[e]] for e in EFF])
        mg = Y[gi].mean(0)
        gap0 = (Yb0.mean(0) - mg).norm(dim=-1).mean()
        gap1 = (yb.mean(0) - mg).norm(dim=-1).mean()
    stats = {"layer": l, "preserve0": float(p0), "steer0": float(s0), "preserve": float(pres), "steer": float(steer),
             "rel_dW": rel, "r_min": float(r.min()), "r_max": float(r.max()),
             "mean_gap_ratio": float(gap1 / gap0), "secs": round(time.time() - t0, 1), "hist": hist}
    del X, Y, Xb, Xg, Yg, Yb
    torch.cuda.empty_cache()
    return A.detach().cpu().contiguous(), B.detach().cpu().contiguous(), r.detach().cpu().contiguous(), stats


tensors, allstats = {}, []
for l in range(int(P["l_min"]), int(P["l_max"]) + 1):
    A, B, r, st = fit_layer(l)
    n = wname(l)
    tensors[f"{n}.A"] = A.float()
    tensors[f"{n}.B"] = B.float()
    if P.get("mp", True):
        tensors[f"{n}.r"] = r.float()
    allstats.append(st)
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in st.items() if k != "hist"}), flush=True)
save_file(tensors, f"{OUT}/lowrank.safetensors")
json.dump({"params": P, "layers": allstats}, open(f"{OUT}/fit.json", "w"), indent=1)
print("FIT_DONE", OUT, flush=True)
