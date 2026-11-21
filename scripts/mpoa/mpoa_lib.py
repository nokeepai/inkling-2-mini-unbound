"""MPOA (C1/C2) candidate construction for Inkling-2-Mini wo_ud (layers 4-23).

Edit per layer l with strength s_l and an orthonormal output-space basis U_l (columns span S_l):
    W'_l = (I - s_l U U^T) diag(r_l) W_l
r_l = 1 without magnitude preservation; with it, r_l is solved so every row of W'_l keeps the original row norm.
Because the projection acts AFTER the row scale, P_S W' = (1-s) P_S diag(r) W exactly (MP does not reopen the leak).
Compact form (CANDIDATE_FORMAT): W' = diag(r) (W + A B^T) with A = -s diag(1/r) U, B = W^T diag(r) U.

Conv-aware subspaces (post_attention_convolution, causal, kernel 4, weight w[c, 0..3]; w[:,3] multiplies h_t):
    y_t = (h_t + sum_j w[:,3-j] * h_{t-j}) / sqrt(5)
    Q^T y_t = 0 for all inputs  <=>  h ⊥ c_j ⊙ Q for j = 0..3, with c_0 = 1 + w[:,3], c_j = w[:,3-j].
  "exact4": S = span{c_j ⊙ Q} (rank 4r, minimal exact).  "exact5": span{Q, w_k ⊙ Q} (rank 5r, GATE2 reference).
  "plain": S = span{Q} (C0-like, leaks through the conv).
"""
from __future__ import annotations

import json
import math
import os

import torch
from safetensors.torch import load_file, save_file

R = os.path.expanduser("~/Code/inkling-abl")
ORIG = f"{R}/methods/mpoa/orig_wo_conv.safetensors"
DIRS = f"{R}/methods/directions/directions.pt"
LAYERS = list(range(4, 24))
LOOP = set(range(6, 18))
LATE_VISITS = list(range(24, 48))


def wname(l: int) -> str:
    return f"model.llm.layers.{l}.attn.wo_ud.weight"


def cname(l: int) -> str:
    return f"model.llm.layers.{l}.attn_sconv.weight"


_cache: dict = {}


def orig() -> dict:
    if "orig" not in _cache:
        _cache["orig"] = load_file(ORIG)
    return _cache["orig"]


def dirs() -> dict:
    if "dirs" not in _cache:
        _cache["dirs"] = torch.load(DIRS, weights_only=False, map_location="cpu")
    return _cache["dirs"]


def orthonormal(M: torch.Tensor, tol: float = 1e-6) -> torch.Tensor:
    """Orthonormal basis of span(M) via SVD (fp64), dropping near-null directions."""
    U, S, _ = torch.linalg.svd(M.double(), full_matrices=False)
    keep = S > tol * S[0]
    return U[:, keep].float()


def unit(v: torch.Tensor) -> torch.Tensor:
    return v / v.norm()


# ---------- direction basis ----------

def protected_group(name: str, k: int) -> torch.Tensor:
    """Top-k orthonormal components of a protected group over late visits (kind 'out', effort 0)."""
    d = dirs()
    if name == "harmfulness":
        X = d["harmfulness"][0.0]["out"][LATE_VISITS]                     # [24, 2304]
    else:
        X = d["atoms"][0.0][name]["out"][LATE_VISITS][:, 0:4].reshape(-1, 2304)  # positions 0-3 only
    X = X / X.norm(dim=1, keepdim=True).clamp_min(1e-12)
    U, S, _ = torch.linalg.svd(X.T.double(), full_matrices=False)
    return U[:, :k].float()


def harmless_dir() -> torch.Tensor:
    """Mean harmless activation direction over late visits, positions 0-3 (kind 'out', effort 0)."""
    h = dirs()["means"][0.0]["out"]["harmless"][LATE_VISITS][:, 0:4]
    h = h / h.norm(dim=-1, keepdim=True)
    return unit(h.mean(dim=(0, 1)))


def basis(p: dict) -> tuple[torch.Tensor, dict]:
    """Refusal basis Q_r [2304, r'] from params. Returns (Q, diagnostics)."""
    d = dirs()
    r = int(p.get("r", 2))
    src = p.get("basis", "Q")
    Q = d["Q"] if src == "Q" else d["bases"]["call_24_47"]["out"][src]
    Q = Q[:, :r].clone()
    info: dict = {}
    tw = float(p.get("think_w", 0.0))
    if tw > 0:
        t = d["refusal_think"][0.9]["out"][LATE_VISITS].mean(0)
        t = unit(t)
        Q = torch.cat([Q, tw * t[:, None]], dim=1)
    if p.get("harmless_proj", True):
        h = harmless_dir()
        info["harmless_cos"] = [round(float(unit(Q[:, i]) @ h), 4) for i in range(Q.shape[1])]
        Q = Q - torch.outer(h, h @ Q)
    # C2: ridge orthogonalisation against protected groups. lam -> 0 removes the overlap, lam -> inf keeps Q.
    groups = p.get("protect", {})          # {"harmfulness": lam, "math": lam, ...}
    if groups:
        k = int(p.get("protect_k", 2))
        G = torch.cat([protected_group(g, k) for g in groups], dim=1).double()   # [2304, k*ng]
        lam = torch.tensor([float(groups[g]) for g in groups for _ in range(k)], dtype=torch.float64)
        Qd = Q.double()
        before = [float((G.T @ unit(Qd[:, i])).norm()) for i in range(Qd.shape[1])]
        coef = torch.linalg.solve(G.T @ G + torch.diag(lam), G.T @ Qd)
        Qd = Qd - G @ coef
        info["protect_overlap_before"] = [round(x, 4) for x in before]
        info["protect_overlap_after"] = [round(float((G.T @ unit(Qd[:, i])).norm()), 4) for i in range(Qd.shape[1])]
        Q = Qd.float()
    return orthonormal(Q), info


# ---------- per-layer subspace and strength ----------

def subspace(Q: torch.Tensor, l: int, mode: str) -> torch.Tensor:
    if mode == "plain":
        return orthonormal(Q)
    w = orig()[cname(l)].float().reshape(2304, 4)
    if mode == "exact4":
        cols = [(1.0 + w[:, 3])[:, None] * Q] + [w[:, 3 - j][:, None] * Q for j in (1, 2, 3)]
    elif mode.startswith("svd"):
        # partial fix: top (k*r) left singular vectors of the unnormalised exact generators [c_j ⊙ Q]
        k = int(mode[3:])
        G = torch.cat([(1.0 + w[:, 3])[:, None] * Q] + [w[:, 3 - j][:, None] * Q for j in (1, 2, 3)], dim=1)
        U, S, _ = torch.linalg.svd(G.double(), full_matrices=False)
        return U[:, : k * Q.shape[1]].float()
    elif mode == "exact5":
        cols = [Q] + [w[:, k][:, None] * Q for k in range(4)]
    else:
        raise ValueError(mode)
    return orthonormal(torch.cat(cols, dim=1))


def strengths(p: dict) -> dict[int, float]:
    """Smooth kernel over layers: floor + (peak-floor)*exp(-(l-mu)^2/(2 w^2)); loop layers x m; late layers x late_m."""
    if "s_list" in p:
        return {l: float(s) for l, s in zip(LAYERS, p["s_list"])}
    lo = int(p.get("l_min", 4))
    out = {}
    for l in LAYERS:
        if l < lo:
            out[l] = 0.0
            continue
        s = p["floor"] + (p["peak"] - p["floor"]) * math.exp(-((l - p["mu"]) ** 2) / (2 * p["width"] ** 2))
        if l in LOOP:
            s *= p.get("loop_m", 1.0)
        if l >= 18:
            s *= p.get("late_m", 1.0)
        out[l] = max(0.0, s)
    return out


# ---------- weights ----------

def apply_lowrank(W: torch.Tensor, A: torch.Tensor, B: torch.Tensor, r: torch.Tensor | None) -> torch.Tensor:
    """Reference for build_cand: W' = bf16(diag(r) (W_fp32 + A B^T)) in fp32 on CPU."""
    X = W.float() + A @ B.T
    if r is not None:
        X = X * r[:, None]
    return X.to(torch.bfloat16)


def solve_rowscale(W: torch.Tensor, U: torch.Tensor, s: float, r_max: float, iters: int = 60) -> torch.Tensor:
    """r such that rows of (I - s U U^T) diag(r) W have the original row norms (fixed point, fp64), r <= r_max.
    A few heavy conv-tap channels lie mostly inside S and cannot be restored; the cap keeps them sane."""
    Wd, Ud = W.double(), U.double()
    target = Wd.norm(dim=1)
    r = torch.ones(Wd.shape[0], dtype=torch.float64)
    for _ in range(iters):
        X = r[:, None] * Wd
        X = X - s * Ud @ (Ud.T @ X)
        n = X.norm(dim=1)
        r = (r * target / n).clamp(1.0 / r_max, r_max)
    return r.float()


def edit_layer(l: int, U: torch.Tensor, s: float, mp) -> dict:
    """mp: False/'none' | True/'pre' (row scale before projection: exact, capped) | 'post' (rescale after: leaks)."""
    W = orig()[wname(l)].float()
    mode = {True: "pre", False: "none"}.get(mp, mp)
    if mode == "pre":
        r = solve_rowscale(W, U, s, r_max=1.5)
        A = (-s * (1.0 / r))[:, None] * U
        B = W.T @ (r[:, None] * U)
    elif mode == "post":
        X = W - s * U @ (U.T @ W)
        r = W.norm(dim=1) / X.norm(dim=1)
        A, B = -s * U, W.T @ U
    elif mode == "none":
        r = None
        A, B = -s * U, W.T @ U
    else:
        raise ValueError(mp)
    return {"A": A.contiguous(), "B": B.contiguous(), "r": None if r is None else r.contiguous()}


def build(p: dict) -> tuple[dict, dict]:
    """params -> ({tensor_name.A/.B/.r: fp32}, diagnostics)."""
    Q, info = basis(p)
    mode = p.get("conv", "exact4")
    mp = p.get("mp", "pre")
    S = strengths(p)
    lr: dict[str, torch.Tensor] = {}
    diag = {"basis": info, "rank_r": Q.shape[1], "s": {}, "k": {}}
    exact_layers = set(p.get("exact_layers", LAYERS))   # layers outside get conv="plain"
    for l in LAYERS:
        s = S[l]
        if s == 0.0:
            continue
        U = subspace(Q, l, mode if l in exact_layers else "plain")
        e = edit_layer(l, U, s, mp)
        n = wname(l)
        lr[f"{n}.A"], lr[f"{n}.B"] = e["A"], e["B"]
        if e["r"] is not None:
            lr[f"{n}.r"] = e["r"]
        diag["s"][l] = round(s, 4)
        diag["k"][l] = U.shape[1]
    return lr, diag


def materialise(lr: dict) -> dict[str, torch.Tensor]:
    out = {}
    for k in lr:
        if k.endswith(".A"):
            n = k[:-2]
            out[n] = apply_lowrank(orig()[n], lr[n + ".A"], lr[n + ".B"], lr.get(n + ".r"))
    return out


def leak(W_new: torch.Tensor, l: int, Q: torch.Tensor, T: int = 64, seed: int = 0) -> float:
    """Post-conv leak ||Q^T y'|| / ||Q^T y|| on random gaussian inputs x (fp32), for s = 1 sanity checks."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(T, 3072, generator=g)
    w = orig()[cname(l)].float().reshape(2304, 4)

    def conv(h):
        y = h.clone() * (1 + w[:, 3])
        for j in (1, 2, 3):
            y[j:] += w[:, 3 - j] * h[:-j]
        return y / math.sqrt(5)
    y0 = conv(x @ orig()[wname(l)].float().T)
    y1 = conv(x @ W_new.float().T)
    return float((y1 @ Q).norm() / (y0 @ Q).norm())


def write_candidate(cand_id: str, family: str, p: dict, parent: str | None = None) -> dict:
    d = f"{R}/methods/mpoa/cands/{cand_id}"
    os.makedirs(d, exist_ok=True)
    lr, diag = build(p)
    W = materialise(lr)
    Q, _ = basis(p)
    diag["leak"] = {l: round(leak(W[wname(l)], l, Q), 4) for l in LAYERS if wname(l) in W}
    diag["rownorm_ratio_min"] = {l: round(float((W[wname(l)].float().norm(dim=1) / orig()[wname(l)].float().norm(dim=1)).min()), 3)
                                 for l in LAYERS if wname(l) in W}
    # per-tensor sha256 of the intended BF16 bytes, for the build_cand bit-exact check
    diag["tensor_sha256"] = {n: __import__("hashlib").sha256(t.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
                             for n, t in W.items()}
    save_file(lr, f"{d}/lowrank.safetensors")
    spec = {"cand_id": cand_id, "family": family, "parent": parent,
            "created": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(timespec="seconds"),
            "agent": "mpoa", "params": p, "kind": "weights", "diag": diag}
    json.dump(spec, open(f"{d}/spec.json", "w"), indent=1, default=str)
    return {"dir": d, "lr": lr, "diag": diag}
