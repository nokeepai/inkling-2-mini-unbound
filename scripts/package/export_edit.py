"""Export a candidate's edit as edit.pt, a PEFT LoRA adapter and a PEFT DoRA adapter, and measure exactness.

  python3 export_edit.py <cand_dir> <orig_dir> <out_dir> [--build-cand PATH] [--name-map hf_name_map.json]
                         [--ranks 2,4,...] [--lora-rank auto|N] [--peft-check]

Edit (CANDIDATE_FORMAT compact form), per edited BF16 tensor W [out,in]:
    W' = bf16( diag(r) · (W_fp32 + A·Bᵀ) )      A [out,k], B [in,k], r [out] (r = 1 if absent)
Outputs in <out_dir>:
  edit.pt                 the edit: A, B, r per tensor, per-layer output subspace and strength, layer map,
                          router edits, hashes; apply with apply_edit.py (bit-exact = build_cand)
  lora/                   PEFT LoRA (adapter_model.safetensors + adapter_config.json): best rank-k approx of the
                          exact delta W' − W (SVD), k chosen per --lora-rank
  dora/                   PEFT DoRA: lora_B = A, lora_A = Bᵀ, magnitude m = r·‖W + A·Bᵀ‖_row. Same function as the
                          edit in exact arithmetic (DoRA's row renormalisation IS the row scale r)
  exactness.json          per tensor and rank: max |error| in BF16 ULPs of W', elements that differ after merge
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import time

import torch
from safetensors.torch import load_file, save_file

DEFAULT_RANKS = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 1536, 2304]
PEFT_PREFIX = "base_model.model."


def load_mod(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def tsha(t: torch.Tensor) -> str:
    return hashlib.sha256(t.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def ulp_bf16(x: torch.Tensor) -> torch.Tensor:
    """Spacing of BF16 numbers at |x| (fp32 result); zero maps to the smallest normal spacing."""
    a = x.float().abs().clamp_min(torch.finfo(torch.bfloat16).tiny)
    e = torch.floor(torch.log2(a))
    return torch.pow(2.0, e - 7)


def layer_of(name: str) -> int:
    return int(re.search(r"layers\.(\d+)\.", name).group(1))


def kind_of(name: str) -> str:
    if name.endswith("attn.wo_ud.weight"):
        return "attn.wo_ud"
    if name.endswith("mlp.gate.weight"):
        return "router"
    return "other"


# ---------- edit.pt ----------

def output_subspace(A: torch.Tensor) -> torch.Tensor:
    U, S, _ = torch.linalg.svd(A.double(), full_matrices=False)
    return U[:, S > 1e-9 * S[0]].float()


def make_edit(spec: dict, parts: dict, W: dict, Wn: dict) -> dict:
    s_diag = {int(k): float(v) for k, v in spec.get("diag", {}).get("s", {}).items()}
    tensors, layer_map, directions, router = {}, {}, {}, {}
    for n in sorted(parts, key=lambda x: (layer_of(x), x)):
        p, l, kd = parts[n], layer_of(n), kind_of(n)
        rec = {"A": p["A"].clone(), "B": p["B"].clone(), "layer": l, "kind": kd, "rank": int(p["A"].shape[1]),
               "sha256_orig": tsha(W[n]), "sha256_edited": tsha(Wn[n])}
        if "r" in p:
            rec["r"] = p["r"].clone()
        tensors[n] = rec
        layer_map.setdefault(l, []).append(n)
        if kd == "attn.wo_ud":
            directions[l] = output_subspace(p["A"])  # columns span the hidden-space subspace the edit writes into
        elif kd == "router":
            router[n] = {"layer": l, "rows_touched": int((p["A"].abs().sum(1) > 0).sum())}
    return {
        "format": "inkling-abl-edit/1",
        "cand_id": spec.get("cand_id"), "family": spec.get("family"), "params": spec.get("params"),
        "formula": "W' = bf16( diag(r) * (W.float() + A @ B.T) ); fp32 on CPU, single thread; r = 1 if absent",
        "base_model": "thinkingmachines/Inkling-2-Mini",
        "tensors": tensors, "layer_map": layer_map,
        "directions": directions,
        "strengths": {l: s_diag[l] for l in sorted(s_diag)},
        "router_edits": router,
        "note": ("Only the listed BF16 tensors change; every expert (NVFP4/MXFP8) and every other tensor is untouched. "
                 "directions[l] is an orthonormal basis of span(A) for layer l's attn.wo_ud edit (hidden space, "
                 "2304-d). For MPOA families: W' = (I - s U U^T) diag(r) W with A = -s diag(1/r) U, B = W^T diag(r) U."),
    }


# ---------- exactness of low-rank / DoRA forms ----------

def merge_err(W: torch.Tensor, Wn: torch.Tensor, approx_delta: torch.Tensor) -> dict:
    """Reference merge: bf16(W_fp32 + delta_fp32) once. Error measured against the canonical W'."""
    merged = (W.float() + approx_delta).to(torch.bfloat16)
    ulp = ulp_bf16(Wn)
    err = (W.float() + approx_delta - Wn.float()).abs() / ulp
    return {"max_err_ulp_prebf16": round(float(err.max()), 4),
            "n_diff_after_bf16_merge": int((merged != Wn).sum()),
            "max_diff_ulp_after_merge": round(float(((merged.float() - Wn.float()).abs() / ulp).max()), 4)}


def lora_sweep(W: torch.Tensor, Wn: torch.Tensor, ranks: list[int]) -> tuple[dict, tuple]:
    D = Wn.double() - W.double()  # exact delta (difference of two BF16 values)
    U, S, Vh = torch.linalg.svd(D, full_matrices=False)
    rows = {}
    for k in ranks:
        k = min(k, S.numel())
        approx = ((U[:, :k] * S[:k]) @ Vh[:k]).float()
        rows[k] = merge_err(W, Wn, approx) | {"rel_fro_err": round(float((approx.double() - D).norm() / D.norm()), 6)}
    return rows, (U, S, Vh)


def min_rank(rows: dict, key: str, thresh: float) -> int | None:
    ok = [k for k, v in sorted(rows.items()) if v[key] <= thresh]
    return ok[0] if ok else None


def dora_parts(W: torch.Tensor, p: dict) -> dict:
    X = W.float() + p["A"] @ p["B"].T
    r = p.get("r", torch.ones(W.shape[0]))
    return {"lora_B": p["A"].clone(), "lora_A": p["B"].T.contiguous(), "mag": (r * X.norm(dim=1)).contiguous()}


def dora_err(W: torch.Tensor, Wn: torch.Tensor, d: dict) -> dict:
    X = W.float() + d["lora_B"] @ d["lora_A"]
    approx = (d["mag"] / X.norm(dim=1))[:, None] * X - W.float()
    return merge_err(W, Wn, approx)


# ---------- PEFT files ----------

def hf_module(name_map: dict, n: str) -> tuple[str, str]:
    m = name_map[n]
    mod = m["hf"].removesuffix(".weight")
    return mod, m.get("hf_dtype", "bfloat16")


def write_adapter(out: str, tensors: dict, modules: dict[str, int], dora: bool) -> None:
    os.makedirs(out, exist_ok=True)
    save_file(tensors, os.path.join(out, "adapter_model.safetensors"), metadata={"format": "pt"})
    ranks = sorted(set(modules.values()))
    base_r = ranks[0]
    cfg = {"peft_type": "LORA", "task_type": None, "base_model_name_or_path": "thinkingmachines/Inkling-2-Mini",
           "r": base_r, "lora_alpha": base_r, "lora_dropout": 0.0, "bias": "none", "fan_in_fan_out": False,
           "target_modules": sorted(modules), "rank_pattern": {m: k for m, k in modules.items() if k != base_r},
           "alpha_pattern": {m: k for m, k in modules.items() if k != base_r},  # scaling = alpha/r = 1 everywhere
           "use_dora": dora, "use_rslora": False, "init_lora_weights": True, "modules_to_save": None,
           "inference_mode": True}
    json.dump(cfg, open(os.path.join(out, "adapter_config.json"), "w"), indent=1)


def peft_native_check(W: torch.Tensor, Wn: torch.Tensor, lA: torch.Tensor, lB: torch.Tensor, mag) -> dict:
    """Merge with PEFT itself on a bare nn.Linear in the base dtype; count elements that differ from W'."""
    import peft
    from torch import nn

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(W.shape[1], W.shape[0], bias=False, dtype=W.dtype)

    m = M()
    with torch.no_grad():
        m.lin.weight.copy_(W)
    k = lA.shape[0]
    cfg = peft.LoraConfig(r=k, lora_alpha=k, target_modules=["lin"], use_dora=mag is not None, init_lora_weights=False)
    pm = peft.get_peft_model(m, cfg)
    layer = pm.base_model.model.lin
    with torch.no_grad():
        layer.lora_A["default"].weight.copy_(lA)
        layer.lora_B["default"].weight.copy_(lB)
        if mag is not None:
            layer.lora_magnitude_vector["default"].weight.copy_(mag)
    merged = pm.merge_and_unload().lin.weight.detach()
    d = (merged.float() - Wn.float()).abs() / ulp_bf16(Wn)
    return {"peft_version": peft.__version__, "lora_dtype": str(lA.dtype), "n_diff": int((merged != Wn).sum()),
            "max_diff_ulp": round(float(d.max()), 3)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cand"), ap.add_argument("orig"), ap.add_argument("out")
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--build-cand", default=os.path.join(here, "build_cand.py"))
    ap.add_argument("--name-map", default=os.path.join(here, "hf_name_map.json"))
    ap.add_argument("--ranks", default=",".join(map(str, DEFAULT_RANKS)))
    ap.add_argument("--lora-rank", default="auto", help="auto = smallest swept rank with max error <= 1 ULP")
    ap.add_argument("--peft-check", action="store_true")
    ap.add_argument("--plain-lora", action="store_true", help="also write lora/ (full rank for row-scaled edits; off by default, lead 2026-11-21)")
    a = ap.parse_args()
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    t0 = time.time()
    bc = load_mod(a.build_cand, "build_cand")
    parts = bc.parse(load_file(os.path.join(a.cand, "lowrank.safetensors")))
    spec = json.load(open(os.path.join(a.cand, "spec.json")))
    Wn = bc.build(parts, a.orig)  # canonical edited tensors (sets 1 thread)
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    files = bc.weight_files(a.orig)
    W = {}
    from safetensors import safe_open
    for n in Wn:
        with safe_open(files[n], "pt") as f:
            W[n] = f.get_tensor(n)
    os.makedirs(a.out, exist_ok=True)
    edit = make_edit(spec, parts, W, Wn)
    torch.save(edit, os.path.join(a.out, "edit.pt"))
    # round trip: edit.pt alone must rebuild W' bit-exactly
    rt = torch.load(os.path.join(a.out, "edit.pt"), weights_only=False)
    rt_parts = {n: {k: v for k, v in t.items() if k in ("A", "B", "r")} for n, t in rt["tensors"].items()}
    rebuilt = bc.build(rt_parts, a.orig)
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    edit_exact = all(torch.equal(rebuilt[n], Wn[n]) for n in Wn)

    name_map = json.load(open(a.name_map))
    ranks = [int(x) for x in a.ranks.split(",")]
    report, lora_t, dora_t, lora_mods, dora_mods, svds = {}, {}, {}, {}, {}, {}
    for n in sorted(Wn, key=lambda x: (layer_of(x), x)):
        rows, svds[n] = lora_sweep(W[n], Wn[n], ranks)
        d = dora_parts(W[n], parts[n])
        report[n] = {"lora": rows, "dora": dora_err(W[n], Wn[n], d) | {"rank": int(d["lora_A"].shape[0])},
                     "min_rank_le_1ulp": min_rank(rows, "max_err_ulp_prebf16", 1.0),
                     "min_rank_exact_merge": min_rank(rows, "n_diff_after_bf16_merge", 0),
                     "delta_rank_numerical": int((svds[n][1] > svds[n][1][0] * 1e-6).sum())}
        mod, hf_dtype = hf_module(name_map, n)
        report[n]["hf_module"], report[n]["hf_dtype"] = mod, hf_dtype
        dora_t[f"{PEFT_PREFIX}{mod}.lora_A.weight"] = d["lora_A"]
        dora_t[f"{PEFT_PREFIX}{mod}.lora_B.weight"] = d["lora_B"]
        dora_t[f"{PEFT_PREFIX}{mod}.lora_magnitude_vector"] = d["mag"]
        dora_mods[mod] = int(d["lora_A"].shape[0])
        print(n, "min_rank<=1ulp", report[n]["min_rank_le_1ulp"], "exact", report[n]["min_rank_exact_merge"],
              "dora", report[n]["dora"], flush=True)

    # one rank for every tensor: the smallest swept rank meeting <= 1 ULP on all tensors (else the largest swept)
    if a.lora_rank == "auto":
        need = [report[n]["min_rank_le_1ulp"] for n in report]
        k = max(need) if all(x is not None for x in need) else max(ranks)
    else:
        k = int(a.lora_rank)
    for n in report:
        U, S, Vh = svds[n]
        kk = min(k, S.numel())
        sq = S[:kk].sqrt()
        mod = report[n]["hf_module"]
        lora_t[f"{PEFT_PREFIX}{mod}.lora_B.weight"] = (U[:, :kk] * sq).float().contiguous()
        lora_t[f"{PEFT_PREFIX}{mod}.lora_A.weight"] = (sq[:, None] * Vh[:kk]).float().contiguous()
        lora_mods[mod] = kk
        report[n]["shipped_lora"] = merge_err(W[n], Wn[n], (lora_t[f"{PEFT_PREFIX}{mod}.lora_B.weight"] @ lora_t[f"{PEFT_PREFIX}{mod}.lora_A.weight"]))
    if a.plain_lora:
        write_adapter(os.path.join(a.out, "lora"), lora_t, lora_mods, dora=False)
    write_adapter(os.path.join(a.out, "dora"), dora_t, dora_mods, dora=True)

    if a.peft_check and a.plain_lora:
        n = sorted(Wn, key=layer_of)[0]
        mod = report[n]["hf_module"]
        report["_peft_native"] = {
            "tensor": n,
            "lora": peft_native_check(W[n], Wn[n], lora_t[f"{PEFT_PREFIX}{mod}.lora_A.weight"], lora_t[f"{PEFT_PREFIX}{mod}.lora_B.weight"], None),
            "dora": peft_native_check(W[n], Wn[n], dora_t[f"{PEFT_PREFIX}{mod}.lora_A.weight"], dora_t[f"{PEFT_PREFIX}{mod}.lora_B.weight"], dora_t[f"{PEFT_PREFIX}{mod}.lora_magnitude_vector"]),
        }
    summary = {"cand_id": spec.get("cand_id"), "n_tensors": len(Wn), "edit_pt_bit_exact": edit_exact,
               "plain_lora_written": a.plain_lora, "lora_rank_shipped": k,
               "lora_exact": all(report[n]["shipped_lora"]["n_diff_after_bf16_merge"] == 0 for n in Wn),
               "lora_max_err_ulp_prebf16": max(report[n]["shipped_lora"]["max_err_ulp_prebf16"] for n in Wn),
               "lora_n_diff_total": sum(report[n]["shipped_lora"]["n_diff_after_bf16_merge"] for n in Wn),
               "dora_n_diff_total": sum(report[n]["dora"]["n_diff_after_bf16_merge"] for n in Wn),
               "dora_max_err_ulp_prebf16": max(report[n]["dora"]["max_err_ulp_prebf16"] for n in Wn),
               "elements_total": sum(t.numel() for t in Wn.values()),
               "sizes_bytes": {x: os.path.getsize(os.path.join(a.out, x)) for x in ("edit.pt", "lora/adapter_model.safetensors", "dora/adapter_model.safetensors") if os.path.exists(os.path.join(a.out, x))},
               "seconds": round(time.time() - t0, 1)}
    json.dump({"summary": summary, "per_tensor": report}, open(os.path.join(a.out, "exactness.json"), "w"), indent=1, default=str)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
