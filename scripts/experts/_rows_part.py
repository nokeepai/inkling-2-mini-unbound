
# ---- row-masked conditions: every row of a batch carries its own condition (as diag's attribution) ----
st = {"B": 1, "conds": ["base"]}
RF = {}  # per-batch tensors built by set_conds


def parse(cond):
    o = {"attn": False, "moepost": False, "moepreS": False, "topexp": 0, "mask": 0, "cond": (0, 0.0)}
    for part in cond.split("+"):
        if part in ("base", ""):
            continue
        if part in ("attn", "moepost", "moepreS"):
            o[part] = True
        elif part.startswith("mask"):
            o["mask"] = int(part[4:])
        elif part.startswith("cond"):
            k, d = part[4:].split("_d")
            o["cond"] = (int(k), float(d))
        elif part.startswith("topexp"):
            o["topexp"] = int(part[6:])
        else:
            raise ValueError(part)
    return o


def set_conds(conds):
    st["B"], st["conds"] = len(conds), conds
    ps = [parse(c) for c in conds]
    dev = "cuda:0"
    for k in ("attn", "moepost", "moepreS"):
        RF[k] = torch.tensor([p[k] for p in ps], device=dev)
    RF["any"] = {k: bool(RF[k].any()) for k in ("attn", "moepost", "moepreS")}
    RF["A"], RF["MB"], RF["TE"] = {}, {}, {}
    for l in range(24):
        A = torch.zeros(len(ps), 128, device=dev)
        MB = torch.zeros(len(ps), 128, device=dev)
        TE = {}
        for r, p in enumerate(ps):
            K, d = p["cond"]
            for (ll, e) in PAIRS[:K]:
                if ll == l:
                    A[r, e] = d / HGAP[l]
            for (ll, e) in PAIRS[:p["mask"]]:
                if ll == l:
                    MB[r, e] = -1e4
            for (ll, e) in PAIRS[:p["topexp"]]:
                if ll == l:
                    TE.setdefault(e, torch.zeros(len(ps), dtype=torch.bool, device=dev))[r] = True
        RF["A"][l] = A if bool(A.any()) else None
        RF["MB"][l] = MB if bool(MB.any()) else None
        RF["TE"][l] = TE


def rowidx(n):
    B = st["B"]
    return torch.arange(n, device="cuda:0") // (n // B)


def proj_rows(o, Bm, key):
    if not RF["any"][key]:
        return None
    y = o.float()
    yp = y - (y @ Bm) @ Bm.T
    m = RF[key][:, None, None]
    return torch.where(m, yp, y).to(o.dtype)


hooks = []
for i in range(24):
    L = tm.layers[i]
    hooks.append(L.post_attention_convolution.register_forward_hook(lambda m, a, o: proj_rows(o, Q, "attn")))
    hooks.append(L.post_mlp_convolution.register_forward_hook(lambda m, a, o: proj_rows(o, Q, "moepost")))
    hooks.append(L.mlp.register_forward_hook((lambda i: lambda m, a, o: proj_rows(o, S[i], "moepreS"))(i)))


def topexp_hook(i):
    def hook(m, inp, out):
        TE = RF["TE"][i]
        if not TE:
            return None
        hs, idx, wts = inp
        ri = rowidx(hs.shape[0])
        corr = torch.zeros_like(out, dtype=torch.float32)
        for ex, rmask in TE.items():
            tokm = rmask[ri]
            rows, slots = torch.where((idx == ex) & tokm[:, None])
            if rows.numel() == 0:
                continue
            h = m._apply_gate(torch.nn.functional.linear(hs[rows], m.gate_up_proj[ex]))
            h = torch.nn.functional.linear(h, m.down_proj[ex]).float() * wts[rows, slots, None].float()
            corr.index_add_(0, rows, (h @ S[i]) @ S[i].T)
        return (out.float() - corr).to(out.dtype)
    return hook


for i in range(24):
    hooks.append(tm.layers[i].mlp.experts.register_forward_hook(topexp_hook(i)))


def make_router_forward(g, l):
    import torch.nn.functional as F

    def forward(hidden_states):
        flat = hidden_states.float().reshape(-1, g.hidden_dim)
        logits = F.linear(flat, g.weight)
        A, MB = RF["A"][l], RF["MB"][l]
        ri = rowidx(flat.shape[0]) if (A is not None or MB is not None) else None
        if A is not None:
            logits = logits - A[ri] * (flat @ HDIR[l].float())[:, None]
        scores = g.score_fn(logits)
        b = g.e_score_correction_bias if MB is None else g.e_score_correction_bias + MB[ri]
        indices = torch.topk(scores + b, g.top_k, dim=-1, sorted=False).indices
        weights = scores.gather(1, indices)
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
        return logits, weights * g.routed_scaling_factor, indices
    return forward


for l in range(24):
    g = tm.layers[l].mlp.gate
    g.forward = make_router_forward(g, l)


def cut(g, stops):
    out = []
    for x in g:
        if x in stops:
            return out, True
        out.append(x)
    return out, False


@torch.no_grad()
def refusal_rows(conds, bs):
    rows = [(c, k) for c in conds for k in range(len(hseqs))]
    res = {c: [] for c in conds}
    for b in range(0, len(rows), bs):
        chunk = rows[b:b + bs]
        set_conds([c for c, _ in chunk])
        ids, mask = left_pad([hseqs[k] for _, k in chunk], PAD)
        t0 = time.time()
        o = model.generate(input_ids=ids, attention_mask=mask, do_sample=False, max_new_tokens=THINK_CAP + ANS,
                           pad_token_id=PAD, eos_token_id=[EOS])
        log("gen chunk", b, len(chunk), f"{time.time() - t0:.0f}s")
        for r, (c, k) in enumerate(chunk):
            g = o[r, ids.shape[1]:].tolist()
            t, e = cut(g, (END_MSG, PAD, EOS))
            rest = g[len(t) + 1:]
            ans = tok.decode(cut(rest[2:2 + ANS], (END_MSG, PAD, EOS))[0]) if rest[:2] == [MSG_MODEL, TEXT] else None
            res[c].append({"id": items[k]["id"], "think_len": len(t), "answer": ans})
        del o
        torch.cuda.empty_cache()
    return res

