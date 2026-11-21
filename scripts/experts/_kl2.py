@torch.no_grad()
def tf_logp(idx, conts):
    seqs = [vprompts[i] + conts[i] for i in idx]
    T = max(len(s) for s in seqs)
    ids = torch.full((len(idx), T), 199999, dtype=torch.long)
    mask = torch.zeros((len(idx), T), dtype=torch.long)
    for j, s in enumerate(seqs):
        ids[j, :len(s)] = torch.tensor(s)
        mask[j, :len(s)] = 1
    h = model.model(input_ids=ids.cuda(), attention_mask=mask.cuda(), use_cache=False).last_hidden_state
    out = []
    for j, i in enumerate(idx):
        p, L = len(vprompts[i]), len(conts[i])
        out.append(model.logit_softcap(model.lm_head(h[j, p - 1:p - 1 + L]).float()).log_softmax(-1))
    return out


BASE_PATH = "/root/exp/out/kl_base.pt"


@torch.no_grad()
def build_base():
    conts = []
    for b in range(0, 50, 25):
        sub = vprompts[b:b + 25]
        ids, mask = left_pad(sub, PAD)
        o = model.generate(input_ids=ids, attention_mask=mask, do_sample=False, max_new_tokens=256,
                           pad_token_id=PAD, eos_token_id=[EOS])
        for r in range(o.shape[0]):
            c = o[r, ids.shape[1]:].tolist()
            c = c[:c.index(EOS) + 1] if EOS in c else c
            c = [x for x in c if x != PAD] if PAD != EOS else c
            conts.append(c)
    lps = []
    for b in range(0, 50, 10):
        lps += [x.half().cpu() for x in tf_logp(list(range(b, min(50, b + 10))), conts)]
    torch.save({"conts": conts, "lp": lps}, BASE_PATH)
    return conts, lps


@torch.no_grad()
def kl_run(conts, base_lp):
    tot, n, agree = 0.0, 0, 0
    for b in range(0, 50, 10):
        idx = list(range(b, min(50, b + 10)))
        for i, q in zip(idx, tf_logp(idx, conts)):
            p = base_lp[i].cuda().float()
            p = p.log_softmax(-1)
            tot += float((p.exp() * (p - q)).sum())
            n += p.shape[0]
            agree += int((p.argmax(-1) == q.argmax(-1)).sum())
    return tot / n, agree / n, n

