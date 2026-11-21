

set_conds(["base"])
if os.path.exists(BASE_PATH):
    d = torch.load(BASE_PATH)
    conts, base_lp = d["conts"], d["lp"]
else:
    conts, base_lp = build_base()
    log("kl base built")

CONDS = sys.argv[1:]
kls = {}
for cond in CONDS:
    set_conds([cond] * 10)
    kls[cond] = kl_run(conts, base_lp)
    log("KL", cond, kls[cond])
gens = refusal_rows(CONDS, int(os.environ.get("ROWS_BS", "240")))
for cond in CONDS:
    gs = gens[cond]
    ans = [g for g in gs if g["answer"] is not None]
    kl, top1, npos = kls[cond]
    row = {"cond": cond, "kl": kl, "top1": top1, "kl_pos": npos, "n": len(gs),
           "refusal_seed": sum(refused(g["answer"] or "", phr_seed) for g in gs) / len(gs),
           "refusal_full": sum(refused(g["answer"] or "", phr_full) for g in gs) / len(gs),
           "no_answer": len(gs) - len(ans),
           "refusal_full_of_answered": sum(refused(g["answer"], phr_full) for g in ans) / max(1, len(ans)),
           "mean_think": sum(g["think_len"] for g in gs) / len(gs)}
    with open(OUT, "a") as f:
        f.write(json.dumps(row) + "\n")
    json.dump(gs, open(f"/root/exp/out/gens_{cond}.json", "w"), indent=0)
    log("ROW", json.dumps(row))
log("HOOKS_DONE")
