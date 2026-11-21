"""Auditor (Mac): quick core numbers from raw gens in the private dir (phrase refusal as already scored on the pod,
per category max, Jev useful via harness/judge_jev.py). For the early core table before metrics.json exists.

  python3 quick_core.py <evaldir> <mode: off|e09> [--jev]
"""
import json
import sys

sys.path.insert(0, "/Users/nikshepsvn/Code/inkling-abl/harness")
P = "/Users/nikshepsvn/.inkling-abl/audit_private/test"


def main() -> None:
    d, mode = sys.argv[1], sys.argv[2]
    rows = [json.loads(l) for l in open(f"{P}/{d}/gens_harmful_{mode}.jsonl")]
    import judge_jev as J
    ref = [float(r["refusal"]) for r in rows]
    cats = {}
    for r in rows:
        cats.setdefault(r["category"], []).append(float(r["refusal"]))
    worst = max(cats, key=lambda c: sum(cats[c]) / len(cats[c]))
    out = {"cand": d, "mode": mode, "refusal": round(sum(ref) / len(ref), 4), "CI": J.boot(ref), "n": len(ref),
           "refusals": int(sum(ref)), "worst_cat": worst, "worst_cat_rate": round(sum(cats[worst]) / len(cats[worst]), 4),
           "cap_hit": round(sum(r["cap_hit"] for r in rows) / len(rows), 4),
           "loop": round(sum(r["loop"] for r in rows) / len(rows), 4)}
    if "--jev" in sys.argv:
        res = J.judge([{"prompt": r["prompt"], "answer": r["answer"]} for r in rows])
        bad = [j for j in res if j.get("error")]
        if bad:
            raise SystemExit(f"{len(bad)} Jev errors")
        u = [float(j["useful"]) for j in res]
        out.update({"jev_useful": round(sum(u) / len(u), 4), "jev_CI": J.boot(u),
                    "jev_pfulfils": round(sum(j["p_fulfils"] for j in res) / len(res), 4)})
        with open(f"{P}/{d}/jev_harmful_{mode}.jsonl", "w") as f:
            for r, j in zip(rows, res):
                f.write(json.dumps({"id": r["id"], "category": r["category"], "phrase_refusal": r["refusal"],
                                    "p_fulfils": j["p_fulfils"], "useful": j["useful"]}) + "\n")
    print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
