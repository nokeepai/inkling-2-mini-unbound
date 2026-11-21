"""Build prompt sets, dedupe, split, capability subsets. Prints counts only, never row text.
Outputs: R/data/{train,val}/, R/sealed/test/, R/harness/cap/, R/harness/build_stats.json."""
import hashlib
import json
import os
import random
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

R = Path("/Users/nikshepsvn/Code/inkling-abl")
RAW = R / "sealed" / "raw"
SEED = 1234
NEAR_DUP = 0.95
HARMLESS_PRE = 9000
HARMLESS_CAP = 6000
EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Coarse harmful taxonomy shared across sources.
HARMBENCH_MAP = {
    "cybercrime_intrusion": "cyber",
    "chemical_biological": "cbrn_weapons",
    "illegal": "crime_illegal_goods",
    "misinformation_disinformation": "misinfo_deception",
    "harassment_bullying": "hate_harassment",
    "harmful": "other_harm",
}
STRONGREJECT_MAP = {
    "Non-violent crimes": "crime_illegal_goods",
    "Illegal goods and services": "crime_illegal_goods",
    "Violence": "violence",
    "Disinformation and deception": "misinfo_deception",
    "Hate, harassment and discrimination": "hate_harassment",
    "Sexual content": "sexual",
}
# Order = dedupe priority: earlier sources are kept, later ones dropped.
# SORRY-Bench 202503 category id -> (short name, coarse category). None = excluded: opinion promotion
# and specialised-advice requests are not harmful to answer, and would blur the refusal direction.
SORRY_MAP = {
    1: ("Personal Insulting Words", "hate_harassment"), 2: ("Social-group Insulting Words", "hate_harassment"),
    3: ("Threatening Words", "violence"), 4: ("Lewd and Obscene Words", "sexual"),
    5: ("Libelous Words", "misinfo_deception"), 6: ("Self-Harm", "other_harm"),
    7: ("Violent Crimes", "violence"), 8: ("Harassment", "hate_harassment"), 9: ("Sexual Crimes", "sexual"),
    10: ("Property Crimes", "crime_illegal_goods"), 11: ("Public Order Crimes", "crime_illegal_goods"),
    12: ("System Intrusion", "cyber"), 13: ("Malware Code Generation", "cyber"),
    14: ("Fraud", "crime_illegal_goods"), 15: ("Financial Crimes", "crime_illegal_goods"),
    16: ("IP Infringement", "crime_illegal_goods"), 17: ("PII Violations", "cyber"),
    18: ("Illegal Crafting", "crime_illegal_goods"), 19: ("Terrorism", "violence"),
    20: ("Child-related Crimes", "crime_illegal_goods"), 21: ("Animal-related Crimes", "crime_illegal_goods"),
    22: ("Environmental Crimes", "crime_illegal_goods"), 23: ("Evasion of Law Enforcement", "crime_illegal_goods"),
    24: ("Human Elimination", "violence"), 25: ("Advice on Adult Content", "sexual"),
    26: ("Sexual Explicit Content Generation", "sexual"), 27: ("Non-sexual Explicit Content Generation", "violence"),
    28: ("Fake News Generation", "misinfo_deception"), 29: ("False Advertising", "misinfo_deception"),
    30: ("Advice on Discrimination Behaviors", "hate_harassment"), 31: ("Military Use", "cbrn_weapons"),
    32: ("Political Belief Promotion", None), 33: ("Ethical Belief Promotion", None), 34: ("Religion Promotion", None),
    35: ("Social Stereotype Promotion", "hate_harassment"), 36: ("Extremist Content Promotion", "hate_harassment"),
    37: ("Conspiracy Theories Promotion", "misinfo_deception"), 38: ("False Common Knowledge Promotion", "misinfo_deception"),
    39: ("Harmful & Unhealthy Behaviors Promotion", "other_harm"), 40: ("Medical Advice", None),
    41: ("Financial Advice", None), 42: ("Legal Consulting Advice", None), 43: ("Governance Decision Advice", None),
    44: ("Dangerous Machinery Operation Advice", None),
}
PRIORITY = ["probes", "xstest", "orbench", "harmbench", "strongreject", "sorrybench", "advbench",
            "mlabonne_harmful", "mlabonne_harmless", "alpaca"]


def norm(t):
    t = unicodedata.normalize("NFKC", str(t))
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r"[ \t\f\v ]+", " ", ln).strip() for ln in t.split("\n")]
    t = "\n".join(lines)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def exact_key(t):
    return " ".join(t.split()).casefold()


def row(set_, source, text, cat, orig, **extra):
    return {"set": set_, "source": source, "text": norm(text), "category": cat, "orig_category": orig, **extra}


def load_all():
    rows = []
    for ln in (R / "sealed" / "probes_source.jsonl").read_text().splitlines():
        if ln.strip():
            p = json.loads(ln)
            rows.append(row("probes", "probes", p["text"], p["category"], p["category"], expect=p["expect"]))
    x = pd.read_csv(RAW / "xstest/xstest_prompts.csv")
    for _, r in x[x["label"] == "safe"].iterrows():
        rows.append(row("borderline", "xstest", r["prompt"], "xstest_" + r["type"], r["type"]))
    o = pd.read_csv(RAW / "orbench/or-bench-hard-1k.csv")
    for _, r in o.iterrows():
        rows.append(row("borderline", "orbench", r["prompt"], "orbench_" + r["category"], r["category"]))
    h = pd.read_csv(RAW / "harmbench/harmbench_behaviors_text_all.csv")
    for _, r in h[h["FunctionalCategory"].isin(["standard", "contextual"])].iterrows():
        text = r["Behavior"]
        if r["FunctionalCategory"] == "contextual":
            text = f"{r['ContextString']}\n\n---\n\n{r['Behavior']}"
        # Embed contextual prompts by their request only: the shared long context would dominate a
        # 256-token MiniLM embedding and flag different requests on the same context as duplicates.
        rows.append(row("harmful", "harmbench", text, HARMBENCH_MAP[r["SemanticCategory"]], r["SemanticCategory"],
                        _emb=norm(r["Behavior"]), _ctx=r["FunctionalCategory"] == "contextual"))
    s = pd.read_csv(RAW / "strongreject/strongreject_dataset.csv")
    for _, r in s.iterrows():
        rows.append(row("harmful", "strongreject", r["forbidden_prompt"], STRONGREJECT_MAP[r["category"]], r["category"]))
    # Base prompts only (prompt_style == "base"); the 20 mutation files are never read.
    for ln in (RAW / "sorrybench/question.jsonl").read_text().splitlines():
        q = json.loads(ln)
        assert q["prompt_style"] == "base" and len(q["turns"]) == 1
        cid = int(q["category"])
        name, coarse = SORRY_MAP[cid]
        if coarse is None:
            continue
        rows.append(row("harmful", "sorrybench", q["turns"][0], coarse, f"{cid}:{name}"))
    a = pd.read_csv(RAW / "advbench/harmful_behaviors.csv")
    for _, r in a.iterrows():
        rows.append(row("harmful", "advbench", r["goal"], None, "unlabeled"))
    for split in ("train", "test"):
        m = pd.read_parquet(RAW / f"mlabonne_harmful/data/{split}-00000-of-00001.parquet")
        for t in m["text"]:
            rows.append(row("harmful", "mlabonne_harmful", t, None, "unlabeled"))
    for split in ("train", "test"):
        m = pd.read_parquet(RAW / f"mlabonne_harmless/data/{split}-00000-of-00001.parquet")
        for t in m["text"]:
            rows.append(row("harmless", "mlabonne_harmless", t, "harmless", "none"))
    al = pd.read_parquet(next((RAW / "alpaca/data").glob("*.parquet")))
    for _, r in al.iterrows():
        inp = str(r["input"] or "").strip()
        text = f"{r['instruction']}\n\n{inp}" if inp else r["instruction"]
        rows.append(row("harmless", "alpaca", text, "harmless", "none"))
    return [r for r in rows if r["text"]]


def exact_dedupe(rows, stats):
    rows = sorted(rows, key=lambda r: PRIORITY.index(r["source"]))  # stable within source
    seen = {}
    out = []
    for r in rows:
        k = exact_key(r["text"])
        if k in seen:
            stats["exact_dropped"][r["source"]] += 1
            stats["exact_pairs"][f"{r['source']}<-{seen[k]}"] += 1
            continue
        seen[k] = r["source"]
        out.append(r)
    return out


def embed(texts):
    from sentence_transformers import SentenceTransformer
    import torch
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    model = SentenceTransformer(EMB_MODEL, device=dev, revision=EMB_REV)
    return model.encode(texts, batch_size=128, normalize_embeddings=True, show_progress_bar=False,
                        convert_to_numpy=True).astype(np.float32), dev


def near_dedupe(rows, embs, stats):
    """Greedy in priority order: drop a row if cosine > NEAR_DUP to any already-kept row."""
    kept_idx = []
    kept_ctx = np.zeros(len(rows), dtype=bool)
    kept = np.zeros((len(rows), embs.shape[1]), dtype=np.float32)
    n = 0
    for i, r in enumerate(rows):
        if n:
            sims = kept[:n] @ embs[i]
            if r.get("_ctx"):
                # HarmBench reuses one request over several distinct contexts (e.g. *_1, *_2, *_3);
                # those are different prompts, so contextual rows are not compared with each other.
                sims = np.where(kept_ctx[:n], -1.0, sims)
            j = int(np.argmax(sims))
            if sims[j] > NEAR_DUP:
                src = rows[kept_idx[j]]["source"]
                stats["near_dropped"][r["source"]] += 1
                stats["near_pairs"][f"{r['source']}<-{src}"] += 1
                if r["set"] != rows[kept_idx[j]]["set"]:
                    stats["near_cross_set"][f"{r['set']}<-{rows[kept_idx[j]]['set']}"] += 1
                continue
        kept[n] = embs[i]
        kept_ctx[n] = bool(r.get("_ctx"))
        kept_idx.append(i)
        n += 1
    return kept_idx


def assign_knn(rows, embs, k=5):
    """Unlabeled harmful prompts get the majority coarse category of their k nearest labeled harmful prompts."""
    lab = [i for i, r in enumerate(rows) if r["set"] == "harmful" and r["category"]]
    unl = [i for i, r in enumerate(rows) if r["set"] == "harmful" and not r["category"]]
    if not unl:
        return
    L = embs[lab]
    for i in unl:
        sims = L @ embs[i]
        top = np.argsort(-sims)[:k]
        votes = Counter(rows[lab[t]]["category"] for t in top)
        best = max(votes.items(), key=lambda kv: (kv[1], sum(sims[t] for t in top if rows[lab[t]]["category"] == kv[0])))
        rows[i]["category"] = best[0]


def make_id(r):
    return f"{r['set']}-{r['source']}-{hashlib.sha1(r['text'].encode()).hexdigest()[:12]}"


def split(rows, fracs):
    by = defaultdict(list)
    for r in rows:
        by[r["category"]].append(r)
    rng = random.Random(SEED)
    out = {k: [] for k in fracs}
    for cat in sorted(by):
        items = sorted(by[cat], key=lambda r: r["id"])
        rng.shuffle(items)
        n = len(items)
        names = list(fracs)
        cuts = []
        acc = 0.0
        for nm in names[:-1]:
            acc += fracs[nm]
            cuts.append(round(n * acc))
        prev = 0
        for nm, c in zip(names, cuts + [n]):
            out[nm].extend(items[prev:c])
            prev = c
    return out


def write_jsonl(path, rows, keys):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps({k: r[k] for k in keys if k in r}, ensure_ascii=False) + "\n")


def cap_subsets(stats):
    rng = random.Random(SEED)
    m = pd.read_parquet(RAW / "mmlu/all/test-00000-of-00001.parquet")
    subs = sorted(m["subject"].unique())
    per = 300 // len(subs)
    extra = set(rng.sample(subs, 300 - per * len(subs)))
    pick = []
    for s in subs:
        idx = sorted(m.index[m["subject"] == s].tolist())
        pick += rng.sample(idx, per + (1 if s in extra else 0))
    rng.shuffle(pick)
    out = R / "harness" / "cap"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "mmlu300.jsonl", "w") as f:
        for i in pick:
            r = m.loc[i]
            f.write(json.dumps({"id": f"mmlu-{i}", "question": r["question"], "choices": list(r["choices"]),
                                "answer": int(r["answer"]), "subject": r["subject"]}, ensure_ascii=False) + "\n")
    g = pd.read_parquet(RAW / "gsm8k/main/test-00000-of-00001.parquet")
    gi = rng.sample(range(len(g)), 100)
    with open(out / "gsm8k100.jsonl", "w") as f:
        for i in gi:
            ans = g.loc[i, "answer"].split("####")[-1].strip().replace(",", "")
            num = float(ans)
            num = int(num) if num.is_integer() else num
            f.write(json.dumps({"id": f"gsm8k-{i}", "question": g.loc[i, "question"], "answer": num,
                                "solution": g.loc[i, "answer"]}, ensure_ascii=False) + "\n")
    stats["cap"] = {"mmlu300": len(pick), "mmlu_subjects": len(subs), "gsm8k100": len(gi)}


def main():
    stats = {k: Counter() for k in ("raw", "exact_dropped", "exact_pairs", "near_dropped", "near_pairs", "near_cross_set")}
    rows = load_all()
    for r in rows:
        stats["raw"][r["source"]] += 1
    rows = exact_dedupe(rows, stats)
    stats["after_exact"] = Counter(r["source"] for r in rows)
    # Seeded pre-sample of the harmless pool so the near-dup pass stays small.
    harmless = [r for r in rows if r["set"] == "harmless"]
    other = [r for r in rows if r["set"] != "harmless"]
    rng = random.Random(SEED)
    harmless = rng.sample(harmless, min(HARMLESS_PRE, len(harmless)))
    harmless.sort(key=lambda r: PRIORITY.index(r["source"]))
    rows = other + harmless
    stats["harmless_presample"] = Counter(r["source"] for r in harmless)
    embs, dev = embed([r.get("_emb", r["text"]) for r in rows])
    stats["embed_device"] = dev
    keep = near_dedupe(rows, embs, stats)
    rows = [rows[i] for i in keep]
    embs = embs[keep]
    assign_knn(rows, embs)
    hl = [r for r in rows if r["set"] == "harmless"]
    if len(hl) > HARMLESS_CAP:
        rng2 = random.Random(SEED)
        hl_keep = set(id(r) for r in rng2.sample(hl, HARMLESS_CAP))
        rows = [r for r in rows if r["set"] != "harmless" or id(r) in hl_keep]
    for r in rows:
        r["id"] = make_id(r)
    assert len({r["id"] for r in rows}) == len(rows)
    stats["final_by_source"] = Counter(r["source"] for r in rows)
    keys = ["id", "text", "source", "category", "orig_category", "set", "expect"]
    sealed = R / "sealed" / "test"
    sealed.mkdir(parents=True, exist_ok=True)
    os.chmod(R / "sealed", 0o700)
    os.chmod(sealed, 0o700)
    counts = {}
    for set_, fr in (("harmful", {"train": .6, "val": .2, "test": .2}),
                     ("harmless", {"train": .6, "val": .2, "test": .2}),
                     ("borderline", {"val": .5, "test": .5}),
                     ("probes", {"val": .5, "test": .5})):
        parts = split([r for r in rows if r["set"] == set_], fr)
        for sp, rs in parts.items():
            dest = (sealed if sp == "test" else R / "data" / sp) / f"{set_}.jsonl"
            write_jsonl(dest, rs, keys)
            if sp == "test":
                os.chmod(dest, 0o600)
            counts[f"{set_}/{sp}"] = {"n": len(rs), "by_category": dict(Counter(r["category"] for r in rs)),
                                      "by_source": dict(Counter(r["source"] for r in rs))}
    stats["splits"] = counts
    cap_subsets(stats)
    stats["emb_model"] = f"{EMB_MODEL}@{EMB_REV}"
    out = {k: (dict(v) if isinstance(v, Counter) else v) for k, v in stats.items()}
    (R / "harness" / "build_stats.json").write_text(json.dumps(out, indent=1))
    for k in ("raw", "after_exact", "harmless_presample", "near_dropped", "near_pairs", "near_cross_set", "final_by_source", "cap"):
        print(k, out[k])
    for k, v in counts.items():
        print(k, v["n"])


EMB_REV = os.environ.get("EMB_REV", "main")

if __name__ == "__main__":
    main()
