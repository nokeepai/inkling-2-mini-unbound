"""Lock permissions on sealed/, write harness/SOURCES.md and harness/DATA_CARD.md (counts + hashes only)."""
import hashlib
import json
import os
from pathlib import Path
import sys
sys.path.insert(0, "/Users/nikshepsvn/Code/inkling-abl/scripts/data")
from build import SORRY_MAP  # noqa: E402

R = Path("/Users/nikshepsvn/Code/inkling-abl")
S = json.loads((R / "harness" / "build_stats.json").read_text())
M = json.loads((R / "sealed" / "raw" / "manifest.json").read_text())

for dp, dns, fns in os.walk(R / "sealed"):
    os.chmod(dp, 0o700)
    for f in fns:
        os.chmod(os.path.join(dp, f), 0o600)

LIC_NOTE = {
    "mlabonne_harmful": "none declared on card (derived from AdvBench, MIT)",
    "mlabonne_harmless": "none declared on card (derived from Alpaca, CC-BY-NC-4.0)",
    "sorrybench": "custom, gated (see repo LICENSE; card says other)",
}
USE = {
    "mlabonne_harmful": "harmful (fully exact-duplicated by AdvBench; contributes 0 rows)",
    "mlabonne_harmless": "harmless", "alpaca": "harmless (instruction + input when input non-empty)",
    "orbench": "borderline (or-bench-hard-1k.csv)", "sorrybench": "harmful (question.jsonl, 440 base prompts; 80 in 8 excluded categories; 20 mutation files not used)",
    "mmlu": "cap/mmlu300.jsonl (all/test)", "gsm8k": "cap/gsm8k100.jsonl (main/test)",
    "advbench": "harmful (goal column)", "harmbench": "harmful (standard + contextual; copyright excluded)",
    "strongreject": "harmful", "xstest": "borderline (label == safe only)",
}
src = ["# Sources (pinned)\n", "Downloaded 2026-11-21 by the data agent (scripts/data/fetch.py). Raw copies: sealed/raw/ (contain test prompts, so sealed).\n",
       "| key | used as | url | pinned revision | licence | raw rows |", "|---|---|---|---|---|---|"]
for k, v in M.items():
    lic = v.get("license") or LIC_NOTE.get(k, "unknown")
    if isinstance(lic, list):
        lic = ",".join(lic)
    rows = S["raw"].get(k, "")
    if k == "sorrybench":
        rows = f"440 ({S['raw'].get(k, 0)} after category exclusion)"
    if k == "mmlu":
        rows = 14042
    if k == "gsm8k":
        rows = 1319
    src.append(f"| {k} | {USE.get(k, '')} | {v.get('url', v.get('repo'))} | `{v.get('revision', 'n/a')}` | {lic} | {rows} |")
src.append(f"| probes | probes (40, written by the data agent) | sealed/probes_source.jsonl | n/a | project-internal | 40 |")
src.append(f"| embedder | near-dup dedupe | https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2 | `{S['emb_model'].split('@')[1]}` | apache-2.0 | n/a |")
src.append("\nGitHub raw files were fetched from `raw.githubusercontent.com/<repo>/<sha>/<path>`; HF files via `hf_hub_download(revision=<sha>)`.\n")
src.append("SORRY-Bench: the gate was accepted by the account owner on 2026-11-21; 202503 pinned. Only `question.jsonl` (prompt_style == base) is used; category names from the repo's `meta_info.py`.\n")
(R / "harness" / "SOURCES.md").write_text("\n".join(src) + "\n")


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


card = ["# Data card\n", "Built by scripts/data/run_build.sh (seed 1234). Counts and hashes only; no test text.\n",
        "## Pipeline",
        "1. Normalise: NFKC, CRLF->LF, runs of spaces/tabs -> one space, strip each line, 3+ newlines -> 2, strip. (Newlines kept so HarmBench contextual tables/code survive.)",
        "2. Exact dedupe on whitespace-collapsed casefolded text, priority order: probes, xstest, orbench, harmbench, strongreject, sorrybench, advbench, mlabonne_harmful, mlabonne_harmless, alpaca (later source dropped).",
        "3. Harmless pool pre-sampled to 9,000 (seeded) before the embedding pass, capped to 6,000 after it (seeded).",
        f"4. Near dedupe: greedy, same priority order, drop a row if cosine > 0.95 to any kept row, across all sets ({S['emb_model']}, device {S['embed_device']}). HarmBench contextual prompts are embedded by their request text only, and contextual-vs-contextual pairs are exempt (HarmBench reuses one request over several distinct contexts).",
        "5. Harmful categories: HarmBench, StrongREJECT and SORRY-Bench labels mapped to a coarse taxonomy (below). AdvBench has no labels: `orig_category = unlabeled`, `category` = majority vote of its 5 nearest labelled harmful prompts by embedding. These are approximate.",
        "6. Split per set, stratified by `category`, seed 1234: harmful/harmless 60/20/20; borderline and probes 50/50 val/test. Test only in sealed/test/ (dir 700, files 600).\n",
        "## Coarse harmful taxonomy",
        "| category | HarmBench SemanticCategory | StrongREJECT category |", "|---|---|---|",
        "| cyber | cybercrime_intrusion | |", "| cbrn_weapons | chemical_biological | |",
        "| crime_illegal_goods | illegal | Non-violent crimes; Illegal goods and services |",
        "| misinfo_deception | misinformation_disinformation | Disinformation and deception |",
        "| hate_harassment | harassment_bullying | Hate, harassment and discrimination |",
        "| violence | | Violence |", "| sexual | | Sexual content |", "| other_harm | harmful | |\n",
        "### SORRY-Bench 202503 (44 categories -> coarse; `orig_category` = `<id>:<short name>`)",
        "| id | SORRY-Bench category | coarse |", "|---|---|---|"] + [
        f"| {i} | {n} | {c or 'EXCLUDED (not harmful to answer: opinion promotion or specialised advice)'} |" for i, (n, c) in SORRY_MAP.items()] + ["",
        "Borderline `category` = `xstest_<type>` or `orbench_<category>`. Probes: `honest_limit` (expect honest_limit) and `crisis` (expect crisis_resources). Harmless: `harmless`.\n",
        "## Counts per source", "| source | raw | after exact | near-dup dropped | final |", "|---|---|---|---|---|"]
for k in S["raw"]:
    card.append(f"| {k} | {S['raw'][k]} | {S['after_exact'].get(k, 0)} | {S['near_dropped'].get(k, 0)} | {S['final_by_source'].get(k, 0)} |")
card.append(f"\nHarmless pre-sample (before near-dup): {S['harmless_presample']}. Exact-dup pairs (dropped<-kept): {S['exact_pairs']}. Near-dup pairs: {S['near_pairs']}. Cross-set near dups: {S['near_cross_set'] or 'none'}.\n")
card += ["## Counts per split", "| set/split | n | by category | by source |", "|---|---|---|---|"]
for k, v in S["splits"].items():
    card.append(f"| {k} | {v['n']} | {v['by_category']} | {v['by_source']} |")
card.append(f"\nCapability subsets (harness/cap/, not split): {S['cap']}. MMLU: 5 per subject, plus 1 more for 15 seeded subjects. GSM8K `answer` is the number after `####`.\n")
card += ["## sha256", "| file | sha256 |", "|---|---|"]
files = sorted([p for p in (R / "data").rglob("*") if p.is_file()] + [p for p in (R / "sealed").rglob("*") if p.is_file() and "/raw/" not in str(p) and ".cache" not in str(p)]
               + sorted((R / "harness" / "cap").glob("*.jsonl")))
for p in files:
    card.append(f"| {p.relative_to(R)} | `{sha(p)}` |")
(R / "harness" / "DATA_CARD.md").write_text("\n".join(card) + "\n")
print("files hashed", len(files))
