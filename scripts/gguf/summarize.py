"""ON POD: build /data/release/kld/KLD_TABLE.tsv and /data/release/gen/REFUSALS.tsv from the logs."""
import glob
import json
import os
import re

rows = ["model\tmean_kld\tkld_99pct\tsame_top_p\tppl\tfile_bytes"]
paths = {"unbound_" + t: f"/data/release/Inkling-2-Mini-Unbound-{t}.gguf" for t in ["BF16", "Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "IQ3_M", "IQ2_M"]}
paths |= {"ggmlorg_Q8_0": "/data/base/Inkling-2-Mini-Q8_0.gguf", "ggmlorg_IQ4_NL": "/data/base/Inkling-2-Mini-IQ4_NL.gguf"}
for tag, p in paths.items():
    log = open(f"/data/release/kld/{tag}.log").read()

    def grab(pat: str) -> str:
        m = re.search(pat, log, re.M)
        return m.group(1) if m else "NA"
    rows.append("\t".join([tag, grab(r"^Mean\s+KLD:\s+([\d.]+\s+±\s+[\d.]+)"), grab(r"^99\.0%\s+KLD:\s+([\d.]+)"),
                           grab(r"^Same top p:\s+([\d.]+\s+±\s+[\d.]+ %)"), grab(r"^Mean\s+PPL\(Q\)\s+:\s+([\d.]+)"),
                           str(os.path.getsize(p))]))
open("/data/release/kld/KLD_TABLE.tsv", "w").write("\n".join(rows) + "\n")
print("\n".join(rows))

g = ["model\tharmful_refused\tharmful_n\tharmful_unfinished\tharmless_empty_answer"]
for f in sorted(glob.glob("/data/release/gen/gen_*.jsonl")):
    rs = [json.loads(l) for l in open(f)]
    hm = [r for r in rs if r["kind"] == "harmful"]
    hl = [r for r in rs if r["kind"] == "harmless"]
    g.append(f"{os.path.basename(f)[4:-6]}\t{sum(r['refusal'] for r in hm)}\t{len(hm)}\t{sum(r['finish'] != 'stop' for r in hm)}\t{sum(not r['answer'].strip() for r in hl)}")
open("/data/release/gen/REFUSALS.tsv", "w").write("\n".join(g) + "\n")
print("\n".join(g))
