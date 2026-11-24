"""ON release pod: put v2.1 cards, charts and the GGUF manifest on each repo's v2.1 branch (one commit per repo).
  python3 cards_up.py <cards_dir>   (cards_dir/{main,gguf,basalt}/README.md, cards_dir/charts/*, cards_dir/gguf_manifest_rows.tsv)"""
import os, sys
from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download

D = sys.argv[1]; api = HfApi(); O = "nokeepai/Inkling-2-Mini-Unbound"
charts = [CommitOperationAdd(f"charts/{f}", f"{D}/charts/{f}") for f in sorted(os.listdir(f"{D}/charts"))]
api.create_commit(O, operations=[CommitOperationAdd("README.md", f"{D}/main/README.md")] + charts, revision="v2.1",
                  commit_message="v2.1: model card and charts")
man = open(hf_hub_download(O + "-GGUF", "MANIFEST.tsv", revision="main")).read().rstrip("\n") + "\n" + open(f"{D}/gguf_manifest_rows.tsv").read()
open(f"{D}/MANIFEST.tsv", "w").write(man)
api.create_commit(O + "-GGUF", operations=[CommitOperationAdd("README.md", f"{D}/gguf/README.md"), CommitOperationAdd("MANIFEST.tsv", f"{D}/MANIFEST.tsv")],
                  revision="v2.1", commit_message="v2.1: model card and manifest")
api.create_commit(O + "-Basalt", operations=[CommitOperationAdd("README.md", f"{D}/basalt/README.md")], revision="v2.1",
                  commit_message="v2.1: model card")
print("CARDS_UP")
