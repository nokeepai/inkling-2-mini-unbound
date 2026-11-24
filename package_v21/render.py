"""Render release docs from templates + values.json. Unknown {{KEY}} -> "pending".

  python3 package_v21/render.py <github_clone_dir>
Writes package_v21/dist/{main,gguf,basalt}/README.md and <clone>/README.md. Prints any key still pending.
"""
import json, os, re, sys

H = os.path.dirname(os.path.abspath(__file__))
V = json.load(open(f"{H}/values.json")) if os.path.exists(f"{H}/values.json") else {}
OUT = {"main_README.md": f"{H}/dist/main/README.md", "gguf_README.md": f"{H}/dist/gguf/README.md",
       "basalt_README.md": f"{H}/dist/basalt/README.md", "github_README.md": f"{sys.argv[1]}/README.md"}
pending = set()


def sub(m):
    k = m.group(1)
    if k in V:
        return str(V[k])
    pending.add(k)
    return "pending"


for t, o in OUT.items():
    s = re.sub(r"\{\{([A-Z0-9_]+)\}\}", sub, open(f"{H}/templates/{t}").read())
    if t == "main_README.md":  # HF resolves relative images against main; point at the branch the charts are on
        rev = V.get("CHART_REV", "main")
        s = s.replace("](charts/", f"](https://huggingface.co/nokeepai/Inkling-2-Mini-Unbound/resolve/{rev}/charts/")
    os.makedirs(os.path.dirname(o), exist_ok=True)
    open(o, "w").write(s)
print("rendered;", "pending:", " ".join(sorted(pending)) or "none")
