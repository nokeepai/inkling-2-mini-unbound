"""Assert harness/judge.py TEMPLATE equals the official strongreject_finetuned template."""
import ast
import json
import sys

official = json.load(open(sys.argv[1]))["strongreject_finetuned"]
src = open("/Users/nikshepsvn/Code/inkling-abl/harness/judge.py").read()
tree = ast.parse(src)
for node in tree.body:
    if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "TEMPLATE":
        ours = ast.literal_eval(node.value)
print("MATCH" if ours == official else "MISMATCH")
