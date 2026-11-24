"""ON release pod: upload a staged folder to a HF repo branch (huggingface_hub >= 1.x: upload_folder, Xet storage).
  python3 up21.py <repo_id> <folder> <revision>"""
import sys
from huggingface_hub import HfApi

repo, folder, rev = sys.argv[1:4]
api = HfApi()
api.create_branch(repo, branch=rev, exist_ok=True)
c = api.upload_folder(repo_id=repo, folder_path=folder, revision=rev, commit_message=f"v2.1: release files ({folder.rsplit('/', 1)[-1]})")
print("UPLOAD_DONE", repo, rev, c.oid if hasattr(c, "oid") else c)
