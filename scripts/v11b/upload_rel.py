"""ON eval pod: upload a release folder to a HF repo branch (resumable, parallel). HF_TOKEN from env.
  python3 upload_rel.py <repo_id> <folder> <revision> <pattern>[,<pattern>...]"""
import sys
from huggingface_hub import HfApi

repo, folder, rev, pats = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4].split(",")
HfApi().upload_large_folder(repo_id=repo, folder_path=folder, repo_type="model", revision=rev,
                            allow_patterns=pats, num_workers=12, print_report=True, print_report_every=120)
print("UPLOAD_DONE", repo, folder, rev)
