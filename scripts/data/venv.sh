#!/bin/bash
set -e
V=/Users/nikshepsvn/Code/inkling-abl/scripts/data/.venv
/Users/nikshepsvn/.local/bin/uv venv --python /opt/homebrew/bin/python3.12 "$V"
/Users/nikshepsvn/.local/bin/uv pip install --python "$V/bin/python" huggingface_hub pandas pyarrow sentence-transformers torch numpy requests
"$V/bin/python" -c "import sentence_transformers, torch; print('OK', torch.__version__)"
