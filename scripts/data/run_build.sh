#!/bin/bash
set -e
PY=/Users/nikshepsvn/Code/inkling-abl/scripts/data/.venv/bin/python
export EMB_REV=$($PY -c "from huggingface_hub import HfApi; print(HfApi().model_info('sentence-transformers/all-MiniLM-L6-v2').sha)")
echo "EMB_REV=$EMB_REV"
$PY /Users/nikshepsvn/Code/inkling-abl/scripts/data/build.py
