#!/usr/bin/env bash
# Upload candidate folders to /srv/cands/<id>/ on the eval pods.  bash push_cands.sh <id> [<id>...]
export LIUM_NONINTERACTIVE=1
R=/Users/nikshepsvn/Code/inkling-abl
for pod in ink-eval-1 ink-eval-2; do
  for id in "$@"; do
    src=$(ls -d $R/methods/*/cands/$id | head -1)
    lium exec $pod "mkdir -p /srv/cands/$id" >/dev/null
    for f in spec.json sha256.txt tensors.safetensors; do
      lium scp $pod "$src/$f" "/srv/cands/$id/$f"
    done
    lium exec $pod "cd /srv/cands/$id && sha256sum tensors.safetensors && cat sha256.txt"
  done
done
echo PUSH_DONE
