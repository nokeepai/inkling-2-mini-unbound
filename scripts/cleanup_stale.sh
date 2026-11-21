#!/bin/bash
# Remove stale, rebuildable search artifacts. Keeps: finalists, baselines used by the auditor (ou),
# all lowrank/spec files (every candidate stays reproducible), directions.pt, results, audit, suite tables.
set -u
R=/Users/nikshepsvn/Code/inkling-abl
S=/private/tmp/claude-501/-Users-nikshepsvn-Code-innervale/140f1e4d-5898-4fb1-9611-0b847d426684/scratchpad
before=$(df -k /Users/nikshepsvn | tail -1 | awk '{print $4}')
# full BF16 tensor files of search candidates (compact lowrank kept, so all are rebuildable); keep ou + finalists
find "$R/methods" -name tensors.safetensors \
  ! -path "*/baselines/cands/ou/*" ! -path "*/c1h_r1_svd1_s3_L4/*" ! -path "*/c1dt001/*" -print -delete | wc -l | sed 's/^/candidate tensor files removed: /'
rm -f "$R/methods/mpoa/orig_wo_conv.safetensors"          # mpoa done; packager bakes on the pod
rm -rf "$S/mini_orig" "$S/fake_c0_lr" "$S/tf" "$S/sr"     # finished scratch builds/tests
rm -rf "$R/vendor/transformers-pr51050/.git" 2>/dev/null   # vendored source stays; its git history isn't needed
after=$(df -k /Users/nikshepsvn | tail -1 | awk '{print $4}')
echo "freed MB: $(( (after - before) / 1024 ))"
df -h /Users/nikshepsvn | tail -1
