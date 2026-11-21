#!/bin/bash
# Download the small release artifacts from ink-gguf-1 into R/package/gguf/.
R=/Users/nikshepsvn/Code/inkling-abl
D=$R/package/gguf
mkdir -p $D/proof $D/kld $D/gen
L="bash $R/scripts/ink_lium.sh"
for f in MANIFEST.tsv llama_cpp_commit.txt proof/tensor_manifest.tsv proof/patch_proof.json; do
  $L scp ink-gguf-1 /data/release/$f $D/$f -d 2>&1 | grep -i -E "fail|error"
done
$L scp ink-gguf-1 /root/gguf/name_map.json $D/name_map.json -d 2>&1 | grep -i -E "fail|error"
for f in "$@"; do $L scp ink-gguf-1 /data/release/$f $D/$f -d 2>&1 | grep -i -E "fail|error"; done
ls -la $D $D/proof
