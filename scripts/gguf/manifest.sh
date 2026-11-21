#!/bin/bash
# ON POD: sha256 + size of every release GGUF, in parallel. Writes /data/release/MANIFEST.tsv
cd /data/release
ls Inkling-2-Mini-Unbound-*.gguf | xargs -P 8 -I{} sh -c 'echo "{}\t$(stat -c %s {})\t$(sha256sum {} | cut -d" " -f1)"' | sort > MANIFEST.tsv.part
{ printf 'file\tbytes\tsha256\n'; cat MANIFEST.tsv.part; } > MANIFEST.tsv
rm MANIFEST.tsv.part
cat MANIFEST.tsv
echo MANIFEST_DONE
