# ON pod: wikitext-2-raw-v1 test parquet -> /data/compare/wiki.test.raw (rows concatenated as-is, the usual wiki.test.raw form)
import glob, hashlib, pyarrow.parquet as pq
f = glob.glob('/data/compare/models/wikitext/wikitext-2-raw-v1/test-*.parquet')[0]
txt = ''.join(pq.read_table(f).column('text').to_pylist())
open('/data/compare/wiki.test.raw', 'w').write(txt)
print(f, len(txt), hashlib.sha256(txt.encode()).hexdigest())
