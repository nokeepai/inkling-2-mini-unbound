#!/usr/bin/env bash
# Mac: push scripts + gpqa mirror, start 8 servers (g0-3 orig, g4-7 ou), inspect gpqa parquet.
set -uo pipefail
R=/Users/nikshepsvn/Code/inkling-abl
ink() { bash "$R/scripts/ink_lium.sh" "$@"; }
bash $R/scripts/suite/push_pod.sh
ink exec ink-suite-1 "mkdir -p /srv/suite/data/gpqa_raw /srv/suite/out" >/dev/null
ink scp ink-suite-1 $R/suite/data/gpqa/fingertap/GPQA-Diamond/test/gpqa_diamond.parquet /srv/suite/data/gpqa_raw/gpqa_diamond.parquet >/dev/null
ink scp ink-suite-1 $R/suite/data/gpqa/hendrydong/gpqa_diamond/data/test-00000-of-00001.parquet /srv/suite/data/gpqa_raw/hendrydong.parquet >/dev/null
ink exec ink-suite-1 "nohup bash /srv/suite/start_servers.sh /w/orig /w/orig /w/orig /w/orig /w/merged_ou /w/merged_ou /w/merged_ou /w/merged_ou > /root/logs/start_servers.log 2>&1 < /dev/null &" >/dev/null
ink exec ink-suite-1 "python3 -c \"import pandas as pd; g=pd.read_parquet('/srv/suite/data/gpqa_raw/gpqa_diamond.parquet'); print(g.shape, list(g.columns)[:20]); h=pd.read_parquet('/srv/suite/data/gpqa_raw/hendrydong.parquet'); print(h.shape, list(h.columns))\""
