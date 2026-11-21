"""Print columns, row counts and category value counts of raw sources (no row text)."""
import pandas as pd
from pathlib import Path

RAW = Path("/Users/nikshepsvn/Code/inkling-abl/sealed/raw")
for p in sorted(RAW.rglob("*")):
    if p.suffix not in (".csv", ".parquet"):
        continue
    df = pd.read_csv(p) if p.suffix == ".csv" else pd.read_parquet(p)
    print("==", p.relative_to(RAW), len(df), list(df.columns))
    for c in df.columns:
        if c.lower() in ("category", "semanticcategory", "functionalcategory", "type", "label", "focus", "source", "subject"):
            vc = df[c].value_counts()
            print("  ", c, dict(vc.head(60)))
