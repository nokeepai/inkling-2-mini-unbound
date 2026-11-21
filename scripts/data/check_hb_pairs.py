"""Show HarmBench within-source near-dup pairs (behaviour ids + functional category + cosine)."""
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

h = pd.read_csv("/Users/nikshepsvn/Code/inkling-abl/sealed/raw/harmbench/harmbench_behaviors_text_all.csv")
h = h[h["FunctionalCategory"].isin(["standard", "contextual"])].reset_index(drop=True)
m = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
e = m.encode(h["Behavior"].tolist(), normalize_embeddings=True)
s = e @ e.T
np.fill_diagonal(s, 0)
for i, j in zip(*np.where(np.triu(s) > 0.95)):
    print(round(float(s[i, j]), 3), h.loc[i, "BehaviorID"], h.loc[i, "FunctionalCategory"], "|", h.loc[j, "BehaviorID"], h.loc[j, "FunctionalCategory"])
