import os
from transformers import AutoTokenizer
import transformers
d = '/data/compare/models/tok'
print(transformers.__version__, sorted(os.listdir(d)))
T = AutoTokenizer.from_pretrained(d)
for e in (0, 0.5, 0.9):
    s = T.apply_chat_template([{"role": "user", "content": "What is 2+2?"}], tokenize=False, reasoning_effort=e)
    print(e, repr(s[-200:]))
print([T.decode([i]) for i in (200001, 200004, 200006, 200008, 200010)])
