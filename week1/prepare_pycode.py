"""Week 1, Day 1: tokenize ~50M tokens of Python into train.bin / val.bin for nanoGPT.
Run from the nanoGPT folder on the pod:  python data/pycode/prepare.py
Day 2: change DATASET to "codeparrot/codeparrot-clean" and TARGET_TOKENS to 500_000_000.
"""
import os, numpy as np, tiktoken
from datasets import load_dataset

DATASET = "codeparrot/codeparrot-clean-valid"   # small, fast
TARGET_TOKENS = 50_000_000
OUT = os.path.dirname(os.path.abspath(__file__))

enc = tiktoken.get_encoding("gpt2")
ds = load_dataset(DATASET, split="train", streaming=True)
toks, n = [], 0
for ex in ds:
    t = enc.encode_ordinary(ex["content"]) + [enc.eot_token]
    toks.extend(t); n += len(t)
    if n >= TARGET_TOKENS:
        break
arr = np.array(toks, dtype=np.uint16)
split = int(0.98 * len(arr))
arr[:split].tofile(os.path.join(OUT, "train.bin"))
arr[split:].tofile(os.path.join(OUT, "val.bin"))
print(f"tokens: {n:,}  train: {split:,}  val: {len(arr)-split:,}")
