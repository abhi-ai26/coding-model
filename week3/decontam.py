"""13-gram overlap between our 5k SFT samples and HumanEval / MBPP (problems + reference solutions + tests)."""
import re, collections
from datasets import load_dataset
N = 13
def grams(text):
    toks = re.findall(r"\w+", text.lower())
    return {" ".join(toks[i:i+N]) for i in range(len(toks) - N + 1)}

bench = {}
for row in load_dataset("openai/openai_humaneval", split="test"):
    bench[row["task_id"]] = grams(row["prompt"] + row["canonical_solution"] + row["test"])
for row in load_dataset("google-research-datasets/mbpp", "sanitized", split="test"):
    bench[f"MBPP/{row['task_id']}"] = grams(row["prompt"] + row["code"] + " ".join(row["test_list"]))
index = collections.defaultdict(set)
for pid, gs in bench.items():
    for g in gs: index[g].add(pid)
print(f"benchmark problems: {len(bench)}, distinct 13-grams: {len(index):,}")

ds = load_dataset("OpenCoder-LLM/opc-sft-stage2", "educational_instruct", split="train").select(range(5000))
hits = collections.Counter(); flagged = []
for i, ex in enumerate(ds):
    overlap = grams(ex["instruction"] + " " + ex["output"]) & index.keys()
    if overlap:
        flagged.append(i)
        for g in overlap:
            for pid in index[g]: hits[pid] += 1
        if len(flagged) <= 5:
            print(f"sample {i}: overlaps {sorted({p for g in overlap for p in index[g]})[:4]}  e.g. '{next(iter(overlap))}'")
print(f"\nflagged samples: {len(flagged)} / 5000 ({100*len(flagged)/5000:.2f}%)")
print("most-hit problems:", hits.most_common(10))
