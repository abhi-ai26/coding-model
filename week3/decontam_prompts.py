"""Same check, but only against the benchmark problem STATEMENTS (docstrings / task descriptions) — the part that cannot be an idiom."""
import re, collections
from datasets import load_dataset
N = 13
def grams(t):
    toks = re.findall(r"\w+", t.lower()); return {" ".join(toks[i:i+N]) for i in range(len(toks) - N + 1)}
bench = {row["task_id"]: grams(row["prompt"]) for row in load_dataset("openai/openai_humaneval", split="test")}
bench.update({f"MBPP/{row['task_id']}": grams(row["prompt"]) for row in load_dataset("google-research-datasets/mbpp", "sanitized", split="test")})
index = collections.defaultdict(set)
for pid, gs in bench.items():
    for g in gs: index[g].add(pid)
ds = load_dataset("OpenCoder-LLM/opc-sft-stage2", "educational_instruct", split="train").select(range(5000))
hits = collections.defaultdict(list)
for i, ex in enumerate(ds):
    for g in grams(ex["instruction"] + " " + ex["output"]) & index.keys():
        for pid in index[g]: hits[pid].append(i)
print(f"problems whose STATEMENT appears in the 5k samples: {len(hits)} / {len(bench)}")
for pid, idx in sorted(hits.items(), key=lambda kv: -len(set(kv[1])))[:15]:
    print(f"  {pid}: in {len(set(idx))} sample(s), e.g. #{sorted(set(idx))[0]}")
