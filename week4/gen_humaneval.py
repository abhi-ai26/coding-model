"""Generate HumanEval answers from the 4-bit Qwen3-30B-A3B base + our adapter (Unsloth inference), in EvalPlus's instruction format."""
import json, os
from unsloth import FastLanguageModel
from datasets import load_dataset
model, tok = FastLanguageModel.from_pretrained("/workspace/coding-model/qwen3-30b-a3b-qlora-adapter", max_seq_length=4096, load_in_4bit=True)
FastLanguageModel.for_inference(model)
tok.padding_side = "left"
probs = load_dataset("openai/openai_humaneval", split="test")
def wrap(p): return f"Please provide a self-contained Python script that solves the following problem in a markdown code block:\n```python\n{p}\n```"
os.makedirs("evalplus_results_30b_sft/humaneval", exist_ok=True)
out = open("evalplus_results_30b_sft/humaneval/samples.jsonl", "w"); B = 8
for i in range(0, len(probs), B):
    batch = probs[i:i+B]
    texts = [tok.apply_chat_template([{"role": "user", "content": wrap(p)}], tokenize=False, add_generation_prompt=True) + "```python\n" for p in batch["prompt"]]
    enc = tok(texts, return_tensors="pt", padding=True).to("cuda")
    gen = model.generate(**enc, max_new_tokens=768, do_sample=False, pad_token_id=tok.pad_token_id or tok.eos_token_id)
    for tid, g in zip(batch["task_id"], gen):
        txt = tok.decode(g[enc.input_ids.shape[1]:], skip_special_tokens=True)
        out.write(json.dumps({"task_id": tid, "solution": txt.split("```")[0]}) + "\n")
    print(f"{min(i+B, len(probs))}/{len(probs)}", flush=True)
out.close(); print("samples written")
