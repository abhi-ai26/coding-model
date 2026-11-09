"""Week 1, Day 3: measure what the KV cache does to decode speed."""
import time, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
m = "Qwen/Qwen2.5-Coder-1.5B"
tok = AutoTokenizer.from_pretrained(m)
model = AutoModelForCausalLM.from_pretrained(m, torch_dtype=torch.bfloat16).cuda()
x = tok("def quicksort(arr):", return_tensors="pt").to("cuda")
for cache in (True, False):
    torch.cuda.synchronize(); t = time.time()
    model.generate(**x, max_new_tokens=400, do_sample=False, use_cache=cache)
    torch.cuda.synchronize(); print(f"use_cache={cache}: {400/(time.time()-t):.1f} tok/s")
c = model.config
kv_kb = 2 * c.num_hidden_layers * c.num_key_value_heads * (c.hidden_size // c.num_attention_heads) * 2 / 1024
print(f"KV cache: {kv_kb:.1f} KB per token  ->  {kv_kb*32768/1024/1024:.2f} GB at 32K tokens")
