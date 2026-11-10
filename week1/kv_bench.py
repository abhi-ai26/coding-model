"""Week 1, Day 3: what the KV cache does to decode speed, and when it matters."""
import time, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
m = "Qwen/Qwen2.5-Coder-1.5B"
tok = AutoTokenizer.from_pretrained(m)
model = AutoModelForCausalLM.from_pretrained(m, dtype=torch.bfloat16).cuda()
c = model.config
kv_kb = 2 * c.num_hidden_layers * c.num_key_value_heads * (c.hidden_size // c.num_attention_heads) * 2 / 1024
print(f"KV cache: {kv_kb:.1f} KB/token -> {kv_kb*32768/2**20:.2f} GB at 32K, {kv_kb*2**20/2**20:.1f} GB at 1M tokens")

def run(ctx, cache, new=200):
    ids = torch.randint(100, 50000, (1, ctx), device="cuda")          # random context of length ctx
    torch.cuda.synchronize(); t = time.time()
    out = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=new,
                         min_new_tokens=new, do_sample=False, use_cache=cache, pad_token_id=tok.eos_token_id)
    torch.cuda.synchronize(); dt = time.time() - t
    n = out.shape[1] - ctx
    print(f"context {ctx:5d} | use_cache={cache!s:5} | {n} new tokens | {n/dt:6.1f} tok/s | {dt:5.1f} s", flush=True)

run(16, True, new=8)                                                   # warm-up, not timed
for ctx in (16, 2048, 8192):
    for cache in (True, False):
        run(ctx, cache)
