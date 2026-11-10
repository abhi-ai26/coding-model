# Week 1 — numbers

| Day | What | Result | GPU time | ₹ |
| --- | --- | --- | --- | --- |
| 1 | nanoGPT 30M on 50M Python tokens | val loss: | | |
| 2 | nanoGPT 85M on 520M tokens | val loss: · predicted h: · actual h: | | |
| 3 | KV cache on/off, Qwen2.5-Coder-1.5B | tok/s with: · without: · KB/token: | | |
| 4 | Unsloth LoRA SFT, 5k OpenCoder | final train loss: | | |
| 5 | HumanEval+ base vs SFT | base pass@1: · sft pass@1: | | |
| 6 | Published to Hugging Face | link: | | |

## What surprised me

## What confused me

## Day 1 log
- A40 48GB, PyTorch 2.13 / CUDA 12.9, RunPod CA-MTL-1
- nanoGPT 51M params (incl. embeddings), 50,000,658 tokens (codeparrot-clean-valid), 3000 steps
- step 3000: train loss 1.2438, val loss 1.4221
- 264 ms/step, MFU ~19.8%, run time ~13 min, cost ≈ ₹10
- sample: correct Python shape (indentation, docstrings, imports), no semantics yet

## Day 2 log
- nanoGPT 124M params (GPT-2 small shape), 500M tokens (codeparrot-clean), 4000 steps, 131k tokens/step
- predicted 65–115 min (6ND at 57 TFLOP/s measured) → actual 102 min; achieved ~67 TFLOP/s, MFU ~24%
- val loss 0.985 at step 3500 (Day 1 final: 1.42); train ≈ val, no memorisation
- samples: docstrings, nested defs, numpy idioms, but repetition loops and a duplicate parameter name; fluency not competence (4 tokens/param)
- cost ≈ ₹60; W&B run d2-85m in project pycode
- lesson: Stop releases the GPU; the network volume is the real home, the pod is disposable

## Day 3 log
- Qwen2.5-Coder-1.5B, A40, HF eager, batch 1: KV cache 28 KB/token (2×28 layers×2 KV heads×128×2 B) → 0.88 GB @32K, 28 GB @1M
- v1 bench said "no difference" — two flaws: assumed 400 tokens (model stopped ~120), and short context is CPU-launch-bound so GPU savings are hidden
- v2 context sweep (200 new tokens): 16 → 47 vs 51 tok/s; 2K → 49 vs 11 (4.5×); 8K → 46 vs 2.7 (17×); no-cache time linear in context
- with cache flat ~47 tok/s (≈5× below the 220 tok/s memory-bound ceiling: eager-mode overhead)

## Day 4 log
- Unsloth LoRA r=16 on Qwen2.5-Coder-1.5B, 5k OpenCoder educational_instruct, 1 epoch, 313 steps, 4 min 28 s on A40 (18.7 samples/s)
- loss 0.70 → 0.30 by step 70, flat 0.27–0.30 after: format learned in ~1k samples; the floor is the answers' own unpredictability
- merged 16-bit model saved to qwen-coder-1.5b-sft (3 GB); W&B run d4-sft-1.5b, project week1
