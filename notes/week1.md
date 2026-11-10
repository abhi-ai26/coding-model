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

## Day 5 log (EvalPlus 0.4.0.dev44, vLLM 0.33, greedy, bf16, A40 — run by us)
| model | prompt | HumanEval | HumanEval+ | MBPP | MBPP+ |
|---|---|---|---|---|---|
| base Qwen2.5-Coder-1.5B | completion | 0.396 | 0.335 | 0.696 | 0.582 |
| ours (5k SFT) | completion | 0.567 | 0.463 | 0.754 | 0.624 |
| ours (5k SFT) | instruction, own template | 0.707 | 0.628 | – | – |
- base MBPP+ matches Qwen's reported 58.6 → harness trusted; HumanEval base 3 pts under their 43.9 (prompt sensitivity)
- gain generalises (+4.2 MBPP+) but HumanEval+ gain is 3× larger: format/stopping + distribution match with textbook data; contamination check pending (Week 5)
- SFT unlocks, it doesn't add: 5k samples ≈ official Instruct on HumanEval (0.707 vs 0.707), 4 pts under on HumanEval+
- EvalPlus caches by model name: use --root for a second prompt mode; vLLM 164 problems in 88 s vs ~40 min eager

## Day 6 log
- published https://huggingface.co/abhi-ai26/qwen-coder-1.5b-sft-week1 (private for now), model card with own-harness numbers, prompt format, caveats
- 7 files, 3.09 GB; card states Claude/ChatGPT tutoring, pending contamination check
- Gate 1 deliverable in place on day 6 of week 1
