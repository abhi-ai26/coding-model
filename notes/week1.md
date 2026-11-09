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
