# Week 4 — MoE, honest harness, sandbox, distillation, re-plan

## Day 0 log
- volume filled to 100% by Unsloth "pre-downloading" the 60 GB bf16 base beside the 4-bit one; hub cache is a shared blob store now → `hf cache ls` / `hf cache prune` (freed 51 GB of orphans) / `hf cache rm` (20 GB of duplicate unsloth mirrors); lesson: keep 30% free, check df before unknown downloads
- fix: give Unsloth a local snapshot path, not a repo id; vLLM server from Day 3 was still holding 41 GB of GPU → pkill before training

## Day 1 log — Qwen3-30B-A3B-Base (bnb-4bit) + QLoRA r=16, same 5k samples
- 313 steps, 2,901 s (48 min), 1.72 samples/s (7B: 9.8, 1.5B: 18.7) — 4-bit MoE experts run as thousands of small dequantized kernels; the 3B-active advantage is real for arithmetic, absent for overhead → Week 11 wants bf16/FP8 experts on an 80 GB card
- loss 0.67 → 0.28 (train avg 0.298); adapter 2.5 GB (LoRA on every expert ≠ thin) → target attention/shared layers on MoEs
- vLLM 0.33 has dropped bitsandbytes → could not serve the 4-bit base; generated HumanEval with Unsloth inference (week4/gen_humaneval.py, batch 8, ~28 min) and scored with evalplus --samples
- HumanEval 0.890 / HumanEval+ 0.817 (instruction mode; runtime differs from the vLLM rows; base number pending)
