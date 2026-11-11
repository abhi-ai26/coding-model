# Week 2 — nanochat on one GPU

## Day 1 log
- nanochat cloned, uv env, rustbpe built; 24 FineWeb-Edu shards (~6 GB, ~1.1B tokens)
- tokenizer: 32,768 BPE on 2B chars in 79 s; bytes/token 4.74 (GPT-2 4.67, GPT-4 4.82); code 3.17 vs GPT-2 2.19; Korean 1.19 (3-byte chars → ~2.5 tok/char: the Indic tokenizer problem)

## Day 2 log — depth-8 pretraining
- config: 8 layers, 512 dim, 4 heads, seq 2048, vocab 32,768; 125.8M total params but 42M "scaling" (matrix) params; value_embeds 67M are lookups
- SSSL (sliding window, SDPA fallback on A40): 85.6k tok/s, MFU 16.9%, ETA 76 min → aborted at step 18
- L (full attention): 205k tok/s, MFU 47.5%, 1,500 steps × 262,144 tok = 393M tokens, 32.25 min; 72 TFLOP/s achieved (nanochat 3.52e8 FLOPs/token)
- predicted 46 min at 50 TFLOP/s → actual 32 min; calibration: A40 ≈ 70 TFLOP/s on compiled bf16 training (nanoGPT Day 2 agrees: 67)
- final train loss 3.12; val bpb 0.957 (d20 speedrun ≈ 0.81); CORE 0.0872 (d20 ≈ 0.22, GPT-2 ≈ 0.26); peak mem 8.2 GB → batch 64 possible
- GPU power pinned at 300 W, SM clock 1.5–1.75 GHz: power-throttled = fully used
- W&B: nanochat/d8_week2_L; cost ≈ ₹20
- gotchas: the `--` separator is for torchrun only; nanochat venv needs WANDB_BASE_URL=https://api.forge.coreweave.com; checkpoints default to ~/.cache (container disk) → cache moved to volume + symlink

## Prediction table (6ND)
| run | N (matrix params) | D | FLOPs | TFLOP/s assumed | predicted | actual |
|---|---|---|---|---|---|---|
| Week 1 nanoGPT 124M | 124M | 524M | 3.9e17 | 57 | 65–115 min | 102 min (67 achieved) |
| Week 2 nanochat d8 | 42M | 393M | 1.39e17 | 50 | 46 min | 32 min (72 achieved) |
| Plan: 30B-A3B SFT | 3B active | 3.6B | 6.5e19 | 160 (H100 FP8, 40%) | ~110 GPU-h | — |

## Day 4 log — SFT (mid-training now lives inside chat_sft)
- chat_sft on d8: 1,867 steps, 39.9 min (ETA said 40), 203k tok/s, MFU 47.9%; loss 2.25 → 1.35; chat val bpb 0.729 → 0.420
- in-training eval (full ARC/MMLU; 24-problem GSM8K/HumanEval): ARC-Easy 32.5%, ARC-Challenge 27.6%, MMLU 29.4%, GSM8K 0, HumanEval 0, ChatCORE 0.038
- same model: ARC-Easy 47.9% by likelihood ranking (base_eval) vs 32.5% by generated answer (chat_eval) — the harness is part of the number
- chat_cli: capital of France → Paris with a stammer; reverse string → correct s[::-1] but wrong example output (cannot execute); "sky blue in two sentences" → paragraph about adding integers ("two" pulled it into the maths SFT data)
- chat_web removed upstream (da32e1d "remove UI bloat") → CLI only; RL skipped: GSM8K reward is 0 at this size, GRPO has no signal
- W&B: nanochat-sft/d8_sft; cost ≈ ₹25

## Day 6 log — autoresearch overnight (Claude Code / Sonnet on Claude Max, one A40)
- autoresearch baseline needed DEVICE_BATCH_SIZE 128→64 on 46 GB (logits buffer OOM); baseline val_bpb 1.1898, 180 steps per 5-min budget (H100 reference: 0.998, 953 steps)
- 57 experiments, 15 keep / 42 discard / 0 crash; best 1.0869 (−8.6%) at 9e0e077; peak VRAM 11.3 GB
- one change did almost everything: TOTAL_BATCH_SIZE 2**19→2**17→2**16 with DBS 32 (180→~1,300 steps) took 1.190→1.099; the other ~50 runs bought 0.012
- other keeps: attention window 1024→512, UNEMBEDDING_LR 0.004→0.008, MATRIX_LR 0.04→0.02, WARMDOWN_RATIO 0.7, leaky-ReLU², ve_gate 32→64; nothing architectural mattered
- agent measured seed noise ≈ 0.004 bpb unprompted and flagged keeps < 0.003 as noise → rule for Weeks 11–13: two baseline seeds before any tuning campaign
- autoresearch's mfu_percent uses an H100 constant: ignore on A40; first commit failed on missing git identity, agent set a repo-local one
- diff and TSV in notes/week2-autoresearch.*; cost ≈ ₹300 of A40 plus Max quota
