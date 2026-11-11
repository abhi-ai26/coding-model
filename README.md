# coding-model

I am building an open coding model from India, as a solo project, learning by doing. The target is a small (about 3B active parameters) mixture-of-experts coding agent with long context that runs on one GPU and does well *inside* an open coding-agent harness (Mini-SWE-Agent for comparable scores, OpenCode as the product). Everything will be open: weights, data, configs, evals, and a report. Budget is ₹20 lakh, no GPU bought, everything rented by the hour. The public releases along the way are the proof-of-execution for Indian AI grants later.

This repo is the working record: scripts, notes with every number I measured, and what went wrong. Claude and ChatGPT are my tutors (they plan the steps and help me debug); I run everything and write the notes. Started 9 November 2026.

## Results so far (weeks 1–3, three calendar days)

Every number names the harness and runtime that produced it. Baselines are my own runs with the same setup, never copied from papers.

### Models I trained

| what | size / data | time, hardware | result |
|---|---|---|---|
| nanoGPT from scratch | 51M params, 50M Python tokens | 13 min, A40 | val loss 1.42 |
| nanoGPT from scratch | 124M params, 500M Python tokens | 102 min, A40 (predicted 65–115) | val loss 0.985, ~67 TFLOP/s achieved |
| nanochat depth-8, pretrain | 126M params (42M matrix), 393M tokens | 32 min, A40 (predicted 46) | val 0.957 bits/byte, CORE 0.087 |
| nanochat depth-8, chat SFT | same model | 40 min, A40 | ChatCORE 0.038; it answers questions, mostly wrongly |
| Qwen2.5-Coder-1.5B + LoRA | 5k OpenCoder samples | 4.5 min, A40 | see EvalPlus table |
| Qwen2.5-Coder-7B + QLoRA | same 5k samples, 4-bit base | 8.5 min, A40 | see EvalPlus table |
| autoresearch overnight | 57 experiments, 5 min each | one night, A40, Claude Code | val_bpb 1.190 → 1.087; seed noise ±0.004 |

### EvalPlus (HumanEval+ / MBPP+), vLLM backend, greedy, bf16, one A40

| model | prompt mode | HumanEval | HumanEval+ | MBPP | MBPP+ |
|---|---|---|---|---|---|
| Qwen2.5-Coder-1.5B base | completion | 0.396 | 0.335 | 0.696 | 0.582 |
| 1.5B + 5k LoRA (mine) | completion | 0.567 | 0.463 | 0.754 | 0.624 |
| 1.5B + 5k LoRA (mine) | instruction | 0.707 | 0.628 | – | – |
| Qwen2.5-Coder-7B base | completion | 0.610 | 0.518 | 0.791 | 0.646 |
| 7B + 5k QLoRA (mine) | instruction | 0.805 | 0.744 | 0.828 | 0.717 |

Caveats I found myself: the same 1.5B weights score 0.579 on HumanEval+ in llama.cpp vs 0.628 in vLLM (runtime matters); ±2–3 points is noise on 164 problems; the 5k training samples contain 0 HumanEval problem statements and 8 MBPP ones, so MBPP+ may be ~2 points inflated (`week3/decontam_prompts.py`).

### Inference numbers (1.5B model)

| measurement | result |
|---|---|
| KV cache | 28 KB/token → 0.88 GB at 32K, 28 GB at 1M tokens |
| decode with vs without KV cache at 8K context | 46 vs 2.7 tok/s (17×) |
| vLLM, one user: TTFT at 128 / 1024 / 4096 prompt tokens | 41 / 69 / 180 ms; TPOT flat at 7.1 ms |
| vLLM, 1 → 32 concurrent users | 137 → 2,426 tok/s (17.7×), TPOT 7.1 → 12.0 ms |
| GGUF quantization (llama.cpp): bf16 / Q8_0 / Q4_K_M | HumanEval+ 0.579 / 0.598 / 0.530 at 3.1 / 1.6 / 1.0 GB |
| MacBook, Ollama, Q4_K_M | 112 tok/s decode |

### Harness

Three SWE-bench Lite (dev) instances with DeepSeek V4.1 Flash through Mini-SWE-Agent: 3/3 resolved by the official harness, $0.07 — but the trajectories show the model found the upstream fix in the container's git history (future commits were present) and copied it (`week3/traj_stats.py`, `notes/week3-traj-flash.txt`). My own 1.5B in OpenCode: answers in chat, never calls a tool. Both findings shape weeks 9–15 of the plan.

## Models

- `abhi-ai26/qwen-coder-1.5b-sft-week1` on Hugging Face (private for now; will be made public before the Week 14 release). Model card has the full recipe, the eval table and the contamination check.

## How this is built

One person, a MacBook as a terminal, a rented RunPod A40 (~$0.60/h) attached to a 100 GB network volume that holds everything, Claude Max for the overnight autoresearch loop, OpenRouter for teacher models. Training stack: nanoGPT, nanochat, Unsloth, TRL. Serving and evaluation: vLLM, llama.cpp, Ollama, EvalPlus, SWE-bench harness, Mini-SWE-Agent, OpenCode. Dashboards on W&B (CoreWeave Forge).

The rules I work by, each one earned the hard way (see `docs/weeks-1-3-build-log.md`): secrets never enter chat, documents or git; every number names its harness, runtime and environment preparation; predict before you run.

## Layout

```
setup/      mac-setup.sh, pod-setup.sh
week1/      nanoGPT configs, KV-cache benchmark, Unsloth SFT
week3/      7B QLoRA, trajectory stats, contamination checks
notes/      week1.md, week2.md, week3.md — every day's numbers and what went wrong
docs/       weeks-1-3-build-log.md — the full build log and teaching guide, with the troubleshooting index
models/     Ollama Modelfile (GGUF files are not committed)
evalplus_results*/  the generated samples behind every EvalPlus number
```

Secrets never live in this repo: `.env`, keys and model weights are ignored by `.gitignore`.

## Plan

24 weeks, solo: foundations (weeks 1–4, done early) → data engine (5–10) → model (11–18) → release (19–24). The complete plan and the week-by-week walkthrough live outside this repo for now; the notes here are the record of what actually happened.
