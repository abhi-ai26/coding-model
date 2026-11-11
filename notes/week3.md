# Week 3 — serving, quantization, the harness, Gate 1

## Day 1 log — vLLM serving, prefill vs decode (A40, 1.5B SFT, bf16)
| prompt tokens | TTFT ms | TPOT ms | tok/s |
|---|---|---|---|
| 128 | 41 | 7.08 | 136 |
| 1024 | 69 | 7.08 | 132 |
| 4096 | 180 | 7.16 | 117 |
- TTFT ≈ 40 ms fixed + ~0.03 ms/token (+ attention's quadratic part at 4K); TPOT flat: decode is memory-bound (3 GB / 700 GB/s ≈ 4.5 ms floor)
- batching at 1024-in/256-out: concurrency 1 → 137 tok/s (TPOT 7.1); 8 → 949 (7.8); 32 → 2,426 (12.0): 17.7× throughput for +69% latency
- vLLM 137 tok/s vs HF eager 47 single-stream; KV reservation 36.8 GB = 1.38M tokens of cache

## Day 2 log — GGUF quantization
| file | size | HumanEval | HumanEval+ |
|---|---|---|---|
| bf16 | 3.1 GB | 0.665 | 0.579 |
| Q8_0 | 1.6 GB | 0.683 | 0.598 |
| Q4_K_M | 1.0 GB | 0.604 | 0.530 |
- llama-server + EvalPlus openai backend, instruction mode; Q8 ≈ bf16 (±3 problems = noise); Q4 −5 pts for 3× smaller
- same bf16 weights: 0.579 in llama.cpp vs 0.628 in vLLM → runtime is part of the number; compare only within one
- MacBook, Ollama, Q4_K_M: 112 tok/s decode, 94 tok/s prefill; identical answer to vLLM

## Day 4 log — QLoRA at 7B
- Qwen2.5-Coder-7B, 4-bit base, LoRA r=16, same 5k samples: 508 s (8.5 min), 9.8 samples/s, loss 0.71 → 0.27
- 7B base (completion): HumanEval+ 0.518, MBPP+ 0.646 (Qwen reports 53.0 / 62.9)
- 7B QLoRA (instruction): HumanEval+ 0.744, MBPP+ 0.717 (official Instruct ≈ 0.84 / 0.72)
- gap to official widened 4 → 10 pts on HumanEval+; MBPP+ at parity; cost per point ≈ ₹0.4
- script bug: deactivate inside a script launched from an active venv keeps the venv python → call /usr/bin/python3 explicitly

## Day 3 log — the harness (Mac, OrbStack Docker, Mini-SWE-Agent 2.4.6)
- 3 SWE-bench Lite dev instances (sqlfluff) with DeepSeek V4.1 Flash via OpenRouter: 6 min 43 s, $0.07; 26/45/36 commands
- all 3 "resolved" by the official harness (swebench.harness.run_evaluation, local Docker, 64 s)
- BUT: 12/11/2 git-history commands per run; the model found the upstream fix in the container's .git (future commits present), verified `git merge-base --is-ancestor HEAD <fix>`, and copied the diff
- rules adopted: strip future history in every environment we collect from (Week 9); audit environments for shortcuts before RL (Week 15); report container preparation with every SWE-bench number
- sb-cli hosted evaluator returned 500 → local harness is the standard
- scripts: week3/traj_stats.py; raw stats notes/week3-traj-flash.txt
- OpenCode + teacher on a parse_iso task: wrote both files, made a temp venv for pytest, 6/6 passed in minutes, reported what it did
- OpenCode + our 1.5B (vLLM through SSH tunnel) needed plumbing first: IPv4 baseURL (Node resolves localhost to ::1), --enable-auto-tool-choice --tool-call-parser hermes, --max-model-len 32768, model limits in opencode.json
- result: failure mode (a) — answered in chat ("Here is the code to solve this problem"), zero tool calls, no files, no tests, 3.9 s; code wrong (date never parsed; finally: return swallows the ValueError)
- Week 9 trajectories + Week 12 template give a model hands; Week 15 RL makes it check; Week 14 ships the serving config
