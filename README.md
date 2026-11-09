# coding-model

An open coding model from India: a small (~3B-active) mixture-of-experts coding agent,
fine-tuned and distilled from open teachers, with long context, released with weights,
data, configs and evals.

Author: abhi-ai26 (GitHub / Hugging Face)

## Status
- Week 0: accounts and machine setup
- Week 1: first models on a rented RTX 4090 (nanoGPT, Unsloth fine-tune, HumanEval+)

## Layout
- `setup/`  one-time setup scripts for the Mac and for rented GPU pods
- `week1/`  scripts for Week 1 (data prep, nanoGPT config, KV-cache timing, SFT)
- `notes/`  weekly write-ups with the numbers (loss, tok/s, pass@1, cost)

Secrets never live in this repo: API keys stay in 1Password and in `~/.env` on each machine.
