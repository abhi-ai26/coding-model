#!/bin/bash
# Run on a fresh RunPod pod after ssh:  bash <(curl -s https://raw.githubusercontent.com/abhi-ai26/coding-model/main/setup/pod-setup.sh)
# or: git clone https://github.com/abhi-ai26/coding-model /workspace/coding-model && bash /workspace/coding-model/setup/pod-setup.sh
set -e
echo "==> GPU"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "==> disk"; df -h /workspace | tail -1
cd /workspace
if ! command -v uv >/dev/null 2>&1; then curl -LsSf https://astral.sh/uv/install.sh | sh; fi
export PATH="$HOME/.local/bin:$PATH"
apt-get update -qq && apt-get install -y -qq tmux git-lfs >/dev/null 2>&1 || true
if [ ! -d nanoGPT ]; then git clone -q https://github.com/karpathy/nanoGPT; fi
cd nanoGPT
if [ ! -d .venv ]; then uv venv -q && . .venv/bin/activate && uv pip install -q torch numpy transformers datasets tiktoken wandb tqdm; fi
. .venv/bin/activate
python -c "import torch;print('cuda:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
mkdir -p data/pycode config
cp -f /workspace/coding-model/week1/prepare_pycode.py data/pycode/prepare.py 2>/dev/null || true
cp -f /workspace/coding-model/week1/train_pycode.py config/train_pycode.py 2>/dev/null || true
echo "Ready. Next:  tmux new -s train ; source .venv/bin/activate ; python data/pycode/prepare.py"
