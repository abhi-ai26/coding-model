#!/bin/bash
# One-time Mac setup for the coding-model project.
# Run in Terminal:  bash ~/Documents/coding-model/setup/mac-setup.sh
# Safe to re-run; each step skips what is already done.
set -e
echo "==> 1/7 Xcode command line tools"
if ! xcode-select -p >/dev/null 2>&1; then
  xcode-select --install || true
  echo "    A dialog appeared: click Install, wait for it to finish, then re-run this script."
  exit 0
fi
echo "    ok"

echo "==> 2/7 Homebrew"
if ! command -v brew >/dev/null 2>&1; then
  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
  if [ -x /opt/homebrew/bin/brew ]; then
    echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
    eval "$(/opt/homebrew/bin/brew shellenv)"
  fi
fi
echo "    $(brew --version | head -1)"

echo "==> 3/7 git, tmux, gh, uv"
brew install git tmux gh >/dev/null 2>&1 || brew upgrade git tmux gh >/dev/null 2>&1 || true
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
echo "    git $(git --version | awk '{print $3}') | $(tmux -V) | gh $(gh --version | head -1 | awk '{print $3}') | uv $(uv --version | awk '{print $2}')"

echo "==> 4/7 git identity (project-local only; global config is left alone)"
REPO="$HOME/Documents/coding-model"
if [ -d "$REPO" ]; then
  cd "$REPO" && git init -q 2>/dev/null || true
  git config user.name "abhi-ai26"
  git config user.email "354550440+abhi-ai26@users.noreply.github.com"
  git config init.defaultBranch main
  echo "    $(git config user.name) <$(git config user.email)>  (in $REPO)"
  cd - >/dev/null
fi

echo "==> 5/7 SSH key"
if [ ! -f ~/.ssh/id_codingmodel ]; then
  ssh-keygen -t ed25519 -C "abhi-ai26" -f ~/.ssh/id_codingmodel -N ""
fi
echo "    project key (paste into RunPod Settings > SSH Public Keys); connect to pods with -i ~/.ssh/id_codingmodel"
echo
cat ~/.ssh/id_codingmodel.pub
echo

echo "==> 6/7 GitHub login"
if ! gh auth status 2>/dev/null | grep -q "account abhi-ai26"; then
  echo "    Choose: GitHub.com > HTTPS > Yes authenticate git > Login with a web browser (signed in as abhi-ai26)"
  gh auth login
fi
gh auth switch -u abhi-ai26 2>/dev/null || true
gh auth status 2>&1 | grep -A1 "abhi-ai26" | head -2

echo "==> 7/7 Hugging Face + W&B command-line tools"
export PATH="$HOME/.local/bin:$PATH"
uv tool install -q huggingface_hub 2>/dev/null || uv tool upgrade -q huggingface_hub
uv tool install -q wandb 2>/dev/null || uv tool upgrade -q wandb
if [ ! -f ~/.env ]; then
  printf 'HF_TOKEN=\nWANDB_API_KEY=\nDEEPSEEK_API_KEY=\n' > ~/.env
  chmod 600 ~/.env
  echo "    created ~/.env (fill the three keys from 1Password: open -e ~/.env)"
fi
echo
echo "Done. Next: hf auth login   (paste the Hugging Face Write token)"
echo "      wandb login         (paste the CoreWeave Forge / W&B key)"
