#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

if [ ! -f .env ]; then
    echo "Missing .env in $REPO_ROOT"
    echo "Add credentials to .env before running train setup."
    exit 1
fi

# Export variables from .env so aws/wandb use them immediately.
set -a
# shellcheck disable=SC1091
source .env
set +a

export PATH="$PATH:$HOME/.local/bin"

if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
# shellcheck disable=SC1091
source "$HOME/.local/bin/env"

uv venv --clear
# shellcheck disable=SC1091
source .venv/bin/activate

mkdir -p .cache data
uv pip install -r requirements.txt --cache-dir=./.cache
uv pip install awscli --cache-dir=./.cache

aws s3 sync s3://lc-inpaint/data ./data --exclude "*" --include "*_final.npz"

echo "Setup complete. Start training with:"
echo "  bash train-multi.sh"
echo "or"
echo "  accelerate launch train.py --config configs/randar_nlcd_128_large.yaml"
