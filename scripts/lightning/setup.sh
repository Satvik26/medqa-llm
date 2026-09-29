#!/usr/bin/env bash
# One-time setup on a Lightning AI Studio (or any Linux + NVIDIA GPU box).
#   bash scripts/lightning/setup.sh train   # Unsloth fine-tuning environment
#   bash scripts/lightning/setup.sh serve   # vLLM inference + evaluation environment
# train and serve pin different torch builds, so a single .venv holds one of them at a time.
set -euo pipefail

MODE="${1:-train}"

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

case "$MODE" in
  train) uv sync --extra train --extra analysis --extra rag ;;
  serve) uv sync --extra serve --extra eval --extra rag --extra demo --extra local --extra voice ;;
  *) echo "usage: $0 [train|serve]" >&2; exit 1 ;;
esac

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
uv run python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

if [ ! -f data/splits/train.parquet ]; then
  uv run medqa data split
fi

cat <<'EOF'

Ready. Long jobs belong in tmux so they survive SSH disconnects:
  tmux new -s train
  uv run medqa train -c configs/models/qwen3-1.7b.yaml
  # detach: Ctrl-b d   reattach: tmux attach -t train
  # interrupted? rerun with --resume auto
EOF
