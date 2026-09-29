# Reproducible fine-tuning environment for any Linux host with an NVIDIA GPU.
#   docker build -f docker/train.Dockerfile -t medqa-train .
#   docker run --gpus all -v $PWD/data:/app/data -v $PWD/runs:/app/runs medqa-train \
#       medqa train -c configs/models/qwen3-1.7b.yaml --resume auto
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_INSTALL_DIR=/opt/python \
    UV_PROJECT_ENVIRONMENT=/opt/venv PATH="/opt/venv/bin:$PATH"
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends git build-essential && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock README.md .python-version ./
RUN uv sync --frozen --no-install-project --no-dev --extra train

COPY src ./src
COPY configs ./configs
RUN uv sync --frozen --no-dev --extra train

CMD ["medqa", "--help"]
