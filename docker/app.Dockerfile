# CPU image with the medqa package: Gradio demo, analysis and metrics. Builds and runs on a Mac.
FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PROJECT_ENVIRONMENT=/opt/venv PATH="/opt/venv/bin:$PATH"
WORKDIR /app

# Dependencies first so code edits do not invalidate this layer.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-install-project --no-dev --extra demo

COPY src ./src
COPY configs ./configs
RUN uv sync --frozen --no-dev --extra demo

EXPOSE 7860
CMD ["medqa", "demo", "--api-base", "http://vllm:8000/v1"]
