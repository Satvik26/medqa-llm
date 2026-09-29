CONFIG ?= configs/models/qwen3-1.7b.yaml

.PHONY: help setup-mac test lint format data analysis index report train serve evaluate demo docker-demo

help:  ## Show targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-14s %s\n", $$1, $$2}'

# ---------- laptop (CPU)
setup-mac:  ## Install CPU extras for analysis, retrieval and metrics
	uv sync --extra analysis --extra rag --extra eval --extra demo

test:  ## Run unit tests
	uv run pytest

lint:  ## Lint and check formatting
	uv run ruff check src tests && uv run ruff format --check src tests

format:  ## Auto-format
	uv run ruff check --fix src tests && uv run ruff format src tests

data:  ## Download dataset and create the fixed split
	uv run medqa data split

analysis: data  ## EDA, clustering and embeddings
	uv run medqa analyze eda && uv run medqa analyze cluster && uv run medqa analyze embeddings

index:  ## Build retrieval index and RAFT splits
	uv run medqa rag build-index && uv run medqa rag build-data

report:  ## Rebuild figures and docs/results.md from runs/
	uv run medqa report

# ---------- GPU (Lightning Studio)
train:  ## Fine-tune CONFIG (resumes automatically if checkpoints exist)
	uv run medqa train -c $(CONFIG) --resume auto

serve:  ## OpenAI-compatible vLLM server for CONFIG's adapter
	uv run medqa serve -c $(CONFIG)

evaluate:  ## Zero-shot and fine-tuned evaluation of CONFIG
	uv run medqa evaluate -c $(CONFIG) --adapter none
	uv run medqa evaluate -c $(CONFIG) --adapter auto

demo:  ## Gradio UI against the running vLLM server
	uv run medqa demo -c $(CONFIG)

docker-demo:  ## vLLM + Gradio with docker compose (NVIDIA host)
	docker compose -f docker/compose.yaml up --build
