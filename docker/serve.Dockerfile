# OpenAI-compatible vLLM server with a medqa LoRA adapter mounted at /adapter.
# Base image already contains CUDA, PyTorch and vLLM.
FROM vllm/vllm-openai:latest

ENV BASE_MODEL=unsloth/Qwen3-1.7B \
    MAX_LORA_RANK=32 \
    MAX_MODEL_LEN=2048

EXPOSE 8000
ENTRYPOINT ["/bin/sh", "-c", "exec vllm serve \"$BASE_MODEL\" --dtype half --max-model-len \"$MAX_MODEL_LEN\" --enable-lora --max-lora-rank \"$MAX_LORA_RANK\" --lora-modules medqa=/adapter --host 0.0.0.0 --port 8000 \"$@\"", "--"]
