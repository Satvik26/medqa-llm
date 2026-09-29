"""Offline batched generation with vLLM: base model plus zero or more hot-swappable LoRA adapters."""

from __future__ import annotations

import time
from collections.abc import Sequence
from pathlib import Path

from medqa.config import ExperimentConfig
from medqa.inference.base import GenerationBatch
from medqa.prompts import clean_generation, render_prompt


class VLLMGenerator:
    def __init__(self, cfg: ExperimentConfig, enable_lora: bool = True) -> None:
        from vllm import LLM, SamplingParams

        s = cfg.serve
        kwargs = {
            "model": s.base_model,
            "dtype": s.dtype,
            "max_model_len": s.max_model_len,
            "gpu_memory_utilization": s.gpu_memory_utilization,
            "seed": 0,
        }
        if s.quantization:
            kwargs["quantization"] = s.quantization
        if enable_lora:
            kwargs.update(enable_lora=True, max_lora_rank=s.max_lora_rank, max_loras=2)
        self.llm = LLM(**kwargs)
        self.tokenizer = self.llm.get_tokenizer()
        g = cfg.generation
        self.sampling = SamplingParams(
            temperature=g.temperature, top_p=g.top_p, max_tokens=g.max_new_tokens
        )
        self.name = s.base_model
        self._adapter: Path | None = None
        self._lora_ids: dict[Path, int] = {}

    def use_adapter(self, adapter: Path | None) -> VLLMGenerator:
        self._adapter = adapter
        return self

    def _lora_request(self):
        if self._adapter is None:
            return None
        from vllm.lora.request import LoRARequest

        lora_id = self._lora_ids.setdefault(self._adapter, len(self._lora_ids) + 1)
        return LoRARequest(f"adapter-{lora_id}", lora_id, str(self._adapter))

    def generate(
        self, questions: Sequence[str], contexts: Sequence[Sequence[str]] | None = None
    ) -> GenerationBatch:
        ctx = contexts or [None] * len(questions)
        prompts = [render_prompt(self.tokenizer, q, c) for q, c in zip(questions, ctx, strict=True)]
        start = time.perf_counter()
        outputs = self.llm.generate(
            prompts, self.sampling, lora_request=self._lora_request(), use_tqdm=True
        )
        seconds = time.perf_counter() - start
        return GenerationBatch(
            texts=[clean_generation(o.outputs[0].text) for o in outputs],
            completion_tokens=[len(o.outputs[0].token_ids) for o in outputs],
            seconds=seconds,
        )


def serve_command(cfg: ExperimentConfig, adapter: Path | None) -> list[str]:
    """`vllm serve` arguments for an OpenAI-compatible server (used by the demo and remote eval)."""
    s = cfg.serve
    cmd = [
        "vllm", "serve", s.base_model,
        "--dtype", s.dtype,
        "--max-model-len", str(s.max_model_len),
        "--gpu-memory-utilization", str(s.gpu_memory_utilization),
        "--host", s.host,
        "--port", str(s.port),
    ]  # fmt: skip
    if s.quantization:
        cmd += ["--quantization", s.quantization]
    if adapter is not None:
        cmd += [
            "--enable-lora",
            "--max-lora-rank", str(s.max_lora_rank),
            "--lora-modules", f"{s.served_lora_name}={adapter}",
        ]  # fmt: skip
    return cmd
