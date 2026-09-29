"""Merge a LoRA adapter into its base model, or export GGUF for llama.cpp / Ollama (CPU and Mac)."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from medqa.config import ExperimentConfig

ExportFormat = Literal["merged_16bit", "merged_4bit", "gguf"]


def export(
    cfg: ExperimentConfig,
    adapter: Path,
    out_dir: Path,
    fmt: ExportFormat,
    gguf_quant: str = "q4_k_m",
) -> Path:
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(adapter),
        max_seq_length=cfg.model.max_seq_length,
        load_in_4bit=cfg.model.load_in_4bit,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    if fmt == "gguf":
        model.save_pretrained_gguf(str(out_dir), tokenizer, quantization_method=gguf_quant)
    else:
        model.save_pretrained_merged(str(out_dir), tokenizer, save_method=fmt)
    return out_dir
