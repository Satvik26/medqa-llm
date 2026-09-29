"""Typed experiment configuration loaded from YAML files with `base:` inheritance."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DataConfig(_Strict):
    dataset: str = "medalpaca/medical_meadow_medical_flashcards"
    split_dir: Path = Path("data/splits")
    test_size: float = Field(0.2, gt=0, lt=1)
    val_size: float = Field(0.05, ge=0, lt=1)
    seed: int = 42
    rag: bool = False
    rag_top_k: int = Field(3, ge=1)
    index_dir: Path = Path("runs/index")

    @property
    def active_split_dir(self) -> Path:
        """Directory holding the splits used for training: context-augmented when `rag` is on."""
        return (
            self.split_dir.with_name(self.split_dir.name + "_rag") if self.rag else self.split_dir
        )


class LoraConfig(_Strict):
    r: int = 32
    alpha: int = 32
    dropout: float = 0.0
    target_modules: list[str] = Field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )


class ModelConfig(_Strict):
    display_name: str = "Qwen3-1.7B"
    base_model: str = "unsloth/Qwen3-1.7B-unsloth-bnb-4bit"
    max_seq_length: int = 512
    load_in_4bit: bool = True
    lora: LoraConfig = Field(default_factory=LoraConfig)


class TrainingConfig(_Strict):
    per_device_train_batch_size: int = 8
    gradient_accumulation_steps: int = 4
    num_train_epochs: float = 3
    max_steps: int = -1  # > 0 overrides epochs; use a small value for a smoke test
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    lr_scheduler_type: str = "linear"
    optim: str = "adamw_8bit"
    logging_steps: int = 10
    save_steps: int = 100
    eval_steps: int = 100
    save_total_limit: int = 3
    early_stopping_patience: int | None = 3
    seed: int = 3407
    report_to: str = "tensorboard"
    push_to_hub: bool = False
    hub_model_id: str | None = None


class ServeConfig(_Strict):
    base_model: str = "unsloth/Qwen3-1.7B"
    dtype: str = "half"
    quantization: str | None = None
    max_model_len: int = 2048
    gpu_memory_utilization: float = Field(0.6, gt=0, le=1)
    max_lora_rank: int = 32
    host: str = "0.0.0.0"
    port: int = 8000
    served_lora_name: str = "medqa"


class GenerationConfig(_Strict):
    max_new_tokens: int = 128
    temperature: float = 0.0
    top_p: float = 1.0


class EvalConfig(_Strict):
    split: str = "test"
    limit: int | None = None
    checkpoint_sample: int = 200
    bertscore_model: str = "roberta-large"
    semantic_model: str = "NeuML/pubmedbert-base-embeddings"


class ApiConfig(_Strict):
    """An OpenAI-compatible endpoint: a frontier model, or a vLLM server started with `medqa serve`."""

    base_url: str = "http://localhost:8000/v1"
    model: str = "medqa"
    api_key_env: str | None = None
    max_concurrency: int = 8
    # Reasoning models spend completion tokens on hidden reasoning, need `max_completion_tokens`
    # and reject a custom temperature, so all three are configurable.
    max_tokens: int | None = None
    max_tokens_param: str = "max_tokens"
    temperature: float | None = 0.0
    extra_body: dict[str, Any] = Field(default_factory=dict)


class ExperimentConfig(_Strict):
    run_name: str = "base"
    output_root: Path = Path("runs")
    data: DataConfig = Field(default_factory=DataConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    serve: ServeConfig = Field(default_factory=ServeConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)
    api: ApiConfig | None = None

    @property
    def run_dir(self) -> Path:
        return self.output_root / self.run_name

    def dump(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(self.model_dump(mode="json"), sort_keys=False))


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_raw(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text()) or {}
    parent = raw.pop("base", None)
    if parent is None:
        return raw
    return deep_merge(_load_raw((path.parent / parent).resolve()), raw)


def load_config(path: str | Path | None = None, **overrides: Any) -> ExperimentConfig:
    """Load a YAML config (following `base:` links); `overrides` use dotted keys, e.g. `data.rag=True`."""
    raw = _load_raw(Path(path)) if path else {}
    for dotted, value in overrides.items():
        node = raw
        *parents, leaf = dotted.split(".")
        for key in parents:
            node = node.setdefault(key, {})
        node[leaf] = value
    return ExperimentConfig.model_validate(raw)
