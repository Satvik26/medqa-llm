from pathlib import Path

import pytest
from pydantic import ValidationError

from medqa.config import deep_merge, load_config

CONFIGS = sorted(Path("configs").rglob("*.yaml"))


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: str(p))
def test_every_config_validates(path):
    cfg = load_config(path)
    assert cfg.run_name


def test_model_config_inherits_base():
    cfg = load_config("configs/models/qwen3-8b.yaml")
    assert cfg.run_name == "qwen3-8b-qlora"
    assert cfg.serve.quantization == "bitsandbytes"
    assert cfg.training.per_device_train_batch_size == 2
    assert cfg.training.learning_rate == 2e-4  # from base.yaml
    assert cfg.model.lora.r == 32


def test_frontier_config_has_api():
    cfg = load_config("configs/frontier.yaml")
    assert cfg.api is not None
    assert cfg.api.max_tokens_param == "max_completion_tokens"
    assert cfg.api.temperature is None


def test_dotted_overrides():
    cfg = load_config(
        "configs/models/qwen3-1.7b.yaml", **{"data.rag": True, "training.save_steps": 50}
    )
    assert cfg.data.rag and cfg.training.save_steps == 50
    assert cfg.data.active_split_dir == Path("data/splits_rag")


def test_unknown_keys_are_rejected():
    with pytest.raises(ValidationError):
        load_config(None, **{"training.learning_rte": 1e-4})


def test_deep_merge_keeps_nested_defaults():
    assert deep_merge({"a": {"x": 1, "y": 2}}, {"a": {"y": 3}}) == {"a": {"x": 1, "y": 3}}
