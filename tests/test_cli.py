from typer.testing import CliRunner

from medqa.cli import app

runner = CliRunner()


def test_help_lists_commands():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in (
        "train",
        "evaluate",
        "serve",
        "report",
        "publish",
        "rag",
        "checkpoints",
        "bench",
    ):
        assert command in result.output


def test_serve_command_line():
    from pathlib import Path

    from medqa.config import load_config
    from medqa.inference.vllm_engine import serve_command

    cmd = serve_command(load_config("configs/models/qwen3-8b.yaml"), Path("runs/x/adapter"))
    assert cmd[:3] == ["vllm", "serve", "unsloth/Qwen3-8B-unsloth-bnb-4bit"]
    assert "--quantization" in cmd and "--enable-lora" in cmd
    assert "medqa=runs/x/adapter" in cmd
