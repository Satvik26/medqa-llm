"""`medqa` command-line interface. Run `medqa --help` or `medqa <command> --help`."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Annotated

import typer
from rich import print
from rich.table import Table

from medqa.config import ExperimentConfig, load_config

app = typer.Typer(
    help="Fine-tune, serve and benchmark LLMs for medical question answering.", no_args_is_help=True
)
data_app = typer.Typer(help="Download and split the dataset.", no_args_is_help=True)
analyze_app = typer.Typer(help="CPU analysis: EDA, clustering, embeddings.", no_args_is_help=True)
rag_app = typer.Typer(help="Retrieval index and RAFT training data.", no_args_is_help=True)
ckpt_app = typer.Typer(help="Inspect and rank training checkpoints.", no_args_is_help=True)
bench_app = typer.Typer(help="Benchmarks (quantization).", no_args_is_help=True)
for sub, name in (
    (data_app, "data"),
    (analyze_app, "analyze"),
    (rag_app, "rag"),
    (ckpt_app, "checkpoints"),
    (bench_app, "bench"),
):
    app.add_typer(sub, name=name)

DEFAULT_CONFIG = Path("configs/models/qwen3-1.7b.yaml")
ConfigOpt = Annotated[
    Path, typer.Option("--config", "-c", help="Experiment YAML.", exists=True, dir_okay=False)
]
AdapterOpt = Annotated[
    str, typer.Option(help="'auto' = <run_dir>/adapter, 'none' = base model, or a path.")
]
SetOpt = Annotated[
    list[str] | None,
    typer.Option("--set", help="Override a config value, e.g. --set training.max_steps=30"),
]


def _cfg(path: Path, overrides: list[str] | None = None) -> ExperimentConfig:
    import yaml

    parsed = {}
    for item in overrides or []:
        key, sep, value = item.partition("=")
        if not sep:
            raise typer.BadParameter(f"--set expects key=value, got {item!r}")
        parsed[key.strip()] = yaml.safe_load(value)
    return load_config(path, **parsed)


def _adapter(cfg: ExperimentConfig, value: str) -> Path | None:
    if value == "none":
        return None
    path = cfg.run_dir / "adapter" if value == "auto" else Path(value)
    if not path.exists():
        raise typer.BadParameter(f"Adapter not found: {path}. Train first or pass --adapter none.")
    return path


# ---------------------------------------------------------------- data
@data_app.command("split")
def data_split(config: ConfigOpt = DEFAULT_CONFIG) -> None:
    """Download the dataset and write the fixed train/val/test split to Parquet."""
    from medqa.data import load_flashcards, make_splits, save_splits
    from medqa.data.split import overlap_stats

    cfg = _cfg(config)
    df = load_flashcards(cfg.data.dataset)
    splits = make_splits(df, cfg.data.test_size, cfg.data.val_size, cfg.data.seed)
    save_splits(splits, cfg.data.split_dir)
    stats = {name: len(f) for name, f in splits.items()} | overlap_stats(splits)
    (cfg.data.split_dir / "stats.json").write_text(json.dumps(stats, indent=2))
    print(f"Saved to {cfg.data.split_dir}: {stats}")


@data_app.command("publish")
def data_publish(
    repo: Annotated[str, typer.Option(help="e.g. your-name/medical-flashcards-splits")],
    config: ConfigOpt = DEFAULT_CONFIG,
) -> None:
    """Push the split to the Hugging Face Hub so results are reproducible by others."""
    from medqa.publish import push_splits

    print(push_splits(_cfg(config).data.split_dir, repo))


# ---------------------------------------------------------------- analysis
def _train_frame(cfg: ExperimentConfig):
    from medqa.data.split import load_splits

    return load_splits(cfg.data.split_dir)["train"]


@analyze_app.command("eda")
def analyze_eda(config: ConfigOpt = DEFAULT_CONFIG, out: Path = Path("runs/analysis/eda")) -> None:
    """Lengths, vocabulary, frequent words, word clouds, n-grams."""
    from medqa.analysis.eda import run_eda

    out.mkdir(parents=True, exist_ok=True)
    stats = run_eda(_train_frame(_cfg(config)), out)
    print({k: v for k, v in stats.items() if k != "top_ngrams"})


@analyze_app.command("cluster")
def analyze_cluster(
    config: ConfigOpt = DEFAULT_CONFIG,
    out: Path = Path("runs/analysis/clustering"),
    k: int | None = None,
) -> None:
    """TF-IDF + k-means topics over the answers."""
    from medqa.analysis.clustering import run_clustering

    out.mkdir(parents=True, exist_ok=True)
    summary = run_clustering(_train_frame(_cfg(config))["answer"].tolist(), out, k=k)
    table = Table("cluster", "size", "top terms")
    for c in summary["clusters"]:
        table.add_row(str(c["id"]), str(c["size"]), ", ".join(c["top_terms"][:8]))
    print(f"k={summary['k']} silhouette={summary['silhouette']:.4f}")
    print(table)


@analyze_app.command("embeddings")
def analyze_embeddings(
    config: ConfigOpt = DEFAULT_CONFIG,
    out: Path = Path("runs/analysis/embeddings"),
    pretrained: Annotated[
        str | None,
        typer.Option(help="gensim-downloader model to compare, e.g. glove-wiki-gigaword-100"),
    ] = None,
) -> None:
    """Train domain Word2Vec and inspect neighbours of medical terms."""
    from medqa.analysis.embeddings import run_embeddings
    from medqa.data.preprocess import preprocess

    out.mkdir(parents=True, exist_ok=True)
    df = _train_frame(_cfg(config))
    sentences = [
        preprocess(t, remove_stopwords=True) for t in (df["question"] + " " + df["answer"])
    ]
    result = run_embeddings(sentences, out, pretrained)
    for term, nbrs in result["domain_neighbours"].items():
        print(f"[bold]{term}[/bold]: {', '.join(w for w, _ in nbrs)}")


# ---------------------------------------------------------------- RAG
@rag_app.command("build-index")
def rag_build_index(config: ConfigOpt = DEFAULT_CONFIG) -> None:
    """Index training flashcards with BM25 + PubMedBERT embeddings."""
    from medqa.retrieval.raft_data import build_index

    cfg = _cfg(config)
    retriever = build_index(cfg)
    print(f"Indexed {len(retriever.corpus)} flashcards into {cfg.data.index_dir}")


@rag_app.command("build-data")
def rag_build_data(config: ConfigOpt = Path("configs/rag_qwen3-1.7b.yaml")) -> None:
    """Attach top-k retrieved passages to every split (self-matches excluded for train)."""
    from medqa.retrieval.raft_data import build_rag_splits

    print(build_rag_splits(_cfg(config)))


@rag_app.command("search")
def rag_search(question: str, config: ConfigOpt = DEFAULT_CONFIG, k: int = 3) -> None:
    """Try the retriever on one question."""
    from medqa.retrieval.hybrid import HybridRetriever

    for p in HybridRetriever.load(_cfg(config).data.index_dir).search([question], k=k)[0]:
        print(f"[{p.score:.4f}] {p.text}")


# ---------------------------------------------------------------- training
@app.command()
def train(
    config: ConfigOpt = DEFAULT_CONFIG,
    resume: Annotated[
        str | None,
        typer.Option(help="'auto' (latest local checkpoint), 'hub', or a checkpoint path."),
    ] = None,
    overrides: SetOpt = None,
) -> None:
    """QLoRA/LoRA fine-tuning with Unsloth (GPU)."""
    from medqa.training.sft import train as run

    run(_cfg(config, overrides), resume)


@app.command()
def export(
    config: ConfigOpt = DEFAULT_CONFIG,
    adapter: AdapterOpt = "auto",
    fmt: Annotated[
        str, typer.Option("--format", help="merged_16bit | merged_4bit | gguf")
    ] = "merged_16bit",
    out: Path | None = None,
) -> None:
    """Merge the adapter into the base model, or export GGUF for llama.cpp / Ollama."""
    from medqa.training.merge_export import export as run

    cfg = _cfg(config)
    print(run(cfg, _adapter(cfg, adapter), out or cfg.run_dir / fmt, fmt))  # type: ignore[arg-type]


@ckpt_app.command("list")
def checkpoints_list(config: ConfigOpt = DEFAULT_CONFIG) -> None:
    """Saved checkpoints with their validation loss; marks the best one."""
    from medqa.training.checkpoints import (
        best_checkpoint,
        eval_losses,
        list_checkpoints,
        read_trainer_state,
    )

    cfg = _cfg(config)
    losses = eval_losses(read_trainer_state(cfg.run_dir) or {})
    best = best_checkpoint(cfg.run_dir)
    table = Table("step", "path", "eval_loss", "best")
    for c in list_checkpoints(cfg.run_dir):
        loss = losses.get(c.step)
        table.add_row(
            str(c.step),
            str(c.path),
            f"{loss:.4f}" if loss is not None else "-",
            "*" if best and best.name == c.path.name else "",
        )
    print(table)


@ckpt_app.command("evaluate")
def checkpoints_evaluate(config: ConfigOpt = DEFAULT_CONFIG, sample: int | None = None) -> None:
    """Score every checkpoint on a validation sample with vLLM (GPU)."""
    from medqa.evaluation.runner import evaluate_checkpoints

    evaluate_checkpoints(_cfg(config), sample)


# ---------------------------------------------------------------- inference & evaluation
@app.command()
def serve(config: ConfigOpt = DEFAULT_CONFIG, adapter: AdapterOpt = "auto") -> None:
    """Start an OpenAI-compatible vLLM server (GPU). The adapter is served as `medqa`."""
    from medqa.inference.vllm_engine import serve_command

    cfg = _cfg(config)
    cmd = serve_command(cfg, _adapter(cfg, adapter))
    print(" ".join(cmd))
    os.execvp(cmd[0], cmd)


@app.command()
def evaluate(
    config: ConfigOpt = DEFAULT_CONFIG,
    backend: Annotated[
        str, typer.Option(help="vllm (offline, GPU) | api (OpenAI-compatible) | hf (transformers)")
    ] = "vllm",
    adapter: AdapterOpt = "none",
    rag: Annotated[bool, typer.Option(help="Put retrieved passages in the prompt.")] = False,
    limit: int | None = None,
    split: str | None = None,
    tag: str | None = None,
    judge_config: Annotated[
        Path | None, typer.Option(help="Config whose `api` block is used as LLM judge.")
    ] = None,
    judge_sample: int = 0,
    overrides: SetOpt = None,
) -> None:
    """Generate answers for the test split and score them (BERTScore, ROUGE-L, semantic cosine)."""
    from medqa.evaluation.runner import evaluate as run

    cfg = _cfg(config, overrides)
    judge_api = load_config(judge_config).api if judge_config else None
    if backend == "api":
        # The adapter lives inside the vLLM server; the client only selects it by served name.
        adapter_path = None if adapter == "none" else Path(f"served:{cfg.serve.served_lora_name}")
    else:
        adapter_path = _adapter(cfg, adapter)
    run(cfg, backend, adapter_path, rag, split, limit, tag, judge_api, judge_sample)  # type: ignore[arg-type]


@app.command()
def rescore(
    predictions: Annotated[Path, typer.Argument(exists=True)], config: ConfigOpt = DEFAULT_CONFIG
) -> None:
    """Recompute metrics for an existing predictions.jsonl."""
    from medqa.evaluation.runner import rescore as run

    print(run(predictions, _cfg(config))["metrics"])


@bench_app.command("quant")
def bench_quant(
    config: ConfigOpt = DEFAULT_CONFIG, adapter: AdapterOpt = "auto", n: int = 200
) -> None:
    """Compare fp16 / int8 / NF4 inference: memory, speed and quality (GPU)."""
    from medqa.evaluation.quant_bench import run_quant_bench

    cfg = _cfg(config)
    run_quant_bench(cfg, _adapter(cfg, adapter), n)


@app.command()
def report(
    runs: Path = Path("runs"),
    figures: Path = Path("reports/figures"),
    results_md: Path = Path("docs/results.md"),
    course_csv: Path = Path("reports/course_results.csv"),
) -> None:
    """Rebuild all result figures and docs/results.md from run artifacts."""
    from medqa.evaluation.report import build_report

    for path in build_report(runs, figures, results_md, course_csv):
        print(f"wrote {path}")
    print(f"wrote {results_md}")


@app.command()
def publish(
    repo: Annotated[str, typer.Option(help="Hub repo id, e.g. your-name/qwen3-1.7b-medqa-lora")],
    tag: Annotated[str, typer.Option(help="Release tag, e.g. v1.0")],
    config: ConfigOpt = DEFAULT_CONFIG,
    adapter: AdapterOpt = "auto",
    public: bool = False,
) -> None:
    """Upload an adapter (or any checkpoint-N) to the Hub with a model card and a version tag."""
    from medqa.publish import push_adapter

    cfg = _cfg(config)
    print(push_adapter(cfg, _adapter(cfg, adapter), repo, tag, private=not public))


# ---------------------------------------------------------------- apps
@app.command()
def demo(
    config: ConfigOpt = DEFAULT_CONFIG,
    api_base: str = "http://localhost:8000/v1",
    model: Annotated[
        str, typer.Option(help="Served model name: 'medqa' (adapter) or the base model id.")
    ] = "medqa",
    rag: bool = False,
    voice: bool = False,
    share: bool = False,
    port: int = 7860,
) -> None:
    """Gradio UI in front of the vLLM server started by `medqa serve`."""
    from medqa.config import ApiConfig
    from medqa.demo import build_app

    cfg = _cfg(config)
    api = ApiConfig(base_url=api_base, model=model)
    build_app(cfg, api, cfg.data.index_dir if rag else None, voice).launch(
        server_name="0.0.0.0", server_port=port, share=share
    )


@app.command()
def voice(
    audio: Annotated[
        Path, typer.Argument(exists=True, help="WAV/MP3 file with a spoken question.")
    ],
    config: ConfigOpt = DEFAULT_CONFIG,
    api_base: str = "http://localhost:8000/v1",
    model: str = "medqa",
    out: Path = Path("answer.wav"),
) -> None:
    """Answer a spoken question and write the spoken answer to a WAV file."""
    import soundfile as sf

    from medqa.config import ApiConfig
    from medqa.inference.openai_engine import OpenAIGenerator
    from medqa.voice.assistant import VoiceAssistant

    cfg = _cfg(config)
    gen = OpenAIGenerator(ApiConfig(base_url=api_base, model=model), cfg.generation)
    question, reply, (sr, wav) = VoiceAssistant(lambda q: gen.generate([q]).texts[0]).respond(
        str(audio)
    )
    sf.write(out, wav, sr)
    print(f"[bold]Q:[/bold] {question}\n[bold]A:[/bold] {reply}\nAudio: {out}")


if __name__ == "__main__":
    app()
