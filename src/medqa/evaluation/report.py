"""Rebuild every results figure and `docs/results.md` from the JSON files that runs leave behind."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from medqa.analysis import plots
from medqa.training.checkpoints import eval_losses, read_trainer_state, train_losses

TAG_ORDER = ["zero-shot", "zero-shot+rag", "finetuned", "finetuned+rag"]


def collect_results(runs_root: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(runs_root.glob("*/eval/*/metrics.json")):
        m = json.loads(path.read_text())
        row = {k: m.get(k) for k in ("display_name", "run_name", "tag", "n", "quantization", "rag")}
        for name, ci in m.get("metrics", {}).items():
            row[name], row[f"{name}_lo"], row[f"{name}_hi"] = ci["mean"], ci["lo"], ci["hi"]
        row.update({f"tp_{k}": v for k, v in m.get("throughput", {}).items()})
        if "judge" in m:
            row["judge_score"] = m["judge"]["mean"]
        rows.append(row)
    return pd.DataFrame(rows)


def _fmt(row: pd.Series, name: str) -> str:
    if name not in row or pd.isna(row[name]):
        return "-"
    return f"{row[name]:.3f} [{row[name + '_lo']:.3f}, {row[name + '_hi']:.3f}]"


def results_markdown(df: pd.DataFrame) -> str:
    header = "| Model | Setting | n | BERTScore F1 [95% CI] | ROUGE-L | Semantic cosine | Judge (1-5) | Answers/s |\n|---|---|---|---|---|---|---|---|"
    lines = [header]
    for _, r in df.sort_values("bertscore_f1", ascending=False).iterrows():
        judge = (
            f"{r['judge_score']:.2f}"
            if "judge_score" in r and pd.notna(r.get("judge_score"))
            else "-"
        )
        qps = r.get("tp_questions_per_s")
        lines.append(
            f"| {r['display_name']} | {r['tag']} | {r['n']} | {_fmt(r, 'bertscore_f1')} | {_fmt(r, 'rouge_l')} "
            f"| {_fmt(r, 'semantic_cosine')} | {judge} | {qps if pd.notna(qps) else '-'} |"
        )
    return "\n".join(lines)


def plot_model_comparison(df: pd.DataFrame, path: Path, metric: str = "bertscore_f1") -> Path:
    models = list(dict.fromkeys(df.sort_values(metric)["display_name"]))
    tags = [t for t in TAG_ORDER if t in set(df["tag"])]
    width = 0.8 / max(len(tags), 1)
    fig, ax = plots.plt.subplots(figsize=(max(8, 1.6 * len(models)), 4.5))
    for i, tag in enumerate(tags):
        sub = df[df["tag"] == tag].set_index("display_name").reindex(models)
        x = np.arange(len(models)) + i * width
        err = np.vstack([sub[metric] - sub[f"{metric}_lo"], sub[f"{metric}_hi"] - sub[metric]])
        ax.bar(x, sub[metric], width, yerr=err, capsize=3, label=tag)
    ax.set_xticks(np.arange(len(models)) + width * (len(tags) - 1) / 2, models, rotation=15)
    lo = np.nanmin(df[f"{metric}_lo"])
    ax.set_ylim(max(0, lo - 0.05), min(1, np.nanmax(df[f"{metric}_hi"]) + 0.02))
    ax.set(title=f"{metric} by model and setting (95% bootstrap CI)", ylabel=metric)
    ax.legend()
    return plots.save(fig, path)


def plot_loss_curves(run_dir: Path, path: Path) -> Path | None:
    state = read_trainer_state(run_dir)
    if not state:
        return None
    tr, ev = train_losses(state), eval_losses(state)
    fig, ax = plots.plt.subplots(figsize=(8, 4))
    ax.plot(list(tr), list(tr.values()), label="train loss", alpha=0.7)
    if ev:
        ax.plot(list(ev), list(ev.values()), marker="o", label="validation loss")
    if best := state.get("best_model_checkpoint"):
        step = int(Path(best).name.split("-")[-1])
        ax.axvline(
            step, color="grey", linestyle="--", linewidth=1, label=f"best checkpoint ({step})"
        )
    ax.set(title=f"Loss curves: {run_dir.name}", xlabel="step", ylabel="loss")
    ax.legend()
    return plots.save(fig, path)


def plot_checkpoint_scores(run_dir: Path, path: Path) -> Path | None:
    f = run_dir / "checkpoint_scores.json"
    if not f.exists():
        return None
    rows = json.loads(f.read_text())["checkpoints"]
    steps = [r["step"] for r in rows]
    return plots.lines(
        steps,
        {"BERTScore F1 (val sample)": [r["bertscore_f1"] for r in rows]},
        f"Checkpoint selection: {run_dir.name}",
        "step",
        "BERTScore F1",
        path,
        mark_best="BERTScore F1 (val sample)",
    )


def plot_quant_bench(results: Path, path: Path) -> Path:
    data = json.loads(results.read_text())
    rows = data["rows"]
    labels = [r["precision"] for r in rows]
    fig, axes = plots.plt.subplots(1, 3, figsize=(13, 3.8))
    panels = [
        ("Weight memory (GB)", [r["weights_gb"] for r in rows]),
        ("Throughput (tokens/s)", [r["tokens_per_s"] for r in rows]),
        ("BERTScore F1", [r["bertscore_f1"]["mean"] for r in rows]),
    ]
    for ax, (title, values) in zip(axes, panels, strict=True):
        ax.bar(labels, values, color=plots.PALETTE[: len(labels)])
        ax.set_title(title)
        for i, v in enumerate(values):
            ax.annotate(f"{v:.3g}", (i, v), ha="center", va="bottom", fontsize=9)
    axes[2].set_ylim(min(panels[2][1]) - 0.02, max(panels[2][1]) + 0.01)
    fig.suptitle(f"Inference precision trade-offs: {data['model']}")
    return plots.save(fig, path)


def plot_training_cost(runs_root: Path, path: Path) -> Path | None:
    """Peak VRAM and wall time per training run (QLoRA vs LoRA, model sizes)."""
    summaries = [json.loads(p.read_text()) for p in sorted(runs_root.glob("*/train_summary.json"))]
    if not summaries:
        return None
    names = [s["display_name"] for s in summaries]
    fig, axes = plots.plt.subplots(1, 2, figsize=(12, max(3, 0.5 * len(names))))
    axes[0].barh(names, [s.get("gpu_peak_reserved_gb") or 0 for s in summaries])
    axes[0].set_title("Peak GPU memory in training (GB)")
    axes[1].barh(names, [s["wall_time_s"] / 3600 for s in summaries], color=plots.PALETTE[1])
    axes[1].set_title("Training wall time (hours)")
    return plots.save(fig, path)


def plot_course_results(csv: Path, path: Path) -> Path:
    df = pd.read_csv(csv)
    pivot = df.pivot_table(index="model", columns="fine_tuned", values="bertscore_f1").sort_values(
        True
    )
    fig, ax = plots.plt.subplots(figsize=(8, 4))
    y = np.arange(len(pivot))
    ax.barh(y - 0.2, pivot[False], 0.4, label="base")
    ax.barh(y + 0.2, pivot[True], 0.4, label="fine-tuned")
    ax.set_yticks(y, pivot.index)
    ax.set_xlim(0.6, 0.95)
    ax.set(title="Course project (2024): BERTScore F1, base vs fine-tuned", xlabel="BERTScore F1")
    ax.legend(loc="lower right")
    return plots.save(fig, path)


def build_report(
    runs_root: Path, figures_dir: Path, results_md: Path, course_csv: Path
) -> list[Path]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    made: list[Path | None] = []
    if course_csv.exists():
        made.append(plot_course_results(course_csv, figures_dir / "course_results.png"))

    df = collect_results(runs_root)
    sections = [
        "# Results\n",
        "Generated by `medqa report` from `runs/*/eval/*/metrics.json`. Do not edit by hand.\n",
    ]
    if not df.empty:
        made.append(plot_model_comparison(df, figures_dir / "model_comparison.png"))
        sections += [
            "## Model comparison\n",
            results_markdown(df),
            "\n![Model comparison](../reports/figures/model_comparison.png)\n",
        ]

    for run_dir in (
        sorted(p for p in runs_root.iterdir() if p.is_dir()) if runs_root.exists() else []
    ):
        made.append(plot_loss_curves(run_dir, figures_dir / f"loss_{run_dir.name}.png"))
        made.append(
            plot_checkpoint_scores(run_dir, figures_dir / f"checkpoints_{run_dir.name}.png")
        )
    made.append(plot_training_cost(runs_root, figures_dir / "training_cost.png"))
    for results in sorted(runs_root.glob("quant_bench/*/results.json")):
        made.append(plot_quant_bench(results, figures_dir / f"quant_{results.parent.name}.png"))

    if course_csv.exists():
        sections += [
            "## Course baseline (2024)\n",
            "![Course results](../reports/figures/course_results.png)\n",
        ]
    results_md.parent.mkdir(parents=True, exist_ok=True)
    results_md.write_text("\n".join(sections))
    return [p for p in made if p is not None]
