"""Inference quantization benchmark: same model and adapter loaded in fp16, int8 and NF4.

Measures weight memory, peak GPU memory, load time, throughput and answer quality, so the
memory-vs-quality trade-off of quantization is visible in one table.
"""

from __future__ import annotations

import gc
import json
import time
from pathlib import Path

from rich import print

from medqa.config import ExperimentConfig
from medqa.evaluation.metrics import bertscore, bootstrap_ci
from medqa.evaluation.runner import eval_frame

MODES = ("fp16", "int8", "nf4")


def run_quant_bench(
    cfg: ExperimentConfig, adapter: Path | None, n: int = 200, modes=MODES
) -> list[dict]:
    import torch

    from medqa.inference.hf_engine import HFGenerator

    if cfg.serve.quantization:
        raise ValueError(
            "Use a config whose serve.base_model holds 16-bit weights (e.g. qwen3-1.7b)"
        )
    frame = eval_frame(cfg, "test", n, rag=False)
    refs = frame["answer"].tolist()
    rows = []
    for mode in modes:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        gen = HFGenerator(
            cfg.serve.base_model,
            adapter,
            precision=mode,
            max_new_tokens=cfg.generation.max_new_tokens,
        )
        load_s = time.perf_counter() - start
        batch = gen.generate(frame["question"].tolist())
        f1 = bertscore(batch.texts, refs, cfg.eval.bertscore_model)["bertscore_f1"]
        rows.append(
            {
                "precision": mode,
                "weights_gb": gen.memory_footprint_gb,
                "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 3),
                "load_s": round(load_s, 1),
                **batch.throughput,
                "bertscore_f1": bootstrap_ci(f1),
            }
        )
        print(rows[-1])
        del gen
        gc.collect()

    out = cfg.output_root / "quant_bench" / cfg.run_name
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(
        json.dumps(
            {
                "model": cfg.serve.base_model,
                "adapter": str(adapter) if adapter else None,
                "n": n,
                "rows": rows,
            },
            indent=2,
        )
    )
    return rows
