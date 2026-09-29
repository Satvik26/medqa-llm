"""transformers + PEFT generation: runs on CUDA, Apple MPS or CPU and supports int8 / NF4 loading.

Slower than vLLM, but it reports exact weight memory, which the quantization benchmark needs.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from medqa.inference.base import GenerationBatch
from medqa.prompts import clean_generation, render_prompt

Precision = Literal["fp16", "int8", "nf4", "fp32"]


def _device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class HFGenerator:
    def __init__(
        self,
        base_model: str,
        adapter: Path | None = None,
        precision: Precision = "fp16",
        max_new_tokens: int = 128,
        batch_size: int = 16,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        self.device = _device()
        kwargs: dict = {}
        if precision == "int8":
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        elif precision == "nf4":
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
        if precision in ("int8", "nf4"):
            if self.device != "cuda":
                raise RuntimeError("bitsandbytes int8/nf4 needs a CUDA GPU")
            kwargs["device_map"] = {"": 0}
        else:
            use_fp32 = precision == "fp32" or self.device == "cpu"
            kwargs["dtype"] = torch.float32 if use_fp32 else torch.float16

        self.tokenizer = AutoTokenizer.from_pretrained(str(adapter or base_model))
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(base_model, **kwargs)
        if "device_map" not in kwargs:
            model = model.to(self.device)
        if adapter is not None:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, str(adapter))
        self.model = model.eval()
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size
        self.name = f"{base_model} [{precision}]"

    @property
    def memory_footprint_gb(self) -> float:
        return round(self.model.get_memory_footprint() / 1024**3, 3)

    def generate(
        self, questions: Sequence[str], contexts: Sequence[Sequence[str]] | None = None
    ) -> GenerationBatch:
        import torch

        ctx = list(contexts or [None] * len(questions))
        prompts = [render_prompt(self.tokenizer, q, c) for q, c in zip(questions, ctx, strict=True)]
        texts, tokens = [], []
        start = time.perf_counter()
        for i in range(0, len(prompts), self.batch_size):
            enc = self.tokenizer(
                prompts[i : i + self.batch_size], return_tensors="pt", padding=True
            ).to(self.model.device)
            with torch.inference_mode():
                out = self.model.generate(
                    **enc,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            new = out[:, enc["input_ids"].shape[1] :]
            texts += [
                clean_generation(t)
                for t in self.tokenizer.batch_decode(new, skip_special_tokens=True)
            ]
            tokens += [int((row != self.tokenizer.pad_token_id).sum()) for row in new]
        return GenerationBatch(
            texts=texts, completion_tokens=tokens, seconds=time.perf_counter() - start
        )
