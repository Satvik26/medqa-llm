from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class GenerationBatch:
    texts: list[str]
    completion_tokens: list[int]
    seconds: float
    latencies: list[float] = field(default_factory=list)

    @property
    def throughput(self) -> dict[str, float]:
        n, tokens = len(self.texts), sum(self.completion_tokens)
        stats = {
            "seconds": round(self.seconds, 2),
            "questions_per_s": round(n / self.seconds, 3) if self.seconds else 0.0,
            "tokens_per_s": round(tokens / self.seconds, 1) if self.seconds else 0.0,
            "mean_completion_tokens": round(tokens / n, 1) if n else 0.0,
        }
        if self.latencies:
            stats["mean_latency_s"] = round(sum(self.latencies) / len(self.latencies), 3)
        return stats


class Generator(Protocol):
    name: str

    def generate(
        self, questions: Sequence[str], contexts: Sequence[Sequence[str]] | None = None
    ) -> GenerationBatch: ...
