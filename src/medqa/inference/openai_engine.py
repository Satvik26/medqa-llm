"""Generation through any OpenAI-compatible chat API: a frontier model or our own vLLM server."""

from __future__ import annotations

import asyncio
import os
import time
from collections.abc import Sequence

from medqa.config import ApiConfig, GenerationConfig
from medqa.inference.base import GenerationBatch
from medqa.prompts import build_messages, clean_generation


class OpenAIGenerator:
    def __init__(self, api: ApiConfig, generation: GenerationConfig) -> None:
        self.api = api
        self.generation = generation
        self.name = api.model

    def _client(self):
        from openai import AsyncOpenAI

        key = os.environ.get(self.api.api_key_env, "") if self.api.api_key_env else "not-needed"
        if self.api.api_key_env and not key:
            raise RuntimeError(f"Set the {self.api.api_key_env} environment variable")
        return AsyncOpenAI(base_url=self.api.base_url, api_key=key, max_retries=5, timeout=120)

    def _request_kwargs(self) -> dict:
        kwargs: dict = {
            self.api.max_tokens_param: self.api.max_tokens or self.generation.max_new_tokens
        }
        if self.api.temperature is not None:
            kwargs["temperature"] = self.api.temperature
        if self.api.extra_body:
            kwargs["extra_body"] = self.api.extra_body
        return kwargs

    async def _run(self, questions, contexts) -> GenerationBatch:
        client = self._client()
        limit = asyncio.Semaphore(self.api.max_concurrency)
        kwargs = self._request_kwargs()

        async def one(question: str, ctx):
            async with limit:
                start = time.perf_counter()
                resp = await client.chat.completions.create(
                    model=self.api.model, messages=build_messages(question, ctx), **kwargs
                )
                latency = time.perf_counter() - start
            text = clean_generation(resp.choices[0].message.content or "")
            tokens = resp.usage.completion_tokens if resp.usage else 0
            return text, tokens, latency

        start = time.perf_counter()
        results = await asyncio.gather(
            *(one(q, c) for q, c in zip(questions, contexts, strict=True))
        )
        seconds = time.perf_counter() - start
        await client.close()
        texts, tokens, latencies = (
            map(list, zip(*results, strict=True)) if results else ([], [], [])
        )
        return GenerationBatch(
            texts=texts, completion_tokens=tokens, seconds=seconds, latencies=latencies
        )

    def generate(
        self, questions: Sequence[str], contexts: Sequence[Sequence[str]] | None = None
    ) -> GenerationBatch:
        return asyncio.run(self._run(list(questions), list(contexts or [None] * len(questions))))
