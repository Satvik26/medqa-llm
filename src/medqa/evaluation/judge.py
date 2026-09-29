"""LLM-as-judge with the Prometheus absolute-grading prompt (1-5 against a reference answer)."""

from __future__ import annotations

import asyncio
import os
import re

from medqa.config import ApiConfig

RUBRIC = """[Is the response medically correct and complete compared with the reference answer?]
Score 1: The response is incorrect or contradicts the reference.
Score 2: The response is mostly incorrect, with a minor correct element.
Score 3: The response is partially correct but misses or distorts key facts.
Score 4: The response is correct with small omissions or imprecisions.
Score 5: The response is fully correct and conveys the same key facts as the reference."""

JUDGE_TEMPLATE = """###Task Description:
An instruction, a response to evaluate, a reference answer that gets a score of 5, and a score rubric representing an evaluation criteria are given.
1. Write a detailed feedback that assesses the quality of the response strictly based on the given score rubric, not evaluating in general.
2. After writing a feedback, write a score that is an integer between 1 and 5. You should refer to the score rubric.
3. The output format should look as follows: "Feedback: (write a feedback for criteria) [RESULT] (an integer number between 1 and 5)"
4. Please do not generate any other opening, closing, and explanations.

###The instruction to evaluate:
{question}

###Response to evaluate:
{response}

###Reference Answer (Score 5):
{reference}

###Score Rubrics:
{rubric}

###Feedback:"""

_RESULT_RE = re.compile(r"\[RESULT\]\s*\(?([1-5])\)?")


def build_judge_prompt(question: str, response: str, reference: str, rubric: str = RUBRIC) -> str:
    return JUDGE_TEMPLATE.format(
        question=question, response=response, reference=reference, rubric=rubric
    )


def parse_score(text: str) -> int | None:
    matches = _RESULT_RE.findall(text)
    return int(matches[-1]) if matches else None


def judge(
    api: ApiConfig, questions: list[str], responses: list[str], references: list[str]
) -> list[int | None]:
    from openai import AsyncOpenAI

    key = os.environ.get(api.api_key_env, "") if api.api_key_env else "not-needed"

    async def run() -> list[int | None]:
        client = AsyncOpenAI(base_url=api.base_url, api_key=key, max_retries=5)
        limit = asyncio.Semaphore(api.max_concurrency)
        extra = {"temperature": api.temperature} if api.temperature is not None else {}

        async def one(q: str, r: str, ref: str) -> int | None:
            async with limit:
                resp = await client.chat.completions.create(
                    model=api.model,
                    messages=[{"role": "user", "content": build_judge_prompt(q, r, ref)}],
                    **{api.max_tokens_param: api.max_tokens or 512},
                    **extra,
                    extra_body=api.extra_body or None,
                )
            return parse_score(resp.choices[0].message.content or "")

        try:
            return await asyncio.gather(
                *(one(*row) for row in zip(questions, responses, references, strict=True))
            )
        finally:
            await client.close()

    return asyncio.run(run())
