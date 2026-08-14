"""A deliberately small LLM abstraction.

Two reasons this exists rather than a direct SDK call:

  * The evaluation gate and the API must use the same completion path, or the
    thing CI measures is not the thing that ships.
  * Deployed systems rarely get to pick the vendor. Being able to point Sift at
    whatever a customer already pays for is a two-line config change here.

It is an interface with two implementations and no plugin registry, because
that is all the problem needs.
"""

from __future__ import annotations

import abc
import os
from dataclasses import dataclass
from functools import lru_cache

from tenacity import retry, stop_after_attempt, wait_exponential

from sift.config import settings


class LLMNotConfigured(RuntimeError):
    """Raised when no API key is present.

    Deliberately distinct from a transport error: ingest and retrieval work
    fine without a key, and the API says so instead of returning a 500.
    """


@dataclass
class Completion:
    text: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0


class LLMClient(abc.ABC):
    model: str

    @abc.abstractmethod
    def complete(self, system: str, user: str, max_tokens: int, temperature: float) -> Completion:
        ...


class AnthropicClient(LLMClient):
    def __init__(self, model: str, api_key: str) -> None:
        import anthropic

        self.model = model
        self._client = anthropic.Anthropic(api_key=api_key)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=15), reraise=True)
    def complete(self, system: str, user: str, max_tokens: int, temperature: float) -> Completion:
        resp = self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        text = "".join(block.text for block in resp.content if block.type == "text")
        return Completion(
            text=text,
            model=self.model,
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
        )


class OpenAIClient(LLMClient):
    def __init__(self, model: str, api_key: str) -> None:
        from openai import OpenAI

        self.model = model
        self._client = OpenAI(api_key=api_key)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=15), reraise=True)
    def complete(self, system: str, user: str, max_tokens: int, temperature: float) -> Completion:
        resp = self._client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        usage = resp.usage
        return Completion(
            text=resp.choices[0].message.content or "",
            model=self.model,
            input_tokens=getattr(usage, "prompt_tokens", 0),
            output_tokens=getattr(usage, "completion_tokens", 0),
        )


@lru_cache(maxsize=2)
def get_llm(provider: str | None = None) -> LLMClient:
    provider = (provider or settings.llm_provider).lower()

    if provider == "anthropic":
        key = os.getenv("ANTHROPIC_API_KEY")
        if not key:
            raise LLMNotConfigured("ANTHROPIC_API_KEY is not set")
        return AnthropicClient(settings.anthropic_model, key)

    if provider == "openai":
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            raise LLMNotConfigured("OPENAI_API_KEY is not set")
        return OpenAIClient(settings.openai_model, key)

    raise LLMNotConfigured(f"Unknown provider {provider!r}; expected 'anthropic' or 'openai'")


def llm_available() -> bool:
    try:
        get_llm()
        return True
    except LLMNotConfigured:
        return False
