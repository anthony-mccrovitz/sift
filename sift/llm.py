"""A deliberately small LLM abstraction.

Two reasons this exists rather than a direct SDK call:

  * The evaluation gate and the API must use the same completion path, or the
    thing CI measures is not the thing that ships.
  * Deployed systems rarely get to pick the vendor. Being able to point Sift at
    whatever a customer already pays for is a two-line config change here.

It is an interface with three implementations and no plugin registry, because
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


class LocalClient(LLMClient):
    """A Hugging Face instruct model running on this machine. No key, no cost.

    Why this exists, when two hosted providers already do. Three of the gate's
    metrics -- citation validity, abstention rate and false abstention rate --
    are properties of a *generated answer*, but none of them asks a model for an
    opinion about quality. Citation validity is computed by our own code against
    the retrieved set; abstention is a property of the text. They need an answer,
    not a judge. Without a local option they sat unmeasured behind a paid key
    along with the two metrics that genuinely do need one.

    It also means anyone who clones this repository can run the whole pipeline,
    end to end, and watch POST /query return a cited answer, rather than an
    error telling them to go and buy something.

    What it is not. This is not a substitute for a judge. A small model grading
    its own output is worthless, so faithfulness and answer_relevancy still
    require a real one. Numbers produced under this client are labelled with the
    model that produced them, because citation discipline is a property of the
    generator and quoting it bare would imply otherwise.

    Greedy decoding, always: an eval gate cannot be sampling.
    """

    def __init__(self, model: str) -> None:
        self.model = model
        self._tokenizer = None
        self._model = None

    def _load(self):
        # Imported and loaded lazily. transformers and torch arrive with
        # sentence-transformers, so this costs nothing until someone selects
        # this provider, and importing them at module scope would push several
        # seconds onto every CLI that touches sift.llm.
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        device = (
            "mps" if torch.backends.mps.is_available()
            else "cuda" if torch.cuda.is_available()
            else "cpu"
        )
        self._tokenizer = AutoTokenizer.from_pretrained(self.model)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model,
            dtype=torch.float16 if device != "cpu" else torch.float32,
        ).to(device)
        self._model.eval()
        self._device = device

    def complete(self, system: str, user: str, max_tokens: int, temperature: float) -> Completion:
        import torch

        self._load()
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        prompt = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._device)

        with torch.no_grad():
            output = self._model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                # `temperature` is accepted and ignored on purpose. The
                # interface carries it because the hosted clients need it; here,
                # sampling would make the gate non-reproducible, which is the
                # one thing this project will not trade away.
                do_sample=False,
                pad_token_id=self._tokenizer.eos_token_id,
            )

        generated = output[0][inputs["input_ids"].shape[1]:]
        return Completion(
            text=self._tokenizer.decode(generated, skip_special_tokens=True).strip(),
            model=self.model,
            input_tokens=int(inputs["input_ids"].shape[1]),
            output_tokens=int(generated.shape[0]),
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

    if provider == "local":
        # No key to check. The failure mode here is a missing model rather than
        # a missing credential, and it surfaces at load time with a message from
        # transformers that says exactly which repo it could not fetch.
        return LocalClient(settings.local_model)

    raise LLMNotConfigured(
        f"Unknown provider {provider!r}; expected 'anthropic', 'openai' or 'local'"
    )


def llm_available() -> bool:
    try:
        get_llm()
        return True
    except LLMNotConfigured:
        return False
