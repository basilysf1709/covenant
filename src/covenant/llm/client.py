"""LLM client abstraction for OpenAI and Anthropic."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass

from covenant.config import Settings


@dataclass
class LLMResponse:
    text: str
    usage: dict[str, int] | None = None


class LLMClient(ABC):
    @abstractmethod
    async def complete(self, messages: list[dict[str, str]], **kwargs) -> LLMResponse:
        ...

    @abstractmethod
    async def embed(self, text: str) -> list[float]:
        ...


class OpenAIClient(LLMClient):
    def __init__(self, settings: Settings):
        from openai import AsyncOpenAI

        self.client = AsyncOpenAI(api_key=settings.llm_api_key)
        self.model = settings.llm_model
        self.embedding_model = settings.embedding_model

    async def complete(self, messages: list[dict[str, str]], **kwargs) -> LLMResponse:
        resp = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            **kwargs,
        )
        choice = resp.choices[0]
        usage = {"prompt": resp.usage.prompt_tokens, "completion": resp.usage.completion_tokens} if resp.usage else None
        return LLMResponse(text=choice.message.content or "", usage=usage)

    async def embed(self, text: str) -> list[float]:
        resp = await self.client.embeddings.create(model=self.embedding_model, input=text)
        return resp.data[0].embedding


class AnthropicClient(LLMClient):
    def __init__(self, settings: Settings):
        from anthropic import AsyncAnthropic

        self.client = AsyncAnthropic(api_key=settings.llm_api_key)
        self.model = settings.llm_model or "claude-sonnet-4-5-20250929"

    async def complete(self, messages: list[dict[str, str]], **kwargs) -> LLMResponse:
        # Convert OpenAI-style messages to Anthropic format
        system = ""
        anthropic_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system = msg["content"]
            else:
                anthropic_messages.append({"role": msg["role"], "content": msg["content"]})

        kwargs.pop("response_format", None)
        resp = await self.client.messages.create(
            model=self.model,
            max_tokens=kwargs.pop("max_tokens", 1024),
            system=system,
            messages=anthropic_messages,
            **kwargs,
        )
        text = resp.content[0].text if resp.content else ""
        usage = {"prompt": resp.usage.input_tokens, "completion": resp.usage.output_tokens}
        return LLMResponse(text=text, usage=usage)

    async def embed(self, text: str) -> list[float]:
        raise NotImplementedError("Anthropic does not provide an embedding API. Use OpenAI for embeddings.")


class FakeLLMClient(LLMClient):
    """Deterministic fake for testing."""

    def __init__(self, responses: list[str] | None = None):
        self.responses = responses or []
        self._call_count = 0
        self.call_log: list[list[dict[str, str]]] = []

    async def complete(self, messages: list[dict[str, str]], **kwargs) -> LLMResponse:
        self.call_log.append(messages)
        if self._call_count < len(self.responses):
            text = self.responses[self._call_count]
        else:
            text = json.dumps({"action": "stop", "payload": {}, "reasoning": "fake done"})
        self._call_count += 1
        return LLMResponse(text=text, usage={"prompt": 10, "completion": 10})

    async def embed(self, text: str) -> list[float]:
        return [0.0] * 1536


def get_llm_client(settings: Settings) -> LLMClient:
    if settings.llm_provider == "anthropic":
        return AnthropicClient(settings)
    return OpenAIClient(settings)
