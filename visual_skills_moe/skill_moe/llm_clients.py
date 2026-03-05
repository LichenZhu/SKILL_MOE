from __future__ import annotations

import os
from typing import Protocol

DEFAULT_MODEL = "gpt-4o-mini"


class LLMClient(Protocol):
    def complete(self, prompt: str, max_tokens: int = 64) -> str:
        ...


class LiteLLMClient:
    """
    Unified LLM client using litellm.  Works with any provider litellm supports.
    When OPENAI_BASE_URL is set (e.g. a LiteLLM proxy), requests are sent there.
    """

    def __init__(
        self,
        model: str | None = None,
        api_base: str | None = None,
        api_key: str | None = None,
    ) -> None:
        self.model = model or os.getenv("OPENAI_MODEL", DEFAULT_MODEL)
        self.api_base = api_base or os.getenv("OPENAI_BASE_URL")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")

    def complete(self, prompt: str, max_tokens: int = 64) -> str:
        import litellm  # lazy import

        # Prefix with openai/ when routing through a proxy so litellm
        # uses the OpenAI-compatible request format.
        litellm_model = f"openai/{self.model}" if self.api_base else self.model

        resp = litellm.completion(
            model=litellm_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            api_base=self.api_base,
            api_key=self.api_key,
        )
        return resp.choices[0].message.content.strip()


def default_llm_client() -> LLMClient | None:
    """Return a LiteLLMClient when a usable endpoint/credential is configured."""
    if os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY") or os.getenv("OPENAI_BASE_URL"):
        return LiteLLMClient()
    return None
