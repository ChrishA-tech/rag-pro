from __future__ import annotations

from typing import Protocol


class LLM(Protocol):
    def complete(self, system: str, user: str) -> str: ...


class AnthropicLLM:
    def __init__(self, model: str, max_tokens: int = 700):
        try:
            import anthropic
        except ImportError as e:  # pragma: no cover
            raise ImportError("needs: pip install 'rag-pro[llm]'") from e
        self.client = anthropic.Anthropic()
        self.model, self.max_tokens = model, max_tokens

    def complete(self, system: str, user: str) -> str:
        r = self.client.messages.create(model=self.model, max_tokens=self.max_tokens, system=system,
                                        messages=[{"role": "user", "content": user}])
        return "".join(b.text for b in r.content if getattr(b, "type", "") == "text")
