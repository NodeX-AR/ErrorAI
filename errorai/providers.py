from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from .config import ModelConfig


class ModelProvider(ABC):
    @abstractmethod
    def suggest_patch(self, snippet: str, error_message: str) -> str | None:
        raise NotImplementedError


class RulesOnlyProvider(ModelProvider):
    def suggest_patch(self, snippet: str, error_message: str) -> str | None:
        lowered = error_message.lower()
        if "expected ':'" in lowered and not snippet.rstrip().endswith(":"):
            return f"{snippet.rstrip()}:"
        return None


@dataclass
class LlamaCppProvider(ModelProvider):
    config: ModelConfig
    model_path: Path
    _llm: object | None = None

    def _get_llm(self):
        if self._llm is not None:
            return self._llm
        from llama_cpp import Llama  # type: ignore

        self._llm = Llama(
            model_path=str(self.model_path),
            n_ctx=self.config.context_window,
            verbose=False,
        )
        return self._llm

    def suggest_patch(self, snippet: str, error_message: str) -> str | None:
        llm = self._get_llm()
        prompt = (
            "You are fixing Python code. Return only the fixed single line.\n"
            f"Error: {error_message}\n"
            f"Code: {snippet}\n"
            "Fix:"
        )
        response = llm(
            prompt=prompt,
            temperature=self.config.temperature,
            max_tokens=128,
            stop=["\n\n"],
        )
        text = response["choices"][0]["text"].strip()
        return text or None
