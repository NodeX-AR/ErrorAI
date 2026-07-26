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


@dataclass
class OnnxProvider(ModelProvider):
    config: ModelConfig
    model_path: Path
    _model: object | None = None
    _tokenizer: object | None = None

    def _get_model(self):
        if self._model is not None and self._tokenizer is not None:
            return self._model, self._tokenizer

        # optimum wraps onnxruntime with a HF-style generate() API, so we
        # don't have to hand-roll KV-cache decoding against a raw .onnx graph.
        from optimum.onnxruntime import ORTModelForCausalLM  # type: ignore
        from transformers import AutoTokenizer  # type: ignore

        model_dir = self.model_path if self.model_path.is_dir() else self.model_path.parent

        self._tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self._model = ORTModelForCausalLM.from_pretrained(
            model_dir,
            use_cache=True,
        )
        return self._model, self._tokenizer

    def suggest_patch(self, snippet: str, error_message: str) -> str | None:
        model, tokenizer = self._get_model()

        prompt = (
            "You are fixing Python code. Return only the fixed single line.\n"
            f"Error: {error_message}\n"
            f"Code: {snippet}\n"
            "Fix:"
        )
        inputs = tokenizer(prompt, return_tensors="pt")
        max_new_tokens = 64
        eos_token_id = tokenizer.eos_token_id

        try:
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=self.config.temperature > 0,
                temperature=max(self.config.temperature, 1e-4),
                eos_token_id=eos_token_id,
                pad_token_id=eos_token_id,
            )
        except Exception:
            return None

        generated = output_ids[0][inputs["input_ids"].shape[-1]:]
        text = tokenizer.decode(generated, skip_special_tokens=True).strip()
        # Model can wander onto a second line/explanation; keep only the fix.
        text = text.splitlines()[0].strip() if text else ""
        return text or None
