from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
import ast
import json
import os
import re
import socket
import urllib.error
import urllib.request
from .config import ModelConfig

# Above this size, don't send the whole file to the model -- extract just the
# enclosing function/class (or a line-window fallback) around the error
# instead. A 10,000-line file sent whole on every fix would mean huge
# latency, a token bill that blows the free tier's budget in one request,
# and real risk of the model rewriting code far from the actual bug.
WHOLE_FILE_CHAR_LIMIT = 3000


def _find_enclosing_block(source: str, lineno: int, max_lines: int = 120) -> tuple[int, int] | None:
    """Return the 1-indexed (start, end) line range of the smallest
    function/class enclosing `lineno`, clamped to max_lines. None if the
    source doesn't parse or nothing encloses the line.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    best: tuple[int, int] | None = None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            start = node.lineno
            end = getattr(node, "end_lineno", None) or start
            if start <= lineno <= end:
                if best is None or (end - start) < (best[1] - best[0]):
                    best = (start, end)
    if best is None:
        return None
    start, end = best
    if end - start + 1 > max_lines:
        half = max_lines // 2
        start = max(start, lineno - half)
        end = min(end, lineno + half)
    return start, end


def _get_patch_window(source: str, lineno: int) -> tuple[str, int, int]:
    """Return (snippet, start_idx, end_idx) -- a 0-indexed line slice of
    source around lineno, preferring the enclosing function/class and
    falling back to a fixed line-count window if that can't be determined.
    """
    lines = source.splitlines(keepends=True)
    block = _find_enclosing_block(source, lineno)
    if block:
        start_idx, end_idx = block[0] - 1, block[1]
    else:
        before, after = 40, 40
        start_idx = max(0, lineno - 1 - before)
        end_idx = min(len(lines), lineno + after)
    return "".join(lines[start_idx:end_idx]), start_idx, end_idx


class ModelProvider(ABC):
    @abstractmethod
    def suggest_patch(self, snippet: str, error_message: str) -> str | None:
        raise NotImplementedError

    def suggest_file_patch(self, source: str, error_message: str, lineno: int) -> str | None:
        """Default: reuse the single-line fix, applied in place in the file.

        Providers that can see and rewrite the whole file (e.g. HttpApiProvider)
        should override this for fixes that need a new line, not just an edit
        to the erroring one (e.g. defining a missing name).
        """
        lines = source.splitlines(keepends=True)
        if lineno < 1 or lineno > len(lines):
            return None
        original_line = lines[lineno - 1]
        fixed_line = self.suggest_patch(original_line.strip(), error_message)
        if not fixed_line or fixed_line.strip() == original_line.strip():
            return None
        indent = original_line[: len(original_line) - len(original_line.lstrip())]
        lines[lineno - 1] = f"{indent}{fixed_line.strip()}\n"
        return "".join(lines)


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
            subfolder="onnx",
            file_name="model_int8.onnx",
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


def _describe_request_error(exc: Exception) -> str:
    if isinstance(exc, urllib.error.HTTPError) and exc.code == 429:
        return "rate limited by the free tier (~2 requests/min per IP) -- wait a bit and try again"
    if isinstance(exc, (socket.timeout, TimeoutError)):
        return (
            "the model took too long to respond (larger files take longer) -- "
            "try raising http_api_timeout in .errorai.toml"
        )
    return f"{type(exc).__name__}: {exc}"


_PROSE_PREFIXES = (
    "here", "here's", "sure", "the fix", "this fixes", "note:", "note that",
    "explanation", "you should", "to fix", "i ", "i'd", "the corrected",
    "certainly", "of course", "this line", "this code", "in python",
)


def _clean_line_response(text: str | None, original_line: str) -> str | None:
    """Reduce a chat-model reply down to a single bare code line.

    Models asked for "just the fix" routinely still wrap it in a markdown
    fence, prepend "Here's the fix:", or tack on an explanatory comment.
    None of that should ever reach Applier.apply_line_change, since the user
    is agreeing to apply exactly one corrected line, not a paragraph.
    """
    if not text:
        return None
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text).strip()

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    lines = [ln for ln in lines if not ln.startswith("```")]
    if not lines:
        return None

    candidate = lines[0]
    lowered = candidate.lower()
    if any(lowered.startswith(p) for p in _PROSE_PREFIXES):
        # Try the next non-prose line instead of giving up outright.
        remaining = [ln for ln in lines[1:] if not any(ln.lower().startswith(p) for p in _PROSE_PREFIXES)]
        if not remaining:
            return None
        candidate = remaining[0]

    # Strip a trailing comment the model added that wasn't in the original --
    # that's commentary, not part of the fix.
    if "#" not in original_line and "#" in candidate:
        candidate = candidate.split("#", 1)[0].rstrip()

    return candidate or None


def _clean_file_response(text: str | None) -> str | None:
    """Strip a markdown code fence, if the model wrapped the file in one."""
    if not text:
        return None
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text).strip()
    return text or None


@dataclass
class HttpApiProvider(ModelProvider):
    """Uses a free, keyless, hosted chat-completions API instead of a local model.

    Default target is OVHcloud AI Endpoints' anonymous tier (no signup, no
    API key, ~2 req/min per IP): https://endpoints.ai.cloud.ovh.net
    Trade-off vs the local ONNX provider: the error line + message leave the
    machine over the network. Only use this provider if that's acceptable.
    """

    config: ModelConfig
    last_error: str | None = None

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        # Anonymous requests are capped at ~2 req/min per IP. A free
        # registered OVHcloud AI Endpoints account gets a higher limit --
        # set http_api_key in config or export OVH_AI_ENDPOINTS_ACCESS_TOKEN.
        key = self.config.http_api_key or os.environ.get("OVH_AI_ENDPOINTS_ACCESS_TOKEN")
        if key:
            headers["Authorization"] = f"Bearer {key}"
        return headers

    def suggest_patch(self, snippet: str, error_message: str) -> str | None:
        prompt = (
            "Fix this single line of Python code so it no longer raises the "
            "error below. Reply with ONLY the corrected line of code and "
            "nothing else -- no explanation, no markdown fences, no comments, "
            "no extra lines.\n\n"
            f"Error: {error_message}\n"
            f"Line: {snippet}\n"
            "Fixed line:"
        )
        payload = json.dumps(
            {
                "model": self.config.http_api_model,
                "messages": [
                    {"role": "system", "content": "You output only corrected code, never prose."},
                    {"role": "user", "content": prompt},
                ],
                "max_tokens": 100,
                "temperature": 0.0,
            }
        ).encode("utf-8")

        request = urllib.request.Request(
            self.config.http_api_base_url,
            data=payload,
            headers=self._headers(),
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.config.http_api_timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
            text = body["choices"][0]["message"]["content"]
        except Exception as exc:
            # Any failure here (network down, SSL/proxy quirks, malformed
            # response, endpoint changed shape, etc.) should degrade to "no
            # fix found", never bubble up and look like a crash. Stash the
            # real reason on the instance so the caller can log it if useful.
            self.last_error = _describe_request_error(exc)
            return None

        return _clean_line_response(text, snippet)

    def suggest_file_patch(self, source: str, error_message: str, lineno: int) -> str | None:
        if len(source) <= WHOLE_FILE_CHAR_LIMIT:
            return self._whole_file_patch(source, error_message, lineno)
        return self._windowed_patch(source, error_message, lineno)

    def _call_chat_api(self, prompt: str, system: str, max_tokens: int) -> str | None:
        payload = json.dumps(
            {
                "model": self.config.http_api_model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                "max_tokens": max_tokens,
                "temperature": 0.0,
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            self.config.http_api_base_url,
            data=payload,
            headers=self._headers(),
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.config.http_api_timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
            return body["choices"][0]["message"]["content"]
        except Exception as exc:
            self.last_error = _describe_request_error(exc)
            return None

    def _whole_file_patch(self, source: str, error_message: str, lineno: int) -> str | None:
        prompt = (
            "Fix the Python error below with the smallest reasonable change. "
            "Reply with ONLY the full corrected file contents -- no "
            "explanation, no markdown fences, no commentary.\n\n"
            f"Error on line {lineno}: {error_message}\n\n"
            f"File:\n{source}"
        )
        text = self._call_chat_api(prompt, "You output only corrected full source files, never prose.", 4000)
        return _clean_file_response(text)

    def _windowed_patch(self, source: str, error_message: str, lineno: int) -> str | None:
        # Large file: only send the enclosing function/class (or a bounded
        # line-window) around the error, not the whole thing. Splice the
        # corrected snippet back into the untouched rest of the file.
        lines = source.splitlines(keepends=True)
        snippet, start_idx, end_idx = _get_patch_window(source, lineno)
        window_lineno = lineno - start_idx
        prompt = (
            "Fix the Python error below with the smallest reasonable change. "
            "This is a SNIPPET taken from a much larger file -- the rest of "
            "the file continues before and after it unchanged, so only fix "
            "what's shown and leave unrelated lines exactly as they are. "
            "Reply with ONLY the corrected snippet -- no explanation, no "
            "markdown fences, no commentary.\n\n"
            f"Error on line {window_lineno} of this snippet: {error_message}\n\n"
            f"Snippet:\n{snippet}"
        )
        text = self._call_chat_api(prompt, "You output only corrected code snippets, never prose.", 1500)
        fixed_snippet = _clean_file_response(text)
        if not fixed_snippet:
            return None
        if snippet.endswith("\n") and not fixed_snippet.endswith("\n"):
            fixed_snippet += "\n"
        return "".join(lines[:start_idx]) + fixed_snippet + "".join(lines[end_idx:])
