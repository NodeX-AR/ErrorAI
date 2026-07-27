from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
import os
import sys
from typing import Any, Dict


def _default_project_root() -> Path:
    """Prefer the directory of the script actually being run.

    Path.cwd() is wrong in IDLE (and many IDEs): "Run Module" does not chdir
    into the script's folder, it just sets sys.argv[0] to the script path.
    Falling back to cwd() there means the running file is almost never
    considered "inside" project_root, and every fix silently gets blocked by
    Applier.can_edit().
    """
    try:
        candidate = sys.argv[0]
        if candidate and Path(candidate).is_file():
            return Path(candidate).resolve().parent
    except Exception:
        pass
    return Path.cwd()

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - py<3.11 fallback
    import tomli as tomllib


def _default_cache_dir() -> Path:
    try:
        from platformdirs import user_cache_dir

        return Path(user_cache_dir("errorai", "ErrorAI"))
    except Exception:
        return Path.home() / ".cache" / "errorai"


@dataclass(frozen=True)
class RuntimeConfig:
    auto_watch: bool = True
    safe_mode: bool = True
    dry_run: bool = False
    project_root: Path = field(default_factory=_default_project_root)
    ignore_patterns: tuple[str, ...] = (
        ".git",
        "__pycache__",
        ".env",
        ".venv",
        "venv",
        "node_modules",
        "*.lock",
        ".errorai",
    )


@dataclass(frozen=True)
class ModelConfig:
    # Default is the free, keyless hosted API -- no local model download,
    # no onnxruntime/optimum/transformers dependency needed. Installing the
    # "pro" extra (`pip install ErrorAI[pro]`) pulls in those deps and lets
    # you set provider = "onnx" for a fully local, offline model instead.
    provider: str = "http_api"
    name: str = "onnx-qwen2.5-coder-0.5b"
    repo_id: str = "onnx-community/Qwen2.5-Coder-0.5B-Instruct"
    model_url: str = (
        "https://huggingface.co/onnx-community/Qwen2.5-Coder-0.5B-Instruct/resolve/main/"
        "onnx/model.onnx"
    )
    context_window: int = 4096
    temperature: float = 0.1
    auto_bootstrap: bool = True
    download_timeout: int = 600
    cache_dir: Path = field(default_factory=_default_cache_dir)
    # Only used when provider = "http_api". Sends the errroring line + message
    # to this endpoint instead of running a local model. Opt-in only: unlike
    # the local onnx/llama_cpp providers, this means code leaves the machine.
    # Model id must match OVHcloud's catalog exactly (case-sensitive) --
    # "qwen2.5-coder-32b-instruct" was retired from their catalog and now
    # 404s; Qwen3-Coder-30B-A3B-Instruct is their current coder-specific
    # model. Check https://www.ovhcloud.com/en/public-cloud/ai-endpoints/catalog/
    # if this ever 404s again, since OVH's lineup changes over time.
    http_api_base_url: str = "https://oai.endpoints.kepler.ai.cloud.ovh.net/v1/chat/completions"
    http_api_model: str = "Qwen3-Coder-30B-A3B-Instruct"
    http_api_timeout: float = 8.0


@dataclass(frozen=True)
class ErrorAIConfig:
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    model: ModelConfig = field(default_factory=ModelConfig)


def _read_toml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("rb") as handle:
        return tomllib.load(handle)


def _coerce_runtime(base: RuntimeConfig, values: Dict[str, Any]) -> RuntimeConfig:
    data = {}
    for key in ("auto_watch", "safe_mode", "dry_run"):
        if key in values:
            data[key] = bool(values[key])
    if "ignore_patterns" in values:
        data["ignore_patterns"] = tuple(str(v) for v in values["ignore_patterns"])
    if "project_root" in values:
        data["project_root"] = Path(values["project_root"]).resolve()
    return replace(base, **data)


def _coerce_model(base: ModelConfig, values: Dict[str, Any]) -> ModelConfig:
    data = {}
    for key in ("provider", "name", "repo_id", "model_url"):
        if key in values:
            data[key] = str(values[key])
    if "context_window" in values:
        data["context_window"] = int(values["context_window"])
    if "temperature" in values:
        data["temperature"] = float(values["temperature"])
    if "auto_bootstrap" in values:
        data["auto_bootstrap"] = bool(values["auto_bootstrap"])
    if "download_timeout" in values:
        data["download_timeout"] = int(values["download_timeout"])
    if "cache_dir" in values:
        data["cache_dir"] = Path(values["cache_dir"]).expanduser().resolve()
    if "http_api_base_url" in values:
        data["http_api_base_url"] = str(values["http_api_base_url"])
    if "http_api_model" in values:
        data["http_api_model"] = str(values["http_api_model"])
    if "http_api_timeout" in values:
        data["http_api_timeout"] = float(values["http_api_timeout"])
    return replace(base, **data)


def load_config(project_root: Path | None = None) -> ErrorAIConfig:
    root = (project_root or _default_project_root()).resolve()
    config = ErrorAIConfig(runtime=RuntimeConfig(project_root=root))
    pyproject = _read_toml(root / "pyproject.toml")
    pyproject_settings = pyproject.get("tool", {}).get("errorai", {})
    local_settings = _read_toml(root / ".errorai.toml")

    runtime_values = {}
    model_values = {}
    for source in (pyproject_settings, local_settings):
        runtime_values.update(source.get("runtime", {}))
        model_values.update(source.get("model", {}))

    return ErrorAIConfig(
        runtime=_coerce_runtime(config.runtime, runtime_values),
        model=_coerce_model(config.model, model_values),
    )


def config_template() -> str:
    return """[runtime]
auto_watch = true
safe_mode = true
# When you approve a fix (Y), it's written for real. If the write can't
# happen (permission denied, outside project_root, etc.) it automatically
# falls back to a preview instead of erroring. Set true to always preview
# only and never write.
dry_run = false
ignore_patterns = [".git", "__pycache__", ".env", ".venv", "venv", "node_modules", "*.lock", ".errorai"]

[model]
# Default: free, keyless hosted API. No local model download, no
# onnxruntime/optimum/transformers dependency required. This sends the
# erroring line + message to http_api_base_url (default: OVHcloud AI
# Endpoints' anonymous tier -- no signup, no key, ~2 req/min). Only keep
# this if you're OK with that one line leaving your machine.
provider = "http_api"
http_api_base_url = "https://oai.endpoints.kepler.ai.cloud.ovh.net/v1/chat/completions"
http_api_model = "Qwen3-Coder-30B-A3B-Instruct"
http_api_timeout = 8.0

# To run a fully local, offline model instead (nothing leaves your
# machine), install the extra:  pip install ErrorAI[pro]
# then switch this block to:
# provider = "onnx"
# name = "onnx-default-python-expert"
# context_window = 4096
# temperature = 0.1
# auto_bootstrap = true
"""


def autostart_enabled() -> bool:
    return os.environ.get("ERRORAI_AUTOSTART", "1") != "0"
