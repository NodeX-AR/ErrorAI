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
    dry_run: bool = True
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
    provider: str = "onnx"
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
    return replace(base, **data)


def load_config(project_root: Path | None = None) -> ErrorAIConfig:
    root = (project_root or Path.cwd()).resolve()
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
dry_run = true
ignore_patterns = [".git", "__pycache__", ".env", ".venv", "venv", "node_modules", "*.lock", ".errorai"]

[model]
provider = "onnx"
name = "onnx-default-python-expert"
context_window = 4096
temperature = 0.1
auto_bootstrap = true
"""


def autostart_enabled() -> bool:
    return os.environ.get("ERRORAI_AUTOSTART", "1") != "0"
