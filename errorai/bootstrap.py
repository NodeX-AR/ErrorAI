from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import urllib.request

from .config import ModelConfig


@dataclass(frozen=True)
class BootstrapStatus:
    ready: bool
    mode: str
    detail: str
    model_path: Path | None = None


def model_path(config: ModelConfig) -> Path:
    return config.cache_dir / "models" / f"{config.name}.gguf"


def _download_model(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=5) as response, destination.open("wb") as handle:
        handle.write(response.read())


def ensure_model(config: ModelConfig, explicit: bool = False) -> BootstrapStatus:
    path = model_path(config)
    if path.exists():
        return BootstrapStatus(True, "model-ready", "Local model available.", path)

    if not explicit and not config.auto_bootstrap:
        return BootstrapStatus(False, "rules-only", "Auto-bootstrap is disabled.", None)

    try:
        _download_model(config.model_url, path)
    except Exception as exc:
        return BootstrapStatus(
            False,
            "rules-only",
            f"Model bootstrap failed ({exc.__class__.__name__}); continuing without model.",
            None,
        )
    return BootstrapStatus(True, "model-ready", "Model bootstrap completed.", path)


def model_status(config: ModelConfig) -> BootstrapStatus:
    path = model_path(config)
    if path.exists():
        return BootstrapStatus(True, "model-ready", "Model file exists.", path)
    return BootstrapStatus(False, "rules-only", "Model file is not installed.", None)
