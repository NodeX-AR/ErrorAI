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
    base = config.cache_dir / "models"
    if config.provider == "onnx":
        # ONNX text generation needs a full model directory (config.json,
        # tokenizer files, model.onnx, etc.), not a single downloaded file.
        return base / config.name
    return base / f"{config.name}.gguf"


def _model_ready(path: Path, config: ModelConfig) -> bool:
    if config.provider == "onnx":
        return path.exists() and path.is_dir() and any(path.iterdir())
    return path.exists()


def _download_onnx_model(config: ModelConfig, destination: Path) -> None:
    from huggingface_hub import snapshot_download

    destination.parent.mkdir(parents=True, exist_ok=True)
    downloaded_dir = Path(
        snapshot_download(
            repo_id=config.repo_id,
            cache_dir=str(config.cache_dir / "hf_cache"),
            allow_patterns=[
                "*.json",
                "*.txt",
                "onnx/model_int8.onnx",
            ],
        )
    )
    if destination.exists() or destination.is_symlink():
        return
    try:
        destination.symlink_to(downloaded_dir, target_is_directory=True)
    except OSError:
        # Symlinks can fail on some Windows setups without dev mode/admin
        # rights; fall back to a plain copy of the downloaded snapshot.
        import shutil

        shutil.copytree(downloaded_dir, destination)


def _download_gguf_model(config: ModelConfig, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(config.model_url, timeout=config.download_timeout) as response:
        with destination.open("wb") as handle:
            while True:
                chunk = response.read(1 << 20)
                if not chunk:
                    break
                handle.write(chunk)


def ensure_model(config: ModelConfig, explicit: bool = False) -> BootstrapStatus:
    path = model_path(config)
    if _model_ready(path, config):
        return BootstrapStatus(True, "model-ready", "Local model available.", path)
    if not explicit and not config.auto_bootstrap:
        return BootstrapStatus(False, "rules-only", "Auto-bootstrap is disabled.", None)
    try:
        if config.provider == "onnx":
            _download_onnx_model(config, path)
        else:
            _download_gguf_model(config, path)
    except Exception as exc:
        return BootstrapStatus(
            False,
            "rules-only",
            f"Model bootstrap failed ({exc.__class__.__name__}: {exc}); continuing without model.",
            None,
        )
    if not _model_ready(path, config):
        return BootstrapStatus(
            False, "rules-only", "Model download finished but expected files are missing.", None
        )
    return BootstrapStatus(True, "model-ready", "Model bootstrap completed.", path)


def model_status(config: ModelConfig) -> BootstrapStatus:
    path = model_path(config)
    if _model_ready(path, config):
        return BootstrapStatus(True, "model-ready", "Model files exist.", path)
    return BootstrapStatus(False, "rules-only", "Model is not installed.", None)
