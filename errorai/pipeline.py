from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import fnmatch
import json
import threading
from typing import Any

from .config import RuntimeConfig
from .environment import Capabilities
from .providers import ModelProvider


class Reporter:
    def __init__(self, project_root: Path):
        self.log_path = project_root / ".errorai" / "logs" / "operations.log"
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def log(self, event: str, payload: dict[str, Any]) -> None:
        record = {
            "time": datetime.now(timezone.utc).isoformat(),
            "event": event,
            "payload": payload,
        }
        with self._lock, self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")


class Analyzer:
    def analyze_exception(self, exc_type, exc_value, exc_tb) -> dict[str, Any]:
        return {
            "type": getattr(exc_type, "__name__", str(exc_type)),
            "message": str(exc_value),
        }


class Planner:
    def __init__(self, provider: ModelProvider):
        self.provider = provider

    def plan_line_fix(self, line: str, error_message: str) -> str | None:
        return self.provider.suggest_patch(line, error_message)


@dataclass
class ApplyResult:
    changed: bool
    detail: str
    preview: str | None = None


class Applier:
    def __init__(self, config: RuntimeConfig):
        self.config = config
        self.project_root = config.project_root.resolve()

    def _is_ignored(self, path: Path) -> bool:
        rel = str(path)
        for pattern in self.config.ignore_patterns:
            if fnmatch.fnmatch(path.name, pattern) or fnmatch.fnmatch(rel, pattern):
                return True
            if pattern in path.parts:
                return True
        return False

    def _is_within_root(self, path: Path) -> bool:
        try:
            path.resolve().relative_to(self.project_root)
        except ValueError:
            return False
        return True

    def can_edit(self, file_path: Path) -> bool:
        path = file_path.resolve()
        return self._is_within_root(path) and not self._is_ignored(path)

    def apply_line_change(self, file_path: Path, lineno: int, new_line: str) -> ApplyResult:
        path = file_path.resolve()
        if not self.can_edit(path):
            return ApplyResult(False, "Blocked by safe mode restrictions.")
        lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
        if lineno < 1 or lineno > len(lines):
            return ApplyResult(False, "Line number out of range.")
        old_line = lines[lineno - 1]
        indentation = old_line[: len(old_line) - len(old_line.lstrip())]
        candidate = f"{indentation}{new_line.strip()}\n"
        preview = f"- {old_line.rstrip()}\n+ {candidate.rstrip()}"
        if self.config.dry_run:
            return ApplyResult(False, "Dry-run mode enabled; no write applied.", preview=preview)
        lines[lineno - 1] = candidate
        path.write_text("".join(lines), encoding="utf-8")
        return ApplyResult(True, "Edit applied.", preview=preview)


class Watcher:
    def __init__(self, capabilities: Capabilities, reporter: Reporter):
        self.capabilities = capabilities
        self.reporter = reporter
        self.started = False

    def start(self) -> bool:
        if not self.capabilities.can_watch_fs:
            self.reporter.log("watcher.skipped", {"reason": "environment unsupported"})
            return False
        try:
            from watchdog.observers import Observer  # type: ignore  # noqa: F401
        except Exception:
            self.reporter.log("watcher.skipped", {"reason": "watchdog unavailable"})
            return False
        self.started = True
        self.reporter.log("watcher.started", {})
        return True
