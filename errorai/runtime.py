from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import linecache
import sys
import threading
import traceback
from types import TracebackType
from typing import Any

from .bootstrap import BootstrapStatus, ensure_model, model_status
from .config import ErrorAIConfig, autostart_enabled, load_config
from .environment import Capabilities, detect_capabilities
from .pipeline import Analyzer, Applier, Planner, Reporter, Watcher
from .providers import LlamaCppProvider, ModelProvider, RulesOnlyProvider


class RuntimeManager:
    _instance: "RuntimeManager | None" = None
    _instance_lock = threading.Lock()

    def __init__(self) -> None:
        self._init_lock = threading.Lock()
        self._initialized = False
        self.config: ErrorAIConfig | None = None
        self.capabilities: Capabilities | None = None
        self.reporter: Reporter | None = None
        self.analyzer: Analyzer | None = None
        self.planner: Planner | None = None
        self.applier: Applier | None = None
        self.watcher: Watcher | None = None
        self.provider: ModelProvider = RulesOnlyProvider()
        self.bootstrap_status = BootstrapStatus(False, "rules-only", "Not initialized.", None)
        self.mode = "rules-only"
        self._orig_sys_hook = sys.excepthook
        self._orig_thread_hook = getattr(threading, "excepthook", None)

    @classmethod
    def get_instance(cls) -> "RuntimeManager":
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def initialize(self, config_override: ErrorAIConfig | None = None) -> "RuntimeManager":
        if not autostart_enabled():
            return self
        with self._init_lock:
            if self._initialized:
                return self

            self.config = config_override or load_config()
            self.capabilities = detect_capabilities()
            self.reporter = Reporter(self.config.runtime.project_root)
            self.analyzer = Analyzer()
            self.applier = Applier(self.config.runtime)
            self.bootstrap_status = ensure_model(self.config.model, explicit=False)
            self.provider = self._select_provider(self.bootstrap_status)
            self.planner = Planner(self.provider)
            self.watcher = Watcher(self.capabilities, self.reporter)
            self._register_hooks()
            watch_started = False
            if self.config.runtime.auto_watch:
                watch_started = self.watcher.start()
            self.mode = "full" if self.bootstrap_status.ready and watch_started else "analyze-only"
            if not self.bootstrap_status.ready:
                self.mode = "rules-only"
            self.reporter.log(
                "runtime.initialized",
                {
                    "environment": self.capabilities.environment,
                    "mode": self.mode,
                    "model_mode": self.bootstrap_status.mode,
                },
            )
            self._initialized = True
        return self

    def _select_provider(self, status: BootstrapStatus) -> ModelProvider:
        if not status.ready or status.model_path is None or self.config is None:
            return RulesOnlyProvider()
        if self.config.model.provider != "llama_cpp":
            return RulesOnlyProvider()
        try:
            return LlamaCppProvider(self.config.model, status.model_path)
        except Exception:
            return RulesOnlyProvider()

    def _register_hooks(self) -> None:
        self._orig_sys_hook = sys.excepthook
        sys.excepthook = self._sys_excepthook
        if hasattr(threading, "excepthook"):
            self._orig_thread_hook = threading.excepthook
            threading.excepthook = self._thread_excepthook

    def _sys_excepthook(self, exc_type, exc_value, exc_tb):
        handled = self.process_exception(exc_type, exc_value, exc_tb)
        if handled:
            print("[errorai] exception intercepted")
            return
        if self._orig_sys_hook:
            self._orig_sys_hook(exc_type, exc_value, exc_tb)

    def _thread_excepthook(self, args):
        handled = self.process_exception(args.exc_type, args.exc_value, args.exc_traceback)
        if handled:
            print("[errorai] thread exception intercepted")
            return
        if self._orig_thread_hook:
            self._orig_thread_hook(args)

    def process_exception(self, exc_type, exc_value, exc_tb: TracebackType | None) -> bool:
        if not exc_tb or not self.analyzer or not self.planner or not self.applier or not self.reporter:
            return False
        tb_list = traceback.extract_tb(exc_tb)
        if not tb_list:
            return False
        frame = tb_list[-1]
        filename = frame.filename
        line = linecache.getline(filename, frame.lineno).strip() if filename else ""
        analysis = self.analyzer.analyze_exception(exc_type, exc_value, exc_tb)
        plan = self.planner.plan_line_fix(line or frame.line or "", analysis["message"])
        if not plan or not filename or "<" in filename:
            self.reporter.log("exception.analyzed", {"analysis": analysis, "fixed": False})
            return False
        result = self.applier.apply_line_change(Path(filename), frame.lineno, plan)
        self.reporter.log(
            "exception.handled",
            {
                "analysis": analysis,
                "changed": result.changed,
                "detail": result.detail,
                "preview": result.preview,
            },
        )
        return result.changed

    def status_report(self) -> dict[str, Any]:
        cfg = self.config or load_config()
        caps = self.capabilities or detect_capabilities()
        model = model_status(cfg.model)
        return {
            "environment": caps.environment,
            "can_watch_fs": caps.can_watch_fs,
            "can_apply_patches": caps.can_apply_patches,
            "mode": self.mode,
            "model_ready": model.ready,
            "model_mode": self.bootstrap_status.mode,
            "model_detail": self.bootstrap_status.detail,
            "dry_run": cfg.runtime.dry_run,
            "project_root": str(cfg.runtime.project_root),
        }

    def install_model(self) -> BootstrapStatus:
        if self.config is None:
            self.config = load_config()
        self.bootstrap_status = ensure_model(self.config.model, explicit=True)
        return self.bootstrap_status

    def configure(self, **overrides) -> None:
        if self.config is None:
            self.config = load_config()
        runtime_values = overrides.get("runtime", {})
        model_values = overrides.get("model", {})
        self.config = ErrorAIConfig(
            runtime=replace(self.config.runtime, **runtime_values),
            model=replace(self.config.model, **model_values),
        )

    def shutdown(self) -> None:
        sys.excepthook = self._orig_sys_hook
        if hasattr(threading, "excepthook") and self._orig_thread_hook is not None:
            threading.excepthook = self._orig_thread_hook
        self._initialized = False


def get_runtime() -> RuntimeManager:
    return RuntimeManager.get_instance()


def configure(**overrides) -> RuntimeManager:
    runtime = get_runtime()
    runtime.configure(**overrides)
    return runtime


def _reset_for_tests() -> None:
    runtime = RuntimeManager._instance
    if runtime is not None:
        runtime.shutdown()
    RuntimeManager._instance = None
