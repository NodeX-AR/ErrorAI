from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys
import threading
import traceback
from types import TracebackType
from typing import Any

from .bootstrap import BootstrapStatus, ensure_model, model_status
from .config import ErrorAIConfig, autostart_enabled, load_config
from .environment import Capabilities, detect_capabilities
from .pipeline import Analyzer, Applier, Planner, Reporter, Watcher
from .providers import ModelProvider, RulesOnlyProvider, HttpApiProvider

try:
    from .providers import LlamaCppProvider
except ImportError:  # pragma: no cover - optional dependency
    LlamaCppProvider = None  # type: ignore[assignment,misc]

try:
    from .providers import OnnxProvider
except ImportError:  # pragma: no cover - optional dependency
    OnnxProvider = None  # type: ignore[assignment,misc]


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
        self._orig_idle_print_exception = None
        self._idle_hook_installed = False

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
            if self.config.model.provider in ("onnx", "llama_cpp"):
                self.bootstrap_status = ensure_model(self.config.model, explicit=False)
            else:
                self.bootstrap_status = BootstrapStatus(
                    True, "remote", "Using remote HTTP API provider; no local model needed.", None
                )
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
                    "idle_hook_installed": self._idle_hook_installed,
                },
            )
            self._initialized = True
        return self

    def _select_provider(self, status: BootstrapStatus) -> ModelProvider:
        if self.config is None:
            return RulesOnlyProvider()

        provider_name = self.config.model.provider

        if provider_name == "http_api":
            return HttpApiProvider(self.config.model)

        if not status.ready or status.model_path is None:
            return RulesOnlyProvider()

        if provider_name == "onnx":
            if OnnxProvider is None:
                return RulesOnlyProvider()
            try:
                return OnnxProvider(self.config.model, status.model_path)
            except Exception:
                return RulesOnlyProvider()

        if provider_name == "llama_cpp":
            if LlamaCppProvider is None:
                return RulesOnlyProvider()
            try:
                return LlamaCppProvider(self.config.model, status.model_path)
            except Exception:
                return RulesOnlyProvider()

        return RulesOnlyProvider()

    def _register_hooks(self) -> None:
        self._orig_sys_hook = sys.excepthook
        sys.excepthook = self._sys_excepthook
        if hasattr(threading, "excepthook"):
            self._orig_thread_hook = threading.excepthook
            threading.excepthook = self._thread_excepthook

        if self.capabilities and self.capabilities.environment == "idle":
            self._install_idle_hook()

    def _install_idle_hook(self) -> None:
        """IDLE runs user code inside idlelib.run's own exec/except loop and
        prints tracebacks itself via idlelib.run.print_exception, so the
        exception never reaches sys.excepthook. Patch that function directly.
        """
        try:
            import idlelib.run as idle_run
        except ImportError:  # pragma: no cover - not actually running under IDLE
            return

        if self._idle_hook_installed:
            return

        self._orig_idle_print_exception = idle_run.print_exception

        def patched_print_exception() -> None:
            exc_type, exc_value, exc_tb = sys.exc_info()
            handled = False
            if exc_type is not None:
                handled = self.process_exception(exc_type, exc_value, exc_tb)
            if handled:
                print("[errorai] exception intercepted")
                return
            if self._orig_idle_print_exception is not None:
                self._orig_idle_print_exception()

        idle_run.print_exception = patched_print_exception
        self._idle_hook_installed = True

    def _uninstall_idle_hook(self) -> None:
        if not self._idle_hook_installed:
            return
        try:
            import idlelib.run as idle_run
        except ImportError:  # pragma: no cover
            self._idle_hook_installed = False
            return
        if self._orig_idle_print_exception is not None:
            idle_run.print_exception = self._orig_idle_print_exception
        self._idle_hook_installed = False

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
        if not filename or "<" in filename:
            return False
        try:
            source = Path(filename).read_text(encoding="utf-8")
        except OSError:
            return False
        analysis = self.analyzer.analyze_exception(exc_type, exc_value, exc_tb)
        try:
            plan = self.planner.plan_file_fix(source, analysis["message"], frame.lineno)
        except Exception as provider_exc:
            # Never let a broken/missing model provider raise out of an
            # exception hook -- that can crash the interpreter (or IDLE's
            # subprocess) while it's in the middle of handling an exception.
            # Capture the full traceback (not just the exception repr) so
            # the real failure point inside the provider is diagnosable
            # from the log instead of just seeing a bare TypeError.
            self.reporter.log(
                "provider.error",
                {
                    "analysis": analysis,
                    "error": f"{type(provider_exc).__name__}: {provider_exc}",
                    "traceback": traceback.format_exc(),
                },
            )
            print(f"[errorai] Model provider failed ({type(provider_exc).__name__}: {provider_exc})")
            print(f"[errorai] See {self.reporter.log_path} for full traceback.")
            return False

        print(f"[errorai] Caught {analysis['type']}: {analysis['message']}")

        if not plan or plan == source:
            self.reporter.log("exception.analyzed", {"analysis": analysis, "fixed": False})
            provider_reason = getattr(self.provider, "last_error", None)
            if provider_reason:
                print(f"[errorai] No fix: provider request failed ({provider_reason})")
            else:
                print("[errorai] No automatic fix found for this error.")
            return False

        print("[errorai] Suggested fix ready (whole-file edit).")

        if self.capabilities and self.capabilities.can_prompt_user:
            try:
                answer = input("[errorai] Should I fix it [Y/N]: ").strip().lower()
            except (EOFError, OSError):
                answer = "n"
            if answer not in {"y", "yes"}:
                self.reporter.log("exception.declined", {"analysis": analysis})
                print("[errorai] Skipped.")
                return False

        result = self.applier.apply_file_change(Path(filename), plan)
        self.reporter.log(
            "exception.handled",
            {
                "analysis": analysis,
                "changed": result.changed,
                "detail": result.detail,
                "preview": result.preview,
            },
        )
        if result.preview and not result.changed:
            print(f"[errorai] {result.detail} ({result.preview})")
        elif result.changed:
            print(f"[errorai] Applied fix to {filename}")
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
            "idle_hook_installed": self._idle_hook_installed,
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
        self._uninstall_idle_hook()
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
