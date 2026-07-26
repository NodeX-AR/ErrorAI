import functools
import json
import linecache
import os
import sys
import traceback
from pathlib import Path
import threading

try:
    from importlib.resources import files
except ImportError:  # pragma: no cover
    files = None

DEFAULT_MODEL = "lightweight-debugger"
DEFAULT_MODEL_INSTALL_DIR = Path.home() / ".errorai" / "models"

class ErrorAI:
    _model_name = DEFAULT_MODEL
    _model_rules = []
    _activated = False

    @staticmethod
    def _source_model_path(model_name):
        if files is None:
            return None
        resource = files("errorai").joinpath("models").joinpath(f"{model_name}.json")
        return resource

    @classmethod
    def ensure_local_model(cls, model_name=None):
        model_name = model_name or cls._model_name
        install_dir = DEFAULT_MODEL_INSTALL_DIR
        install_dir.mkdir(parents=True, exist_ok=True)
        destination = install_dir / f"{model_name}.json"

        if not destination.exists():
            source = cls._source_model_path(model_name)
            if source is not None and source.is_file():
                destination.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")

        if destination.exists():
            try:
                payload = json.loads(destination.read_text(encoding="utf-8"))
                cls._model_rules = payload.get("rules", [])
            except Exception:
                cls._model_rules = []
        else:
            cls._model_rules = []

    @classmethod
    def configure(cls, model=DEFAULT_MODEL):
        cls._model_name = model or DEFAULT_MODEL
        cls.ensure_local_model(cls._model_name)

    @staticmethod
    def _can_prompt():
        stdin = getattr(sys, "stdin", None)
        return bool(stdin and hasattr(stdin, "isatty") and stdin.isatty())

    @classmethod
    def _auto_fix_from_model(cls, snippet, exc_type, error_msg):
        normalized = error_msg.lower()
        for rule in cls._model_rules:
            contains = [token.lower() for token in rule.get("contains", [])]
            if contains and not any(token in normalized for token in contains):
                continue
            if rule.get("type") == "append_if_missing":
                suffix = rule.get("value", "")
                if suffix and not snippet.rstrip().endswith(suffix):
                    return snippet.rstrip() + suffix
            elif rule.get("type") == "replace":
                old = rule.get("old", "")
                new = rule.get("new", "")
                if old and old in snippet:
                    return snippet.replace(old, new, 1)
            elif rule.get("type") == "wrap_name_error":
                token = rule.get("value", "")
                if token and token in str(exc_type):
                    return snippet
        return snippet

    @staticmethod
    def inspect_and_fix(exc_type, exc_value, exc_tb):
        """Core logic to analyze any caught exception, suggest a fix, and prompt."""
        if not exc_tb:
            return False

        # Extract the exact frame where the exception happened
        tb_list = traceback.extract_tb(exc_tb)
        if not tb_list:
            return False
        
        last_frame = tb_list[-1]
        filename = last_frame.filename
        lineno = last_frame.lineno

        # Prevent editing built-in libraries, site-packages, or interactive strings
        if not filename or "site-packages" in filename or "<" in filename:
            return False

        code_snippet = linecache.getline(filename, lineno).strip()
        if not code_snippet:
            code_snippet = last_frame.line or "Unknown code line"

        ErrorAI.ensure_local_model()

        print(f"\n[ERRORAI] Line {lineno} does not have the correct syntax or triggered an error.")
        print(f"    > Code: {code_snippet}")
        print(f"    > Error: {exc_type.__name__}: {exc_value}")

        fixed_code = ErrorAI.mock_ai_fix(code_snippet, str(exc_value), exc_type=exc_type)
        if not fixed_code or fixed_code == code_snippet:
            print("[ERRORAI] Could not determine an automatic fix for this error.")
            return False

        should_fix = True
        if ErrorAI._can_prompt():
            choice = input("\n[ERRORAI] Apply suggested fix? [Y/N]: ").strip().upper()
            should_fix = choice == "Y"

        if should_fix:
            ErrorAI.apply_file_fix(filename, lineno, fixed_code)
            print(f"[ERRORAI] Fix applied to {filename} successfully!")
            return True

        return False

    @staticmethod
    def mock_ai_fix(snippet, error_msg, exc_type=None):
        """Lightweight local model fixer for Python debugging."""
        fixed_from_model = ErrorAI._auto_fix_from_model(snippet, exc_type, error_msg)
        if fixed_from_model != snippet:
            return fixed_from_model

        if "expected ':'" in error_msg or "invalid syntax" in error_msg.lower():
            if not snippet.endswith(":"):
                return snippet + ":"
        return snippet

    @staticmethod
    def apply_file_fix(filename, lineno, new_code):
        """Safely rewrites the specific line in the target file while retaining indentation."""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            if 0 <= lineno - 1 < len(lines):
                old_line = lines[lineno - 1]
                indentation = old_line[:len(old_line) - len(old_line.lstrip())]
                lines[lineno - 1] = indentation + new_code.strip() + "\n"

                with open(filename, 'w', encoding='utf-8') as f:
                    f.writelines(lines)
        except Exception as e:
            print(f"[ERRORAI] Failed to modify file: {e}")

# --- Universal Integration Methods ---

def watch(func):
    """Decorator to wrap individual functions so errors are caught safely anywhere."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception:
            exc_type, exc_value, exc_tb = sys.exc_info()
            fixed = ErrorAI.inspect_and_fix(exc_type, exc_value, exc_tb)
            if fixed:
                print("[ERRORAI] Re-running execution with the applied fix...")
                return func(*args, **kwargs)
            raise
    return wrapper

class catch_errors:
    """Context manager to wrap specific code blocks (great for IDLE or custom scopes)."""
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        if exc_type is not None:
            ErrorAI.inspect_and_fix(exc_type, exc_value, exc_tb)
            return True 
        return False

def global_activate():
    """Global system hook for terminal execution scripts."""
    if ErrorAI._activated:
        return

    ErrorAI.ensure_local_model()

    def global_hook(exc_type, exc_value, exc_tb):
        traceback.print_exception(exc_type, exc_value, exc_tb)
        ErrorAI.inspect_and_fix(exc_type, exc_value, exc_tb)

    sys.excepthook = global_hook
    if hasattr(threading, "excepthook"):
        _threading_hook = threading.excepthook

        def thread_hook(args):
            _threading_hook(args)
            ErrorAI.inspect_and_fix(args.exc_type, args.exc_value, args.exc_traceback)

        threading.excepthook = thread_hook

    try:
        import tkinter

        def tk_hook(self, exc_type, exc_value, exc_tb):
            traceback.print_exception(exc_type, exc_value, exc_tb)
            ErrorAI.inspect_and_fix(exc_type, exc_value, exc_tb)

        tkinter.Tk.report_callback_exception = tk_hook
    except Exception:
        pass

    ErrorAI._activated = True


def enable():
    """Enable ErrorAI globally using lightweight local model."""
    global_activate()


if os.getenv("ERRORAI_AUTO_ACTIVATE", "1") == "1":
    enable()
