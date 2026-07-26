from __future__ import annotations

import functools
import sys

from .runtime import get_runtime


def watch(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception:
            exc_type, exc_value, exc_tb = sys.exc_info()
            get_runtime().process_exception(exc_type, exc_value, exc_tb)
            raise

    return wrapper


class catch_errors:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        if exc_type is None:
            return False
        get_runtime().process_exception(exc_type, exc_value, exc_tb)
        return True


def global_activate():
    runtime = get_runtime()
    runtime.initialize()
    return runtime
