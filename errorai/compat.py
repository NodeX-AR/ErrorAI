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
    def __init__(self, func=None):
        self.func = func

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        if exc_type is None:
            return False
        get_runtime().process_exception(exc_type, exc_value, exc_tb)
        return True

    def __call__(self, *args, **kwargs):
        if callable(self.func):
            @functools.wraps(self.func)
            def inner(*wargs, **wkwargs):
                try:
                    return self.func(*wargs, **wkwargs)
                except Exception as e:
                    get_runtime().process_exception(type(e), e, e.__traceback__)
                    raise
            return inner(*args, **kwargs)


def global_activate():
    runtime = get_runtime()
    runtime.initialize()
    return runtime


# Automatically trigger global activation on import so zero-config works!
global_activate()
