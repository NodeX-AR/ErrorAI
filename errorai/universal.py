from .compat import catch_errors, global_activate, watch
from .runtime import get_runtime


class ErrorAI:
    @staticmethod
    def inspect_and_fix(exc_type, exc_value, exc_tb):
        return get_runtime().process_exception(exc_type, exc_value, exc_tb)
