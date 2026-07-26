from .compat import catch_errors, global_activate, watch
from .runtime import configure, get_runtime

__version__ = "2.0.0"
__all__ = ["get_runtime", "configure", "watch", "catch_errors", "global_activate"]

# Import-time auto-start for zero-decorator usage.
runtime = get_runtime()
runtime.initialize()
