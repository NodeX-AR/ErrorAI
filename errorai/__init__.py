from .compat import catch_errors, global_activate, watch
from .runtime import configure, get_runtime

__version__ = "2.1.0"
__all__ = ["get_runtime", "configure", "watch", "catch_errors", "global_activate"]

# Auto-start runtime on import
get_runtime().initialize()
