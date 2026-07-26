from __future__ import annotations

from dataclasses import dataclass
import os
import sys


@dataclass(frozen=True)
class Capabilities:
    environment: str
    can_watch_fs: bool
    can_apply_patches: bool
    can_prompt_user: bool


def detect_capabilities() -> Capabilities:
    env = "generic"
    interactive = bool(getattr(sys, "ps1", None) or sys.flags.interactive)
    if "ipykernel" in sys.modules or "JPY_PARENT_PID" in os.environ:
        env = "notebook"
    elif "idlelib" in sys.modules:
        env = "idle"
    elif interactive:
        env = "interactive"
    elif sys.stdin and sys.stdin.isatty():
        env = "terminal"
    else:
        env = "ide"

    can_watch_fs = env not in {"notebook"}
    can_apply_patches = env != "notebook"
    can_prompt_user = env in {"terminal", "interactive", "idle", "ide"}
    return Capabilities(
        environment=env,
        can_watch_fs=can_watch_fs,
        can_apply_patches=can_apply_patches,
        can_prompt_user=can_prompt_user,
    )
