from __future__ import annotations

import argparse
from pathlib import Path

from .bootstrap import ensure_model
from .config import config_template, load_config
from .runtime import get_runtime


def cmd_init(args) -> int:
    root = Path.cwd()
    config_path = root / ".errorai.toml"
    if not config_path.exists() or args.force:
        config_path.write_text(config_template(), encoding="utf-8")
        print(f"[errorai] Wrote {config_path}")
    else:
        print(f"[errorai] Config already exists: {config_path}")
    return 0


def cmd_doctor(_args) -> int:
    runtime = get_runtime().initialize()
    status = runtime.status_report()
    print("ErrorAI Doctor")
    for key in (
        "environment",
        "can_watch_fs",
        "can_apply_patches",
        "mode",
        "model_ready",
        "model_mode",
        "model_detail",
        "dry_run",
        "project_root",
    ):
        print(f"- {key}: {status[key]}")
    return 0


def cmd_install_model(_args) -> int:
    cfg = load_config()
    status = ensure_model(cfg.model, explicit=True)
    print(status.detail)
    return 0 if status.ready else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="errorai")
    sub = parser.add_subparsers(dest="command", required=True)

    init_parser = sub.add_parser("init", help="Create default .errorai.toml config.")
    init_parser.add_argument("--force", action="store_true", help="Overwrite existing config.")
    init_parser.set_defaults(func=cmd_init)

    doctor_parser = sub.add_parser("doctor", help="Check runtime readiness and fallback state.")
    doctor_parser.set_defaults(func=cmd_doctor)

    install_parser = sub.add_parser("install-model", help="Install or retry local model bootstrap.")
    install_parser.set_defaults(func=cmd_install_model)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)
