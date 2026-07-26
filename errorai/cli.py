from __future__ import annotations

import argparse
import ast
import traceback
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


def _prompt_fix() -> bool:
    answer = input("Should I fix it [Y/N]: ").strip().lower()
    return answer in {"y", "yes"}


def _attempt_simple_syntax_fix(content: str, err: SyntaxError) -> tuple[str, bool, str]:
    if err.msg and "expected ':'" in err.msg and err.lineno:
        lines = content.splitlines()
        idx = err.lineno - 1
        if 0 <= idx < len(lines):
            line = lines[idx].rstrip()
            if line and not line.endswith(":"):
                lines[idx] = line + ":"
                return "\n".join(lines) + ("\n" if content.endswith("\n") else ""), True, "Added missing ':'"
    return content, False, "No safe automatic syntax fix available"


def _attempt_simple_runtime_fix(content: str, exc: Exception) -> tuple[str, bool, str]:
    if isinstance(exc, TypeError) and "unsupported operand type(s) for +:" in str(exc):
        # Minimal safe stub: no automatic source mutation yet.
        return content, False, "Detected TypeError (+ with incompatible types), but no safe auto-fix rule matched"
    return content, False, "No safe automatic runtime fix available"


def cmd_check(args) -> int:
    file_path = Path(args.file_path).expanduser().resolve()
    if not file_path.exists() or not file_path.is_file():
        print(f"[errorai] File not found: {file_path}")
        return 2

    source = file_path.read_text(encoding="utf-8")

    try:
        ast.parse(source, filename=str(file_path))
        print(f"[errorai] ✅ No syntax errors in {file_path}")
        return 0
    except SyntaxError as err:
        print(f"[errorai] ❌ SyntaxError in {file_path}")
        print(f"[errorai] Line {err.lineno}, Col {err.offset}: {err.msg}")
        if err.text:
            print(f"[errorai] >> {err.text.rstrip()}")

        if not _prompt_fix():
            return 1

        fixed_source, changed, detail = _attempt_simple_syntax_fix(source, err)
        print(f"[errorai] {detail}")

        if changed:
            file_path.write_text(fixed_source, encoding="utf-8")
            print(f"[errorai] Applied fix to {file_path}")
            try:
                ast.parse(fixed_source, filename=str(file_path))
                print("[errorai] ✅ Re-check passed")
                return 0
            except SyntaxError as second:
                print(f"[errorai] Still invalid after fix: {second}")
                return 1
        return 1


def cmd_run(args) -> int:
    file_path = Path(args.file_path).expanduser().resolve()
    if not file_path.exists() or not file_path.is_file():
        print(f"[errorai] File not found: {file_path}")
        return 2

    # Run syntax check first
    check_result = cmd_check(argparse.Namespace(file_path=str(file_path)))
    if check_result != 0:
        return check_result

    source = file_path.read_text(encoding="utf-8")
    globals_dict = {"__name__": "__main__", "__file__": str(file_path)}

    try:
        exec(compile(source, str(file_path), "exec"), globals_dict, globals_dict)
        return 0
    except Exception as exc:
        print(f"[errorai] ❌ Runtime error: {type(exc).__name__}: {exc}")
        if not _prompt_fix():
            return 1
        fixed_source, changed, detail = _attempt_simple_runtime_fix(source, exc)
        print(f"[errorai] {detail}")
        if changed:
            file_path.write_text(fixed_source, encoding="utf-8")
            print(f"[errorai] Applied runtime fix to {file_path}")
            print("[errorai] Re-run with: errorai run <file path>")
            return 0
        print("[errorai] No fix applied")
        traceback.print_exc()
        return 1


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

    check_parser = sub.add_parser("check", help="Check a Python file for syntax errors and offer auto-fix.")
    check_parser.add_argument("file_path", help="Path to .py file")
    check_parser.set_defaults(func=cmd_check)

    run_parser = sub.add_parser("run", help="Check then execute a Python file with runtime error interception.")
    run_parser.add_argument("file_path", help="Path to .py file")
    run_parser.set_defaults(func=cmd_run)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)
