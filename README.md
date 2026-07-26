# ErrorAI v2

`ErrorAI` is a Python-only autonomous runtime that starts on import, catches exceptions, and operates with safe-by-default watch/read/write behavior.

## Quickstart

```bash
pip install ErrorAI
```

```python
import errorai
```

That import initializes a singleton runtime, registers exception hooks, and starts background watch services when supported.

## Optional setup commands


```bash
errorai init
errorai doctor
errorai install-model
```

- `init`: writes default `.errorai.toml`
- `doctor`: reports environment/runtime/model readiness and fallback mode
- `install-model`: retries local model bootstrap

## Model bootstrap strategy

- Core package does not bundle large model weights.
- On import/first use, ErrorAI attempts to bootstrap a lightweight local coding model into user cache.
- If bootstrap fails (offline/network/permissions), runtime continues in `rules-only` mode with clear status.

## Safety defaults

- Safe mode is enabled by default.
- Writes are restricted to the project root.
- Sensitive/common ignore patterns are blocked by default.
- Dry-run mode is on by default to preview edits before writing.

## Migration notes from v1 beta

- `import errorai` now auto-starts the runtime; explicit `global_activate()` is optional.
- `@watch` and `catch_errors` remain available for compatibility but are no longer required.
