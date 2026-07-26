# ErrorAI 

`ErrorAI` is a lightweight Python package that intercepts runtime errors and syntax crashes in your terminal, installs a local lightweight debugging model, and safely patches your files on the fly.

## Installation

```bash
pip install ErrorAI
```
Usage
1. Import-Only Activation (works in script terminals and IDLE-style sessions)
```Python
import errorai
# Auto-activates on import
print(x / y)  # Triggers interactive prompt on failure
```
On first import, ErrorAI installs its lightweight local model to:
`~/.errorai/models/lightweight-debugger.json`

2. Optional model selection
```Python
import errorai
errorai.ErrorAI.configure("lightweight-debugger")
```
3. Function Watcher Decorator
```Python
from errorai import watch

@watch
def risky_operation():
    print(undefined_variable)

risky_operation()
```
4. Context Manager (For IDLE & Custom Blocks)
```Python
from errorai import catch_errors

with catch_errors():
    # Experimental code block
    result = 10 / 0
```
