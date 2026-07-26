# ErrorAI 

`ErrorAI` is a lightweight Python package that intercepts runtime errors and syntax crashes in your terminal, prompts you interactively, and safely patches your files on the fly.

## Installation

```bash
pip install ErrorAI
```
Usage
1. Global Terminal Hook
```Python
import errorai
errorai.global_activate()

# Your code here...
print(x / y)  # Triggers interactive prompt on failure
```
2. Function Watcher Decorator
```Python
from errorai import watch

@watch
def risky_operation():
    print(undefined_variable)

risky_operation()
```
3. Context Manager (For IDLE & Custom Blocks)
```Python
from errorai import catch_errors

with catch_errors():
    # Experimental code block
    result = 10 / 0
```
