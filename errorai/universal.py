import sys
import traceback
import linecache
import functools

class ErrorAI:
    @staticmethod
    def inspect_and_fix(exc_type, exc_value, exc_tb):
        """Core logic to analyze any caught exception, suggest a fix, and prompt."""
        if not exc_tb:
            return False

        # Extract the exact frame where the exception happened
        tb_list = traceback.extract_tb(exc_tb)
        if not tb_list:
            return False
        
        last_frame = tb_list[-1]
        filename = last_frame.filename
        lineno = last_frame.lineno

        # Prevent editing built-in libraries, site-packages, or interactive strings
        if not filename or "site-packages" in filename or "<" in filename:
            return False

        code_snippet = linecache.getline(filename, lineno).strip()
        if not code_snippet:
            code_snippet = last_frame.line or "Unknown code line"

        # Display interactive prompt to user matching your vision
        print(f"\n[ERRORAI] Line {lineno} does not have the correct syntax or triggered an error.")
        print(f"    > Code: {code_snippet}")
        print(f"    > Error: {exc_type.__name__}: {exc_value}")
        
        choice = input("\nShould I fix it? [Y/N]: ").strip().upper()
        if choice == 'Y':
            # Plug in your AI backend / LLM API call here
            fixed_code = ErrorAI.mock_ai_fix(code_snippet, str(exc_value))
            
            if fixed_code and fixed_code != code_snippet:
                ErrorAI.apply_file_fix(filename, lineno, fixed_code)
                print(f"[ERRORAI] Fix applied to {filename} successfully!")
                return True
            else:
                print("[ERRORAI] Could not determine an automatic fix for this error.")
        
        return False

    @staticmethod
    def mock_ai_fix(snippet, error_msg):
        """Placeholder for AI logic. (Easily replaceable with OpenAI, Anthropic, or Ollama API calls)."""
        # Simple heuristic examples for demonstration (e.g., missing semicolon/colon)
        if "expected ':'" in error_msg or "invalid syntax" in error_msg.lower():
            if not snippet.endswith(":"):
                return snippet + ":"
        return snippet

    @staticmethod
    def apply_file_fix(filename, lineno, new_code):
        """Safely rewrites the specific line in the target file while retaining indentation."""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            if 0 <= lineno - 1 < len(lines):
                old_line = lines[lineno - 1]
                indentation = old_line[:len(old_line) - len(old_line.lstrip())]
                lines[lineno - 1] = indentation + new_code.strip() + "\n"

                with open(filename, 'w', encoding='utf-8') as f:
                    f.writelines(lines)
        except Exception as e:
            print(f"[ERRORAI] Failed to modify file: {e}")

# --- Universal Integration Methods ---

def watch(func):
    """Decorator to wrap individual functions so errors are caught safely anywhere."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception:
            exc_type, exc_value, exc_tb = sys.exc_info()
            fixed = ErrorAI.inspect_and_fix(exc_type, exc_value, exc_tb)
            if fixed:
                print("[ERRORAI] Re-running execution with the applied fix...")
                return func(*args, **kwargs)
            raise
    return wrapper

class catch_errors:
    """Context manager to wrap specific code blocks (great for IDLE or custom scopes)."""
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        if exc_type is not None:
            ErrorAI.inspect_and_fix(exc_type, exc_value, exc_tb)
            return True 
        return False

def global_activate():
    """Global system hook for terminal execution scripts."""
    def global_hook(exc_type, exc_value, exc_tb):
        traceback.print_exception(exc_type, exc_value, exc_tb)
        ErrorAI.inspect_and_fix(exc_type, exc_value, exc_tb)
    
    sys.excepthook = global_hook
