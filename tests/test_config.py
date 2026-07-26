import os
import tempfile
from pathlib import Path
import unittest

os.environ["ERRORAI_AUTOSTART"] = "0"

from errorai.config import load_config


class ConfigTests(unittest.TestCase):
    def test_config_parsing_with_file_override(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "pyproject.toml").write_text(
                "[tool.errorai.runtime]\nauto_watch = false\n\n[tool.errorai.model]\ncontext_window = 2048\n",
                encoding="utf-8",
            )
            (root / ".errorai.toml").write_text(
                "[runtime]\ndry_run = false\n\n[model]\ntemperature = 0.2\n",
                encoding="utf-8",
            )
            config = load_config(root)
            self.assertFalse(config.runtime.auto_watch)
            self.assertFalse(config.runtime.dry_run)
            self.assertEqual(config.model.context_window, 2048)
            self.assertEqual(config.model.temperature, 0.2)


if __name__ == "__main__":
    unittest.main()
