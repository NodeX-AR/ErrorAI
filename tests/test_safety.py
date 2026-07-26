import os
import tempfile
from pathlib import Path
import unittest

os.environ["ERRORAI_AUTOSTART"] = "0"

from errorai.config import RuntimeConfig
from errorai.pipeline import Applier


class SafetyTests(unittest.TestCase):
    def test_applier_restricts_writes_to_project_root(self):
        with tempfile.TemporaryDirectory() as root_dir, tempfile.TemporaryDirectory() as outside_dir:
            root = Path(root_dir)
            outside = Path(outside_dir)
            target = root / "main.py"
            target.write_text("print('x')\n", encoding="utf-8")
            outside_target = outside / "outside.py"
            outside_target.write_text("print('y')\n", encoding="utf-8")

            applier = Applier(RuntimeConfig(project_root=root, dry_run=False))
            self.assertTrue(applier.can_edit(target))
            self.assertFalse(applier.can_edit(outside_target))

    def test_applier_respects_sensitive_ignore_defaults(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            sensitive = root / ".env"
            sensitive.write_text("A=B\n", encoding="utf-8")
            applier = Applier(RuntimeConfig(project_root=root))
            self.assertFalse(applier.can_edit(sensitive))


if __name__ == "__main__":
    unittest.main()
