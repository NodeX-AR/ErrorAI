import os
import tempfile
from pathlib import Path
import unittest

os.environ["ERRORAI_AUTOSTART"] = "0"

from errorai import bootstrap
from errorai.config import ModelConfig


class BootstrapTests(unittest.TestCase):
    def test_bootstrap_fallback_to_rules_only_on_failure(self):
        with tempfile.TemporaryDirectory() as td:
            cfg = ModelConfig(cache_dir=Path(td))

            original_download = bootstrap._download_model

            def fail_download(url, destination):
                raise OSError("offline")

            bootstrap._download_model = fail_download
            try:
                status = bootstrap.ensure_model(cfg, explicit=True)
            finally:
                bootstrap._download_model = original_download

            self.assertFalse(status.ready)
            self.assertEqual(status.mode, "rules-only")
            self.assertIn("failed", status.detail.lower())


if __name__ == "__main__":
    unittest.main()
