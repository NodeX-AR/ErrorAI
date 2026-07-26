import os
import sys
import unittest

os.environ["ERRORAI_AUTOSTART"] = "0"

from errorai.runtime import _reset_for_tests, get_runtime


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        os.environ["ERRORAI_AUTOSTART"] = "1"
        _reset_for_tests()
        self._original_sys_hook = sys.excepthook

    def tearDown(self):
        _reset_for_tests()
        os.environ.pop("ERRORAI_AUTOSTART", None)
        sys.excepthook = self._original_sys_hook

    def test_singleton_runtime_manager(self):
        first = get_runtime()
        second = get_runtime()
        self.assertIs(first, second)

    def test_hook_registration_on_initialize(self):
        runtime = get_runtime().initialize()
        self.assertIsNot(sys.excepthook, self._original_sys_hook)
        self.assertTrue(runtime._initialized)


if __name__ == "__main__":
    unittest.main()
