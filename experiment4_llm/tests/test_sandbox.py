from __future__ import annotations

import unittest

from experiment4_llm.src.sandbox import SandboxLimits, execute_in_sandbox


class SandboxTests(unittest.TestCase):
    def test_stdin_execution(self) -> None:
        code = """
import sys
text = sys.stdin.read().strip()
print(text.upper())
"""
        result = execute_in_sandbox(
            code=code,
            input_data="hello\n",
            fn_name=None,
            timeout_seconds=2,
            memory_limit_mb=128,
        )
        self.assertFalse(result.timed_out)
        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.stdout.strip(), "HELLO")

    def test_timeout_kills_process(self) -> None:
        code = "while True:\n    pass\n"
        result = execute_in_sandbox(
            code=code,
            input_data="",
            fn_name=None,
            timeout_seconds=1,
            memory_limit_mb=64,
        )
        self.assertTrue(result.timed_out)

    def test_memory_limit(self) -> None:
        code = "x = 'a' * (400 * 1024 * 1024)\nprint(len(x))\n"
        result = execute_in_sandbox(
            code=code,
            input_data="",
            fn_name=None,
            timeout_seconds=3,
            memory_limit_mb=64,
        )
        # Platform behavior differs: either explicit MemoryError or killed process.
        self.assertNotEqual(result.returncode, 0)

    def test_function_call_mode(self) -> None:
        code = """
def add(a, b):
    return a + b
"""
        result = execute_in_sandbox(
            code=code,
            input_data=[3, 4],
            fn_name="add",
            timeout_seconds=2,
            memory_limit_mb=128,
        )
        self.assertFalse(result.timed_out)
        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.stdout.strip(), "7")


if __name__ == "__main__":
    unittest.main()
