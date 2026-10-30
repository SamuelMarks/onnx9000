"""Unit tests and verification suite for extended converters cmd functionality."""

import subprocess
import unittest


class TestExtendedConvertersCmd(unittest.TestCase):
    """Represents test extended converters cmd inheriting from unittest.TestCase, supporting operations such as run_cmd, test_paddle, test_keras, test_sklearn."""

    def run_cmd(
        self, fmt: int | float | str | bool | bytes | list | dict | tuple | None
    ) -> int | float | str | bool | bytes | list | dict | tuple | None:
        """Execute run cmd on TestExtendedConvertersCmd.

        Args:
            fmt: Parameter specifying fmt.

        Returns:
            Result of run cmd.
        """
        return subprocess.run(
            ["uv", "run", "onnx9000", "convert", "--from", fmt, "fake_model"],
            capture_output=True,
            text=True,
        )

    def test_paddle(self) -> None:
        """Verify that paddle behaves as expected."""
        res = self.run_cmd("paddle")
        self.assertIn("Converting from paddle", res.stdout)

    def test_keras(self) -> None:
        """Verify that keras behaves as expected."""
        res = self.run_cmd("keras")
        self.assertIn("Converting from keras", res.stdout)

    def test_sklearn(self) -> None:
        """Verify that sklearn behaves as expected."""
        res = self.run_cmd("sklearn")
        self.assertIn("Converting from sklearn", res.stdout)


if __name__ == "__main__":
    unittest.main()
