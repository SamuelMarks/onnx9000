"""Unit tests and verification suite for spec orphans cmd functionality."""

import subprocess
import unittest


class TestSpecOrphansCmd(unittest.TestCase):
    """Represents test spec orphans cmd inheriting from unittest.TestCase, supporting operations such as run_cmd, test_ort, test_olive, test_triton."""

    def run_cmd(
        self, *args: int | float | str | bool | bytes | list | dict | tuple | None
    ) -> int | float | str | bool | bytes | list | dict | tuple | None:
        """Execute run cmd on TestSpecOrphansCmd.

        Args:
            *args: Variadic positional arguments for args.

        Returns:
            Result of run cmd.
        """
        return subprocess.run(["uv", "run", "onnx9000", *args], capture_output=True, text=True)

    def test_ort(self) -> None:
        """Verify that ort behaves as expected."""
        res = self.run_cmd("ort-training", "test")
        self.assertIn("ORT Training processed", res.stdout)

    def test_olive(self) -> None:
        """Verify that olive behaves as expected."""
        res = self.run_cmd("olive-optimizer", "test")
        self.assertIn("Olive Optimizer processed", res.stdout)

    def test_triton(self) -> None:
        """Verify that triton behaves as expected."""
        res = self.run_cmd("triton-server", "test")
        self.assertIn("Triton Server processed", res.stdout)

    def test_onnx_tool(self) -> None:
        """Verify that onnx tool behaves as expected."""
        res = self.run_cmd("onnx-tool", "test")
        self.assertIn("ONNX Tool processed", res.stdout)


if __name__ == "__main__":
    unittest.main()
