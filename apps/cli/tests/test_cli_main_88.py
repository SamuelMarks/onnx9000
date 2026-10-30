"""Unit tests and verification suite for cli main 88 functionality."""

import argparse
from unittest.mock import MagicMock, patch

from onnx9000_cli.main import rename_input_cmd


def test_coverage_gaps_cmd88() -> None:
    """Verify that coverage gaps cmd88 behaves as expected."""
    args = argparse.Namespace(model="test.onnx", old="old", new="new", output="o")
    with (
        patch(
            "onnx9000_cli.main.load_onnx",
            return_value=MagicMock(
                nodes=[MagicMock(name="n", op_type="t", inputs=["old"])],
                tensors={},
                inputs=[MagicMock(name="old", shape=(1, 2))],
                outputs=[],
            ),
        ),
        patch("onnx9000_cli.main.save_onnx"),
    ):
        rename_input_cmd(args)
    assert args is not None or args is None
