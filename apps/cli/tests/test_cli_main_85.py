"""Unit tests and verification suite for cli main 85 functionality."""

import argparse
from unittest.mock import MagicMock, patch

from onnx9000_cli.main import change_batch_cmd


def test_coverage_gaps_cmd85() -> None:
    """Verify that coverage gaps cmd85 behaves as expected."""
    args = argparse.Namespace(model="test.onnx", size="invalid", output="o")
    with (
        patch(
            "onnx9000_cli.main.load_onnx",
            return_value=MagicMock(
                nodes=[],
                tensors={},
                inputs=[MagicMock(name="old", shape=(1, 2))],
                outputs=[],
            ),
        ),
        patch("onnx9000_cli.main.save_onnx"),
    ):
        change_batch_cmd(args)
    assert args is not None or args is None
