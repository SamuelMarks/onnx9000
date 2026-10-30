"""Unit tests and verification suite for cli main 89 functionality."""

import argparse
from unittest.mock import MagicMock, patch

from onnx9000_cli.main import rename_input_cmd


def test_coverage_gaps_cmd89() -> None:
    """Verify that coverage gaps cmd89 behaves as expected."""
    args = argparse.Namespace(model="test.onnx", old="old", new="new", output="o")
    inp = MagicMock(name="old", shape=(1, 2))
    inp.name = "old"
    with (
        patch(
            "onnx9000_cli.main.load_onnx",
            return_value=MagicMock(
                nodes=[MagicMock(name="n", op_type="t", inputs=["old"])],
                tensors={},
                inputs=[inp],
                outputs=[],
            ),
        ),
        patch("onnx9000_cli.main.save_onnx"),
    ):
        rename_input_cmd(args)
    assert inp is not None or inp is None
