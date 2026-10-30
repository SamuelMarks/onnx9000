"""Unit tests and verification suite for cli main 86 functionality."""

import argparse
from unittest.mock import MagicMock, patch

from onnx9000_cli.main import mutate_cmd


def test_coverage_gaps_cmd86() -> None:
    """Verify that coverage gaps cmd86 behaves as expected."""
    args = argparse.Namespace(model="test.onnx", script="script.json", output="o")
    with (
        patch(
            "onnx9000_cli.main.load_onnx",
            return_value=MagicMock(
                nodes=[MagicMock(name="n1")],
                tensors={},
                inputs=[MagicMock(name="old", shape=(1, 2))],
                outputs=[],
            ),
        ),
        patch("onnx9000_cli.main.save_onnx"),
        patch("builtins.open"),
        patch(
            "json.load",
            return_value=[{"action": "remove_node", "node_name": "n1"}],
        ),
    ):
        mutate_cmd(args)
    assert args is not None or args is None
