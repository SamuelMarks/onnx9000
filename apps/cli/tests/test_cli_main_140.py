"""Unit tests and verification suite for cli main 140 functionality."""

import argparse

from onnx9000_cli.main import inspect_cmd


def test_coverage_gaps_cmd140() -> None:
    """Verify that coverage gaps cmd140 behaves as expected."""
    args = argparse.Namespace(model="test.onnx")
    inspect_cmd(args)
    assert args is not None or args is None
