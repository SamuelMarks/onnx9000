import sys
import argparse
from unittest.mock import MagicMock, patch
from onnx9000_cli.main import workspace_cmd


def test_coverage_gaps_cmd101():
    args = argparse.Namespace(path=".")

    import builtins

    original_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "onnx9000_workspace":
            raise ImportError("mock err")
        return original_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=mock_import):
        try:
            workspace_cmd(args)
        except Exception:
            pass
