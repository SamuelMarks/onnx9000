import sys
import argparse
from unittest.mock import MagicMock, patch
from onnx9000_cli.main import inspect_cmd


def test_coverage_gaps_cmd140():
    args = argparse.Namespace(model="test.onnx")
    inspect_cmd(args)
