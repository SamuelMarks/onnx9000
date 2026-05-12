import argparse
from unittest.mock import patch

from onnx9000_cli.main import triton_cmd


def test_triton_cmd():
    args = argparse.Namespace(model="dummy.onnx")
    with patch("builtins.print") as mock_print:
        triton_cmd(args)
    mock_print.assert_any_call("Generating Triton code from dummy.onnx...")
    mock_print.assert_any_call("@triton.jit")
