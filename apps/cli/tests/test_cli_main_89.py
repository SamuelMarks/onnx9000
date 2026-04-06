import sys
import argparse
from unittest.mock import MagicMock, patch
from onnx9000_cli.main import main, rename_input_cmd


def test_coverage_gaps_cmd89():
    args = argparse.Namespace(model="test.onnx", old="old", new="new", output="o")
    inp = MagicMock(name="old", shape=(1, 2))
    inp.name = "old"
    with patch(
        "onnx9000_cli.main.load_onnx",
        return_value=MagicMock(
            nodes=[MagicMock(name="n", op_type="t", inputs=["old"])],
            tensors={},
            inputs=[inp],
            outputs=[],
        ),
    ):
        with patch("onnx9000_cli.main.save_onnx"):
            rename_input_cmd(args)
