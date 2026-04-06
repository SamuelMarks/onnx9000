import sys
import argparse
from unittest.mock import MagicMock, patch
from onnx9000_cli.main import main, change_batch_cmd


def test_coverage_gaps_cmd85():
    args = argparse.Namespace(model="test.onnx", size="invalid", output="o")
    with patch(
        "onnx9000_cli.main.load_onnx",
        return_value=MagicMock(
            nodes=[], tensors={}, inputs=[MagicMock(name="old", shape=(1, 2))], outputs=[]
        ),
    ):
        with patch("onnx9000_cli.main.save_onnx"):
            change_batch_cmd(args)
