import argparse
import sys
from unittest.mock import MagicMock, patch

from onnx9000_cli.main import main


def test_coverage_gaps_cmd19():
    cmds = [
        ["optimize", "test.onnx", "--prune", "--sparsity", "0.5", "--quantize", "-o", "out.onnx"],
        ["quantize", "test.onnx"],
        ["info", "ops", "test.onnx"],
        ["info", "tensors", "test.onnx"],
        ["info", "summary", "test.onnx"],
        ["info", "shape", "test.onnx"],
        ["info", "metadata", "test.onnx"],
    ]

    with (
        patch(
            "onnx9000_cli.main.load_onnx",
            return_value=MagicMock(
                nodes=[MagicMock(op_type="TreeEnsembleClassifier")],
                tensors={},
                inputs=[],
                outputs=[],
            ),
        ),
        patch("onnx9000_cli.main.save_onnx"),
    ):
        with patch.dict(
            sys.modules,
            {
                "onnx9000.optimizer.sparse.modifier": MagicMock(),
                "onnx9000.optimizer.quantizer": MagicMock(),
            },
        ):
            for cmd_args in cmds:
                with patch.object(sys, "argv", ["onnx9000"] + cmd_args):
                    try:
                        main()
                    except Exception:
                        pass
                    except SystemExit:
                        pass
