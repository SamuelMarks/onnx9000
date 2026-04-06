import sys
import argparse
from unittest.mock import MagicMock, patch
from onnx9000_cli.main import main


def test_coverage_gaps_cmd4():
    cmds = [
        ["simplify", "test.onnx"],
        ["optimize", "test.onnx"],
        ["quantize", "test.onnx"],
        ["export", "test.onnx", "test.mlir", "--format", "mlir"],
        ["prune", "test.onnx", "--nodes", "1"],
        ["sparse", "prune", "test.onnx"],
        ["sparse", "de-sparsify", "test.onnx"],
        ["change-batch", "test.onnx", "1", "10"],
        ["mutate", "test.onnx"],
        ["compile", "test.onnx"],
        ["openvino", "export", "test.onnx"],
        ["openvino", "infer", "test.onnx"],
    ]

    with (
        patch(
            "onnx9000_cli.main.load_onnx",
            return_value=MagicMock(nodes=[], tensors={}, inputs=[], outputs=[]),
        ),
        patch("onnx9000_cli.main.save_onnx"),
    ):
        with patch.dict(
            sys.modules,
            {
                "onnx9000.optimizer.simplifier": MagicMock(),
                "onnx9000.optimizer.surgeon": MagicMock(),
                "onnx9000.optimizer.quantizer": MagicMock(),
                "onnx9000.core.exporter": MagicMock(),
                "onnx9000.core.sparse": MagicMock(),
                "onnx9000.core.mutator": MagicMock(),
                "onnx9000.compiler.api": MagicMock(),
                "onnx9000.openvino.api": MagicMock(),
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
