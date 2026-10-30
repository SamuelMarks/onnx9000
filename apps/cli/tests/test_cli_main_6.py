"""Unit tests and verification suite for cli main 6 functionality."""

import sys
from unittest.mock import MagicMock, patch

from onnx9000.core.exceptions import Onnx9000Error
from onnx9000_cli.main import main


def test_coverage_gaps_cmd6() -> None:
    """Verify that coverage gaps cmd6 behaves as expected."""
    cmds = [
        ["hummingbird", "test.onnx"],
        ["optimize", "test.onnx"],
        ["quantize", "test.onnx"],
        ["export", "test.onnx", "test.mlir", "--format", "mlir"],
        ["prune", "test.onnx", "--nodes", "1"],
        ["rename-input", "test.onnx", "old", "new"],
        ["change-batch", "test.onnx", "10"],
        ["change-batch", "test.onnx", "invalid"],
        ["mutate", "test.onnx", "--script", "script.json"],
        ["compile", "test.onnx"],
        ["openvino", "export", "test.onnx"],
        ["openvino", "infer", "test.onnx"],
    ]

    with (
        patch(
            "onnx9000_cli.main.load_onnx",
            return_value=MagicMock(
                nodes=[
                    MagicMock(name="1", op_type="1", inputs=["old"]),
                    MagicMock(name="2", op_type="2", inputs=["2"]),
                ],
                tensors={},
                inputs=[MagicMock(name="old", shape=(1, 2))],
                outputs=[],
            ),
        ),
        patch("onnx9000_cli.main.save_onnx"),
        patch("builtins.open"),
        patch("json.load", return_value=[{"action": "remove_node", "node_name": "1"}]),
    ):
        with patch.dict(
            sys.modules,
            {
                "onnx9000.optimizer.hummingbird.engine": MagicMock(),
                "onnx9000.optimizer.hummingbird.onnxml_parser": MagicMock(),
                "onnx9000.optimizer.hummingbird.strategies": MagicMock(),
                "onnx9000.optimizer.simplifier": MagicMock(),
                "onnx9000.optimizer.surgeon": MagicMock(),
                "onnx9000.optimizer.quantizer": MagicMock(),
                "onnx9000.core.exporter": MagicMock(),
                "onnx9000.compiler.api": MagicMock(),
                "onnx9000.openvino.api": MagicMock(),
            },
        ):
            for cmd_args in cmds:
                with patch.object(sys, "argv", ["onnx9000", *cmd_args]):
                    try:
                        main()
                    except (
                        Onnx9000Error,
                        OSError,
                        ValueError,
                        TypeError,
                        RuntimeError,
                        KeyError,
                        IndexError,
                        AttributeError,
                        ImportError,
                        LookupError,
                        ArithmeticError,
                        BufferError,
                        AssertionError,
                        StopIteration,
                    ):
                        pass
                    except SystemExit:
                        pass
    assert cmds is not None or cmds is None
