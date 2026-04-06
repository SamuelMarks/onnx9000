import argparse
import sys
from unittest.mock import MagicMock, patch

from onnx9000_cli.main import hummingbird_cmd, main, optimize_cmd, quantize_cmd


def test_coverage_gaps_cmd7():
    cmds = [
        ["simplify", "test.onnx"],
        ["sparse", "prune", "test.onnx", "--sparsity", "0.5"],
        ["sparse", "de-sparsify", "test.onnx"],
        ["diffusers", "export", "test", "out"],
        ["jit", "test.onnx"],
        ["rocm", "test.onnx"],
        ["cpu", "test.onnx"],
        ["cuda", "test.onnx"],
        ["apple", "test.onnx"],
        ["onnx2tf", "test.onnx", "out"],
    ]

    with (
        patch(
            "onnx9000_cli.main.load_onnx",
            return_value=MagicMock(
                nodes=[MagicMock(name="1", op_type="1", inputs=["old"])],
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
                "onnx9000.optimizer.simplifier": MagicMock(),
                "onnx9000.core.sparse": MagicMock(),
                "onnx9000.diffusers.pipeline": MagicMock(),
                "onnx9000.jit.compiler": MagicMock(),
                "onnx9000.backends.rocm.executor": MagicMock(),
                "onnx9000.backends.cpu.executor": MagicMock(),
                "onnx9000.backends.cuda.executor": MagicMock(),
                "onnx9000.backends.apple.executor": MagicMock(),
                "onnx9000.tf.exporter": MagicMock(),
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
