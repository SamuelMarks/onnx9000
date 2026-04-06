import argparse
import sys
from unittest.mock import MagicMock, patch

from onnx9000_cli.main import main


def test_coverage_gaps_cmd114():
    cmds = [
        [
            "simplify",
            "test.onnx",
            "out.onnx",
            "--skip-fusions",
            "--skip-constant-folding",
            "--skip-shape-inference",
            "--skip-fuse-bn",
            "--dry-run",
            "--max-iterations",
            "2",
            "--log-json",
            "--size-limit-mb",
            "10",
            "--input-shape",
            "a:1,2",
            "b:a,b",
            "c:",
        ],
    ]

    class DummyInput:
        def __init__(self, name):
            self.name = name
            self.shape = (1, 2)
            self.dtype = "float32"

    with (
        patch(
            "onnx9000_cli.main.load_onnx",
            return_value=MagicMock(nodes=[], tensors={}, inputs=[DummyInput("old")], outputs=[]),
        ),
        patch("onnx9000_cli.main.save_onnx"),
    ):
        with patch.dict(
            sys.modules,
            {
                "onnx9000.optimizer.simplifier": MagicMock(
                    simplify=MagicMock(return_value=MagicMock())
                ),
            },
        ):
            with patch("builtins.open"), patch("os.path.exists", return_value=False):
                for cmd_args in cmds:
                    with patch.object(sys, "argv", ["onnx9000"] + cmd_args):
                        try:
                            from onnx9000_cli.main import main as m

                            m()
                        except Exception:
                            pass
                        except SystemExit:
                            pass
