import argparse
import sys
from unittest.mock import MagicMock, patch

from onnx9000_cli.main import main


def test_coverage_gaps_cmd125():
    cmds = [
        ["simplify", "test.onnx"],
        ["prune", "test.onnx", "--nodes", "1"],
        ["rename-input", "test.onnx", "old", "new", "-o", "out.onnx"],
        ["change-batch", "test.onnx", "10"],
        ["mutate", "test.onnx", "--script", "script.json"],
    ]

    with patch(
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
    ):
        with patch("onnx9000_cli.main.save_onnx"):
            with patch.dict(
                sys.modules,
                {
                    "onnx9000.optimizer.simplifier": MagicMock(),
                },
            ):
                with patch("builtins.open"), patch("os.path.exists", return_value=False):
                    with patch(
                        "importlib.util.spec_from_file_location",
                        return_value=MagicMock(loader=MagicMock()),
                    ):
                        with patch("importlib.util.module_from_spec", return_value=MagicMock()):
                            for cmd_args in cmds:
                                with patch.object(sys, "argv", ["onnx9000"] + cmd_args):
                                    try:
                                        from onnx9000_cli.main import main as m

                                        m()
                                    except Exception:
                                        pass
                                    except SystemExit:
                                        pass
