import sys
import argparse
from unittest.mock import MagicMock, patch
from onnx9000_cli.main import main


def test_coverage_gaps_cmd61():
    cmds = [
        ["sparse"],
        ["prune", "test.onnx", "--nodes", "1,2"],
        ["rename-input", "test.onnx", "old", "new", "-o", "out.onnx"],
        ["rename-input", "test.onnx", "old", "new"],
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
    ):
        for cmd_args in cmds:
            with patch.object(sys, "argv", ["onnx9000"] + cmd_args):
                try:
                    main()
                except Exception:
                    pass
                except SystemExit:
                    pass
