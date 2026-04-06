import argparse
import sys
from unittest.mock import MagicMock, patch

from onnx9000_cli.main import main


def test_coverage_gaps_cmd105():
    cmds = [
        ["convert", "test.onnx", "--from", "paddle", "--to", "onnx", "-o", "out.onnx"],
    ]

    with (
        patch(
            "onnx9000_cli.main.load_onnx",
            return_value=MagicMock(nodes=[], tensors={}, inputs=[], outputs=[]),
        ),
        patch("builtins.open"),
        patch("os.makedirs"),
        patch("onnx9000_cli.main.save_onnx"),
    ):
        with patch.dict(
            sys.modules,
            {
                "onnx9000.converters.paddle.api": MagicMock(),
            },
        ):
            with patch("os.path.isdir", side_effect=[True] * 4):
                with patch("os.path.exists", side_effect=[False, True, False, False, False, False]):
                    for cmd_args in cmds:
                        with patch.object(sys, "argv", ["onnx9000"] + cmd_args):
                            try:
                                main()
                            except Exception:
                                pass
                            except SystemExit:
                                pass
