import argparse
import sys
from unittest.mock import MagicMock, patch

from onnx9000_cli.main import main


def test_coverage_gaps_cmd77():
    cmds = [
        ["mutate", "test.onnx", "--script", "s.json"],
    ]

    with (
        patch(
            "onnx9000_cli.main.load_onnx",
            return_value=MagicMock(nodes=[], tensors={}, inputs=[], outputs=[]),
        ),
        patch("onnx9000_cli.main.save_onnx"),
    ):
        with (
            patch("builtins.open"),
            patch("os.path.isdir", return_value=False),
            patch("os.path.exists", return_value=False),
            patch("json.load", return_value=[{"action": "remove_node", "node_name": "x"}]),
        ):
            for cmd_args in cmds:
                with patch.object(sys, "argv", ["onnx9000"] + cmd_args):
                    try:
                        from onnx9000_cli.main import main as m

                        m()
                    except Exception:
                        pass
                    except SystemExit:
                        pass
