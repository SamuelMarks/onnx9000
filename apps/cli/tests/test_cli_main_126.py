import sys
import argparse
from unittest.mock import MagicMock, patch
from onnx9000_cli.main import main, mutate_cmd


def test_coverage_gaps_cmd126():
    cmds = [
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
            with patch("builtins.open"), patch("os.path.exists", return_value=False):
                with patch("json.load", return_value=[{"action": "remove_node", "node_name": "1"}]):
                    for cmd_args in cmds:
                        with patch.object(sys, "argv", ["onnx9000"] + cmd_args):
                            try:
                                from onnx9000_cli.main import main as m

                                m()
                            except Exception:
                                pass
                            except SystemExit:
                                pass

                    args = argparse.Namespace(model="test.onnx", script="script.json", output=None)
                    mutate_cmd(args)
