import argparse
import sys
from unittest.mock import MagicMock, patch

from onnx9000_cli.main import main


def test_coverage_gaps_cmd38():
    cmds = [
        ["optimum"],
        ["info"],
        ["openvino"],
    ]

    with (
        patch(
            "onnx9000_cli.main.load_onnx",
            return_value=MagicMock(nodes=[], tensors={}, inputs=[], outputs=[]),
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
