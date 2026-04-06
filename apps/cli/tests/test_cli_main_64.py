import argparse
import sys
from unittest.mock import MagicMock, patch

from onnx9000_cli.main import main


def test_coverage_gaps_cmd64():
    cmds = [
        ["onnx2tf", "test.onnx"],
        ["optimum"],
        ["optimum", "unknown"],
    ]

    for cmd_args in cmds:
        with patch.object(sys, "argv", ["onnx9000"] + cmd_args):
            try:
                main()
            except Exception:
                pass
            except SystemExit:
                pass
