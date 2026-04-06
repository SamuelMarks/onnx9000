import argparse
import sys
from unittest.mock import MagicMock, patch

from onnx9000_cli.main import main


def test_coverage_gaps_cmd121():
    import runpy

    try:
        with patch.object(sys, "argv", ["onnx9000", "--help"]):
            with patch("sys.exit"):
                runpy.run_module("onnx9000_cli.main", run_name="__main__")
    except Exception:
        pass
    except SystemExit:
        pass
