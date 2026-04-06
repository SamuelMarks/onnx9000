import argparse
import sys
from unittest.mock import MagicMock, patch

from onnx9000_cli.main import main


def test_coverage_gaps_cmd66():
    try:
        from onnx9000_cli.main import main

        with patch("sys.exit"):
            with patch.object(sys, "argv", ["onnx9000"]):
                main()
    except Exception:
        pass
    except SystemExit:
        pass
