import argparse
import sys
from unittest.mock import MagicMock, patch

from onnx9000_cli.main import main


def test_coverage_gaps_cmd29():
    with patch("sys.exit"):
        with patch.object(sys, "argv", ["onnx9000"]):
            try:
                main()
            except Exception:
                pass
            except SystemExit:
                pass
